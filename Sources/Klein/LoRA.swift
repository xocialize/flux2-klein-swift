// In-engine DiT-LoRA for the FLUX.2-klein-4B transformer (Flux2Transformer).
//
// Loads one or more LoRAs and applies them to `Flux2Transformer` at runtime (activation-path
// adapter, NOT fused into the base weights — so the low-rank term survives bf16/int4). This is
// R3 step 1 for the self-trained pose RefControl LoRA; the target adapter is the ai-toolkit
// BFL-fused output (e.g. thedeoxen/refcontrol-FLUX.2-klein-4B-reference-depth-lora).
//
// Why this exists instead of `MLXLMCommon.LoRAContainer`: that container is typed against
// `LanguageModel`, so a DiT can't call it. The underlying primitives (`LoRALinear.from`, the
// replace-children pattern, `Module.update(parameters:)`) are generic, so we mirror the
// container's replace/update logic here, klein-DiT-typed. `LoRALinear.from` auto-dispatches to
// `QLoRALinear` when the base is a `QuantizedLinear`, so int4 works for free — no special case.
//
// TWO LoRA dialects are handled:
//   * BFL-fused (ai-toolkit): keys like `diffusion_model.double_blocks.<i>.img_attn.qkv.lora_A`
//     — the img/txt attention qkv is ONE fused projection; we split its lora_B output axis into
//     thirds (to_q/to_k/to_v resp. add_q/add_k/add_v_proj) sharing the same lora_A. Exact: a
//     fused qkv is [W_q;W_k;W_v] stacked on the output axis, and the delta B·A splits row-wise
//     with a shared A.
//   * diffusers-split (already-separated q/k/v, linear_in/out, …): mapped 1:1, no split.
//
// klein's Flux2Transformer has TWO block arrays with different attention shapes:
//   * `transformerBlocks`      (double-stream, SPLIT projections — Flux2Attention)
//   * `singleTransformerBlocks`(single-stream, FUSED qkv+mlp — Flux2ParallelSelfAttention;
//     its `to_qkv_mlp_proj` is NOT split here — BFL `linear1` maps to it verbatim).
// …and top-level Linears outside the blocks (the in/out embedders, the timestep embedder, the
// three modulations, the final projection), which fal's PEFT exports also adapt (v0.10.0).
//
// Every tensor must map (v0.10.0, AB-A-0192): a key the applicator can't place throws
// `unknownKeys` instead of being skipped. Before, fal's adapters silently lost their top-level
// targets, and an adapter could apply partially with nothing said.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public enum KleinLoRA {

    public enum LoRAError: Error, LocalizedError {
        case incompletePair(String)
        case noTargets(String)
        /// Tensors the applicator can't place on the klein DiT (file name, keys).
        case unknownKeys(String, [String])

        public var errorDescription: String? {
            switch self {
            case .incompletePair(let p):
                return "LoRA layer \(p) is missing its lora_A or lora_B tensor."
            case .noTargets(let url):
                return "No recognizable klein LoRA tensors found in \(url)."
            case .unknownKeys(let file, let keys):
                let shown = keys.prefix(6).joined(separator: ", ")
                return "LoRA \(file) has \(keys.count) tensor(s) that don't map onto the FLUX.2-klein-4B DiT "
                    + "(\(shown)\(keys.count > 6 ? ", …" : "")). It is for another model, or a dialect this "
                    + "package doesn't read."
            }
        }
    }

    /// A/B factor suffixes across the LoRA dialects in the wild. `A` is the [rank, in] down
    /// projection; `B` is the [out, rank] up projection.
    private static let aSuffixes = [
        ".lora_A.weight", ".lora_A.default.weight", ".lora.down.weight", ".lora_down.weight",
    ]
    private static let bSuffixes = [
        ".lora_B.weight", ".lora_B.default.weight", ".lora.up.weight", ".lora_up.weight",
    ]
    private static let alphaSuffix = ".alpha"

    /// BFL / diffusers double-block submodule -> klein block-relative Linear path(s). A `third`
    /// of nil is a straight rename; 0/1/2 selects the q/k/v (resp. add_q/k/v) slice of a fused
    /// qkv lora_B. Entries with `third` set MUST share the same source key (the fused qkv).
    private static let doubleMap: [String: [(rel: String, third: Int?)]] = [
        // BFL-fused
        "img_attn.qkv": [("attn.to_q", 0), ("attn.to_k", 1), ("attn.to_v", 2)],
        "img_attn.proj": [("attn.to_out", nil)],
        "img_mlp.0": [("ff.linear_in", nil)],
        "img_mlp.2": [("ff.linear_out", nil)],
        "txt_attn.qkv": [("attn.add_q_proj", 0), ("attn.add_k_proj", 1), ("attn.add_v_proj", 2)],
        "txt_attn.proj": [("attn.to_add_out", nil)],
        "txt_mlp.0": [("ff_context.linear_in", nil)],
        "txt_mlp.2": [("ff_context.linear_out", nil)],
        // diffusers-split (already separated) — identity map, no surgery
        "attn.to_q": [("attn.to_q", nil)],
        "attn.to_k": [("attn.to_k", nil)],
        "attn.to_v": [("attn.to_v", nil)],
        "attn.to_out": [("attn.to_out", nil)],
        "attn.to_out.0": [("attn.to_out", nil)],
        "attn.add_q_proj": [("attn.add_q_proj", nil)],
        "attn.add_k_proj": [("attn.add_k_proj", nil)],
        "attn.add_v_proj": [("attn.add_v_proj", nil)],
        "attn.to_add_out": [("attn.to_add_out", nil)],
        "ff.linear_in": [("ff.linear_in", nil)],
        "ff.linear_out": [("ff.linear_out", nil)],
        "ff_context.linear_in": [("ff_context.linear_in", nil)],
        "ff_context.linear_out": [("ff_context.linear_out", nil)],
    ]

    /// BFL / diffusers single-block submodule -> klein block-relative Linear path. The single
    /// block's projection is fused qkv+mlp in BOTH conventions, so `linear1` is a pure rename to
    /// `attn.to_qkv_mlp_proj` — NO split.
    private static let singleMap: [String: [(rel: String, third: Int?)]] = [
        "linear1": [("attn.to_qkv_mlp_proj", nil)],
        "linear2": [("attn.to_out", nil)],
        "attn.to_qkv_mlp_proj": [("attn.to_qkv_mlp_proj", nil)],
        "attn.to_out": [("attn.to_out", nil)],
    ]

    /// BFL / diffusers top-level Linear (outside the blocks) -> klein module path. The BFL names
    /// follow diffusers' FLUX.2 conversion, which renames these without reordering their rows.
    /// BFL's `final_layer.adaLN_modulation.1` is absent on purpose: diffusers swaps its scale and
    /// shift halves, so its delta can't be a rename.
    private static let topLevelMap: [String: String] = [
        // BFL
        "img_in": "x_embedder",
        "txt_in": "context_embedder",
        "time_in.in_layer": "time_guidance_embed.linear_1",
        "time_in.out_layer": "time_guidance_embed.linear_2",
        "double_stream_modulation_img.lin": "double_stream_modulation_img.linear",
        "double_stream_modulation_txt.lin": "double_stream_modulation_txt.linear",
        "single_stream_modulation.lin": "single_stream_modulation.linear",
        "final_layer.linear": "proj_out",
        // diffusers
        "x_embedder": "x_embedder",
        "context_embedder": "context_embedder",
        "time_guidance_embed.timestep_embedder.linear_1": "time_guidance_embed.linear_1",
        "time_guidance_embed.timestep_embedder.linear_2": "time_guidance_embed.linear_2",
        "double_stream_modulation_img.linear": "double_stream_modulation_img.linear",
        "double_stream_modulation_txt.linear": "double_stream_modulation_txt.linear",
        "single_stream_modulation.linear": "single_stream_modulation.linear",
        "norm_out.linear": "norm_out.linear",
        "proj_out": "proj_out",
    ]

    /// Expand a LoRA source base (suffix already stripped) into the klein full module path(s) it
    /// adapts, with an optional qkv-slice index. Handles the `diffusion_model.`/`transformer.`
    /// prefixes, both `double_blocks`/`transformer_blocks` and
    /// `single_blocks`/`single_transformer_blocks` block names, and the top-level Linears.
    /// Returns [] for anything unrecognized; `factors` refuses those (`unknownKeys`).
    static func expand(base: String) -> [(path: String, third: Int?)] {
        var s = base
        // Strip known container prefixes, outermost first: PEFT's `base_model.model.` (as emitted
        // by diffusers/peft LoRA exports, e.g. fal's), then the BFL `diffusion_model.` /
        // diffusers `transformer.` roots. A nested `base_model.model.diffusion_model.…` collapses
        // in one pass because the list is ordered outer→inner.
        for p in ["base_model.model.", "diffusion_model.", "transformer."] where s.hasPrefix(p) {
            s.removeFirst(p.count)
        }
        if let top = topLevelMap[s] { return [(top, nil)] }
        let comps = s.split(separator: ".").map(String.init)
        guard comps.count >= 3, let idx = Int(comps[1]) else { return [] }
        let head = comps[0]
        let rest = comps[2...].joined(separator: ".")
        if head == "double_blocks" || head == "transformer_blocks" {
            guard let rels = doubleMap[rest] else { return [] }
            return rels.map { ("transformer_blocks.\(idx).\($0.rel)", $0.third) }
        }
        if head == "single_blocks" || head == "single_transformer_blocks" {
            guard let rels = singleMap[rest] else { return [] }
            return rels.map { ("single_transformer_blocks.\(idx).\($0.rel)", $0.third) }
        }
        return []
    }

    /// The tensors in an adapter file the applicator can't place: keys that are neither a lora A/B
    /// factor nor an alpha, and A/B bases that map to no klein module. Empty = the file maps
    /// completely. Pure key logic, so a registry's headers can be checked without the weights.
    public static func unmappedKeys(_ keys: [String]) -> [String] {
        var unknown: [String] = []
        for key in keys {
            if let base = factorBase(key) {
                if expand(base: base).isEmpty { unknown.append(key) }
            } else if !key.hasSuffix(alphaSuffix) {
                unknown.append(key)
            }
        }
        return unknown.sorted()
    }

    /// The module base of an A or B factor key, or nil when `key` is neither.
    static func factorBase(_ key: String) -> String? {
        for s in aSuffixes + bSuffixes where key.hasSuffix(s) { return String(key.dropLast(s.count)) }
        return nil
    }

    /// Strip the `transformer_blocks.<i>.` / `single_transformer_blocks.<i>.` prefix off a full
    /// model path to get the block-relative module key (e.g. `attn.to_q`). Used only to build the
    /// `namedModules()` match set; ranks/params are keyed by the full path so the `attn.to_out`
    /// key shared by both arrays can never cross-wire. nil for a top-level path.
    static func blockRelative(_ path: String) -> String? {
        for head in ["single_transformer_blocks", "transformer_blocks"] {
            let prefix = head + "."
            guard path.hasPrefix(prefix) else { continue }
            let after = path.dropFirst(prefix.count)          // "<i>.<rel>"
            guard let dot = after.firstIndex(of: ".") else { return nil }
            return String(after[after.index(after: dot)...])
        }
        return nil
    }

    /// One LoRA's per-target low-rank factors, keyed by klein full model path. `a` is [in, rank];
    /// `b` is [rank, out] with the layer's effective scale baked in. For a split qkv, `b` is the
    /// slice for this q/k/v third and `a` is the shared down-projection.
    struct Factors { var a: MLXArray; var b: MLXArray }

    static func factors(
        from url: URL, dtype: DType, strength: Float
    ) throws -> [String: Factors] {
        let raw = try MLX.loadArrays(url: url)
        func match(_ key: String, _ suffixes: [String]) -> String? {
            for s in suffixes where key.hasSuffix(s) {
                return String(key.dropLast(s.count))   // BFL/diffusers base, pre-expand
            }
            return nil
        }
        var aMats: [String: MLXArray] = [:]
        var bMats: [String: MLXArray] = [:]
        var alphas: [String: MLXArray] = [:]
        var unknown: [String] = []
        for (key, value) in raw {
            if let base = match(key, aSuffixes) { aMats[base] = value }
            else if let base = match(key, bSuffixes) { bMats[base] = value }
            else if key.hasSuffix(alphaSuffix) {
                alphas[String(key.dropLast(alphaSuffix.count))] = value
            } else {
                unknown.append(key)
            }
        }
        guard !aMats.isEmpty else { throw LoRAError.noTargets(url.path) }
        for base in bMats.keys where aMats[base] == nil { throw LoRAError.incompletePair(base) }

        var out: [String: Factors] = [:]
        for (base, aMat) in aMats {
            let targets = expand(base: base)
            if targets.isEmpty { unknown.append(base); continue }
            guard let bMat = bMats[base] else { throw LoRAError.incompletePair(base) }
            let rank = aMat.dim(0)            // lora_A is [rank, in]
            // alpha/rank when present; alpha-less adapters apply at scale 1.0.
            let scale = strength * (alphas[base].map { $0.item(Float.self) / Float(rank) } ?? 1.0)
            let aT = aMat.T.asType(dtype)              // [in, rank] (shared across a qkv split)
            let bT = (scale * bMat.T).asType(dtype)    // [rank, out] (scale baked in pre-split)
            for (path, third) in targets {
                let bSlice: MLXArray
                if let t = third {
                    // Fused qkv: b is [rank, 3·D]; slice the output axis into equal thirds. D is
                    // derived from the tensor (= model hidden, 3072 for klein-4B) — never hardcoded.
                    let d = bT.dim(1) / 3
                    bSlice = bT[0..., (t * d)..<((t + 1) * d)]
                } else {
                    bSlice = bT
                }
                out[path] = Factors(a: aT, b: bSlice)
            }
        }
        guard unknown.isEmpty else { throw LoRAError.unknownKeys(url.lastPathComponent, unknown.sorted()) }
        return out
    }

    /// Rank-stacked LoRA parameters ready to attach (see `combined`).
    struct Combined {
        /// Keyed by full path + `.lora_a` / `.lora_b`.
        var params: [String: MLXArray]
        /// Combined rank per full path.
        var ranks: [String: Int]
        /// Block-relative keys, for the per-block `namedModules()` filter.
        var blockKeys: Set<String>
        /// Full paths of top-level Linears.
        var topLevelPaths: Set<String>
    }

    /// Combine one or more LoRAs into per-module `lora_a`/`lora_b` parameters by rank-stacking:
    /// concat the `a` factors along rank (axis 1) and the `b` factors along rank (axis 0), so the
    /// LoRALinear contribution is the exact SUM of each adapter's low-rank term.
    static func combined(_ loras: [(url: URL, strength: Float)], dtype: DType) throws -> Combined {
        let perLoRA = try loras.map { try factors(from: $0.url, dtype: dtype, strength: $0.strength) }
        var paths = Set<String>()
        perLoRA.forEach { paths.formUnion($0.keys) }

        var out = Combined(params: [:], ranks: [:], blockKeys: [], topLevelPaths: [])
        for path in paths {
            let present = perLoRA.compactMap { $0[path] }   // same order for a and b
            let aCat = present.count == 1 ? present[0].a : concatenated(present.map(\.a), axis: 1)
            let bCat = present.count == 1 ? present[0].b : concatenated(present.map(\.b), axis: 0)
            out.params[path + ".lora_a"] = aCat
            out.params[path + ".lora_b"] = bCat
            out.ranks[path] = aCat.dim(1)
            if let rel = blockRelative(path) { out.blockKeys.insert(rel) } else { out.topLevelPaths.insert(path) }
        }
        return out
    }

    /// Summary of an `apply`: how many Linears were adapted in each block array and outside them.
    public struct AppliedSummary {
        public var doubleTargets: Int
        public var singleTargets: Int
        public var topLevelTargets: Int = 0
        public var total: Int { doubleTargets + singleTargets + topLevelTargets }
    }

    /// Apply one or more LoRAs to the klein DiT in place (runtime adapter). The low-rank term is
    /// added in the activation path, so it survives bf16 AND int4 (LoRALinear.from dispatches to
    /// QLoRALinear on a quantized base). Overhead is one rank-r matmul per adapted Linear.
    @discardableResult
    public static func apply(
        loRAs loras: [(url: URL, strength: Float)],
        to model: Flux2Transformer,
        dtype: DType = .bfloat16
    ) throws -> AppliedSummary {
        let c = try combined(loras, dtype: dtype)
        var dbl = 0, sgl = 0, top = 0
        replaceTargets(in: model, keys: c.blockKeys, topLevel: c.topLevelPaths) { path, linear in
            guard let r = c.ranks[path] else { return nil }   // key shared by both arrays: skip if no param
            if path.hasPrefix("single_transformer_blocks") { sgl += 1 }
            else if path.hasPrefix("transformer_blocks") { dbl += 1 }
            else { top += 1 }
            return LoRALinear.from(linear: linear, rank: r, scale: 1.0)
        }
        try model.update(parameters: ModuleParameters.unflattened(c.params), verify: .noUnusedKeys)
        return AppliedSummary(doubleTargets: dbl, singleTargets: sgl, topLevelTargets: top)
    }

    /// Convenience: apply a single LoRA at `strength` (1.0 = the diffusers `load_lora_weights`
    /// default).
    @discardableResult
    public static func apply(
        loRA url: URL,
        to model: Flux2Transformer,
        dtype: DType = .bfloat16,
        strength: Float = 1.0
    ) throws -> AppliedSummary {
        try apply(loRAs: [(url, strength)], to: model, dtype: dtype)
    }

    /// Replace each targeted leaf `Linear` in every block of BOTH arrays, and each targeted
    /// top-level Linear, with a transformed module. The transform receives the full path (so it can
    /// look up a per-layer rank) and may return nil to skip.
    static func replaceTargets(
        in model: Flux2Transformer,
        keys: Set<String>,
        topLevel: Set<String> = [],
        _ transform: (_ path: String, _ linear: Linear) -> Module?
    ) {
        if !topLevel.isEmpty {
            let modules = Dictionary(model.namedModules(), uniquingKeysWith: { first, _ in first })
            var update: [(String, Module)] = []
            for path in topLevel.sorted() {
                if let linear = modules[path] as? Linear, let m = transform(path, linear) {
                    update.append((path, m))
                }
            }
            if !update.isEmpty { model.update(modules: .unflattened(update)) }
        }
        func process<B: Module>(_ blocks: [B], _ prefix: String) {
            for (i, block) in blocks.enumerated() {
                var update: [(String, Module)] = []
                for (key, child) in block.namedModules() where keys.contains(key) {
                    if let linear = child as? Linear,
                        let m = transform("\(prefix).\(i).\(key)", linear) {
                        update.append((key, m))
                    }
                }
                if !update.isEmpty { block.update(modules: .unflattened(update)) }
            }
        }
        process(model.transformerBlocks, "transformer_blocks")
        process(model.singleTransformerBlocks, "single_transformer_blocks")
    }
}

/// Stateful LoRA hot-swapper for a resident klein DiT: switch the active adapter set (e.g. a
/// pose LoRA picked from a dropdown) WITHOUT reloading the base. `set(_:)` detaches the current
/// adapter — restoring the pristine base modules captured on first use — then applies the new
/// combo. Because `LoRALinear` shares the base weight `MLXArray` by reference, neither the
/// capture nor a swap duplicates base weights: only the small lora_a/lora_b factors are added and
/// freed. Not thread-safe; drive it from the same actor that owns the DiT.
public final class KleinLoRASwapper {
    private let model: Flux2Transformer
    private let dtype: DType
    /// full path -> pristine base module (Linear/QuantizedLinear).
    private var pristine: [String: Module] = [:]
    /// block-relative keys currently realized as a LoRALinear.
    private var applied: Set<String> = []
    /// top-level full paths currently realized as a LoRALinear.
    private var appliedTopLevel: Set<String> = []

    public init(model: Flux2Transformer, dtype: DType = .bfloat16) {
        self.model = model
        self.dtype = dtype
    }

    /// The block-relative keys and top-level paths currently adapted (empty = pure base).
    public var activeKeys: Set<String> { applied.union(appliedTopLevel) }

    /// Make `loras` the active adapter set. An empty array leaves the pristine base in place.
    ///
    /// The files are read and checked before the model is touched, so an adapter that doesn't map
    /// (`KleinLoRA.LoRAError.unknownKeys`) throws with the previous set still applied.
    public func set(_ loras: [(url: URL, strength: Float)]) throws {
        let c = loras.isEmpty ? nil : try KleinLoRA.combined(loras, dtype: dtype)
        detach()
        guard let c else { return }
        func attach<B: Module>(_ blocks: [B], _ prefix: String) {
            for (i, block) in blocks.enumerated() {
                var update: [(String, Module)] = []
                for (key, child) in block.namedModules() where c.blockKeys.contains(key) {
                    let path = "\(prefix).\(i).\(key)"
                    guard let linear = child as? Linear, let rank = c.ranks[path] else { continue }
                    // First touch: the child IS the pristine base — capture (shares weights by
                    // reference, so free) so detach can restore it.
                    pristine[path] = pristine[path] ?? linear
                    update.append((key, LoRALinear.from(linear: linear, rank: rank, scale: 1.0)))
                }
                if !update.isEmpty { block.update(modules: .unflattened(update)) }
            }
        }
        // Recorded before attaching, so a failure part-way still detaches cleanly next time.
        applied = c.blockKeys
        appliedTopLevel = c.topLevelPaths
        attach(model.transformerBlocks, "transformer_blocks")
        attach(model.singleTransformerBlocks, "single_transformer_blocks")
        if !c.topLevelPaths.isEmpty {
            let modules = Dictionary(model.namedModules(), uniquingKeysWith: { first, _ in first })
            var update: [(String, Module)] = []
            for path in c.topLevelPaths.sorted() {
                guard let linear = modules[path] as? Linear, let rank = c.ranks[path] else { continue }
                pristine[path] = pristine[path] ?? linear
                update.append((path, LoRALinear.from(linear: linear, rank: rank, scale: 1.0)))
            }
            if !update.isEmpty { model.update(modules: .unflattened(update)) }
        }
        try model.update(parameters: ModuleParameters.unflattened(c.params), verify: .noUnusedKeys)
    }

    /// Restore the pristine base in every currently-adapted target.
    public func detach() {
        if !appliedTopLevel.isEmpty {
            let restore = appliedTopLevel.sorted().compactMap { path in pristine[path].map { (path, $0) } }
            if !restore.isEmpty { model.update(modules: .unflattened(restore)) }
            appliedTopLevel = []
        }
        guard !applied.isEmpty else { return }
        func restore<B: Module>(_ blocks: [B], _ prefix: String) {
            for (i, block) in blocks.enumerated() {
                var upd: [(String, Module)] = []
                for key in applied {
                    if let orig = pristine["\(prefix).\(i).\(key)"] { upd.append((key, orig)) }
                }
                if !upd.isEmpty { block.update(modules: .unflattened(upd)) }
            }
        }
        restore(model.transformerBlocks, "transformer_blocks")
        restore(model.singleTransformerBlocks, "single_transformer_blocks")
        applied = []
    }
}
