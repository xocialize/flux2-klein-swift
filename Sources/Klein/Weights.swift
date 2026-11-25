// Weight loading for Flux2Transformer. The diffusers klein checkpoint keys are near-identical
// to the (mflux-derived) Swift module names; only two remaps are needed. All params are
// bias-free Linears (`.weight` only) — no conv-layout transpose in the DiT.

import Foundation
import MLX
import MLXNN

public enum KleinWeights {

    /// Precision-sensitive DiT projections kept at bf16 when quantizing (Lens/Z-Image doctrine:
    /// skip in/out embedders, modulation, timestep/guidance embed, norm_out). The bulk —
    /// attention + FFN Linears in the 5 double + 20 single blocks — is quantized.
    static func keepHiPrecision(path: String) -> Bool {
        let keep = ["x_embedder", "context_embedder", "proj_out", "norm_out",
                    "time_guidance_embed", "double_stream_modulation_img",
                    "double_stream_modulation_txt", "single_stream_modulation"]
        return keep.contains { path.hasPrefix($0) }
    }

    /// Quantize a loaded DiT in place (affine, group 64) skipping keepHiPrecision layers.
    public static func quantizeDiT(_ model: Flux2Transformer, bits: Int, groupSize: Int = 64) {
        quantize(model: model, filter: { path, module in
            guard module is Linear else { return nil }
            if keepHiPrecision(path: path) { return nil }
            return (groupSize, bits, .affine)
        })
        eval(model)
    }

    /// diffusers key → Swift module path.
    static func remapDiTKey(_ key: String) -> String {
        var key = key
        // time_guidance_embed.timestep_embedder.linear_{1,2} → time_guidance_embed.linear_{1,2}
        key = key.replacingOccurrences(
            of: "time_guidance_embed.timestep_embedder.", with: "time_guidance_embed.")
        // double-block attn.to_out.0.weight → attn.to_out.weight (single block already has no index)
        key = key.replacingOccurrences(of: ".attn.to_out.0.", with: ".attn.to_out.")
        return key
    }

    static func loadShards(directory: URL) throws -> [String: MLXArray] {
        let files = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        precondition(!files.isEmpty, "no safetensors in \(directory.path)")
        var weights: [String: MLXArray] = [:]
        for file in files { weights.merge(try MLX.loadArrays(url: file)) { _, new in new } }
        return weights
    }

    /// Load the DiT from `<snapshot>/transformer/`, strict.
    public static func loadTransformer(snapshotPath: String, dtype: DType? = .bfloat16) throws -> Flux2Transformer {
        let dir = URL(fileURLWithPath: snapshotPath).appendingPathComponent("transformer")
        var weights: [String: MLXArray] = [:]
        for (key, value) in try loadShards(directory: dir) {
            weights[remapDiTKey(key)] = dtype.map { value.asType($0) } ?? value
        }
        let model = Flux2Transformer()
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(model)
        return model
    }

    /// Load the FLUX.2 VAE ENCODER (encoder.* + quant_conv.*) from `<snapshot>/vae/` — the keys
    /// flux2-vae-mlx-swift (decoder-only) skips. fp32. Conv2d weights transpose (O,I,kH,kW)→(O,kH,kW,I).
    public static func loadVAEEncoder(snapshotPath: String, dtype: DType = .float32) throws -> KleinVAEEncoder {
        let dir = URL(fileURLWithPath: snapshotPath).appendingPathComponent("vae")
        var weights: [String: MLXArray] = [:]
        for (key, rawValue) in try loadShards(directory: dir) {
            guard key.hasPrefix("encoder.") || key.hasPrefix("quant_conv.") else { continue }
            var value = rawValue
            if key.hasSuffix(".weight"), value.ndim == 4 { value = value.transposed(0, 2, 3, 1) }
            weights[key] = value.asType(dtype)
        }
        let enc = KleinVAEEncoder()
        try enc.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(enc)
        return enc
    }

    /// Load the Qwen3-4B encoder from `<snapshot>/text_encoder/`, keeping only the layers up to
    /// the deepest tap (klein: 0..26). Drops deeper layers, final norm, and the (tied) lm_head.
    public static func loadTextEncoder(
        snapshotPath: String, tapLayers: [Int] = [9, 18, 27], dtype: DType? = .bfloat16
    ) throws -> Qwen3HiddenStateEncoder {
        let dir = URL(fileURLWithPath: snapshotPath).appendingPathComponent("text_encoder")
        let config = try Qwen3EncoderConfiguration.load(dir.appendingPathComponent("config.json"))
        let keepBelow = tapLayers.max()!        // keep layer indices 0..<keepBelow

        var weights: [String: MLXArray] = [:]
        for (key, value) in try loadShards(directory: dir) {
            guard let stripped = textEncoderKey(key, keepBelow: keepBelow) else { continue }
            weights[stripped] = dtype.map { value.asType($0) } ?? value
        }
        let encoder = Qwen3HiddenStateEncoder(config, tapLayers: tapLayers)
        try encoder.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(encoder)
        return encoder
    }

    // MARK: - Pre-packed weights (Bonsai Image; AB-R-1130)

    /// A packed tensor file's quantization, as written beside it: `quantization_config.json` (Prism ML's DiT pack:
    /// `bits`, `group_size`) or mlx-lm's `config.json` (`quantization.bits`, `quantization.group_size`).
    public struct PackedQuantization: Sendable, Equatable {
        public let bits: Int
        public let groupSize: Int
        public init(bits: Int, groupSize: Int) { (self.bits, self.groupSize) = (bits, groupSize) }

        public static func load(_ url: URL) throws -> PackedQuantization {
            struct Flat: Decodable { let bits: Int?; let group_size: Int?; let quantization: Nested? }
            struct Nested: Decodable { let bits: Int; let group_size: Int }
            let file = try JSONDecoder().decode(Flat.self, from: Data(contentsOf: url))
            if let q = file.quantization { return PackedQuantization(bits: q.bits, groupSize: q.group_size) }
            guard let bits = file.bits, let groupSize = file.group_size else {
                throw CocoaError(.fileReadCorruptFile, userInfo: [NSFilePathErrorKey: url.path])
            }
            return PackedQuantization(bits: bits, groupSize: groupSize)
        }
    }

    /// A `Flux2Transformer` whose block Linears are `QuantizedLinear` at `quantization`, with the same layers kept
    /// at full precision as `quantizeDiT`. The parameters are the lazy random initialisation, never evaluated:
    /// the caller replaces them (`loadPackedTransformer`) or reads only their keys and shapes.
    static func packedTransformerSkeleton(_ quantization: PackedQuantization) -> Flux2Transformer {
        let model = Flux2Transformer()
        quantize(model: model, filter: { path, module in
            guard module is Linear, !keepHiPrecision(path: path) else { return nil }
            return (quantization.groupSize, quantization.bits, .affine)
        })
        return model
    }

    /// The same for the Qwen3 tap encoder: every Linear and the token embedding packed, as mlx-lm writes them.
    static func packedTextEncoderSkeleton(
        _ config: Qwen3EncoderConfiguration, tapLayers: [Int], quantization: PackedQuantization
    ) -> Qwen3HiddenStateEncoder {
        let encoder = Qwen3HiddenStateEncoder(config, tapLayers: tapLayers)
        quantize(model: encoder, filter: { _, module in
            (module is Linear || module is Embedding) ? (quantization.groupSize, quantization.bits, .affine) : nil
        })
        return encoder
    }

    /// Load a pre-packed DiT from `directory`: MLX `QuantizedLinear` triples (`weight` U32, `scales`, `biases`) in
    /// diffusers key names, with `quantization_config.json` beside them. Strict: every tensor must land.
    public static func loadPackedTransformer(directory: URL) throws -> Flux2Transformer {
        let quantization = try PackedQuantization.load(directory.appendingPathComponent("quantization_config.json"))
        var weights: [String: MLXArray] = [:]
        for (key, value) in try loadShards(directory: directory) { weights[remapDiTKey(key)] = value }
        let model = packedTransformerSkeleton(quantization)
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(model)
        return model
    }

    /// Model keys of an mlx-lm Qwen3 checkpoint → the tap encoder's: `model.` stripped, the final norm, the head
    /// and the layers past the deepest tap dropped. Shared by the bf16 and the packed loaders.
    static func textEncoderKey(_ key: String, keepBelow: Int) -> String? {
        guard key.hasPrefix("model.") else { return nil }      // drops lm_head.*
        let stripped = String(key.dropFirst("model.".count))
        if stripped.hasPrefix("norm.") { return nil }
        if stripped.hasPrefix("layers.") {
            let idx = Int(stripped.dropFirst("layers.".count).prefix { $0 != "." }) ?? -1
            if idx < 0 || idx >= keepBelow { return nil }
        }
        return stripped
    }

    /// Load a pre-quantized mlx-lm Qwen3-4B from `directory` (its `config.json` names the quantization) as the
    /// tap encoder, keeping only the layers up to the deepest tap.
    public static func loadPackedTextEncoder(
        directory: URL, tapLayers: [Int] = [9, 18, 27]
    ) throws -> Qwen3HiddenStateEncoder {
        let configURL = directory.appendingPathComponent("config.json")
        let config = try Qwen3EncoderConfiguration.load(configURL)
        let quantization = try PackedQuantization.load(configURL)
        var weights: [String: MLXArray] = [:]
        for (key, value) in try loadShards(directory: directory) {
            if let mapped = textEncoderKey(key, keepBelow: tapLayers.max()!) { weights[mapped] = value }
        }
        let encoder = packedTextEncoderSkeleton(config, tapLayers: tapLayers, quantization: quantization)
        try encoder.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(encoder)
        return encoder
    }
}
