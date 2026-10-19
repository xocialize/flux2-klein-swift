// klein-cli — GPU render lane for the Klein port.
//   swift run -c release klein-cli --size 1024 --steps 4 --seed 42 [--snapshot <dir>] [--prompt ...]

import CoreGraphics
import Foundation
import Flux2VAE
import ImageIO
import Klein
import MLX
import MLXKlein
import MLXToolKit
import Tokenizers
import UniformTypeIdentifiers

@main
struct KleinCLI {
    static func main() async throws {
        var args = Array(CommandLine.arguments.dropFirst())
        func opt(_ n: String) -> String? {
            guard let i = args.firstIndex(of: n), i + 1 < args.count else { return nil }
            let v = args[i + 1]; args.removeSubrange(i...(i + 1)); return v
        }
        func flag(_ n: String) -> Bool {
            if let i = args.firstIndex(of: n) { args.remove(at: i); return true }; return false
        }
        // --nax-probe: weights-free NAX split-K GEMM probe at the exact Klein single-block
        // to_out shape (K = innerDim 3072 + mlpHidden 9216 = 12288, N = 3072). mlx-swift
        // ≤0.31.6 JIT-miscompiles the steel split-K NAX GEMM (mlx#3797, fixed by mlx#3810):
        // dispatch window M·N ≥ 2048², K ≥ 10240, K ≥ 3·max(M,N), half precision only →
        // M ∈ [1366, 4096] broken (NaN/1e21 on this machine, incl. the [1,M,K] Linear call
        // shape), M=1024 / M≥4608 clean. NOTE: production renders are currently clean
        // anyway — fp32 temb promotion makes the DiT activation stream fp32, outside the
        // half-precision window (see Flux2ParallelSelfAttention.outProjected). Run after
        // any mlx-swift bump to decide whether the row-chunk can be dropped. GPU stream on
        // purpose — the bug is in the JIT-compiled Metal kernel.
        if flag("--nax-probe") {
            var lcg: UInt64 = 0x9E37_79B9_7F4A_7C15
            func rand(_ n: Int) -> [Float] {
                (0 ..< n).map { _ in
                    lcg = lcg &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
                    return Float(Int64(bitPattern: lcg >> 11)) / Float(Int64.max >> 11)
                }
            }
            func cosine(_ a: MLXArray, _ b: MLXArray) -> Float {
                let x = a.asType(.float32).flattened()
                let y = b.asType(.float32).flattened()
                let d = (x * y).sum().item(Float.self)
                let nx = sqrt((x * x).sum()).item(Float.self)
                let ny = sqrt((y * y).sum()).item(Float.self)
                return nx * ny == 0 ? 1 : d / (nx * ny)
            }
            let (K, N) = (12288, 3072)   // Klein single-block to_out
            let b = MLXArray(rand(N * K), [N, K]).asType(.bfloat16)
            var ok = true
            for m in [1024, 1366, 2048, 2816, 4096, 4608] {
                let a = MLXArray(rand(m * K), [m, K]).asType(.bfloat16)
                let y = matmul(a, b.T)
                let yRef = matmul(a.asType(.float32), b.asType(.float32).T)
                eval(y, yRef)
                let mab = abs(y.asType(.float32) - yRef).max().item(Float.self)
                let c = cosine(y, yRef)
                let pass = c > 0.999 && mab.isFinite && mab < 100
                if !pass { ok = false }
                print(String(format: "  M=%d K=%d N=%d bf16: cos %.8f max_abs %.3e  %@",
                             m, K, N, c, mab, (pass ? "OK" : "BROKEN") as NSString))
                // 3-D Linear-shaped variant [1, M, K] — the production call shape.
                let y3 = matmul(a.reshaped([1, m, K]), b.T)
                eval(y3)
                let mab3 = abs(y3.asType(.float32).reshaped([m, N]) - yRef).max().item(Float.self)
                let c3 = cosine(y3, yRef)
                let pass3 = c3 > 0.999 && mab3.isFinite && mab3 < 100
                if !pass3 { ok = false }
                print(String(format: "  M=%d K=%d N=%d bf16 [1,M,K]: cos %.8f max_abs %.3e  %@",
                             m, K, N, c3, mab3, (pass3 ? "OK" : "BROKEN") as NSString))
            }
            print("[nax-probe] \(ok ? "PASS — kernel fixed, row-chunk removable" : "FAIL — keep the row-chunk")")
            exit(ok ? 0 : 1)
        }

        // Repeated option: collects every `--slot role=path` (or any repeated flag) in order.
        func opts(_ n: String) -> [String] {
            var vals: [String] = []
            while let i = args.firstIndex(of: n), i + 1 < args.count {
                vals.append(args[i + 1]); args.removeSubrange(i...(i + 1))
            }
            return vals
        }
        // lora-list: print the bundled Klein LoRA registry (id, surface, kind, trigger, slots).
        if args.first == "lora-list" {
            let reg = try LoRARegistry.bundled()
            print("[lora-list] schema v\(reg.schemaVersion) base=\(reg.base) — \(reg.adapters.count) adapter(s)")
            for e in reg.adapters {
                let slots = e.slots.map { "\($0.slot):\($0.role)\($0.required ? "" : "?")" }.joined(separator: " ")
                let trig = e.trigger.count > 40 ? String(e.trigger.prefix(37)) + "…" : e.trigger
                print("  • \(e.id)  [\(e.surface)/\(e.kind)]  trigger=\(trig)  strength=\(e.defaultStrength)"
                    + (slots.isEmpty ? "" : "  slots=[\(slots)]"))
            }
            return
        }
        // lora-info <id>: full detail for one adapter incl. required conditioning slots.
        if args.first == "lora-info", args.count >= 2 {
            let reg = try LoRARegistry.bundled()
            guard let e = reg.entry(id: args[1]) else { throw LoRARegistryError.unknownAdapter(args[1]) }
            print("[lora-info] \(e.id) — \(e.displayName)")
            print("  repo=\(e.repo)/\(e.weightFile)")
            print("  surface=\(e.surface) kind=\(e.kind) trigger=\(e.trigger) defaultStrength=\(e.defaultStrength)")
            if let s = e.status { print("  status=\(s) license=\(e.license ?? "?") gated=\(e.gatedDownload ?? false)") }
            if let src = e.source { print("  source=\(src)") }
            if let t = e.promptTemplate { print("  promptTemplate=\"\(t)\"") }
            for s in e.slots {
                print("  slot \(s.slot): \(s.role)\(s.required ? " (required)" : " (optional)") "
                    + "media=\(s.media) ingest=\(s.ingest ?? "-")" + (s.note.map { " — \($0)" } ?? ""))
            }
            if let v = e.verification {
                print("  verify: dialect=\(v.headerDialect ?? "?") rank=\(v.rank.map(String.init) ?? "?") "
                    + "tensors=\(v.tensors.map(String.init) ?? "?") "
                    + "targets=\(v.doubleTargets.map(String.init) ?? "?")d+\(v.singleTargets.map(String.init) ?? "?")s "
                    + "gate=\(v.gate ?? "?")")
            }
            return
        }
        // lora-smoke <lora.safetensors>: STRUCTURAL smoke — build a Flux2Transformer at config
        // dims (random init, no weights), apply the LoRA, print how many Linears were adapted per
        // block array. No inference / no Metal compute.
        if args.first == "lora-smoke" {
            guard args.count >= 2 else {
                print("usage: klein-cli lora-smoke <lora.safetensors>"); return
            }
            let loraURL = URL(fileURLWithPath: args[1])
            let dit = Flux2Transformer()   // config dims, random init — structural only
            let s = try KleinLoRA.apply(loRA: loraURL, to: dit)
            print("[lora-smoke] \(loraURL.lastPathComponent): adapted "
                + "double=\(s.doubleTargets) single=\(s.singleTargets) total=\(s.total)")
            return
        }
        // --pkg-edit <ref.png>: drive the real package's imageEdit surface (IEditRequest).
        if let editPath = opt("--pkg-edit") {
            let snap = opt("--snapshot") ?? URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
                .deletingLastPathComponent().deletingLastPathComponent()
                .appendingPathComponent("weights/FLUX.2-klein-4B").path
            // --quant 4|8 and --evict pick the tier (as --pkg-e2e does); --size sets the edit's output side.
            let quantBits = opt("--quant").flatMap { Int($0) }   // opt() consumes the flag: read it once
            let q: Quant = quantBits == 4 ? .int4 : quantBits == 8 ? .int8 : .bf16
            let evict = flag("--evict")
            let pkg = Klein4BT2IPackage(configuration: .init(quant: q, snapshotPath: snap, evictEncoder: evict))
            let surfaces = Klein4BT2IPackage.manifest.surfaces.map(\.name).joined(separator: ", ")
            print("[pkg-edit] surfaces=[\(surfaces)] quant=\(q) evict=\(evict)")
            let t0 = Date(); try await pkg.load()
            print("[pkg-edit] load: \(String(format: "%.2f", Date().timeIntervalSince(t0)))s | resident floor \(MLX.Memory.activeMemory / (1 << 20))MB")
            let refData = try Data(contentsOf: URL(fileURLWithPath: editPath))
            let sz = Int(opt("--size") ?? "1024")!
            let t1 = Date()
            let resp = try await pkg.run(IEditRequest(
                images: [Image(format: .png, data: refData, width: sz, height: sz)],
                prompt: opt("--prompt") ?? "the fox sitting on a sunny tropical beach, photorealistic",
                width: sz, height: sz, seed: 5)) as! IEditResponse
            print("[pkg-edit] run: \(String(format: "%.2f", Date().timeIntervalSince(t1)))s → "
                  + "\(resp.image.data.count) bytes .\(resp.image.format)")
            try resp.image.data.write(to: URL(fileURLWithPath: opt("--out") ?? "klein_pkg_edit.png"))
            print("[pkg-edit] PEAK \(MLX.Memory.peakMemory / (1 << 20))MB | quant=\(q) evict=\(evict) size=\(sz)")
            await pkg.unload()
            return
        }
        // --pkg-e2e: drive the real Klein4BT2IPackage (load→run→decode).
        if flag("--pkg-e2e") {
            let snap = opt("--snapshot") ?? URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
                .deletingLastPathComponent().deletingLastPathComponent()
                .appendingPathComponent("weights/FLUX.2-klein-4B").path
            let q: Quant = opt("--quant").flatMap { Int($0) } == 4 ? .int4 : .bf16
            let evict = flag("--evict")
            let pkg = Klein4BT2IPackage(configuration: .init(quant: q, snapshotPath: snap, evictEncoder: evict))
            print("[pkg-e2e] surface=\(Klein4BT2IPackage.manifest.surfaces[0].name) quant=\(q) evict=\(evict)")
            let t0 = Date(); try await pkg.load()
            let residentMB = MLX.Memory.activeMemory / (1 << 20)
            print("[pkg-e2e] load: \(String(format: "%.2f", Date().timeIntervalSince(t0)))s | resident floor \(residentMB)MB")
            let t1 = Date()
            let sz = Int(opt("--size") ?? "1024")!
            let resp = try await pkg.run(T2IRequest(
                prompt: opt("--prompt") ?? "a red fox sitting in a snowy forest at sunrise, photorealistic",
                width: sz, height: sz, seed: 42)) as! T2IResponse
            print("[pkg-e2e] run: \(String(format: "%.2f", Date().timeIntervalSince(t1)))s → "
                  + "\(resp.image.width)x\(resp.image.height) \(resp.image.data.count) bytes .\(resp.image.format)")
            try resp.image.data.write(to: URL(fileURLWithPath: opt("--out") ?? "klein_pkg_e2e.png"))
            print("[pkg-e2e] PEAK \(MLX.Memory.peakMemory / (1 << 20))MB | resident-floor \(residentMB)MB | quant=\(q) evict=\(evict)")
            await pkg.unload()
            return
        }
        let snapshot = opt("--snapshot")
            ?? URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent().deletingLastPathComponent()
                .deletingLastPathComponent().deletingLastPathComponent()
                .deletingLastPathComponent()
                .appendingPathComponent("weights/FLUX.2-klein-4B").path
        let size = Int(opt("--size") ?? "1024")!
        let steps = Int(opt("--steps") ?? "4")!
        let seed = UInt64(opt("--seed") ?? "42")!
        let promptFlag = opt("--prompt")
        let out = opt("--out") ?? "klein_\(size)_s\(seed).png"

        // --lora-id <id> [--slot role=path ...] [--lora-cache <dir>]: registry-driven LoRA.
        // The entry is self-describing — one path serves every adapter: resolve → cache.ensure →
        // (for `ic`) bind role→file slots in the manifest's declared (load-bearing) order → feed
        // the edit pipeline. Adding a LoRA is a JSON entry, never new CLI code.
        var registryLoRAPath: String? = nil
        var registryEditRefs: String? = nil
        var registryPrompt: String? = nil
        var registryStrength: Float? = nil
        if let loraId = opt("--lora-id") {
            let reg = try LoRARegistry.bundled()
            guard let entry = reg.entry(id: loraId) else { throw LoRARegistryError.unknownAdapter(loraId) }
            let cacheDir = opt("--lora-cache").map { URL(fileURLWithPath: $0) }
                ?? FileManager.default.homeDirectoryForCurrentUser
                    .appendingPathComponent("Development/MLXModelStore/loras")
            let cache = LoRACache(directory: cacheDir)
            let url = try await cache.ensure(entry)
            registryLoRAPath = url.path
            registryStrength = entry.defaultStrength
            registryPrompt = entry.promptTemplate
            print("[cli] registry LoRA '\(entry.id)' [\(entry.surface)/\(entry.kind)] → \(url.lastPathComponent)")
            if entry.isReferenceConditioned {
                // Parse `--slot role=path` pairs, then order them by the manifest's slots.
                var bindings: [String: URL] = [:]
                for pair in opts("--slot") {
                    guard let eq = pair.firstIndex(of: "=") else {
                        throw LoRARegistryError.unknownSlot(pair, adapter: entry.id)
                    }
                    let role = String(pair[..<eq])
                    let path = String(pair[pair.index(after: eq)...])
                    bindings[role] = URL(fileURLWithPath: path)
                }
                let ordered = try entry.orderedInputs(bindings: bindings)
                registryEditRefs = ordered.map(\.path).joined(separator: ",")
                let roles = entry.slots.compactMap { bindings[$0.role] != nil ? $0.role : nil }
                print("[cli] slots (in order): \(roles.joined(separator: " → "))")
            }
        }

        let prompt = promptFlag ?? registryPrompt
            ?? "A lighthouse on a stormy coast at dusk, dramatic clouds, crashing waves, warm lamp glow, photorealistic"

        func timed<T>(_ label: String, _ body: () throws -> T) rethrows -> T {
            let t0 = Date(); let r = try body()
            print("[cli] \(label): \(String(format: "%.2f", Date().timeIntervalSince(t0)))s"); return r
        }

        // --edit <ref.png>: compositional multi-reference edit (comma-separate for multiple refs).
        // A registry `ic` LoRA supplies its slot-ordered refs here when --edit is not given directly.
        let editRefs = opt("--edit") ?? registryEditRefs
        let quantBits = Int(opt("--quant") ?? "0")!
        // Base tier: --guidance >1 enables two-pass CFG; --negative sets the negative prompt.
        let guidance = Float(opt("--guidance") ?? "1.0")!
        let negativePrompt = opt("--negative")
        let transformer = try timed("load DiT") { try KleinWeights.loadTransformer(snapshotPath: snapshot, dtype: .bfloat16) }
        if quantBits == 8 || quantBits == 4 {
            timed("quantize DiT int\(quantBits)") { KleinWeights.quantizeDiT(transformer, bits: quantBits) }
            MLX.Memory.clearCache()
            print("[cli] DiT resident post-quant: \(MLX.Memory.activeMemory / (1 << 20)) MB")
        }
        // --lora <path> [--lora-strength <s>]: apply a RefControl (or any) LoRA over the
        // (optionally int4) base — validates the runtime activation-path applicator incl. the
        // QLoRALinear path. For RefControl: --edit <control.png>,<reference.png>.
        if let loraPath = opt("--lora") ?? registryLoRAPath {
            let strength = opt("--lora-strength").flatMap(Float.init) ?? registryStrength ?? 1.0
            let summary = try timed("apply LoRA") {
                try KleinLoRA.apply(loRA: URL(fileURLWithPath: loraPath), to: transformer,
                                    dtype: .bfloat16, strength: strength)
            }
            print("[cli] LoRA applied: double=\(summary.doubleTargets) single=\(summary.singleTargets) total=\(summary.total) @strength \(strength)")
        }
        let vae = try timed("load VAE") { try Flux2VAEWeights.loadVAE(directory: URL(fileURLWithPath: snapshot).appendingPathComponent("vae"), dtype: .float32) }
        let encoder = try timed("load encoder") { try KleinWeights.loadTextEncoder(snapshotPath: snapshot, dtype: .bfloat16) }
        let tok = try await AutoTokenizer.from(modelFolder: URL(fileURLWithPath: snapshot).appendingPathComponent("tokenizer"))
        let textEncoder = KleinTextEncoder(encoder: encoder, tokenizer: tok)

        let embeds = textEncoder.encode(prompt)
        eval(embeds)
        print("[cli] prompt embeds: \(embeds.shape)")

        func loadRefImage(_ path: String, _ dim: Int) -> MLXArray {
            let cs = CGColorSpace(name: CGColorSpace.sRGB)!
            let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil)!
            let cg = CGImageSourceCreateImageAtIndex(src, 0, nil)!
            var buf = [UInt8](repeating: 0, count: dim * dim * 4)
            let ctx = CGContext(data: &buf, width: dim, height: dim, bitsPerComponent: 8,
                bytesPerRow: dim * 4, space: cs, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
            ctx.draw(cg, in: CGRect(x: 0, y: 0, width: dim, height: dim))   // scales to dim×dim
            var rgb = [Float](repeating: 0, count: 3 * dim * dim)
            for p in 0..<(dim * dim) {
                rgb[p] = Float(buf[p*4]) / 127.5 - 1               // R plane
                rgb[dim*dim + p] = Float(buf[p*4+1]) / 127.5 - 1   // G plane
                rgb[2*dim*dim + p] = Float(buf[p*4+2]) / 127.5 - 1 // B plane
            }
            return MLXArray(rgb, [1, 3, dim, dim])
        }

        let t0 = Date()
        let result: KleinPipeline.Result
        if let editRefs {
            let vaeEnc = try timed("load VAE encoder") { try KleinWeights.loadVAEEncoder(snapshotPath: snapshot, dtype: .float32) }
            let vaeArrays = try MLX.loadArrays(url: URL(fileURLWithPath: snapshot).appendingPathComponent("vae/diffusion_pytorch_model.safetensors"))
            let bnMean = vaeArrays["bn.running_mean"]!.asType(.float32).reshaped(1, -1, 1, 1)
            let bnStd = MLX.sqrt(vaeArrays["bn.running_var"]!.asType(.float32).reshaped(1, -1, 1, 1) + 1e-4)
            let refs = editRefs.split(separator: ",").map { loadRefImage(String($0), size) }
            print("[cli] EDIT with \(refs.count) reference(s)")
            let negEmbeds = guidance > 1.0 ? textEncoder.encode(negativePrompt ?? "") : nil
            if guidance > 1.0 { print("[cli] BASE EDIT CFG guidance=\(guidance)") }
            result = KleinEditPipeline.generate(
                transformer: transformer, vae: vae, vaeEncoder: vaeEnc, bnMean: bnMean, bnStd: bnStd,
                promptEmbeds: embeds, negativeEmbeds: negEmbeds, referenceImages: refs, height: size, width: size,
                numInferenceSteps: steps, guidanceScale: guidance, seed: seed, transformerDtype: .bfloat16,
                onStep: { i, n in print("[cli] step \(i)/\(n)") })
        } else {
            let negEmbeds = guidance > 1.0 ? textEncoder.encode(negativePrompt ?? "") : nil
            if guidance > 1.0 { print("[cli] BASE CFG guidance=\(guidance) neg=\"\(negativePrompt ?? "")\"") }
            result = KleinPipeline.generate(
                transformer: transformer, vae: vae, promptEmbeds: embeds, negativeEmbeds: negEmbeds,
                height: size, width: size, numInferenceSteps: steps, guidanceScale: guidance,
                seed: seed, transformerDtype: .bfloat16,
                onStep: { i, n in print("[cli] step \(i)/\(n)") })
        }
        print("[cli] generate: \(String(format: "%.2f", Date().timeIntervalSince(t0)))s | peak \(MLX.Memory.peakMemory / (1 << 20)) MB")

        // [-1,1] NCHW → PNG
        let img = MLX.clip(result.image![0] / 2 + 0.5, min: 0, max: 1).transposed(1, 2, 0) * 255
        let u8 = img.asType(.uint8); eval(u8)
        let (h, w) = (u8.shape[0], u8.shape[1])
        var rgba = [UInt8](repeating: 255, count: h * w * 4)
        let rgb = u8.asArray(UInt8.self)
        for p in 0..<(h * w) { rgba[p*4]=rgb[p*3]; rgba[p*4+1]=rgb[p*3+1]; rgba[p*4+2]=rgb[p*3+2] }
        let cs = CGColorSpace(name: CGColorSpace.sRGB)!
        let ctx = CGContext(data: &rgba, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w*4, space: cs, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        let dest = CGImageDestinationCreateWithURL(URL(fileURLWithPath: out) as CFURL, UTType.png.identifier as CFString, 1, nil)!
        CGImageDestinationAddImage(dest, ctx.makeImage()!, nil); CGImageDestinationFinalize(dest)
        print("[cli] wrote \(out)")
    }
}
