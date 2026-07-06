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
        // --pkg-edit <ref.png>: drive the real package's imageEdit surface (IEditRequest).
        if let editPath = opt("--pkg-edit") {
            let snap = opt("--snapshot") ?? URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
                .deletingLastPathComponent().deletingLastPathComponent()
                .appendingPathComponent("weights/FLUX.2-klein-4B").path
            let pkg = Klein4BT2IPackage(configuration: .init(quant: .bf16, snapshotPath: snap))
            let surfaces = Klein4BT2IPackage.manifest.surfaces.map(\.name).joined(separator: ", ")
            print("[pkg-edit] surfaces=[\(surfaces)]")
            try await pkg.load()
            let refData = try Data(contentsOf: URL(fileURLWithPath: editPath))
            let t1 = Date()
            let resp = try await pkg.run(IEditRequest(
                images: [Image(format: .png, data: refData, width: 1024, height: 1024)],
                prompt: opt("--prompt") ?? "the fox sitting on a sunny tropical beach, photorealistic",
                width: 1024, height: 1024, seed: 5)) as! IEditResponse
            print("[pkg-edit] run: \(String(format: "%.2f", Date().timeIntervalSince(t1)))s → "
                  + "\(resp.image.data.count) bytes .\(resp.image.format)")
            try resp.image.data.write(to: URL(fileURLWithPath: opt("--out") ?? "klein_pkg_edit.png"))
            await pkg.unload()
            print("[pkg-edit] peak \(MLX.Memory.peakMemory / (1 << 20))MB; done")
            return
        }
        // --pkg-e2e: drive the real Klein4BT2IPackage (load→run→decode).
        if flag("--pkg-e2e") {
            let snap = opt("--snapshot") ?? URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
                .deletingLastPathComponent().deletingLastPathComponent()
                .appendingPathComponent("weights/FLUX.2-klein-4B").path
            let q: Quant = opt("--quant").flatMap { Int($0) } == 4 ? .int4 : .bf16
            let pkg = Klein4BT2IPackage(configuration: .init(quant: q, snapshotPath: snap))
            print("[pkg-e2e] surface=\(Klein4BT2IPackage.manifest.surfaces[0].name) quant=\(q)")
            let t0 = Date(); try await pkg.load()
            print("[pkg-e2e] load: \(String(format: "%.2f", Date().timeIntervalSince(t0)))s")
            let t1 = Date()
            let resp = try await pkg.run(T2IRequest(
                prompt: opt("--prompt") ?? "a red fox sitting in a snowy forest at sunrise, photorealistic",
                width: 1024, height: 1024, seed: 42)) as! T2IResponse
            print("[pkg-e2e] run: \(String(format: "%.2f", Date().timeIntervalSince(t1)))s → "
                  + "\(resp.image.width)x\(resp.image.height) \(resp.image.data.count) bytes .\(resp.image.format)")
            try resp.image.data.write(to: URL(fileURLWithPath: opt("--out") ?? "klein_pkg_e2e.png"))
            await pkg.unload()
            print("[pkg-e2e] peak \(MLX.Memory.peakMemory / (1 << 20))MB; done")
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
        let prompt = opt("--prompt")
            ?? "A lighthouse on a stormy coast at dusk, dramatic clouds, crashing waves, warm lamp glow, photorealistic"
        let out = opt("--out") ?? "klein_\(size)_s\(seed).png"

        func timed<T>(_ label: String, _ body: () throws -> T) rethrows -> T {
            let t0 = Date(); let r = try body()
            print("[cli] \(label): \(String(format: "%.2f", Date().timeIntervalSince(t0)))s"); return r
        }

        // --edit <ref.png>: compositional multi-reference edit (comma-separate for multiple refs).
        let editRefs = opt("--edit")
        let quantBits = Int(opt("--quant") ?? "0")!
        let transformer = try timed("load DiT") { try KleinWeights.loadTransformer(snapshotPath: snapshot, dtype: .bfloat16) }
        if quantBits == 8 || quantBits == 4 {
            timed("quantize DiT int\(quantBits)") { KleinWeights.quantizeDiT(transformer, bits: quantBits) }
            MLX.Memory.clearCache()
            print("[cli] DiT resident post-quant: \(MLX.Memory.activeMemory / (1 << 20)) MB")
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
            result = KleinEditPipeline.generate(
                transformer: transformer, vae: vae, vaeEncoder: vaeEnc, bnMean: bnMean, bnStd: bnStd,
                promptEmbeds: embeds, referenceImages: refs, height: size, width: size,
                numInferenceSteps: steps, seed: seed, transformerDtype: .bfloat16,
                onStep: { i, n in print("[cli] step \(i)/\(n)") })
        } else {
            result = KleinPipeline.generate(
                transformer: transformer, vae: vae, promptEmbeds: embeds,
                height: size, width: size, numInferenceSteps: steps, guidanceScale: 1.0,
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
