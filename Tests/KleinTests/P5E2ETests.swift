// P5 e2e isolation — split denoise vs decode against the diffusers e2e golden.
//  (1) decode: my decode() on golden final_latents [1,32,128,128], compared in [0,1] range.
//  (2) denoise: inject golden init, run my denoise, unpack the diffusers way (unpack → bn-denorm
//      → unpatchify) → [1,32,128,128], compare to golden final_latents.
// Gated KLEIN_PARITY=1.

import Foundation
import Flux2VAE
import MLX
import Tokenizers
import XCTest

@testable import Klein

final class P5E2ETests: XCTestCase {

    static let goldensDir = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("Tests/goldens")
    static let snapshotPath = ProcessInfo.processInfo.environment["KLEIN_SNAPSHOT"]
        ?? URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("weights/FLUX.2-klein-4B").path
    static let prompt = "A lighthouse on a stormy coast at dusk, dramatic clouds, crashing waves, warm lamp glow, photorealistic"

    func cos(_ a: MLXArray, _ b: MLXArray) -> Float {
        let x = a.asType(.float32).flattened(), y = b.asType(.float32).flattened()
        return MLX.sum(x * y).item(Float.self) / (Foundation.sqrt(MLX.sum(x*x).item(Float.self)) * Foundation.sqrt(MLX.sum(y*y).item(Float.self)) + 1e-12)
    }
    func psnr01(_ a: MLXArray, _ b: MLXArray) -> Float {   // both in [0,1]
        let mse = MLX.mean(MLX.square(a.asType(.float32) - b.asType(.float32))).item(Float.self)
        return 10 * log10(1.0 / mse)
    }

    func testE2EIsolation() async throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["KLEIN_PARITY"] == "1", "KLEIN_PARITY=1")
        Device.setDefault(device: .cpu)
        let vae = try Flux2VAEWeights.loadVAE(directory: URL(fileURLWithPath: Self.snapshotPath).appendingPathComponent("vae"), dtype: .float32)
        let g = try MLX.loadArrays(url: Self.goldensDir.appendingPathComponent("klein_e2e.safetensors"))
        let finalLatents = g["final_latents"]!.asType(.float32)   // [1,32,128,128]
        let goldenImg01 = g["decoded"]!.asType(.float32)          // [1,3,1024,1024] in [0,1] (pt postprocess)

        // (1) DECODE isolation — plain decode(), compare in [0,1].
        let dec = vae.decode(finalLatents); eval(dec)
        let dec01 = MLX.clip(dec / 2 + 0.5, min: 0, max: 1)
        print("  [P5] decode(final)→[0,1] PSNR = \(String(format: "%.2f", psnr01(dec01, goldenImg01))) dB")

        // (2) DENOISE isolation — inject golden init, run my denoise, unpack diffusers-way.
        let transformer = try KleinWeights.loadTransformer(snapshotPath: Self.snapshotPath, dtype: .float32)
        let encoder = try KleinWeights.loadTextEncoder(snapshotPath: Self.snapshotPath, dtype: .float32)
        let tok = try await AutoTokenizer.from(modelFolder: URL(fileURLWithPath: Self.snapshotPath).appendingPathComponent("tokenizer"))
        let textEnc = KleinTextEncoder(encoder: encoder, tokenizer: tok)
        let embeds = textEnc.encode(Self.prompt); eval(embeds)

        // pack golden init [1,128,64,64] → [1,4096,128]
        let init0 = g["init_unpacked"]!.asType(.float32)
        let initPacked = init0.reshaped(1, 128, 4096).transposed(0, 2, 1)
        let result = KleinPipeline.generate(
            transformer: transformer, vae: nil, promptEmbeds: embeds,
            height: 1024, width: 1024, numInferenceSteps: 4, guidanceScale: 1.0,
            initPacked: initPacked, transformerDtype: .float32)
        // diffusers-way unpack of my packed [1,4096,128]:
        //   reshape→[1,64,64,128]→[1,128,64,64]; bn-denorm; unpatchify→[1,32,128,128]
        var mine = result.latents.reshaped(1, 64, 64, 128).transposed(0, 3, 1, 2)
        // bn stats loaded straight from the vae safetensors (bn accessor is internal).
        let vaeArrays = try MLX.loadArrays(url: URL(fileURLWithPath: Self.snapshotPath)
            .appendingPathComponent("vae/diffusion_pytorch_model.safetensors"))
        let bnMean = vaeArrays["bn.running_mean"]!.asType(.float32).reshaped(1, -1, 1, 1)
        let bnStd = MLX.sqrt(vaeArrays["bn.running_var"]!.asType(.float32).reshaped(1, -1, 1, 1) + 1e-4)
        mine = mine * bnStd + bnMean
        // unpatchify [1,128,h,w] → [1,32,2h,2w]  (diffusers _unpatchify_latents)
        let (hh, wwd) = (mine.shape[2], mine.shape[3])
        mine = mine.reshaped(1, 32, 2, 2, hh, wwd).transposed(0, 1, 4, 2, 5, 3).reshaped(1, 32, hh * 2, wwd * 2)
        eval(mine)
        print("  [P5] denoise final-latents cos=\(String(format: "%.6f", cos(mine, finalLatents))) shape=\(mine.shape) vs \(finalLatents.shape)")
    }
}
