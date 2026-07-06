// P5 decode-path probe — isolate the correct final-latent→image path by decoding the
// diffusers e2e golden's final_latents [1,32,128,128] two ways vs the golden image.
// Gated KLEIN_PARITY=1.

import Foundation
import Flux2VAE
import MLX
import XCTest

@testable import Klein

final class P5DecodeProbeTests: XCTestCase {

    static let goldensDir = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("Tests/goldens")
    static let snapshotPath = ProcessInfo.processInfo.environment["KLEIN_SNAPSHOT"]
        ?? URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("weights/FLUX.2-klein-4B").path

    func psnr(_ a: MLXArray, _ b: MLXArray) -> Float {
        let mse = MLX.mean(MLX.square(a.asType(.float32) - b.asType(.float32))).item(Float.self)
        return 10 * log10(4.0 / mse)
    }

    func testDecodePathProbe() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["KLEIN_PARITY"] == "1", "KLEIN_PARITY=1")
        Device.setDefault(device: .cpu)
        let vae = try Flux2VAEWeights.loadVAE(
            directory: URL(fileURLWithPath: Self.snapshotPath).appendingPathComponent("vae"), dtype: .float32)
        let g = try MLX.loadArrays(url: Self.goldensDir.appendingPathComponent("klein_e2e.safetensors"))
        let finalLatents = g["final_latents"]!.asType(.float32)   // [1,32,128,128]
        let golden = g["decoded"]!.asType(.float32)               // [1,3,1024,1024]

        // Path A: plain decode() on the 32-ch latents.
        let a = vae.decode(finalLatents); eval(a)
        print("  [P5probe] plain decode() PSNR = \(String(format: "%.2f", psnr(a, golden))) dB")

        // Path B: patchify 32→128 then decodePackedLatents (bn-denorm + unpatch + decode).
        let (h, w) = (finalLatents.shape[2], finalLatents.shape[3])
        let packed128 = finalLatents.reshaped(1, 32, h/2, 2, w/2, 2)
            .transposed(0, 1, 3, 5, 2, 4).reshaped(1, 128, h/2, w/2)
        let b = vae.decodePackedLatents(packed128); eval(b)
        print("  [P5probe] decodePackedLatents PSNR = \(String(format: "%.2f", psnr(b, golden))) dB")
    }
}
