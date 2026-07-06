// P3 gate — FLUX.2 VAE decode via the in-house flux2-vae-mlx-swift package (reuse, not
// re-ported) vs the diffusers klein VAE golden (klein_vae.safetensors). Gated KLEIN_PARITY=1.

import Foundation
import Flux2VAE
import MLX
import XCTest

@testable import Klein

final class P3VAETests: XCTestCase {

    static let goldensDir = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("Tests/goldens")

    static let snapshotPath = ProcessInfo.processInfo.environment["KLEIN_SNAPSHOT"]
        ?? URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("weights/FLUX.2-klein-4B").path

    func testVAEDecodeParity() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["KLEIN_PARITY"] == "1",
                          "set KLEIN_PARITY=1 to run the P3 gate")
        Device.setDefault(device: .cpu)

        let vaeDir = URL(fileURLWithPath: Self.snapshotPath).appendingPathComponent("vae")
        let vae = try Flux2VAEWeights.loadVAE(directory: vaeDir, dtype: .float32)

        let g = try MLX.loadArrays(url: Self.goldensDir.appendingPathComponent("klein_vae.safetensors"))
        let latent = g["in_latent"]!.asType(.float32)     // [1,32,32,32] NCHW
        let golden = g["decoded"]!.asType(.float32)       // [1,3,256,256]

        let decoded = vae.decode(latent)
        eval(decoded)
        XCTAssertEqual(decoded.shape, golden.shape)
        let mse = MLX.mean(MLX.square(decoded - golden)).item(Float.self)
        let psnr = 10 * log10(4.0 / mse)
        print("  [P3] klein VAE decode PSNR = \(String(format: "%.2f", psnr)) dB (gate ≥ 60)")
        XCTAssertGreaterThanOrEqual(psnr, 60.0)
    }
}
