// E1 gate — FLUX.2 VAE encoder (for the edit path) vs diffusers latent_dist.mean.
// Gated KLEIN_PARITY=1.

import Foundation
import MLX
import XCTest

@testable import Klein

final class E1VAEEncoderTests: XCTestCase {
    static let goldensDir = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("Tests/goldens")
    static let snapshotPath = ProcessInfo.processInfo.environment["KLEIN_SNAPSHOT"]
        ?? URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("weights/FLUX.2-klein-4B").path

    func testVAEEncodeParity() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["KLEIN_PARITY"] == "1", "KLEIN_PARITY=1")
        Device.setDefault(device: .cpu)
        let enc = try KleinWeights.loadVAEEncoder(snapshotPath: Self.snapshotPath, dtype: .float32)
        let g = try MLX.loadArrays(url: Self.goldensDir.appendingPathComponent("klein_vae_encode.safetensors"))
        let image = g["image"]!.asType(.float32)   // [1,3,256,256] NCHW
        let golden = g["mean"]!.asType(.float32)    // [1,32,32,32]
        let mean = enc.encodeMean(image)
        eval(mean)
        XCTAssertEqual(mean.shape, golden.shape)
        let mse = MLX.mean(MLX.square(mean - golden)).item(Float.self)
        let psnr = 10 * log10((golden.max().item(Float.self) - golden.min().item(Float.self))
                              * (golden.max().item(Float.self) - golden.min().item(Float.self)) / mse)
        let a = mean.flattened(), b = golden.flattened()
        let cos = MLX.sum(a*b).item(Float.self) / (Foundation.sqrt(MLX.sum(a*a).item(Float.self)) * Foundation.sqrt(MLX.sum(b*b).item(Float.self)) + 1e-12)
        print("  [E1] VAE encode cos=\(String(format: "%.7f", cos)) PSNR=\(String(format: "%.1f", psnr)) dB")
        XCTAssertGreaterThan(cos, 0.9999, "VAE encode cosine")
    }
}
