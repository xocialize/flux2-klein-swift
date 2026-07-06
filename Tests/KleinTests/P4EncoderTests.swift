// P4 gate — Qwen3-4B 3-layer-tap encoder vs diffusers golden (klein_encoder.safetensors).
// Injects the golden input_ids (tokenization tested separately) and checks the concatenated
// 9/18/27 → 7680 features. Gated KLEIN_PARITY=1.

import Foundation
import MLX
import XCTest

@testable import Klein

final class P4EncoderTests: XCTestCase {

    static let goldensDir = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("Tests/goldens")

    static let snapshotPath = ProcessInfo.processInfo.environment["KLEIN_SNAPSHOT"]
        ?? URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("weights/FLUX.2-klein-4B").path

    func testEncoderTapFeatures() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["KLEIN_PARITY"] == "1",
                          "set KLEIN_PARITY=1 to run the P4 gate")
        Device.setDefault(device: .cpu)

        let encoder = try KleinWeights.loadTextEncoder(snapshotPath: Self.snapshotPath, dtype: .float32)
        let g = try MLX.loadArrays(url: Self.goldensDir.appendingPathComponent("klein_encoder.safetensors"))
        let ids = g["input_ids"]!.asType(.int32)          // [1, 512], right-padded
        let golden = g["features"]!.asType(.float32)      // [512, 7680]
        // Compare only the REAL tokens: under causal mask + right padding, real-token features
        // are padding-mask-independent (padding is trailing, never attended). The pipeline packs
        // real tokens, so this is the load-bearing region.
        let validLen = MLX.sum(g["attention_mask"]!.asType(.int32)).item(Int.self)

        let feats = encoder(ids)[0][..<validLen]
        eval(feats)
        let goldenValid = golden[..<validLen]
        XCTAssertEqual(feats.shape, goldenValid.shape, "feature shape (must be 7680-wide)")
        let a = feats.asType(.float32).flattened(), b = goldenValid.flattened()
        let cos = MLX.sum(a * b).item(Float.self)
            / (Foundation.sqrt(MLX.sum(a * a).item(Float.self))
               * Foundation.sqrt(MLX.sum(b * b).item(Float.self)) + 1e-12)
        let maxAbs = MLX.abs(a - b).max().item(Float.self)
        print("  [P4] encoder 3-layer-tap cos=\(String(format: "%.7f", cos)) max_abs=\(String(format: "%.3e", maxAbs))")
        XCTAssertGreaterThan(cos, 0.999, "encoder features cosine")
    }
}
