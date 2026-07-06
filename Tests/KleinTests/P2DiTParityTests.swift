// P2 gate — klein DiT parity vs fp32/CPU diffusers goldens (klein_dit.safetensors).
// Whole-forward + staged sub-op checkpoints. Gated behind KLEIN_PARITY=1.

import Foundation
import MLX
import XCTest

@testable import Klein

final class P2DiTParityTests: XCTestCase {

    static let goldensDir = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("Tests/goldens")

    static let snapshotPath = ProcessInfo.processInfo.environment["KLEIN_SNAPSHOT"]
        ?? URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("weights/FLUX.2-klein-4B").path

    nonisolated(unsafe) static let model: Flux2Transformer? = {
        guard ProcessInfo.processInfo.environment["KLEIN_PARITY"] == "1" else { return nil }
        Device.setDefault(device: .cpu)
        return try! KleinWeights.loadTransformer(snapshotPath: snapshotPath, dtype: .float32)
    }()

    func cos(_ a: MLXArray, _ b: MLXArray) -> Float {
        let x = a.asType(.float32).flattened(), y = b.asType(.float32).flattened()
        return MLX.sum(x * y).item(Float.self)
            / (Foundation.sqrt(MLX.sum(x * x).item(Float.self))
               * Foundation.sqrt(MLX.sum(y * y).item(Float.self)) + 1e-12)
    }

    func check(_ name: String, _ ours: MLXArray, _ golden: MLXArray, cosMin: Float = 0.9999) {
        XCTAssertEqual(ours.shape, golden.shape, "\(name) shape")
        let c = cos(ours, golden)
        let m = MLX.abs(ours.asType(.float32) - golden.asType(.float32)).max().item(Float.self)
        print("  [P2] \(name.padding(toLength: 16, withPad: " ", startingAt: 0)) cos=\(String(format: "%.7f", c)) max_abs=\(String(format: "%.3e", m))")
        XCTAssertGreaterThan(c, cosMin, "\(name) cosine")
    }

    func testDiTParity() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["KLEIN_PARITY"] == "1",
                          "set KLEIN_PARITY=1 to run the P2 gate")
        Device.setDefault(device: .cpu)
        let model = Self.model!
        let g = try MLX.loadArrays(url: Self.goldensDir.appendingPathComponent("klein_dit.safetensors"))

        let hidden = g["in_hidden"]!.asType(.float32)
        let encoder = g["in_encoder"]!.asType(.float32)
        let t = g["in_t"]!.asType(.float32)
        let imgIds = g["in_img_ids"]!.asType(.float32)
        let txtIds = g["in_txt_ids"]!.asType(.float32)

        // --- staged: temb / embedders ---
        let tScaled = t * (t.max().item(Float.self) <= 1 ? Float(1000) : 1)
        let temb = model.timeGuidanceEmbed(tScaled, nil)
        check("temb", temb, g["temb"]!)
        let xh = model.xEmbedder(hidden); check("x_embed", xh, g["x_embed"]!)
        let eh = model.contextEmbedder(encoder); check("context_embed", eh, g["context_embed"]!)

        // --- staged: double blocks ---
        let (imgC, imgS) = model.posEmbed(imgIds)
        let (txtC, txtS) = model.posEmbed(txtIds)
        let cosT = concatenated([txtC, imgC], axis: 0)
        let sinT = concatenated([txtS, imgS], axis: 0)
        let modImg = model.dsModImg(temb), modTxt = model.dsModTxt(temb)
        var h = xh, e = eh
        for (i, block) in model.transformerBlocks.enumerated() {
            (e, h) = block(h, encoder: e, modImg: modImg, modTxt: modTxt, cos: cosT, sin: sinT)
            eval(h); eval(e)
            if i == 0 || i == 4 {
                check("double\(i)_hid", h, g["double\(i)_hid"]!)
                check("double\(i)_enc", e, g["double\(i)_enc"]!)
            }
        }

        // --- staged: single blocks ---
        var comb = concatenated([e, h], axis: 1)
        let modS = model.ssMod(temb)[0]
        for (i, block) in model.singleTransformerBlocks.enumerated() {
            comb = block(comb, mod: modS, cos: cosT, sin: sinT)
            eval(comb)
            if i == 0 || i == 10 || i == 19 { check("single\(i)", comb, g["single\(i)"]!) }
        }

        // --- norm_out + whole-forward ---
        let outTokens = comb[0..., e.shape[1]..., 0...]
        check("norm_out", model.normOut(outTokens, temb), g["norm_out"]!)

        let apiOut = model(hidden, encoder: encoder, timestep: t, guidance: nil, imgIds: imgIds, txtIds: txtIds)
        check("out_final", apiOut, g["out_final"]!, cosMin: 0.9999)
    }
}
