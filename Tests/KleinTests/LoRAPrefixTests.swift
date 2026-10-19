// Regression: the LoRA key-prefix stripping in KleinLoRA.expand must handle PEFT's
// `base_model.model.` container prefix (as emitted by diffusers/peft exports, e.g. fal's
// object-removal LoRA) identically to the BFL `diffusion_model.` root — else those adapters
// map to zero targets and silently no-op.

import XCTest

@testable import Klein

final class LoRAPrefixTests: XCTestCase {

    func testPeftBaseModelPrefixMapsLikeBFL() {
        let peft = KleinLoRA.expand(base: "base_model.model.double_blocks.0.img_attn.proj")
        let bfl = KleinLoRA.expand(base: "diffusion_model.double_blocks.0.img_attn.proj")
        XCTAssertEqual(peft.map(\.path), ["transformer_blocks.0.attn.to_out"])
        XCTAssertEqual(peft.map(\.path), bfl.map(\.path))  // same target, different container prefix
    }

    func testFusedQkvSplitsUnderPeftPrefix() {
        // A fused img_attn.qkv must still split into q/k/v thirds regardless of the prefix.
        let t = KleinLoRA.expand(base: "base_model.model.double_blocks.0.img_attn.qkv")
        XCTAssertEqual(t.map(\.path), [
            "transformer_blocks.0.attn.to_q",
            "transformer_blocks.0.attn.to_k",
            "transformer_blocks.0.attn.to_v",
        ])
        XCTAssertEqual(t.map(\.third), [0, 1, 2])
    }

    func testSingleBlockUnderPeftPrefix() {
        let t = KleinLoRA.expand(base: "base_model.model.single_blocks.3.linear1")
        XCTAssertEqual(t.map(\.path), ["single_transformer_blocks.3.attn.to_qkv_mlp_proj"])
    }

    func testUnrecognizedKeyStillSkips() {
        // Prefix stripping must not turn a non-block key into a false match.
        XCTAssertTrue(KleinLoRA.expand(base: "base_model.model.final_layer.weight").isEmpty)
    }
}
