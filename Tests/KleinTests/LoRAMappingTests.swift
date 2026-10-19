// Every tensor of an adapter must land on the klein DiT (v0.10.0, AB-A-0192): the applicator
// throws on a key it can't place instead of skipping it. Offline: key logic over the registry's
// real tensor names (read from the hub headers into Fixtures/), and a tiny Flux2Transformer for
// the swapper.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXRandom
import XCTest

@testable import Klein

final class LoRAMappingTests: XCTestCase {

    /// Each registry adapter's header maps completely, so the strict applicator keeps all seven.
    /// Before v0.10.0, fal's four plain adapters lost their 8 top-level targets silently.
    func testRegistryAdaptersMapCompletely() throws {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appendingPathComponent("Fixtures/registry-adapter-keys.json")
        struct Fixture: Decodable { let adapters: [String: [String]] }
        let fixture = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url))
        XCTAssertEqual(fixture.adapters.count, 7)
        for (id, keys) in fixture.adapters {
            XCTAssertFalse(keys.isEmpty, id)
            XCTAssertEqual(KleinLoRA.unmappedKeys(keys), [], "\(id) has tensors klein can't place")
        }
    }

    /// The top-level Linears, in BFL (fal's PEFT exports) and diffusers names.
    func testTopLevelTargets() {
        let cases: [(String, String)] = [
            ("base_model.model.img_in", "x_embedder"),
            ("base_model.model.txt_in", "context_embedder"),
            ("base_model.model.time_in.in_layer", "time_guidance_embed.linear_1"),
            ("base_model.model.time_in.out_layer", "time_guidance_embed.linear_2"),
            ("base_model.model.double_stream_modulation_img.lin", "double_stream_modulation_img.linear"),
            ("base_model.model.double_stream_modulation_txt.lin", "double_stream_modulation_txt.linear"),
            ("base_model.model.single_stream_modulation.lin", "single_stream_modulation.linear"),
            ("base_model.model.final_layer.linear", "proj_out"),
            ("transformer.time_guidance_embed.timestep_embedder.linear_1", "time_guidance_embed.linear_1"),
            ("transformer.proj_out", "proj_out"),
        ]
        for (base, path) in cases {
            XCTAssertEqual(KleinLoRA.expand(base: base).map(\.path), [path], base)
            XCTAssertNil(KleinLoRA.blockRelative(path), "\(path) is top-level")
        }
        // BFL's final adaLN swaps scale and shift under diffusers' conversion: not a rename.
        XCTAssertTrue(KleinLoRA.expand(base: "diffusion_model.final_layer.adaLN_modulation.1").isEmpty)
    }

    func testUnmappedKeysAreReported() {
        let keys = [
            "diffusion_model.double_blocks.0.img_attn.proj.lora_A.weight",
            "diffusion_model.double_blocks.0.img_attn.proj.lora_B.weight",
            "diffusion_model.double_blocks.0.img_attn.proj.alpha",
            "lora_te1_text_model_encoder_layers_0_mlp_fc1.lora_down.weight",   // a text-encoder LoRA
            "diffusion_model.final_layer.adaLN_modulation.1.lora_A.weight",
            "diffusion_model.some_metadata_tensor",
        ]
        XCTAssertEqual(KleinLoRA.unmappedKeys(keys), [
            "diffusion_model.final_layer.adaLN_modulation.1.lora_A.weight",
            "diffusion_model.some_metadata_tensor",
            "lora_te1_text_model_encoder_layers_0_mlp_fc1.lora_down.weight",
        ])
    }

    // MARK: - The swapper on a tiny transformer

    private static func tinyModel() -> Flux2Transformer {
        Flux2Transformer(inChannels: 8, numLayers: 1, numSingleLayers: 1, numHeads: 2, headDim: 8,
                         jointAttentionDim: 12, mlpRatio: 2)
    }

    /// A rank-2 adapter on one double-block projection and two top-level Linears, plus `extra`.
    private static func writeAdapter(_ extra: [String: MLXArray] = [:]) throws -> URL {
        func pair(_ base: String, inDim: Int, outDim: Int) -> [String: MLXArray] {
            [base + ".lora_A.weight": MLXRandom.normal([2, inDim]),
             base + ".lora_B.weight": MLXRandom.normal([outDim, 2])]
        }
        var arrays = pair("base_model.model.double_blocks.0.img_attn.proj", inDim: 16, outDim: 16)
        arrays.merge(pair("base_model.model.img_in", inDim: 8, outDim: 16)) { $1 }
        arrays.merge(pair("base_model.model.final_layer.linear", inDim: 16, outDim: 8)) { $1 }
        arrays.merge(extra) { $1 }
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("klein-lora-\(UUID().uuidString).safetensors")
        try MLX.save(arrays: arrays, url: url)
        return url
    }

    func testSwapperAttachesTopLevelAndDetaches() throws {
        let model = Self.tinyModel()
        let url = try Self.writeAdapter()
        defer { try? FileManager.default.removeItem(at: url) }
        let swapper = KleinLoRASwapper(model: model, dtype: .float32)

        try swapper.set([(url, 1.0)])
        XCTAssertTrue(model.xEmbedder is LoRALinear)
        XCTAssertTrue(model.projOut is LoRALinear)
        XCTAssertTrue(model.transformerBlocks[0].attn.toOut is LoRALinear)
        XCTAssertEqual(swapper.activeKeys, ["attn.to_out", "x_embedder", "proj_out"])

        swapper.detach()
        XCTAssertFalse(model.xEmbedder is LoRALinear)
        XCTAssertFalse(model.projOut is LoRALinear)
        XCTAssertFalse(model.transformerBlocks[0].attn.toOut is LoRALinear)
        XCTAssertTrue(swapper.activeKeys.isEmpty)
    }

    /// An adapter with a tensor klein can't place throws before the DiT is touched: the set that
    /// was applied stays applied.
    func testUnknownKeyThrowsAndKeepsTheAppliedSet() throws {
        let model = Self.tinyModel()
        let good = try Self.writeAdapter()
        let bad = try Self.writeAdapter(["base_model.model.final_layer.adaLN_modulation.1.lora_A.weight":
                                            MLXRandom.normal([2, 16])])
        defer {
            try? FileManager.default.removeItem(at: good)
            try? FileManager.default.removeItem(at: bad)
        }
        let swapper = KleinLoRASwapper(model: model, dtype: .float32)
        try swapper.set([(good, 1.0)])
        XCTAssertThrowsError(try swapper.set([(bad, 1.0)])) { error in
            guard case KleinLoRA.LoRAError.unknownKeys(_, let keys)? = error as? KleinLoRA.LoRAError else {
                return XCTFail("expected unknownKeys, got \(error)")
            }
            XCTAssertEqual(keys, ["base_model.model.final_layer.adaLN_modulation.1"])
        }
        XCTAssertTrue(model.xEmbedder is LoRALinear, "the earlier set is untouched")
    }
}
