// The packed (pre-quantized) loaders against Bonsai Image Ternary 4B's real tensor names and shapes (AB-R-1130),
// read from the hub's safetensors headers into Fixtures/bonsai-packed-headers.json. Offline and weights-free: the
// skeletons are lazy random initialisations, and only their keys and shapes are read, never their values.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import Klein

final class PackedWeightsTests: XCTestCase {

    private struct Fixture: Decodable {
        struct Tensor: Decodable {
            let dtype: String, shape: [Int]
            init(from decoder: Decoder) throws {
                var c = try decoder.unkeyedContainer()
                dtype = try c.decode(String.self)
                shape = try c.decode([Int].self)
            }
        }
        let transformer: [String: Tensor]
        let text_encoder: [String: Tensor]
    }

    private func fixture() throws -> Fixture {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appendingPathComponent("Fixtures/bonsai-packed-headers.json")
        return try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url))
    }

    private func shapes(_ module: Module) -> [String: [Int]] {
        Dictionary(uniqueKeysWithValues: module.parameters().flattened().map { ($0.0, $0.1.shape) })
    }

    /// The 2-bit / group-128 skeleton has exactly the pack's 369 tensors, each at the pack's shape: 100 packed
    /// Linears (U32 weight + scales + biases) and 69 full-precision tensors. `update(verify: .all)` relies on it.
    func testTransformerSkeletonMatchesThePack() throws {
        let pack = try fixture().transformer
        XCTAssertEqual(pack.count, 369)
        XCTAssertEqual(pack.values.filter { $0.dtype == "U32" }.count, 100)
        let expected = Dictionary(uniqueKeysWithValues: pack.map { (KleinWeights.remapDiTKey($0.key), $0.value.shape) })
        let skeleton = KleinWeights.packedTransformerSkeleton(.init(bits: 2, groupSize: 128))
        XCTAssertEqual(shapes(skeleton), expected)
    }

    /// Every layer the pack leaves at full precision is one `quantizeDiT` keeps too, and the reverse.
    func testPackSkipsWhatQuantizeDiTSkips() throws {
        for (key, tensor) in try fixture().transformer where key.hasSuffix(".weight") && tensor.shape.count == 2 {
            let path = KleinWeights.remapDiTKey(String(key.dropLast(".weight".count)))
            XCTAssertEqual(tensor.dtype != "U32", KleinWeights.keepHiPrecision(path: path), key)
        }
    }

    /// The 4-bit / group-64 tap encoder takes the pack's layers 0–26 and embedding, and nothing else of it.
    func testTextEncoderSkeletonMatchesThePack() throws {
        let pack = try fixture().text_encoder
        var expected: [String: [Int]] = [:]
        for (key, tensor) in pack {
            if let mapped = KleinWeights.textEncoderKey(key, keepBelow: 27) { expected[mapped] = tensor.shape }
        }
        XCTAssertLessThan(expected.count, pack.count)             // layers 27–35 and the final norm are dropped
        XCTAssertNil(KleinWeights.textEncoderKey("model.layers.27.mlp.up_proj.weight", keepBelow: 27))
        XCTAssertNil(KleinWeights.textEncoderKey("model.norm.weight", keepBelow: 27))
        XCTAssertNil(KleinWeights.textEncoderKey("lm_head.weight", keepBelow: 27))

        // Qwen3-4B, as the pack's config.json states it.
        let config = try JSONDecoder().decode(Qwen3EncoderConfiguration.self, from: Data("""
            {"hidden_size": 2560, "num_hidden_layers": 36, "intermediate_size": 9728, "num_attention_heads": 32,
             "rms_norm_eps": 1e-06, "vocab_size": 151936, "num_key_value_heads": 8, "rope_theta": 1000000,
             "head_dim": 128, "max_position_embeddings": 40960}
            """.utf8))
        let skeleton = KleinWeights.packedTextEncoderSkeleton(
            config, tapLayers: [9, 18, 27], quantization: .init(bits: 4, groupSize: 64))
        XCTAssertEqual(shapes(skeleton), expected)
    }

    /// Both places a pack states its quantization: Prism ML's quantization_config.json and mlx-lm's config.json.
    func testPackedQuantizationReadsBothFiles() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("klein-packed-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        func write(_ name: String, _ json: String) throws -> URL {
            let url = dir.appendingPathComponent(name)
            try Data(json.utf8).write(to: url)
            return url
        }
        let flat = try write("quantization_config.json",
                             #"{"format": "mlx-packed-affine", "solver": "ternary", "bits": 2, "group_size": 128}"#)
        XCTAssertEqual(try KleinWeights.PackedQuantization.load(flat), .init(bits: 2, groupSize: 128))
        let nested = try write("config.json", #"{"model_type": "qwen3", "quantization": {"group_size": 64, "bits": 4}}"#)
        XCTAssertEqual(try KleinWeights.PackedQuantization.load(nested), .init(bits: 4, groupSize: 64))
        let neither = try write("plain.json", #"{"model_type": "qwen3"}"#)
        XCTAssertThrowsError(try KleinWeights.PackedQuantization.load(neither))
    }
}
