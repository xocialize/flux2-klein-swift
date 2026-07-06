// Init-time configuration for the FLUX.2-klein-4B MLXEngine package (C9). A snapshot is a
// diffusers tree: transformer/ text_encoder/ vae/ tokenizer/. `snapshotPath` is an explicit
// local override; nil ⇒ resolve against the ModelStore after auto-materializing `weightSources`.

import Foundation
import MLXToolKit

public struct KleinConfiguration: PackageConfiguration, ModelStorable, QuantConfigured {
    /// mlx-community repo (bf16; int8/int4 quantize at load from it).
    public var repo: String
    public var revision: String?
    /// DiT quant tier (bf16/int8/int4). Surfaced to the MemoryGovernor via QuantConfigured.
    public var quant: Quant
    public var snapshotPath: String?
    public var defaultSteps: Int
    public var modelsRootDirectory: URL?

    public init(
        repo: String = "mlx-community/FLUX.2-klein-4B-bf16",
        revision: String? = nil,
        quant: Quant = .bf16,
        snapshotPath: String? = nil,
        defaultSteps: Int = 4,
        modelsRootDirectory: URL? = nil
    ) {
        self.repo = repo
        self.revision = revision
        self.quant = quant
        self.snapshotPath = snapshotPath
        self.defaultSteps = defaultSteps
        self.modelsRootDirectory = modelsRootDirectory
    }

    private enum CodingKeys: String, CodingKey { case repo, revision, quant, defaultSteps }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        repo = try c.decode(String.self, forKey: .repo)
        revision = try c.decodeIfPresent(String.self, forKey: .revision)
        quant = try c.decode(Quant.self, forKey: .quant)
        defaultSteps = try c.decodeIfPresent(Int.self, forKey: .defaultSteps) ?? 4
    }
}

extension KleinConfiguration: WeightSourcing {
    /// bf16 snapshot (one repo; klein-4B ships all-bf16). int8/int4 quantize at load from it —
    /// pre-quantized repos are a later download-size optimization.
    public var weightSources: [WeightSource] {
        [WeightSource(role: "snapshot", repo: repo, revision: revision,
                      matching: ["transformer/*", "text_encoder/*", "vae/*", "tokenizer/*", "*.json"])]
    }

    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        let fm = FileManager.default
        if let snapshotPath,
           fm.fileExists(atPath: URL(fileURLWithPath: snapshotPath).appendingPathComponent("transformer").path) {
            return []
        }
        guard let dir = ModelStore(root: storeRoot).directory(for: repo) else { return weightSources }
        let present = fm.fileExists(atPath: dir.appendingPathComponent("transformer").path)
            && fm.fileExists(atPath: dir.appendingPathComponent("vae").path)
        return present ? [] : weightSources
    }

    public func resolvedSnapshotDirectory(storeRoot: URL?) -> URL? {
        if let snapshotPath { return URL(fileURLWithPath: snapshotPath) }
        return ModelStore(root: storeRoot).directory(for: repo)
    }
}
