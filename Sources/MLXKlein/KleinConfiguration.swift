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
    /// Default CFG scale. Distilled tier = 1.0 (guidance ignored, single forward). Base tier > 1
    /// enables two-pass classifier-free guidance + negative prompt (see KleinPipeline).
    public var guidanceScale: Float
    public var modelsRootDirectory: URL?

    public init(
        repo: String = "mlx-community/FLUX.2-klein-4B-bf16",
        revision: String? = nil,
        quant: Quant = .bf16,
        snapshotPath: String? = nil,
        defaultSteps: Int = 4,
        guidanceScale: Float = 1.0,
        modelsRootDirectory: URL? = nil
    ) {
        self.repo = repo
        self.revision = revision
        self.quant = quant
        self.snapshotPath = snapshotPath
        self.defaultSteps = defaultSteps
        self.guidanceScale = guidanceScale
        self.modelsRootDirectory = modelsRootDirectory
    }

    /// Distilled fast tier (the default): klein-4B, 4-step, guidance 1.0 (no CFG).
    public static func turbo(quant: Quant = .bf16, snapshotPath: String? = nil) -> KleinConfiguration {
        KleinConfiguration(repo: "mlx-community/FLUX.2-klein-4B-bf16", quant: quant,
                           snapshotPath: snapshotPath, defaultSteps: 4, guidanceScale: 1.0)
    }

    /// Base quality tier: klein-base-4B, ~28-step with two-pass CFG (guidance 4.0) + negative prompts.
    public static func base(quant: Quant = .bf16, snapshotPath: String? = nil) -> KleinConfiguration {
        KleinConfiguration(repo: "mlx-community/FLUX.2-klein-base-4B-bf16", quant: quant,
                           snapshotPath: snapshotPath, defaultSteps: 28, guidanceScale: 4.0)
    }

    private enum CodingKeys: String, CodingKey { case repo, revision, quant, defaultSteps, guidanceScale }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        repo = try c.decode(String.self, forKey: .repo)
        revision = try c.decodeIfPresent(String.self, forKey: .revision)
        quant = try c.decode(Quant.self, forKey: .quant)
        defaultSteps = try c.decodeIfPresent(Int.self, forKey: .defaultSteps) ?? 4
        guidanceScale = try c.decodeIfPresent(Float.self, forKey: .guidanceScale) ?? 1.0
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
