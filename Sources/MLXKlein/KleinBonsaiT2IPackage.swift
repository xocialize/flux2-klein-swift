// MLXEngine `textToImage` package for **Bonsai Image Ternary 4B** (Prism ML, Apache-2.0) — the compact tier.
//
// FLUX.2 klein 4B with the transformer RETRAINED to ternary weights: the same 5 double + 20 single MMDiT, the same
// Qwen3-4B 3-layer conditioner and FLUX.2 VAE, 4 steps, guidance 1. It ships pre-packed for MLX: a 2-bit / group-128
// DiT (1.43 GB) and a 4-bit Qwen3-4B (2.26 GB), 3.88 GB in all. So this is a weight swap on the klein core with its
// own loaders (`KleinWeights.loadPackedTransformer` / `loadPackedTextEncoder`), not a new architecture.
//
// What it is for, and what it is not (evaluation AB-R-1130, memo mlxengine-image/BONSAI-IMAGE-EVAL.md):
// - A REDRAW, not a quantization of klein: about 15 dB against klein bf16 at the same seed. Never a stand-in where a
//   result must match another klein tier.
// - Text-to-image only. Its one measured edit kept half the source scene, and Prism ML claims text-to-image only, so
//   there is no edit surface.
// - A 1024²-class model: 10 / 12 usable at 1024² (klein int4 10, bf16 11) but 8 / 12 at 512² (klein 10–11).
// - No adapters: klein LoRAs were trained against klein's weights and are untested on the retrained base.

import Flux2VAE
import Foundation
import Klein
import MLX
import MLXProfiling
import MLXToolKit
import Tokenizers

/// Init-time configuration of `KleinBonsaiT2IPackage`. The snapshot is Prism ML's repo as published:
/// `transformer-packed-mflux/`, `text_encoder-mlx-4bit/`, `vae/`, `tokenizer/`.
public struct KleinBonsaiConfiguration: PackageConfiguration, ModelStorable {
    public static let defaultRepo = "prism-ml/bonsai-image-ternary-4B-mlx-2bit"
    /// The revision the gates and the footprints were measured on (2026-06-01).
    public static let measuredRevision = "2c24c81b934a658ba5590cf39088ba929985b4a8"
    static let transformerDirectory = "transformer-packed-mflux"
    static let textEncoderDirectory = "text_encoder-mlx-4bit"

    public var repo: String
    public var revision: String?
    /// An explicit local snapshot; nil ⇒ the ModelStore's copy of `repo`.
    public var snapshotPath: String?
    public var defaultSteps: Int
    public var modelsRootDirectory: URL?
    /// Latent upscaling (`metaData["latentUpscale"] = 1 | 2`), as on the klein tiers: declares FlowUpscaler's
    /// weights as a second source. Off ⇒ the option is refused and nothing extra is downloaded.
    public var latentUpscaler: Bool
    /// A local `flow_upscaler.safetensors` (or its folder), overriding the store. Env-specific ⇒ not persisted.
    public var latentUpscalerPath: String?

    public init(
        repo: String = KleinBonsaiConfiguration.defaultRepo,
        revision: String? = KleinBonsaiConfiguration.measuredRevision,
        snapshotPath: String? = nil,
        defaultSteps: Int = 4,
        modelsRootDirectory: URL? = nil,
        latentUpscaler: Bool = true,
        latentUpscalerPath: String? = nil
    ) {
        self.repo = repo
        self.revision = revision
        self.snapshotPath = snapshotPath
        self.defaultSteps = defaultSteps
        self.modelsRootDirectory = modelsRootDirectory
        self.latentUpscaler = latentUpscaler
        self.latentUpscalerPath = latentUpscalerPath
    }

    private enum CodingKeys: String, CodingKey {
        case repo, revision, defaultSteps, latentUpscaler
    }

    public init(from decoder: Swift.Decoder) throws {   // Tokenizers also exports a `Decoder`
        let c = try decoder.container(keyedBy: CodingKeys.self)
        repo = try c.decodeIfPresent(String.self, forKey: .repo) ?? Self.defaultRepo
        revision = try c.decodeIfPresent(String.self, forKey: .revision)
        snapshotPath = nil      // env-specific; not persisted
        defaultSteps = try c.decodeIfPresent(Int.self, forKey: .defaultSteps) ?? 4
        modelsRootDirectory = nil
        latentUpscaler = try c.decodeIfPresent(Bool.self, forKey: .latentUpscaler) ?? true
        latentUpscalerPath = nil
    }
}

extension KleinBonsaiConfiguration: WeightSourcing {
    public var weightSources: [WeightSource] {
        [snapshotSource] + (latentUpscaler ? [upscalerSource] : [])
    }

    var snapshotSource: WeightSource {
        WeightSource(role: "snapshot", repo: repo, revision: revision,
                     matching: ["\(Self.transformerDirectory)/*", "\(Self.textEncoderDirectory)/*", "vae/*",
                                "tokenizer/*", "*.json", "LICENSE", "NOTICE.md"])
    }

    var upscalerSource: WeightSource {
        WeightSource(role: "latent-upscaler", repo: KleinConfiguration.latentUpscalerRepo,
                     matching: [KleinConfiguration.latentUpscalerFile, "LICENSE", "README.md"])
    }

    public func resolvedSnapshotDirectory(storeRoot: URL?) -> URL? {
        if let snapshotPath { return URL(fileURLWithPath: snapshotPath) }
        return ModelStore(root: storeRoot).directory(for: repo)
    }

    /// True when `directory` holds the three components load() reads.
    static func isComplete(_ directory: URL) -> Bool {
        [transformerDirectory, textEncoderDirectory, "vae"].allSatisfy {
            FileManager.default.fileExists(atPath: directory.appendingPathComponent($0).path)
        }
    }

    /// FlowUpscaler's weights, resolved as the klein tiers resolve them.
    public func resolvedLatentUpscalerFile(storeRoot: URL?) -> URL? {
        KleinConfiguration(latentUpscaler: latentUpscaler, latentUpscalerPath: latentUpscalerPath)
            .resolvedLatentUpscalerFile(storeRoot: storeRoot)
    }

    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        var missing: [WeightSource] = []
        if let dir = resolvedSnapshotDirectory(storeRoot: storeRoot), Self.isComplete(dir) {
        } else {
            missing.append(snapshotSource)
        }
        if latentUpscaler, resolvedLatentUpscalerFile(storeRoot: storeRoot) == nil { missing.append(upscalerSource) }
        return missing
    }
}

extension KleinBonsaiConfiguration: WorkloadDeclaring {
    /// The request's output pixels, the axis of the manifest's `ActivationScaling`.
    public func workloadUnits(for request: any CapabilityRequest) -> Double? {
        guard let t2i = request as? T2IRequest else { return nil }
        return KleinLatentUpscaleOption.outputPixels(width: t2i.width, height: t2i.height, metaData: t2i.metaData)
    }
}

/// Declared footprint of the Bonsai tier, from `klein-cli --pkg-bonsai` (AB-R-1138):
/// Release, M5 Max, MLX pool capped at the engine's 2 GB, process `phys_footprint`, three runs per size.
///
/// - Resident: 3.57 GB after load (MLX floor 3.22 GB: 2-bit DiT 1.36, 4-bit encoder layers 0–26 1.67, fp32 VAE).
/// - ≤ 1024²: process peak 7.57 / 7.93 / 7.88 GB at 512² / 768² / 1024² ⇒ 4.36 GB over resident at most. The
///   peak is the denoise loop plus the pool, not the decode (row bands), so it barely moves with size.
/// - Latent upscale: 2048² 11.57 GB (8.0 over resident), 4096² 30.03 GB (26.46 over; + 4 % ⇒ 27.5 GB).
///
/// The line runs through 4.4 GB at 1024² and 27.5 GB at 4096², and sits above the 2048² point (9.0 vs 8.0 GB).
/// These are CLI figures: an in-app run should confirm them before a 16 GB or 8 GB product tier leans on them.
enum KleinBonsaiFootprint {
    static let residentBytes: UInt64 = 3_600_000_000
    static let activationBytes: UInt64 = 4_400_000_000
    static let activationScaling = ActivationScaling(
        axis: .outputPixels, baseBytes: 2_860_000_000, bytesPerUnit: 1468.7,
        measuredCeiling: Double(KleinLatentUpscaleOption.maxOutputPixels))
}

@InferenceActor
public final class KleinBonsaiT2IPackage: ModelPackage {
    public typealias Configuration = KleinBonsaiConfiguration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // Apache-2.0 on Prism ML's weights and on both ingredients (FLUX.2 klein 4B, Qwen3-4B; NOTICE.md).
            // Prism ML asks for, and does not require, the credit "Created using Bonsai Image by Prism ML."
            license: LicenseDeclaration(weightLicense: .apache2, portCodeLicense: .mit),
            provenance: Provenance(sourceRepo: KleinBonsaiConfiguration.defaultRepo,
                                   revision: KleinBonsaiConfiguration.measuredRevision, tier: 1),
            requirements: RequirementsManifest(
                // One lane, one footprint (`KleinBonsaiFootprint`). The contract's `Quant` has no 2-bit case, so the
                // entry is filed under `.int4`, the lowest tier it can name; nothing selects on it (the configuration
                // is not `QuantConfigured`).
                footprints: [
                    QuantFootprint(quant: .int4, residentBytes: KleinBonsaiFootprint.residentBytes,
                                   peakActivationBytes: KleinBonsaiFootprint.activationBytes,
                                   activationScaling: KleinBonsaiFootprint.activationScaling),
                ],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: nil
            ),
            specialties: [],
            surfaces: [
                T2IContract.descriptor(
                    name: "flux2-klein-4b-bonsai-t2i",
                    summary: "Bonsai Image Ternary 4B (Prism ML, Apache-2.0) text-to-image: FLUX.2 klein 4B with the "
                        + "transformer retrained to ternary weights, a 3.9 GB download and the lightest klein-class "
                        + "tier. 4 steps, guidance 1.0, no negative prompt. Draw at 1024²-class sizes: it follows "
                        + "prompts about as well as klein int4 there and clearly worse at 512². It draws a different "
                        + "picture from the klein tiers at the same seed. No editing, no adapters. "
                        + "metaData[\"latentUpscale\"] = 1 | 2 as on the klein tiers.",
                    modes: []
                )
            ]
        )
    }

    let configuration: Configuration
    private var generator: KleinGenerator?
    /// FlowUpscaler, loaded on the first request that asks for `latentUpscale`.
    private var flowUpscaler: FlowUpscaler?

    public nonisolated init(configuration: Configuration) { self.configuration = configuration }

    public func load() async throws {
        guard generator == nil else { return }
        guard let snapshot = configuration.resolvedSnapshotDirectory(storeRoot: configuration.modelsRootDirectory),
              KleinBonsaiConfiguration.isComplete(snapshot)
        else { throw KleinPackageError.unreadableSnapshot(configuration.snapshotPath ?? configuration.repo) }

        let transformer = try KleinWeights.loadPackedTransformer(
            directory: snapshot.appendingPathComponent(KleinBonsaiConfiguration.transformerDirectory))
        let vae = try Flux2VAEWeights.loadVAE(directory: snapshot.appendingPathComponent("vae"), dtype: .float32)

        // The 4-bit encoder stays resident (1.7 GB). Evicting it per request was measured and dropped: it takes
        // 1.6 GB off the MLX peak, but the reload makes the process footprint spike past the resident lane's.
        let encoder = try KleinWeights.loadPackedTextEncoder(
            directory: snapshot.appendingPathComponent(KleinBonsaiConfiguration.textEncoderDirectory))
        let tokenizer = try await AutoTokenizer.from(modelFolder: snapshot.appendingPathComponent("tokenizer"))
        let textEncoder = KleinTextEncoder(encoder: encoder, tokenizer: tokenizer)
        generator = KleinGenerator(transformer: transformer, vae: vae, textEncoder: textEncoder,
                                   transformerDtype: .bfloat16)
    }

    public func unload() async {
        generator = nil
        flowUpscaler = nil
        MLX.Memory.clearCache()
    }

    /// The request's latent upscale, validated against the draw; nil when not asked for.
    private func latentUpscale(metaData: MetaData, width: Int, height: Int, seed: UInt64) throws
        -> KleinLatentUpscale? {
        let passes = try KleinLatentUpscaleOption.passes(metaData)
        try KleinLatentUpscaleOption.validate(width: width, height: height, passes: passes)
        guard passes > 0 else { return nil }
        guard configuration.latentUpscaler else {
            throw PackageError.unsupportedRequestFeature(
                "metaData[\"latentUpscale\"] — this configuration has latentUpscaler off")
        }
        if flowUpscaler == nil {
            guard let url = configuration.resolvedLatentUpscalerFile(storeRoot: configuration.modelsRootDirectory)
            else {
                throw KleinPackageError.unreadableSnapshot(
                    "FlowUpscaler weights (\(KleinConfiguration.latentUpscalerRepo)) not found")
            }
            flowUpscaler = try FlowUpscaler.load(url: url, dtype: .float32)
        }
        return KleinLatentUpscale(upscaler: flowUpscaler!, passes: passes, seed: seed)
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        try Task.checkCancellation()                 // CAN-1: the entry checkpoint, before any validation
        guard let generator else { throw PackageError.notLoaded }
        guard let t2i = request as? T2IRequest else { throw PackageError.unsupportedCapability(request.capability) }
        guard t2i.resolvedAdapters.isEmpty else {
            throw PackageError.unsupportedRequestFeature(
                "adapters — klein adapters are untested on the Bonsai base; use a klein tier")
        }
        let width = ((t2i.width ?? 1024) / 16) * 16
        let height = ((t2i.height ?? 1024) / 16) * 16
        let steps = t2i.steps ?? configuration.defaultSteps
        let seed = t2i.seed ?? 0
        let upscale = try latentUpscale(metaData: t2i.metaData, width: width, height: height, seed: seed)
        let prof = MLXProfiler.shared
        prof.beginRun("flux2-klein-bonsai textToImage steps=\(steps) \(width)x\(height) "
            + " latentUpscale=\(upscale?.passes ?? 0)")
        let (pixels, w, h) = KleinProgress.$sink.withValue(KleinProgressBridge.forward) {
            generator.generate(prompt: t2i.prompt, width: width, height: height, steps: steps, seed: seed,
                               negativePrompt: nil, guidanceScale: 1.0, latentUpscale: upscale)
        }
        prof.endRun(denominators: ["step": Double(steps)])
        try Task.checkCancellation()
        RunProgress.report(.postprocess)
        return T2IResponse(image: Image(
            format: .png, data: try Klein4BT2IPackage.encodePNG(pixels: pixels, width: w, height: h),
            width: w, height: h))
    }
}

extension KleinBonsaiT2IPackage {
    /// The author one-liner the engine registers (its own PackageID, beside the klein tiers).
    public nonisolated static var registration: PackageRegistration { .of(KleinBonsaiT2IPackage.self) }
}
