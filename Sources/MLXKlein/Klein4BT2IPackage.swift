// MLXEngine `textToImage` package over the FLUX.2-klein-4B core (Apache-2.0): a compact
// rectified-flow MMDiT (5 double + 20 single blocks) + Qwen3-4B 3-layer-tap conditioner +
// FLUX.2 VAE. Distilled 4-step, guidance 1.0. Multi-reference EDITING is the tier's
// differentiator (P6, follow-on). The fourth public textToImage backer (Lens, ERNIE, Z-Image).

import CoreGraphics
import Flux2VAE
import Foundation
import ImageIO
import MLX
import MLXProfiling
import MLXToolKit
import Tokenizers
import UniformTypeIdentifiers
import Klein

public enum KleinPackageError: Error, LocalizedError {
    case unreadableSnapshot(String)
    case pngEncode
    public var errorDescription: String? {
        switch self {
        case .unreadableSnapshot(let p): return "FLUX.2-klein snapshot not readable at \(p)."
        case .pngEncode: return "PNG encoding failed."
        }
    }
}

@InferenceActor
public final class Klein4BT2IPackage: ModelPackage {
    public typealias Configuration = KleinConfiguration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // 4B is plain Apache-2.0 (weights) + MIT (port). NOTE: the 9B variants are FLUX
            // Non-Commercial and are intentionally NOT wrapped here.
            license: LicenseDeclaration(weightLicense: .apache2, portCodeLicense: .mit),
            provenance: Provenance(sourceRepo: "black-forest-labs/FLUX.2-klein-4B", revision: "main", tier: 1),
            requirements: RequirementsManifest(
                // Split footprint (efficiency contract 1.14.0). Resident = DiT + Qwen3-4B encoder
                // + FLUX.2 VAE; activation = the 1024² denoise scratch (~quant-independent). Numbers
                // RE-BASELINED to in-app phys (IMAGE_AUTORUN, 2026-07-06): int4 floor 9.32 / peak 20.54
                // (activation 11.22). int8/bf16 scale resident by the DiT-weight delta (int4 2.35 →
                // int8 ~4.5 → bf16 7.75), activation held. Activation baselines assume the DiT's
                // fp32 activation stream (temb promotion — see Flux2ParallelSelfAttention.
                // outProjected): a future true-bf16 activation flow would shrink activation but
                // MUST keep the NAX row-chunk armed (mlx#3797). NOTE: the encoder-evict light tier is FAR
                // lower (phys @768² floor 5.13 / peak 10.06) — surfaced dynamically via
                // KleinConfiguration.FootprintConfigured when evictEncoder is set, so the 16 GB tier is
                // admissible without changing these resident-path defaults.
                footprints: [
                    QuantFootprint(quant: .bf16, residentBytes: 14_700_000_000, peakActivationBytes: 11_200_000_000),
                    QuantFootprint(quant: .int8, residentBytes: 11_500_000_000, peakActivationBytes: 11_200_000_000),
                    QuantFootprint(quant: .int4, residentBytes: 9_300_000_000, peakActivationBytes: 11_200_000_000),
                ],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: nil
            ),
            specialties: [],
            surfaces: [
                T2IContract.descriptor(
                    name: "flux2-klein-4b-t2i",
                    summary: "FLUX.2-klein-4B text-to-image (Apache-2.0): compact 4B rectified-flow "
                        + "MMDiT distilled to 4 steps (guidance 1.0, no negative prompt), Qwen3-4B "
                        + "3-layer conditioning + FLUX.2 VAE; ~6 s @1024² int4 on a 16 GB Mac.",
                    modes: []
                ),
                IEditContract.descriptor(
                    name: "flux2-klein-4b-edit",
                    summary: "FLUX.2-klein-4B multi-reference EDITING (its differentiator): compose a "
                        + "new image from one or more reference images (subject/style/scene) via "
                        + "reference-token conditioning — 'the subject from image 1, on a beach'. "
                        + "Pass conditioning images in prompt order.",
                    modes: []
                )
            ]
        )
    }

    let configuration: Configuration
    private var generator: KleinGenerator?
    /// Encoder-evict mode only: loads the Qwen3-4B conditioner on demand (nil when the encoder is
    /// held resident inside the generator).
    private var encoderProvider: (() async throws -> KleinTextEncoder)?

    public nonisolated init(configuration: Configuration) { self.configuration = configuration }

    public func load() async throws {
        guard generator == nil else { return }
        guard let snapshot = configuration.resolvedSnapshotDirectory(storeRoot: configuration.modelsRootDirectory),
            FileManager.default.fileExists(atPath: snapshot.appendingPathComponent("transformer").path)
        else { throw KleinPackageError.unreadableSnapshot(configuration.snapshotPath ?? configuration.repo) }

        let transformer = try KleinWeights.loadTransformer(snapshotPath: snapshot.path, dtype: .bfloat16)
        switch configuration.quant {
        case .int8: KleinWeights.quantizeDiT(transformer, bits: 8)
        case .int4: KleinWeights.quantizeDiT(transformer, bits: 4)
        default: break
        }
        // Optional LoRA: applied AFTER quantize so the activation-path add rides the QLoRALinear
        // (int4-safe) — turns this klein into a RefControl/style specialty. Same code path the CLI
        // `--lora` exercises. loraStrength scales the ΔW contribution.
        if let loraPath = configuration.loraPath {
            let url = URL(fileURLWithPath: loraPath)
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw KleinPackageError.unreadableSnapshot("LoRA not found: \(loraPath)")
            }
            try KleinLoRA.apply(loRAs: [(url, configuration.loraStrength)], to: transformer, dtype: .bfloat16)
        }
        let vae = try Flux2VAEWeights.loadVAE(directory: snapshot.appendingPathComponent("vae"), dtype: .float32)
        // Edit path: VAE encoder + bn stats (small; enables the imageEdit surface).
        let vaeEncoder = try KleinWeights.loadVAEEncoder(snapshotPath: snapshot.path, dtype: .float32)
        let vaeArrays = try MLX.loadArrays(url: snapshot.appendingPathComponent("vae/diffusion_pytorch_model.safetensors"))
        let bnMean = vaeArrays["bn.running_mean"]!.asType(.float32).reshaped(1, -1, 1, 1)
        let bnStd = MLX.sqrt(vaeArrays["bn.running_var"]!.asType(.float32).reshaped(1, -1, 1, 1) + 1e-4)
        eval(bnMean, bnStd)

        let snapPath = snapshot.path
        var textEncoder: KleinTextEncoder? = nil
        if configuration.evictEncoder {
            // Light tier: capture a per-request loader; keep the ~8 GB encoder out of the resident set.
            encoderProvider = {
                let enc = try KleinWeights.loadTextEncoder(snapshotPath: snapPath, dtype: .bfloat16)
                let tok = try await AutoTokenizer.from(
                    modelFolder: URL(fileURLWithPath: snapPath).appendingPathComponent("tokenizer"))
                return KleinTextEncoder(encoder: enc, tokenizer: tok)
            }
        } else {
            let enc = try KleinWeights.loadTextEncoder(snapshotPath: snapPath, dtype: .bfloat16)
            let tok = try await AutoTokenizer.from(
                modelFolder: URL(fileURLWithPath: snapPath).appendingPathComponent("tokenizer"))
            textEncoder = KleinTextEncoder(encoder: enc, tokenizer: tok)
        }
        generator = KleinGenerator(transformer: transformer, vae: vae, textEncoder: textEncoder,
            transformerDtype: .bfloat16, vaeEncoder: vaeEncoder, bnMean: bnMean, bnStd: bnStd)
    }

    public func unload() async {
        generator = nil
        encoderProvider = nil
        MLX.Memory.clearCache()
    }

    /// Encoder-evict: load the conditioner, encode prompt (+ negative for CFG), force-materialize the
    /// embeds off the encoder graph (`eval`), then drop the encoder + clear the pool BEFORE returning —
    /// so the ~8 GB encoder is reclaimed ahead of the DiT denoise peak. Parity-identical to the
    /// resident path (only WHEN the encoder frees changes).
    private func encodeAndEvict(prompt: String, negativePrompt: String?, guidance: Float) async throws
        -> (pos: MLXArray, neg: MLXArray?) {
        guard let encoderProvider else { throw PackageError.notLoaded }
        var enc: KleinTextEncoder? = try await encoderProvider()
        let pos = enc!.encode(prompt)
        let neg = guidance > 1.0 ? enc!.encode(negativePrompt ?? "") : nil
        eval([pos] + (neg.map { [$0] } ?? []))   // materialize off the encoder graph
        enc = nil                                 // release the conditioner (last strong ref)
        MLX.Memory.clearCache()                   // reclaim the ~8 GB before the denoise peak
        try Task.checkCancellation()              // CAN seam: encode done, before the denoise loop
        return (pos, neg)
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        // CAN-1: the entry checkpoint is the FIRST act of run() — before notLoaded validation
        // (engine ≥ 0.27.0). Mid-run cadence: post-encode seam in encodeAndEvict, per-denoise-step
        // `Task.isCancelled` break in KleinPipeline/KleinEditPipeline, cancelled-task decode skip,
        // and the post-generate checkpoints below rethrow CancellationError unchanged.
        try Task.checkCancellation()
        guard let generator else { throw PackageError.notLoaded }
        let prof = MLXProfiler.shared

        if let t2i = request as? T2IRequest {
            let width = ((t2i.width ?? 1024) / 16) * 16
            let height = ((t2i.height ?? 1024) / 16) * 16
            let steps = t2i.steps ?? configuration.defaultSteps
            // Base tier: request guidance overrides the config default (>1 ⇒ two-pass CFG + negative
            // prompt). Distilled config default is 1.0 ⇒ single forward, negative ignored.
            let guidance = Float(t2i.guidanceScale ?? Double(configuration.guidanceScale))
            prof.beginRun("flux2-klein textToImage steps=\(steps) g=\(guidance) evict=\(configuration.evictEncoder) \(width)x\(height)")
            let pixels: [UInt8], w: Int, h: Int
            if configuration.evictEncoder {
                let (pos, neg) = try await encodeAndEvict(
                    prompt: t2i.prompt, negativePrompt: t2i.negativePrompt, guidance: guidance)
                (pixels, w, h) = generator.generate(
                    promptEmbeds: pos, negativeEmbeds: neg, width: width, height: height,
                    steps: steps, seed: t2i.seed ?? 0, guidanceScale: guidance)
            } else {
                (pixels, w, h) = generator.generate(
                    prompt: t2i.prompt, width: width, height: height, steps: steps, seed: t2i.seed ?? 0,
                    negativePrompt: t2i.negativePrompt, guidanceScale: guidance)
            }
            prof.endRun(denominators: ["step": Double(steps)])
            try Task.checkCancellation()
            return T2IResponse(image: Image(format: .png, data: try Self.encodePNG(pixels: pixels, width: w, height: h), width: w, height: h))
        }

        if let edit = request as? IEditRequest {
            let width = ((edit.width ?? 1024) / 16) * 16
            let height = ((edit.height ?? 1024) / 16) * 16
            let steps = edit.steps ?? configuration.defaultSteps
            // Decode each conditioning image → [1,3,height,width] in [-1,1] (scaled to target).
            let refs = try edit.images.map { try Self.decodeReference($0, dim: width) }
            let guidance = Float(edit.guidanceScale ?? Double(configuration.guidanceScale))
            prof.beginRun("flux2-klein imageEdit refs=\(refs.count) steps=\(steps) g=\(guidance) evict=\(configuration.evictEncoder) \(width)x\(height)")
            let pixels: [UInt8], w: Int, h: Int
            if configuration.evictEncoder {
                let (pos, neg) = try await encodeAndEvict(
                    prompt: edit.prompt, negativePrompt: edit.negativePrompt, guidance: guidance)
                (pixels, w, h) = generator.generateEdit(
                    promptEmbeds: pos, negativeEmbeds: neg, referenceImages: refs, width: width,
                    height: height, steps: steps, seed: edit.seed ?? 0, guidanceScale: guidance)
            } else {
                (pixels, w, h) = generator.generateEdit(
                    prompt: edit.prompt, referenceImages: refs, width: width, height: height,
                    steps: steps, seed: edit.seed ?? 0,
                    negativePrompt: edit.negativePrompt, guidanceScale: guidance)
            }
            prof.endRun(denominators: ["step": Double(steps)])
            try Task.checkCancellation()
            return IEditResponse(image: Image(format: .png, data: try Self.encodePNG(pixels: pixels, width: w, height: h), width: w, height: h))
        }

        throw PackageError.unsupportedCapability(request.capability)
    }

    /// Decode a conditioning `Image` (PNG/etc.) → [1,3,dim,dim] in [-1,1], scaled to dim×dim.
    nonisolated static func decodeReference(_ image: Image, dim: Int) throws -> MLXArray {
        guard let src = CGImageSourceCreateWithData(image.data as CFData, nil),
              let cg = CGImageSourceCreateImageAtIndex(src, 0, nil)
        else { throw KleinPackageError.pngEncode }
        let cs = CGColorSpace(name: CGColorSpace.sRGB)!
        var buf = [UInt8](repeating: 0, count: dim * dim * 4)
        let ctx = CGContext(data: &buf, width: dim, height: dim, bitsPerComponent: 8,
            bytesPerRow: dim * 4, space: cs, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: dim, height: dim))
        var rgb = [Float](repeating: 0, count: 3 * dim * dim)
        for p in 0..<(dim * dim) {
            rgb[p] = Float(buf[p*4]) / 127.5 - 1
            rgb[dim*dim + p] = Float(buf[p*4+1]) / 127.5 - 1
            rgb[2*dim*dim + p] = Float(buf[p*4+2]) / 127.5 - 1
        }
        return MLXArray(rgb, [1, 3, dim, dim])
    }

    nonisolated static func encodePNG(pixels: [UInt8], width: Int, height: Int) throws -> Data {
        let cs = CGColorSpace(name: CGColorSpace.sRGB)!
        guard let ctx = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
            bytesPerRow: width * 4, space: cs, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else { throw KleinPackageError.pngEncode }
        let buf = ctx.data!.bindMemory(to: UInt8.self, capacity: width * height * 4)
        for i in 0..<(width * height) {
            buf[i * 4] = pixels[i * 3]; buf[i * 4 + 1] = pixels[i * 3 + 1]
            buf[i * 4 + 2] = pixels[i * 3 + 2]; buf[i * 4 + 3] = 255
        }
        guard let image = ctx.makeImage() else { throw KleinPackageError.pngEncode }
        let out = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(out, UTType.png.identifier as CFString, 1, nil)
        else { throw KleinPackageError.pngEncode }
        CGImageDestinationAddImage(dest, image, nil)
        guard CGImageDestinationFinalize(dest) else { throw KleinPackageError.pngEncode }
        return out as Data
    }
}

extension Klein4BT2IPackage {
    public nonisolated static var registration: PackageRegistration { .of(Klein4BT2IPackage.self) }
}
