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
                // + FLUX.2 VAE. Measured via klein-cli @1024²/4-step:
                //   bf16: DiT 7.75 GB + encoder ~8 GB + VAE ~0.3 GB ≈ 16 GB; peak ~24 GB.
                //   int4: DiT 2.35 GB + encoder ~8 GB (bf16) + VAE 0.3 ≈ 11 GB; peak 19.2 GB.
                // [residentBytes = measured active post-load; peakActivationBytes GPU-smoke —
                //  under-reads process phys ~2.7× (BiRefNet); phys re-baseline pending in-app.]
                footprints: [
                    QuantFootprint(quant: .bf16, residentBytes: 16_000_000_000, peakActivationBytes: 8_000_000_000),
                    QuantFootprint(quant: .int8, residentBytes: 12_000_000_000, peakActivationBytes: 8_000_000_000),
                    QuantFootprint(quant: .int4, residentBytes: 11_000_000_000, peakActivationBytes: 8_000_000_000),
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
                        + "3-layer conditioning + FLUX.2 VAE; ~6 s @1024² int4 on a 16 GB Mac. "
                        + "Multi-reference editing (its differentiator) ships as a follow-on.",
                    modes: []
                )
            ]
        )
    }

    let configuration: Configuration
    private var generator: KleinGenerator?

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
        let vae = try Flux2VAEWeights.loadVAE(directory: snapshot.appendingPathComponent("vae"), dtype: .float32)
        let encoder = try KleinWeights.loadTextEncoder(snapshotPath: snapshot.path, dtype: .bfloat16)
        let tokenizer = try await AutoTokenizer.from(modelFolder: snapshot.appendingPathComponent("tokenizer"))
        let textEncoder = KleinTextEncoder(encoder: encoder, tokenizer: tokenizer)
        generator = KleinGenerator(transformer: transformer, vae: vae, textEncoder: textEncoder, transformerDtype: .bfloat16)
    }

    public func unload() async {
        generator = nil
        MLX.Memory.clearCache()
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        guard let generator else { throw PackageError.notLoaded }
        guard request.capability == .textToImage, let t2i = request as? T2IRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }
        try Task.checkCancellation()
        let width = ((t2i.width ?? 1024) / 16) * 16
        let height = ((t2i.height ?? 1024) / 16) * 16
        let steps = t2i.steps ?? configuration.defaultSteps

        let prof = MLXProfiler.shared
        prof.beginRun("flux2-klein textToImage steps=\(steps) \(width)x\(height)")
        let (pixels, w, h) = generator.generate(
            prompt: t2i.prompt, width: width, height: height, steps: steps, seed: t2i.seed ?? 0)
        prof.endRun(denominators: ["step": Double(steps)])

        try Task.checkCancellation()
        let png = try Self.encodePNG(pixels: pixels, width: w, height: h)
        return T2IResponse(image: Image(format: .png, data: png, width: w, height: h))
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
