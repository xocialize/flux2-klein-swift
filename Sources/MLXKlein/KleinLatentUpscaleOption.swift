// `metaData["latentUpscale"] = 1 | 2` (AB-A-0257): the request's width × height is the DRAW (at most 1024² of area);
// FlowUpscaler doubles the latents once or twice before the decode, so the image is 2× or 4× that on each side, up to
// 4096². It keeps the 1024² composition: downscaled back, the 2048² result is DISTS 0.015–0.034 from the 1024² decode
// (AB-R-0795), where Klein drawing 2048² itself draws another picture.

import Foundation
import Klein
import MLXToolKit

public enum KleinLatentUpscaleOption {
    /// The metaData key.
    public static let key = "latentUpscale"
    /// The largest draw a latent upscale starts from: FlowUpscaler was distilled at 512 → 1024, and 1024² × 4² is the
    /// 4096² ceiling the footprint is measured to.
    public static let maxDrawPixels = 1024 * 1024
    /// The largest image: 4096² (two passes from 1024²).
    public static let maxOutputPixels = 4096 * 4096

    /// Activation over output pixels (see the manifest): the line through the representative 1024² point (the 11.2 GB
    /// scalar) and 27.0 GB at 4096², so a ≤ 1024² run still reserves the scalar and a latent-upscaled image its decode.
    /// It sits above every measured point: 2048² upscaled 7.1 GB vs 14.4 GB; 4096² 25.96 GB vs 27.0 GB.
    static let activationScaling = ActivationScaling(
        axis: .outputPixels, baseBytes: 10_146_000_000, bytesPerUnit: 1004.6, measuredCeiling: Double(maxOutputPixels))

    /// The base (CFG) tier's line: through its 12.4 GB scalar at 1024² and the same 27.0 GB at 4096² (CFG doubles the
    /// denoise passes, not the decode).
    static let baseTierActivationScaling = ActivationScaling(
        axis: .outputPixels, baseBytes: 11_426_700_000, bytesPerUnit: 928.2, measuredCeiling: Double(maxOutputPixels))

    /// The encoder-evict tier's line: through its 5.0 GB hint at 768² (the tier's intended size) and the same 27.0 GB
    /// at 4096² — the upscale and the fp32 decode do not depend on the tier.
    static let evictActivationScaling = ActivationScaling(
        axis: .outputPixels, baseBytes: 4_198_000_000, bytesPerUnit: 1359.1, measuredCeiling: Double(maxOutputPixels))

    /// Passes asked for: 0 when absent (or null / 0), else 1 or 2. Throws on anything else.
    public static func passes(_ metaData: MetaData) throws -> Int {
        let n: Int
        switch metaData[key] {
        case nil, .null?: return 0
        case .int(let v)?: n = v
        case .double(let v)? where v == v.rounded() && abs(v) <= 16: n = Int(v)
        case .string(let v)? where Int(v) != nil: n = Int(v)!
        default:
            throw PackageError.unsupportedRequestFeature("metaData[\"\(key)\"] must be 1 or 2 (×2 or ×4)")
        }
        guard (0 ... KleinLatentUpscale.maxPasses).contains(n) else {
            throw PackageError.unsupportedRequestFeature(
                "metaData[\"\(key)\"] = \(n): 1 (×2) or 2 (×4) passes; a third mostly sharpens and 8192² costs ~52 GB")
        }
        return n
    }

    /// The largest plain draw: 2048², measured through this package with the row-band decode (MLX peak 6.9 GB over
    /// resident, under the line's 14.4 GB). Past it nothing was measured, so run() refuses rather than admit a draw the
    /// line no longer describes (AB-L-0202, AB-T-0204).
    public static let maxNativeDrawPixels = 2048 * 2048

    /// Check a draw of width × height for `passes`: with an upscale, at most 1024² of area; without, at most 2048².
    static func validate(width: Int, height: Int, passes: Int) throws {
        guard passes > 0 else {
            guard width * height <= maxNativeDrawPixels else {
                throw PackageError.unsupportedRequestFeature(
                    "flux2-klein draws at most 2048² of area (got \(width)×\(height)); for a larger image draw ≤ 1024² "
                        + "with metaData[\"\(key)\"] = 1 | 2")
            }
            return
        }
        guard width * height <= maxDrawPixels else {
            throw PackageError.unsupportedRequestFeature(
                "metaData[\"\(key)\"] draws at most 1024² of area and upscales that (got \(width)×\(height)); "
                    + "ask for the draw size, the image comes back \(1 << passes)× on each side")
        }
    }

    /// Output pixels of a request, sized as run() sizes it (draw /16, then 2^passes per side). A malformed option counts
    /// as no upscale here; run() refuses it.
    static func outputPixels(width: Int?, height: Int?, metaData: MetaData) -> Double {
        let w = ((width ?? 1024) / 16) * 16, h = ((height ?? 1024) / 16) * 16
        let p = (try? passes(metaData)) ?? 0
        return Double(w * h) * Double(1 << (2 * p))
    }
}

extension KleinConfiguration: WorkloadDeclaring {
    /// The request's output pixels (the manifest's `ActivationScaling` axis): the engine reserves for this job's image
    /// and refuses one past the 4096² ceiling before admission.
    public func workloadUnits(for request: any CapabilityRequest) -> Double? {
        switch request {
        case let t2i as T2IRequest:
            return KleinLatentUpscaleOption.outputPixels(width: t2i.width, height: t2i.height, metaData: t2i.metaData)
        case let edit as IEditRequest:
            return KleinLatentUpscaleOption.outputPixels(width: edit.width, height: edit.height, metaData: edit.metaData)
        default:
            return nil
        }
    }
}
