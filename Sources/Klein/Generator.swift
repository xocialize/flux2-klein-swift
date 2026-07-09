// KleinGenerator — loaded-model holder that drives KleinPipeline.generate and returns
// interleaved RGB8 pixels. Keeps the MLXEngine wrapper thin (mirrors ZImageGenerator).

import Foundation
import Flux2VAE
import MLX

public final class KleinGenerator {
    public let transformer: Flux2Transformer
    public let vae: Flux2VAE
    /// The Qwen3-4B conditioner. Optional: nil in **encoder-evict** mode, where the caller loads the
    /// encoder, computes embeds, evicts it, then drives the embeds-based `generate`/`generateEdit`
    /// overloads directly — so the ~8 GB encoder never co-resides with the denoise activation peak.
    public let textEncoder: KleinTextEncoder?
    public let transformerDtype: DType
    // Edit path (lazy): VAE encoder + bn stats for reference-image conditioning.
    public let vaeEncoder: KleinVAEEncoder?
    public let bnMean: MLXArray?
    public let bnStd: MLXArray?

    public init(transformer: Flux2Transformer, vae: Flux2VAE, textEncoder: KleinTextEncoder?,
                transformerDtype: DType,
                vaeEncoder: KleinVAEEncoder? = nil, bnMean: MLXArray? = nil, bnStd: MLXArray? = nil) {
        self.transformer = transformer
        self.vae = vae
        self.textEncoder = textEncoder
        self.transformerDtype = transformerDtype
        self.vaeEncoder = vaeEncoder
        self.bnMean = bnMean
        self.bnStd = bnStd
    }

    /// [-1,1] NCHW → interleaved RGB8.
    private static func toPixels(_ img: MLXArray) -> (pixels: [UInt8], width: Int, height: Int) {
        let rgb = MLX.clip(img / 2 + 0.5, min: 0, max: 1).transposed(1, 2, 0) * 255
        let u8 = rgb.asType(.uint8); eval(u8)
        return (u8.asArray(UInt8.self), img.shape[2], img.shape[1])
    }

    // MARK: - Embeds-based entry points (encoder already run + evicted by the caller)

    /// T2I from pre-computed conditioning. `negativeEmbeds` + `guidanceScale > 1` ⇒ two-pass CFG.
    public func generate(
        promptEmbeds: MLXArray, negativeEmbeds: MLXArray?, width: Int, height: Int, steps: Int,
        seed: UInt64, guidanceScale: Float = 1.0, onStep: ((Int, Int) -> Void)? = nil
    ) -> (pixels: [UInt8], width: Int, height: Int) {
        let result = KleinPipeline.generate(
            transformer: transformer, vae: vae, promptEmbeds: promptEmbeds, negativeEmbeds: negativeEmbeds,
            height: height, width: width, numInferenceSteps: steps, guidanceScale: guidanceScale,
            seed: seed, transformerDtype: transformerDtype, onStep: onStep)
        // image is nil only when the run's task was cancelled (decode skipped); the engine
        // wrapper's post-call checkpoint rethrows before these pixels are consumed.
        guard let image = result.image else { return ([], width, height) }
        return Self.toPixels(image[0])
    }

    /// Multi-reference edit from pre-computed conditioning.
    public func generateEdit(
        promptEmbeds: MLXArray, negativeEmbeds: MLXArray?, referenceImages: [MLXArray],
        width: Int, height: Int, steps: Int, seed: UInt64, guidanceScale: Float = 1.0,
        onStep: ((Int, Int) -> Void)? = nil
    ) -> (pixels: [UInt8], width: Int, height: Int) {
        guard let vaeEncoder, let bnMean, let bnStd else {
            fatalError("edit path requires the VAE encoder + bn stats")
        }
        let result = KleinEditPipeline.generate(
            transformer: transformer, vae: vae, vaeEncoder: vaeEncoder, bnMean: bnMean, bnStd: bnStd,
            promptEmbeds: promptEmbeds, negativeEmbeds: negativeEmbeds, referenceImages: referenceImages,
            height: height, width: width, numInferenceSteps: steps, guidanceScale: guidanceScale,
            seed: seed, transformerDtype: transformerDtype, onStep: onStep)
        // Cancelled-task decode skip — see generate(promptEmbeds:...).
        guard let image = result.image else { return ([], width, height) }
        return Self.toPixels(image[0])
    }

    // MARK: - Prompt-based entry points (resident-encoder path; used by the CLI + non-evict tier)

    /// Compositional multi-reference edit → interleaved RGB8. referenceImages: each [1,3,H,W] [-1,1].
    public func generateEdit(
        prompt: String, referenceImages: [MLXArray], width: Int, height: Int, steps: Int, seed: UInt64,
        negativePrompt: String? = nil, guidanceScale: Float = 1.0,
        onStep: ((Int, Int) -> Void)? = nil
    ) -> (pixels: [UInt8], width: Int, height: Int) {
        guard let textEncoder else { fatalError("prompt path requires a resident textEncoder") }
        let embeds = textEncoder.encode(prompt)
        let negEmbeds = (guidanceScale > 1.0) ? textEncoder.encode(negativePrompt ?? "") : nil
        return generateEdit(promptEmbeds: embeds, negativeEmbeds: negEmbeds, referenceImages: referenceImages,
                            width: width, height: height, steps: steps, seed: seed,
                            guidanceScale: guidanceScale, onStep: onStep)
    }

    /// Returns interleaved RGB8 pixels + dimensions. `guidanceScale > 1` + a `negativePrompt`
    /// enables base-tier two-pass CFG; the distilled tier calls with the defaults (guidance 1.0,
    /// no negative) for the single-forward fast path.
    public func generate(
        prompt: String, width: Int, height: Int, steps: Int, seed: UInt64,
        negativePrompt: String? = nil, guidanceScale: Float = 1.0,
        onStep: ((Int, Int) -> Void)? = nil
    ) -> (pixels: [UInt8], width: Int, height: Int) {
        guard let textEncoder else { fatalError("prompt path requires a resident textEncoder") }
        let embeds = textEncoder.encode(prompt)
        let negEmbeds = (guidanceScale > 1.0) ? textEncoder.encode(negativePrompt ?? "") : nil
        return generate(promptEmbeds: embeds, negativeEmbeds: negEmbeds, width: width, height: height,
                        steps: steps, seed: seed, guidanceScale: guidanceScale, onStep: onStep)
    }
}
