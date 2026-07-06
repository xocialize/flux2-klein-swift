// KleinGenerator — loaded-model holder that drives KleinPipeline.generate and returns
// interleaved RGB8 pixels. Keeps the MLXEngine wrapper thin (mirrors ZImageGenerator).

import Foundation
import Flux2VAE
import MLX

public final class KleinGenerator {
    public let transformer: Flux2Transformer
    public let vae: Flux2VAE
    public let textEncoder: KleinTextEncoder
    public let transformerDtype: DType

    public init(transformer: Flux2Transformer, vae: Flux2VAE,
                textEncoder: KleinTextEncoder, transformerDtype: DType) {
        self.transformer = transformer
        self.vae = vae
        self.textEncoder = textEncoder
        self.transformerDtype = transformerDtype
    }

    /// Returns interleaved RGB8 pixels + dimensions.
    public func generate(
        prompt: String, width: Int, height: Int, steps: Int, seed: UInt64,
        onStep: ((Int, Int) -> Void)? = nil
    ) -> (pixels: [UInt8], width: Int, height: Int) {
        let embeds = textEncoder.encode(prompt)
        let result = KleinPipeline.generate(
            transformer: transformer, vae: vae, promptEmbeds: embeds,
            height: height, width: width, numInferenceSteps: steps, guidanceScale: 1.0,
            seed: seed, transformerDtype: transformerDtype, onStep: onStep)
        let img = result.image![0]   // [-1,1] NCHW
        let rgb = MLX.clip(img / 2 + 0.5, min: 0, max: 1).transposed(1, 2, 0) * 255
        let u8 = rgb.asType(.uint8)
        eval(u8)
        return (u8.asArray(UInt8.self), width, height)
    }
}
