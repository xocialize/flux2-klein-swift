// Klein T2I pipeline — mirrors mflux flux2 txt2img (Flux2Klein.generate_image +
// Flux2LatentCreator). Distilled: 4-step, guidance 1.0 (no CFG), dynamic-shift FlowMatchEuler
// (diffusers convention: linear calculate_shift mu). FLUX latent packing: 32-ch → 128 (2×2
// patchify), sequence of gridH·gridW tokens.

import Foundation
import Flux2VAE
import MLX
import MLXRandom

public enum KleinPipeline {

    /// Diffusers calculate_shift (linear mu) — the canonical klein schedule (verified: mu 1.15
    /// @ 4096 tokens reproduces diffusers sigmas exactly). NOT mflux's empirical mu.
    public static func calculateShift(
        imageSeqLen: Int, baseSeqLen: Int = 256, maxSeqLen: Int = 4096,
        baseShift: Double = 0.5, maxShift: Double = 1.15
    ) -> Double {
        let m = (maxShift - baseShift) / Double(maxSeqLen - baseSeqLen)
        let b = baseShift - m * Double(baseSeqLen)
        return Double(imageSeqLen) * m + b
    }

    /// Port of diffusers FlowMatchEulerDiscreteScheduler.set_timesteps (dynamic, exponential
    /// time-shift, no terminal stretch): base sigmas = linspace(sigma_max=1, sigma_min=0.001, N);
    /// sigmas = exp(mu)/(exp(mu)+(1/s−1)); append 0. Returns (timesteps[N]=sigma·1000, sigmas[N+1]).
    static func timestepsAndSigmas(imageSeqLen: Int, numSteps: Int) -> (timesteps: [Float], sigmas: [Float]) {
        let mu = calculateShift(imageSeqLen: imageSeqLen)
        let emu = exp(mu)
        let (sMax, sMin) = (1.0, 0.0010000000474974513)   // scheduler sigma_max / sigma_min
        var sigmas = (0..<numSteps).map { i -> Float in
            let s = numSteps == 1 ? sMax
                : sMax - Double(i) * (sMax - sMin) / Double(numSteps - 1)   // linspace(1, 0.001, N)
            return Float(emu / (emu + (1.0 / s - 1.0)))
        }
        let timesteps = sigmas.map { $0 * 1000 }
        sigmas.append(0)
        return (timesteps, sigmas)
    }

    /// 4-axis ids. Image tokens: [0, h, w, 0] over the gridH×gridW grid. Text tokens:
    /// [0, 0, 0, pos].
    static func imageIds(gridH: Int, gridW: Int) -> MLXArray {
        var host = [Int32](); host.reserveCapacity(gridH * gridW * 4)
        for h in 0..<gridH {
            for w in 0..<gridW { host.append(contentsOf: [0, Int32(h), Int32(w), 0]) }
        }
        return MLXArray(host, [gridH * gridW, 4])
    }

    static func textIds(_ len: Int) -> MLXArray {
        var host = [Int32](); host.reserveCapacity(len * 4)
        for i in 0..<len { host.append(contentsOf: [0, 0, 0, Int32(i)]) }
        return MLXArray(host, [len, 4])
    }

    public struct Result {
        public let latents: MLXArray   // packed [1, seq, 128]
        public let image: MLXArray?    // decoded [-1,1] NCHW
    }

    /// promptEmbeds: [txtLen, 7680] (real tokens). initPacked injects the packed initial
    /// latents [1, seq, 128] (parity path); nil → drawn from MLXRandom.
    public static func generate(
        transformer: Flux2Transformer,
        vae: Flux2VAE?,
        promptEmbeds: MLXArray,
        height: Int = 1024, width: Int = 1024,
        numInferenceSteps: Int = 4,
        guidanceScale: Float = 1.0,
        seed: UInt64 = 0,
        initPacked: MLXArray? = nil,
        transformerDtype: DType = .bfloat16,
        onStep: ((Int, Int) -> Void)? = nil
    ) -> Result {
        let gridH = height / 16, gridW = width / 16   // 2×(H//16)/2 = H//16
        let seqLen = gridH * gridW

        var packed: MLXArray
        if let initPacked {
            precondition(initPacked.shape == [1, seqLen, 128])
            packed = initPacked.asType(.float32)
        } else {
            MLXRandom.seed(seed)
            // [1,128,gridH,gridW] → pack → [1, seq, 128]
            let l = MLXRandom.normal([1, 128, gridH, gridW], dtype: .float32)
            packed = l.reshaped(1, 128, seqLen).transposed(0, 2, 1)
        }

        let imgIds = imageIds(gridH: gridH, gridW: gridW)
        let txtIds = textIds(promptEmbeds.shape[0])
        let encoder = promptEmbeds[.newAxis, 0..., 0...]   // [1, txtLen, 7680]

        let (timesteps, sigmas) = Self.timestepsAndSigmas(imageSeqLen: seqLen, numSteps: numInferenceSteps)
        for i in 0..<numInferenceSteps {
            let noise = transformer(
                packed.asType(transformerDtype), encoder: encoder.asType(transformerDtype),
                timestep: MLXArray([timesteps[i]] as [Float]), guidance: nil,
                imgIds: imgIds, txtIds: txtIds).asType(.float32)
            // mflux Euler step: _step(noise, latents, sigmas[t+1], sigmas[t]) =
            // latents + (sigma_next − sigma) · noise  (dt negative, sigmas descending).
            let dt = sigmas[i + 1] - sigmas[i]
            packed = packed + dt * noise
            eval(packed)
            MLX.Memory.clearCache()
            onStep?(i + 1, numInferenceSteps)
        }

        guard let vae else { return Result(latents: packed, image: nil) }
        // unpack [1, seq, 128] → [1, 128, gridH, gridW] then decodePackedLatents (bn-denorm + unpatch + decode)
        let unpacked = packed.asType(.float32).reshaped(1, gridH, gridW, 128).transposed(0, 3, 1, 2)
        let image = vae.decodePackedLatents(unpacked)
        eval(image)
        return Result(latents: packed, image: image)
    }
}
