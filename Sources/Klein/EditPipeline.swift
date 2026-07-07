// Klein multi-reference EDIT — the tier differentiator. Mirrors mflux flux2 edit (plain path,
// no KV cache): each reference image is VAE-encoded → patchified (32→128) → bn-normalized →
// packed → appended to the target token sequence with a 4D-RoPE t-offset (t = 10·(i+1), so
// references sit apart from the target at t=0). The DiT runs over [target, refs]; only the
// target tokens' velocity is kept per step. Target starts from noise (compositional edit).

import Foundation
import Flux2VAE
import MLX
import MLXRandom

public enum KleinEditPipeline {

    /// Reference conditioning for one image [1,3,H,W] in [-1,1] (scaled to the target dims by the
    /// caller). Returns (packed [1, refSeq, 128], ids [refSeq, 4]) at t-offset `tCoord`.
    public static func prepareReference(
        image: MLXArray, vaeEncoder: KleinVAEEncoder,
        bnMean: MLXArray, bnStd: MLXArray, tCoord: Int32
    ) -> (packed: MLXArray, ids: [Int32]) {
        var lat = vaeEncoder.encodeMean(image)            // [1,32,h,w] NCHW, h=H/8
        // crop to even spatial
        if lat.shape[2] % 2 != 0 { lat = lat[0..., 0..., ..<(lat.shape[2] - 1), 0...] }
        if lat.shape[3] % 2 != 0 { lat = lat[0..., 0..., 0..., ..<(lat.shape[3] - 1)] }
        // patchify [1,32,h,w] → [1,128,h/2,w/2]
        let (h, w) = (lat.shape[2], lat.shape[3])
        var enc = lat.reshaped(1, 32, h / 2, 2, w / 2, 2).transposed(0, 1, 3, 5, 2, 4).reshaped(1, 128, h / 2, w / 2)
        // bn-normalize (forward): (x - mean)/std
        enc = (enc - bnMean) / bnStd
        let (gh, gw) = (h / 2, w / 2)
        let packed = enc.reshaped(1, 128, gh * gw).transposed(0, 2, 1)   // [1, gh*gw, 128]
        // ids: [tCoord, hh, ww, 0] over the gh×gw grid (row-major, matches pack)
        var ids = [Int32](); ids.reserveCapacity(gh * gw * 4)
        for hh in 0..<gh { for ww in 0..<gw { ids.append(contentsOf: [tCoord, Int32(hh), Int32(ww), 0]) } }
        return (packed, ids)
    }

    /// Compositional edit: generate a target (from noise) conditioned on reference images.
    /// referenceImages: each [1,3,targetH,targetW] in [-1,1].
    /// Base tier: pass `negativeEmbeds` + `guidanceScale > 1` for two-pass CFG on the edit
    /// (reference tokens are identical in both passes; only the text conditioning differs). Leave
    /// `negativeEmbeds` nil for the distilled single-forward path.
    public static func generate(
        transformer: Flux2Transformer, vae: Flux2VAE, vaeEncoder: KleinVAEEncoder,
        bnMean: MLXArray, bnStd: MLXArray,
        promptEmbeds: MLXArray, negativeEmbeds: MLXArray? = nil, referenceImages: [MLXArray],
        height: Int = 1024, width: Int = 1024, numInferenceSteps: Int = 4,
        guidanceScale: Float = 1.0,
        seed: UInt64 = 0, transformerDtype: DType = .bfloat16,
        onStep: ((Int, Int) -> Void)? = nil
    ) -> KleinPipeline.Result {
        let gridH = height / 16, gridW = width / 16
        let targetSeq = gridH * gridW

        // target latents from noise (same as T2I)
        MLXRandom.seed(seed)
        var packed = MLXRandom.normal([1, 128, gridH, gridW], dtype: .float32).reshaped(1, 128, targetSeq).transposed(0, 2, 1)
        var imgIdsHost = [Int32](); imgIdsHost.reserveCapacity(targetSeq * 4)
        for h in 0..<gridH { for w in 0..<gridW { imgIdsHost.append(contentsOf: [0, Int32(h), Int32(w), 0]) } }

        // reference tokens (t-offset 10·(i+1)) appended to the sequence
        var refPacked: [MLXArray] = []
        for (i, refImg) in referenceImages.enumerated() {
            let (rp, rids) = prepareReference(image: refImg, vaeEncoder: vaeEncoder,
                                              bnMean: bnMean, bnStd: bnStd, tCoord: Int32(10 * (i + 1)))
            refPacked.append(rp)
            imgIdsHost.append(contentsOf: rids)
        }
        let totalImgSeq = targetSeq + refPacked.reduce(0) { $0 + $1.shape[1] }
        let imgIds = MLXArray(imgIdsHost, [totalImgSeq, 4])
        let encoder = promptEmbeds[.newAxis, 0..., 0...]
        let txtIds = KleinPipeline.textIds(promptEmbeds.shape[0])
        let refCat = refPacked.isEmpty ? nil : concatenated(refPacked, axis: 1)   // [1, refSeqTotal, 128]
        let doCFG = guidanceScale > 1.0 && negativeEmbeds != nil
        let negEncoder = negativeEmbeds.map { $0[.newAxis, 0..., 0...] }

        let (timesteps, sigmas) = KleinPipeline.timestepsAndSigmas(imageSeqLen: targetSeq, numSteps: numInferenceSteps)
        for i in 0..<numInferenceSteps {
            var hidden = packed
            if let refCat { hidden = concatenated([packed, refCat], axis: 1) }
            let t = MLXArray([timesteps[i]] as [Float])
            let posOut = transformer(
                hidden.asType(transformerDtype), encoder: encoder.asType(transformerDtype),
                timestep: t, guidance: nil,
                imgIds: imgIds, txtIds: txtIds).asType(.float32)
            var noise = posOut[0..., ..<targetSeq, 0...]        // keep only target tokens
            if doCFG, let negEncoder {
                let negOut = transformer(
                    hidden.asType(transformerDtype), encoder: negEncoder.asType(transformerDtype),
                    timestep: t, guidance: nil,
                    imgIds: imgIds, txtIds: txtIds).asType(.float32)
                let negNoise = negOut[0..., ..<targetSeq, 0...]
                noise = negNoise + guidanceScale * (noise - negNoise)
            }
            let dt = sigmas[i + 1] - sigmas[i]
            packed = packed + dt * noise
            eval(packed); MLX.Memory.clearCache()
            onStep?(i + 1, numInferenceSteps)
        }

        let unpacked = packed.asType(.float32).reshaped(1, gridH, gridW, 128).transposed(0, 3, 1, 2)
        let image = vae.decodePackedLatents(unpacked)
        eval(image)
        return KleinPipeline.Result(latents: packed, image: image)
    }
}
