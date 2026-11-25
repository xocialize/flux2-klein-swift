// Latent upscaling before the decode (AB-A-0257): draw at about 1024² of area, double the latents one or two times with
// FlowUpscaler (flux2-vae-mlx-swift 0.8.0, TensorForger, Unlicense), then decode the larger latents with the row-band
// decode. It keeps the picture the DiT drew at 1024², where drawing 2048² natively draws another one (AB-R-0795).

import Flux2VAE
import MLX

/// One or two ×2 passes of FlowUpscaler on the denoised latents, before the decode.
public struct KleinLatentUpscale {
    public let upscaler: FlowUpscaler
    /// ×2 passes: 1 (×2) or 2 (×4).
    public let passes: Int
    /// Seed of the first pass's noise; pass i draws from `seed + i`.
    public let seed: UInt64

    public init(upscaler: FlowUpscaler, passes: Int, seed: UInt64) {
        precondition((1 ... 2).contains(passes), "latent upscale passes: 1 or 2")
        self.upscaler = upscaler
        self.passes = passes
        self.seed = seed
    }

    /// The largest supported pass count: ×4. FlowUpscaler has only seen 512 → 1024, a third pass mostly sharpens,
    /// and the exact decode at 8192² costs about 52 GB even in bf16.
    public static let maxPasses = 2
}

enum KleinDecode {
    /// Packed NCHW latents [1, 128, gridH, gridW] → image [1, 3, H, W] in [-1, 1], always through the row-band
    /// decode, which is exact (it matches the untiled decode up to float reassociation, with no seams).
    ///
    /// The untiled decode's peak is temporaries: about 8.5 GB over resident at 1024², 42 GB at 2048². In row bands
    /// the 1024² decode stays about 2.2 GB over resident, so the denoise loop sets a run's peak. It takes the same
    /// time from 512² up (AB-R-1130; int4 with the encoder evicted at 1024²: 13.3 → 4.8 GB MLX peak).
    static func decode(vae: Flux2VAE, unpacked: MLXArray, upscale: KleinLatentUpscale?) -> MLXArray {
        guard let upscale else { return vae.decodePackedLatents(unpacked, tile: .standard) }
        let up = upscale.upscaler.upscalePacked(unpacked, passes: upscale.passes, seed: upscale.seed)
        MLX.Memory.clearCache()
        return vae.decodePackedLatents(up, tile: .standard)
    }
}
