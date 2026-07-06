// FLUX.2 VAE encoder — the standard diffusers AutoencoderKL Encoder (NOT Flux2-specific;
// the Flux2 bn/pixel-unshuffle lives in latent packing, not the conv encoder). Needed for the
// edit path (reference images must be VAE-encoded); flux2-vae-mlx-swift is decoder-only.
// Block structure copy-adapted from z-image-swift Autoencoder.swift (FLUX.1 AE encoder, 118 dB),
// with 32 latent channels + quant_conv (use_quant_conv=true). Returns the deterministic MEAN.

import Foundation
import MLX
import MLXNN

private func silu(_ x: MLXArray) -> MLXArray { x * sigmoid(x) }
private func groupNorm(_ groups: Int, _ channels: Int, _ eps: Float) -> GroupNorm {
    GroupNorm(groupCount: groups, dimensions: channels, eps: eps, pytorchCompatible: true)
}

private final class EncResnetBlock2D: Module {
    @ModuleInfo var norm1: GroupNorm
    @ModuleInfo var conv1: Conv2d
    @ModuleInfo var norm2: GroupNorm
    @ModuleInfo var conv2: Conv2d
    @ModuleInfo(key: "conv_shortcut") var convShortcut: Conv2d?

    init(_ inC: Int, _ outC: Int, groups: Int = 32, eps: Float = 1e-6) {
        self._norm1.wrappedValue = groupNorm(groups, inC, eps)
        self._conv1.wrappedValue = Conv2d(inputChannels: inC, outputChannels: outC, kernelSize: 3, stride: 1, padding: 1)
        self._norm2.wrappedValue = groupNorm(groups, outC, eps)
        self._conv2.wrappedValue = Conv2d(inputChannels: outC, outputChannels: outC, kernelSize: 3, stride: 1, padding: 1)
        if inC != outC {
            self._convShortcut.wrappedValue = Conv2d(inputChannels: inC, outputChannels: outC, kernelSize: 1, stride: 1, padding: 0)
        }
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var residual = x
        var h = conv1(silu(norm1(x)))
        h = conv2(silu(norm2(h)))
        if let convShortcut { residual = convShortcut(residual) }
        return residual + h
    }
}

private final class EncDownsample2D: Module {
    @ModuleInfo var conv: Conv2d
    init(_ channels: Int) {
        self._conv.wrappedValue = Conv2d(inputChannels: channels, outputChannels: channels, kernelSize: 3, stride: 2, padding: 0)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let padded = MLX.padded(x, widths: [.init((0, 0)), .init((0, 1)), .init((0, 1)), .init((0, 0))])
        return conv(padded)
    }
}

private final class EncVAEAttention: Module {
    @ModuleInfo(key: "group_norm") var groupNorm_: GroupNorm
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: [Linear]
    let scale: Float
    init(_ channels: Int, groups: Int = 32, eps: Float = 1e-6) {
        self._groupNorm_.wrappedValue = groupNorm(groups, channels, eps)
        self._toQ.wrappedValue = Linear(channels, channels)
        self._toK.wrappedValue = Linear(channels, channels)
        self._toV.wrappedValue = Linear(channels, channels)
        self._toOut.wrappedValue = [Linear(channels, channels)]
        self.scale = 1.0 / Float(channels).squareRoot()
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let residual = x
        var y = groupNorm_(x).reshaped(b, h * w, c)
        let q = toQ(y), k = toK(y), v = toV(y)
        let attn = softmax(matmul(q, k.transposed(0, 2, 1)) * scale, axis: -1)
        y = toOut[0](matmul(attn, v)).reshaped(b, h, w, c)
        return residual + y
    }
}

private final class EncMidBlock2D: Module {
    @ModuleInfo var resnets: [EncResnetBlock2D]
    @ModuleInfo var attentions: [EncVAEAttention]
    init(_ channels: Int, groups: Int = 32, eps: Float = 1e-6) {
        self._resnets.wrappedValue = [EncResnetBlock2D(channels, channels, groups: groups, eps: eps),
                                      EncResnetBlock2D(channels, channels, groups: groups, eps: eps)]
        self._attentions.wrappedValue = [EncVAEAttention(channels, groups: groups, eps: eps)]
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { resnets[1](attentions[0](resnets[0](x))) }
}

private final class EncDownBlock2D: Module {
    @ModuleInfo var resnets: [EncResnetBlock2D]
    @ModuleInfo var downsamplers: [EncDownsample2D]?
    init(_ inC: Int, _ outC: Int, numLayers: Int, addDownsample: Bool, groups: Int = 32, eps: Float = 1e-6) {
        self._resnets.wrappedValue = (0..<numLayers).map { i in EncResnetBlock2D(i == 0 ? inC : outC, outC, groups: groups, eps: eps) }
        self._downsamplers.wrappedValue = addDownsample ? [EncDownsample2D(outC)] : nil
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for r in resnets { h = r(h) }
        if let downsamplers { h = downsamplers[0](h) }
        return h
    }
}

private final class Encoder: Module {
    @ModuleInfo(key: "conv_in") var convIn: Conv2d
    @ModuleInfo(key: "down_blocks") var downBlocks: [EncDownBlock2D]
    @ModuleInfo(key: "mid_block") var midBlock: EncMidBlock2D
    @ModuleInfo(key: "conv_norm_out") var convNormOut: GroupNorm
    @ModuleInfo(key: "conv_out") var convOut: Conv2d

    init(inC: Int, latentC: Int, blockOut: [Int], layersPerBlock: Int, groups: Int = 32, eps: Float = 1e-6) {
        self._convIn.wrappedValue = Conv2d(inputChannels: inC, outputChannels: blockOut[0], kernelSize: 3, stride: 1, padding: 1)
        var blocks: [EncDownBlock2D] = []
        var outC = blockOut[0]
        for (i, boc) in blockOut.enumerated() {
            let inC2 = outC; outC = boc
            blocks.append(EncDownBlock2D(inC2, outC, numLayers: layersPerBlock, addDownsample: i != blockOut.count - 1, groups: groups, eps: eps))
        }
        self._downBlocks.wrappedValue = blocks
        self._midBlock.wrappedValue = EncMidBlock2D(blockOut.last!, groups: groups, eps: eps)
        self._convNormOut.wrappedValue = groupNorm(groups, blockOut.last!, eps)
        self._convOut.wrappedValue = Conv2d(inputChannels: blockOut.last!, outputChannels: 2 * latentC, kernelSize: 3, stride: 1, padding: 1)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = convIn(x)
        for b in downBlocks { h = b(h) }
        h = midBlock(h)
        return convOut(silu(convNormOut(h)))
    }
}

/// FLUX.2 VAE encoder + quant_conv. `encodeMean` takes an image in [-1,1] NCHW and returns the
/// deterministic latent MEAN [B, latentC, h, w] (NCHW). Loaded from the same vae/ snapshot as the
/// decoder (encoder.* + quant_conv.* keys that flux2-vae-mlx-swift skips).
public final class KleinVAEEncoder: Module {
    @ModuleInfo fileprivate var encoder: Encoder
    @ModuleInfo(key: "quant_conv") var quantConv: Conv2d
    public let latentChannels: Int

    public init(inChannels: Int = 3, latentChannels: Int = 32,
                blockOutChannels: [Int] = [128, 256, 512, 512], layersPerBlock: Int = 2, normGroups: Int = 32) {
        self.latentChannels = latentChannels
        self._encoder.wrappedValue = Encoder(inC: inChannels, latentC: latentChannels, blockOut: blockOutChannels, layersPerBlock: layersPerBlock, groups: normGroups)
        // quant_conv: 1x1 over the 2*latent moments.
        self._quantConv.wrappedValue = Conv2d(inputChannels: 2 * latentChannels, outputChannels: 2 * latentChannels, kernelSize: 1, stride: 1, padding: 0)
        super.init()
    }

    /// image: [B, 3, H, W] in [-1,1] → mean [B, latentC, H/8, W/8] (NCHW).
    public func encodeMean(_ imageNCHW: MLXArray) -> MLXArray {
        let x = imageNCHW.transposed(0, 2, 3, 1)         // NHWC
        var moments = encoder(x)                          // [B, h, w, 2*latentC]
        moments = quantConv(moments)
        moments = moments.transposed(0, 3, 1, 2)          // NCHW [B, 2*latentC, h, w]
        return moments[0..., ..<latentChannels, 0..., 0...]   // mean = first latentC channels
    }
}
