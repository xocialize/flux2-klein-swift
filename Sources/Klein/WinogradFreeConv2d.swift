// A Conv2d that never takes mlx's Winograd path (mlx-swift ≤ 0.31.6 Metal numerics).
//
// mlx's Metal conv2d (mlx/backend/metal/conv.cpp `dispatch_conv_2D_gpu`) runs a Winograd
// F(6×6,3×3) kernel when ALL of these hold: kernel 3×3, stride 1, dilation 1, groups 1,
// C % 32 == 0, O % 32 == 0, C + O ≥ 256, N·H·W ≥ 4096. On M5 that path is lossy — relL2 per
// conv against an exact reference: fp32 6.4e-3 (its batched GEMM runs TF32, MLX_ENABLE_TF32
// defaults on, and the output transform amplifies that ~8×), bf16 5.8e-2, fp16 7.5e-3. Every
// other conv path is exact-class (fp32 ~1e-6; bf16 1.7e-3 = output rounding). conv3d with
// kT = 1 is the same conv on the implicit-GEMM path: exact, but slower than Winograd.
//
// The edit-path VAE encoder hits the window in nearly every 3×3 conv: 21 convs per 1024²
// reference image (resnets 128/256/512-ch + conv_out 512→64). The decoder lives in
// flux2-vae-mlx-swift, which carries its own copy of this class. Probe: `swift test --filter
// WinogradProbeTests` (weight-free). Removal: when the probe reports raw conv2d exact on a new
// mlx-swift pin, go back to plain Conv2d. `KLEIN_VAE_WINOGRAD=1` restores the raw conv2d path
// (validation only).

import Foundation
import MLX
import MLXNN

final class WinogradFreeConv2d: Conv2d {
    /// `false` restores the raw conv2d (Winograd) path — A/B validation only.
    var enabled = getenv("KLEIN_VAE_WINOGRAD") == nil

    /// mlx's Winograd dispatch predicate, evaluated on the actual input (NHWC).
    static func takesWinograd(
        input x: MLXArray, weight: MLXArray, stride: (Int, Int), dilation: (Int, Int),
        groups: Int
    ) -> Bool {
        guard x.ndim == 4, groups == 1, stride == (1, 1), dilation == (1, 1),
            weight.dim(1) == 3, weight.dim(2) == 3
        else { return false }
        let (c, o) = (x.dim(3), weight.dim(0))
        return c % 32 == 0 && o % 32 == 0 && c + o >= 256 && x.dim(0) * x.dim(1) * x.dim(2) >= 4096
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        guard enabled,
            Self.takesWinograd(
                input: x, weight: weight, stride: stride, dilation: dilation, groups: groups)
        else { return super.callAsFunction(x) }
        var y = conv3d(
            x.expandedDimensions(axis: 1), weight.expandedDimensions(axis: 1),
            stride: [1, 1, 1], padding: [0, padding.0, padding.1]
        ).squeezed(axis: 1)
        if let bias { y = y + bias }
        return y
    }
}
