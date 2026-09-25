// Route for 3×3 convs inside mlx's Winograd conv2d window (mlx-swift ≤ 0.31.6 Metal numerics).
//
// mlx's Metal conv2d (mlx/backend/metal/conv.cpp `dispatch_conv_2D_gpu`) runs a Winograd
// F(6×6,3×3) kernel when ALL of these hold: kernel 3×3, stride 1, dilation 1, groups 1,
// C % 32 == 0, O % 32 == 0, C + O ≥ 256, N·H·W ≥ 4096. On M5 that path is lossy — relL2 per
// conv against an exact reference: fp32 6.4e-3 (its batched GEMM runs TF32, MLX_ENABLE_TF32
// defaults on, and the output transform amplifies that ~8×), bf16 5.8e-2, fp16 7.5e-3. Every
// other conv path is exact-class (fp32 ~1e-6; bf16 1.7e-3 = output rounding). conv3d with
// kT = 1 is the same conv on the implicit-GEMM path: exact, but 1.3–4× slower than Winograd at
// 256/512-channel shapes.
//
// The edit-path VAE encoder hits the window in nearly every 3×3 conv: 21 convs per 1024²
// reference image (resnets 128/256/512-ch + conv_out 512→64). Default `.conv3d` (+150 ms per
// 1024² reference). The decoder lives in flux2-vae-mlx-swift, which has its own copy of this.
// Probe: `swift test --filter WinogradProbeTests` (weight-free). Removal: when the probe reports
// raw conv2d exact on a new mlx-swift pin, go back to plain Conv2d.
// `KLEIN_VAE_CONV_ROUTE=winograd|conv3d|fp32Winograd` overrides the defaults (validation).

import Foundation
import MLX
import MLXNN

/// How a 3×3 conv inside mlx's Winograd window runs. Shapes outside the window always take plain
/// conv2d, which is mlx's exact implicit-GEMM path.
public enum KleinVAEConvRoute: String, Sendable {
    /// mlx's default Winograd kernel — fastest; on M5 ~6.4e-3 relL2 per conv in fp32, ~5.8e-2 in bf16.
    case winograd
    /// conv3d with kT = 1 on the implicit-GEMM path — exact; 1.3–4× slower than Winograd.
    case conv3d
    /// Half-precision input upcast to fp32 for the Winograd kernel and the result cast back:
    /// ~6.8e-3 per conv instead of bf16's ~5.8e-2, at fp32-Winograd speed. `.winograd` for fp32.
    case fp32Winograd

    /// `KLEIN_VAE_CONV_ROUTE` = winograd | conv3d | fp32Winograd, if set.
    static var environmentOverride: KleinVAEConvRoute? {
        getenv("KLEIN_VAE_CONV_ROUTE").flatMap { KleinVAEConvRoute(rawValue: String(cString: $0)) }
    }
}

final class WinogradFreeConv2d: Conv2d {
    var route: KleinVAEConvRoute = .conv3d

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
        guard route != .winograd,
            Self.takesWinograd(
                input: x, weight: weight, stride: stride, dilation: dilation, groups: groups)
        else { return super.callAsFunction(x) }
        if route == .fp32Winograd {
            guard x.dtype != .float32 else { return super.callAsFunction(x) }
            var y = conv2d(
                x.asType(.float32), weight.asType(.float32), stride: .init(stride),
                padding: .init(padding), dilation: .init(dilation), groups: groups)
            if let bias { y = y + bias.asType(.float32) }
            return y.asType(x.dtype)
        }
        var y = conv3d(
            x.expandedDimensions(axis: 1), weight.expandedDimensions(axis: 1),
            stride: [1, 1, 1], padding: [0, padding.0, padding.1]
        ).squeezed(axis: 1)
        if let bias { y = y + bias }
        return y
    }
}
