// Edit-reference preprocessing: aspect kept, as diffusers' Flux2KleinPipeline does (AB-A-0213).
//
// diffusers (pipeline_flux2_klein.py, the `condition_images` loop): a reference over 1024² area is
// Lanczos-resized to that area with its aspect kept, then each side is floored to a multiple of 16
// (vae_scale_factor · 2), and `preprocess(..., resize_mode="crop")` covers that size and
// center-crops. A smaller reference is never upscaled. Up to 0.10.1 this package drew every
// reference into an output-width square, which stretched any non-square reference.
//
// One addition here: the area cap is also bounded by the OUTPUT's area. For outputs of 1024² and up
// that is exactly diffusers. Below 1024² it means a reference never carries more tokens than the
// output does, so the 1024²-measured edit footprints, and the 768² light tier's, keep holding.

import CoreGraphics
import Foundation
import MLX

public enum KleinReference {

    /// The model-input size of one reference: aspect kept, area ≤ `maxArea`, never upscaled, each
    /// side floored to a multiple of 16 (at least 16).
    public static func size(width: Int, height: Int, maxArea: Int) -> (width: Int, height: Int) {
        var w = Double(width), h = Double(height)
        if width * height > maxArea {
            let s = (Double(maxArea) / Double(width * height)).squareRoot()
            w = (w * s).rounded(.down)        // Python int() truncation
            h = (h * s).rounded(.down)
        }
        return (max(16, Int(w) / 16 * 16), max(16, Int(h) / 16 * 16))
    }

    /// The area cap for an edit producing `outputWidth × outputHeight`: min(1024², output area).
    public static func maxArea(outputWidth: Int, outputHeight: Int) -> Int {
        min(1024 * 1024, outputWidth * outputHeight)
    }

    /// Decode `cg` → `[1, 3, h, w]` in [-1, 1] at `size(...)`. The image is scaled to cover the target
    /// and center-cropped (diffusers' `resize_mode="crop"`), so the crop is under 16 px per side and
    /// nothing is stretched.
    public static func pixels(_ cg: CGImage, maxArea: Int) -> MLXArray {
        let (tw, th) = size(width: cg.width, height: cg.height, maxArea: maxArea)
        let cs = CGColorSpace(name: CGColorSpace.sRGB)!
        var buf = [UInt8](repeating: 0, count: tw * th * 4)
        buf.withUnsafeMutableBytes { raw in
            let ctx = CGContext(data: raw.baseAddress, width: tw, height: th, bitsPerComponent: 8,
                bytesPerRow: tw * 4, space: cs, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
            ctx.interpolationQuality = .high
            // Cover then center-crop. The drawn rect contains the canvas, so no source edge lands
            // inside it, and CG's edge antialiasing can't blend in the empty fill (AB-L-0244).
            let scale = max(Double(tw) / Double(cg.width), Double(th) / Double(cg.height))
            let dw = Double(cg.width) * scale, dh = Double(cg.height) * scale
            ctx.draw(cg, in: CGRect(x: (Double(tw) - dw) / 2, y: (Double(th) - dh) / 2, width: dw, height: dh))
        }
        let n = tw * th
        var rgb = [Float](repeating: 0, count: 3 * n)
        for p in 0..<n {
            rgb[p] = Float(buf[p * 4]) / 127.5 - 1
            rgb[n + p] = Float(buf[p * 4 + 1]) / 127.5 - 1
            rgb[2 * n + p] = Float(buf[p * 4 + 2]) / 127.5 - 1
        }
        return MLXArray(rgb, [1, 3, th, tw])
    }
}
