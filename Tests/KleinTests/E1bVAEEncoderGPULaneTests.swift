// E1b gate — the edit-path VAE encoder on the GPU lane (Winograd-free route on / raw Winograd
// conv2d) vs the CPU lane, fp32 (the production regime). E1 runs on the CPU lane only and its
// golden (klein_vae_encode.safetensors) is not kept, so the fp32 CPU lane is the reference.
//
// Input: a real DIV2K photo, center-cropped to 512² (gated) and 1024² (the default reference
// size; informational). At 1024² the CPU lane is not a clean reference for this net: MLX's CPU
// GroupNorm accumulates fp32 error with the group size (relL2 vs a float64 reference: 1.2e-6 @256²,
// 1.1e-5 @512², 8.2e-5 @1024² for 128 ch / 32 groups; the GPU's is ~1e-7 at every size), and the
// encoder amplifies what its full-resolution norms inject. So the gate asserts at 512².
//
// Measured 2026-09-24 (M5 Max, mlx-swift 0.31.6), encode mean vs the CPU lane: raw conv2d 3.4e-3,
// route 1.2e-3 (512² and 1024²). The route's residual is the mid-block attention's TF32 matmuls,
// not a conv: under MLX_ENABLE_TF32=0 the route is 4.6e-6 (512²) / 2.2e-5 (1024², CPU GroupNorm).
//
// Run: KLEIN_PARITY=1 KLEIN_SNAPSHOT=<FLUX.2-klein-4B snapshot with vae/> \
//      swift test -c release -Xswiftc -enable-testing --filter E1bVAEEncoderGPULaneTests
// Override: KLEIN_REAL_IMAGE.

import CoreGraphics
import Foundation
import ImageIO
import MLX
import XCTest

@testable import Klein

final class E1bVAEEncoderGPULaneTests: XCTestCase {
    static let env = ProcessInfo.processInfo.environment
    static let realImage = URL(
        fileURLWithPath: env["KLEIN_REAL_IMAGE"]
            ?? "/Volumes/Satechi/Development/mlxengine-image/corpus/sr-bench/DIV2K_valid_HR/0801.png")

    static func stats(_ a: MLXArray, _ ref: MLXArray) -> (relL2: Float, maxAbs: Float) {
        let d = a.asType(.float32) - ref.asType(.float32)
        let rel = sqrt(sum(d * d)) / sqrt(sum(square(ref.asType(.float32))))
        let mx = abs(d).max()
        eval(rel, mx)
        return (rel.item(Float.self), mx.item(Float.self))
    }

    /// GPU run with the route on/off; warm-up, then the mean of `reps` timed runs.
    static func gpuRun(_ enc: KleinVAEEncoder, route: Bool, reps: Int = 3, _ f: () -> MLXArray)
        -> (MLXArray, Double)
    {
        enc.winogradFreeConvs = route
        defer { enc.winogradFreeConvs = true }
        var out = f()
        eval(out)
        let t0 = Date()
        for _ in 0..<reps {
            out = f()
            eval(out)
        }
        return (out, Date().timeIntervalSince(t0) / Double(reps) * 1000)
    }

    /// Center crop to side×side → [1, 3, side, side] in [-1, 1].
    static func loadCrop(_ url: URL, side: Int) throws -> MLXArray {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
            let cg = CGImageSourceCreateImageAtIndex(src, 0, nil)
        else { throw NSError(domain: "E1b", code: 1, userInfo: [NSLocalizedDescriptionKey: "unreadable \(url.path)"]) }
        let (w, h) = (cg.width, cg.height)
        precondition(w >= side && h >= side, "image smaller than crop")
        var rgba = [UInt8](repeating: 0, count: w * h * 4)
        let ctx = CGContext(
            data: &rgba, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
            space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
        let (x0, y0) = ((w - side) / 2, (h - side) / 2)
        let plane = side * side
        var chw = [Float](repeating: 0, count: 3 * plane)
        for y in 0..<side {
            for x in 0..<side {
                let p = ((y0 + y) * w + (x0 + x)) * 4
                let i = y * side + x
                for c in 0..<3 { chw[c * plane + i] = Float(rgba[p + c]) / 127.5 - 1 }
            }
        }
        return MLXArray(chw, [1, 3, side, side])
    }

    func testRealPhotoEncode() throws {
        try XCTSkipUnless(Self.env["KLEIN_PARITY"] == "1", "set KLEIN_PARITY=1 to run")
        let snapshot = try XCTUnwrap(Self.env["KLEIN_SNAPSHOT"], "set KLEIN_SNAPSHOT")
        let enc = try KleinWeights.loadVAEEncoder(snapshotPath: snapshot, dtype: .float32)
        for side in [512, 1024] {
            let pixels = try Self.loadCrop(Self.realImage, side: side)
            let t0 = Date()
            let ref = Device.withDefaultDevice(.cpu) { () -> MLXArray in
                let r = enc.encodeMean(pixels)
                eval(r)
                return r
            }
            print(String(format: "[real %d² photo: %@ — CPU-lane fp32 encode %.1f s]",
                         side, Self.realImage.lastPathComponent, Date().timeIntervalSince(t0)))
            Memory.clearCache()  // the CPU lane leaves GBs pooled; don't time against that
            var route = [Double](), raw = [Double]()
            var outRoute = ref, outRaw = ref
            for _ in 0..<3 {  // interleaved rounds for the timing
                let r = Self.gpuRun(enc, route: true) { enc.encodeMean(pixels) }
                let w = Self.gpuRun(enc, route: false) { enc.encodeMean(pixels) }
                (outRoute, outRaw) = (r.0, w.0)
                route.append(r.1)
                raw.append(w.1)
            }
            let sr = Self.stats(outRoute, ref), sw = Self.stats(outRaw, ref)
            let rr = Self.stats(outRoute, outRaw)
            func fmt(_ v: [Double]) -> String { v.map { String(format: "%.0f", $0) }.joined(separator: "/") }
            print(String(format: "  encode mean vs CPU lane: route relL2 %.2e max %.2e | raw conv2d relL2 %.2e max %.2e | route vs raw %.2e",
                         sr.relL2, sr.maxAbs, sw.relL2, sw.maxAbs, rr.relL2))
            print("  GPU fp32 encode time: route \(fmt(route)) ms | raw \(fmt(raw)) ms")
            if side == 512 {
                if getenv("MLX_ENABLE_TF32").map({ String(cString: $0) }) == "0" {
                    // TF32 off: raw Winograd is exact-class too; only bound the route.
                    XCTAssertLessThan(sr.relL2, 1e-4, "route vs CPU lane (TF32 off)")
                } else {
                    // Default (TF32 on): the mid-block attention keeps ~1.2e-3 on both paths.
                    XCTAssertLessThan(sr.relL2, sw.relL2 / 2, "route vs raw, against the CPU lane")
                    XCTAssertLessThan(sr.relL2, 2e-3, "route vs CPU lane")
                }
            }
            Memory.clearCache()
        }
    }
}
