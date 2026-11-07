// AB-A-0213: edit references keep their aspect (diffusers' Flux2KleinPipeline rule) instead of
// being drawn into an output-width square.

import CoreGraphics
import MLX
import XCTest

@testable import Klein

final class ReferenceSizingTests: XCTestCase {

    func testDiffusersRuleAtA1024Output() {
        let cap = KleinReference.maxArea(outputWidth: 1024, outputHeight: 1024)
        XCTAssertEqual(cap, 1024 * 1024)
        // The ask's case: 1536×1024 → area 1024², aspect kept. sqrt(2/3)·1536 = 1254.1 → 1248;
        // ·1024 = 836.1 → 832.
        XCTAssertTrue(KleinReference.size(width: 1536, height: 1024, maxArea: cap) == (1248, 832))
        // At or under the cap, nothing is rescaled; sides are only floored to /16.
        XCTAssertTrue(KleinReference.size(width: 1280, height: 768, maxArea: cap) == (1280, 768))
        XCTAssertTrue(KleinReference.size(width: 1000, height: 700, maxArea: cap) == (992, 688))
        // Never upscaled.
        XCTAssertTrue(KleinReference.size(width: 512, height: 384, maxArea: cap) == (512, 384))
        // Portrait stays portrait.
        XCTAssertTrue(KleinReference.size(width: 1024, height: 1536, maxArea: cap) == (832, 1248))
    }

    func testAreaCapFollowsASmallerOutput() {
        // A 768² edit: a 1024² reference is held to the output's area (the footprint bound).
        let cap = KleinReference.maxArea(outputWidth: 768, outputHeight: 768)
        XCTAssertEqual(cap, 768 * 768)
        XCTAssertTrue(KleinReference.size(width: 1024, height: 1024, maxArea: cap) == (768, 768))
        // Outputs over 1024² are capped at 1024², as in diffusers.
        XCTAssertEqual(KleinReference.maxArea(outputWidth: 1312, outputHeight: 800), 1024 * 1024)
        // No reference ever exceeds the cap.
        for (w, h) in [(4000, 3000), (6000, 1000), (1000, 6000), (1025, 1025)] {
            let s = KleinReference.size(width: w, height: h, maxArea: cap)
            XCTAssertLessThanOrEqual(s.width * s.height, cap, "\(w)×\(h)")
            XCTAssertEqual(s.width % 16, 0); XCTAssertEqual(s.height % 16, 0)
        }
    }

    /// The pixels are not stretched: a 1536×1024 image whose left 2/3 is red and right 1/3 blue
    /// decodes to 1248×832 with the edge still at 2/3 of the width, and nothing comes out squashed.
    func testDecodeKeepsAspectAndContent() {
        let (sw, sh) = (1536, 1024)
        let cs = CGColorSpace(name: CGColorSpace.sRGB)!
        let ctx = CGContext(data: nil, width: sw, height: sh, bitsPerComponent: 8, bytesPerRow: sw * 4,
            space: cs, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.setFillColor(red: 1, green: 0, blue: 0, alpha: 1); ctx.fill(CGRect(x: 0, y: 0, width: 1024, height: sh))
        ctx.setFillColor(red: 0, green: 0, blue: 1, alpha: 1); ctx.fill(CGRect(x: 1024, y: 0, width: 512, height: sh))
        let image = ctx.makeImage()!

        withMLXCPU {
            let x = KleinReference.pixels(image, maxArea: 1024 * 1024)
            XCTAssertEqual(x.shape, [1, 3, 832, 1248])
            let r = x[0, 0, 416, 0...].asArray(Float.self)       // red plane, middle row
            let b = x[0, 2, 416, 0...].asArray(Float.self)       // blue plane
            // The edge sits at 2/3·1248 = 832, within a pixel.
            XCTAssertEqual(r[820], 1, accuracy: 1e-3); XCTAssertEqual(b[820], -1, accuracy: 1e-3)
            XCTAssertEqual(r[844], -1, accuracy: 1e-3); XCTAssertEqual(b[844], 1, accuracy: 1e-3)
            // The corners carry content, not the empty fill (black would read -1 in every plane).
            XCTAssertEqual(r[0], 1, accuracy: 1e-3)
            XCTAssertEqual(b[1247], 1, accuracy: 1e-3)
        }
    }
}
