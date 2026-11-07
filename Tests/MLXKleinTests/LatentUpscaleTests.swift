// metaData["latentUpscale"] (AB-A-0257): parsing, the draw limit, workload units and the footprint line. Offline.

import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest

@testable import MLXKlein

final class LatentUpscaleTests: XCTestCase {
    func testPasses() throws {
        XCTAssertEqual(try KleinLatentUpscaleOption.passes([:]), 0)
        XCTAssertEqual(try KleinLatentUpscaleOption.passes(["latentUpscale": .null]), 0)
        XCTAssertEqual(try KleinLatentUpscaleOption.passes(["latentUpscale": .int(1)]), 1)
        XCTAssertEqual(try KleinLatentUpscaleOption.passes(["latentUpscale": .double(2)]), 2)
        XCTAssertEqual(try KleinLatentUpscaleOption.passes(["latentUpscale": .string("2")]), 2)
        for bad: MetaValue in [.int(3), .int(-1), .double(1.5), .bool(true), .string("x4")] {
            XCTAssertThrowsError(try KleinLatentUpscaleOption.passes(["latentUpscale": bad]), "\(bad)")
        }
    }

    func testDrawLimit() {
        XCTAssertNoThrow(try KleinLatentUpscaleOption.validate(width: 1024, height: 1024, passes: 2))
        XCTAssertNoThrow(try KleinLatentUpscaleOption.validate(width: 1280, height: 768, passes: 1))
        XCTAssertThrowsError(try KleinLatentUpscaleOption.validate(width: 1280, height: 1024, passes: 1))
        XCTAssertNoThrow(try KleinLatentUpscaleOption.validate(width: 2048, height: 2048, passes: 0))
        XCTAssertThrowsError(try KleinLatentUpscaleOption.validate(width: 2064, height: 2048, passes: 0))
    }

    func testWorkloadUnits() {
        let c = KleinConfiguration()
        XCTAssertEqual(c.workloadUnits(for: T2IRequest(prompt: "x")), 1024 * 1024)
        XCTAssertEqual(c.workloadUnits(for: T2IRequest(prompt: "x", width: 1024, height: 1024,
                                                        metaData: ["latentUpscale": .int(1)])), 2048 * 2048)
        XCTAssertEqual(c.workloadUnits(for: T2IRequest(prompt: "x", width: 1024, height: 768,
                                                        metaData: ["latentUpscale": .int(2)])), 4096 * 3072)
        XCTAssertEqual(c.workloadUnits(for: T2IRequest(prompt: "x", width: 1030, height: 1030)), 1024 * 1024)
    }

    func testFootprintLine() {
        for f in Klein4BT2IPackage.manifest.requirements.footprints
            + Klein4BBaseT2IPackage.manifest.requirements.footprints {
            let line = try! XCTUnwrap(f.activationScaling)
            XCTAssertEqual(line.axis, .outputPixels)
            XCTAssertEqual(line.measuredCeiling, 4096 * 4096)
            // A ≤ 1024² run reserves the scalar it did before; 4096² its measured decode (25.96 GB) with margin.
            XCTAssertLessThanOrEqual(line.projectedBytes(at: 1024 * 1024), f.peakActivationBytes)
            XCTAssertGreaterThanOrEqual(line.bytesAtCeiling, 25_960_000_000)
            XCTAssertGreaterThanOrEqual(line.projectedBytes(at: 2048 * 2048), 7_100_000_000)
        }
        let evict = KleinConfiguration.turbo(quant: .int4, evictEncoder: true)
        let line = try! XCTUnwrap(evict.activationScalingHint)
        XCTAssertLessThanOrEqual(line.projectedBytes(at: 768 * 768), evict.peakActivationBytesHint!)
        XCTAssertGreaterThanOrEqual(line.bytesAtCeiling, 25_960_000_000)
    }

    func testFootprintConformance() {
        for manifest in [Klein4BT2IPackage.manifest, Klein4BBaseT2IPackage.manifest] {
            for c in [KleinConfiguration(quant: .int4), .turbo(quant: .int4, evictEncoder: true)] {
                let report = FootprintConformance.check(manifest: manifest, configuration: c)
                XCTAssertTrue(report.passed, report.summary)
            }
        }
    }

    func testCodableKeepsLatentUpscaler() throws {
        let off = KleinConfiguration(latentUpscaler: false)
        let decoded = try JSONDecoder().decode(KleinConfiguration.self, from: JSONEncoder().encode(off))
        XCTAssertFalse(decoded.latentUpscaler)
        let legacy = try JSONDecoder().decode(KleinConfiguration.self,
                                              from: Data(#"{"repo":"r","quant":"int4"}"#.utf8))
        XCTAssertTrue(legacy.latentUpscaler)
    }
}
