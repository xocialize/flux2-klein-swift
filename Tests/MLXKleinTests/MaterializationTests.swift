// MLXKlein through the engine MAT gate (v0.19.0) + WeightSourcing shape. Offline.

import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest

@testable import MLXKlein

final class MaterializationTests: XCTestCase {

    private func satisfiedSnapshot() throws -> (dir: URL, cleanup: () -> Void) {
        let base = FileManager.default.temporaryDirectory.appending(path: "klein-mat-\(UUID().uuidString)")
        for sub in ["transformer", "vae"] {
            try FileManager.default.createDirectory(at: base.appending(path: sub), withIntermediateDirectories: true)
        }
        return (base, { try? FileManager.default.removeItem(at: base) })
    }

    func testMATGatePasses() throws {
        let (dir, cleanup) = try satisfiedSnapshot(); defer { cleanup() }
        let report = MaterializationConformance.check(
            freshConfiguration: KleinConfiguration(quant: .int4),
            satisfiedConfiguration: KleinConfiguration(quant: .int4, snapshotPath: dir.path))
        XCTAssertTrue(report.passed, report.summary)
    }

    func testWeightSourcesSingleSnapshot() {
        let s = KleinConfiguration().weightSources
        XCTAssertEqual(s.map(\.role), ["snapshot"])
        XCTAssertEqual(s[0].repo, "mlx-community/FLUX.2-klein-4B-bf16")
        XCTAssertTrue(s[0].matching!.contains("transformer/*"))
    }

    func testStoreLayoutAndExplicitPath() throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "klein-store-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let cfg = KleinConfiguration()
        XCTAssertEqual(cfg.missingWeightSources(storeRoot: root).count, 1)
        let dir = root.appending(path: "mlx-community/FLUX.2-klein-4B-bf16")
        for sub in ["transformer", "vae"] {
            try FileManager.default.createDirectory(at: dir.appending(path: sub), withIntermediateDirectories: true)
        }
        XCTAssertTrue(cfg.missingWeightSources(storeRoot: root).isEmpty)
        XCTAssertEqual(cfg.resolvedSnapshotDirectory(storeRoot: root)?.path, dir.path)
    }

    func testManifestApacheAndSurfaces() {
        XCTAssertEqual(Klein4BT2IPackage.manifest.license.weightLicense, .apache2)
        let names = Klein4BT2IPackage.manifest.surfaces.map(\.name)
        XCTAssertTrue(names.contains("flux2-klein-4b-t2i"))
        XCTAssertTrue(names.contains("flux2-klein-4b-edit"))   // multi-ref edit surface
    }

    func testCodableRoundTrip() throws {
        let cfg = KleinConfiguration(quant: .int4)
        let decoded = try JSONDecoder().decode(KleinConfiguration.self, from: JSONEncoder().encode(cfg))
        XCTAssertEqual(decoded.quant, .int4)
        XCTAssertEqual(decoded.defaultSteps, 4)
    }
}
