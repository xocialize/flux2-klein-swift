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

    // MARK: - Base (quality) tier

    func testBaseMATGatePasses() throws {
        let (dir, cleanup) = try satisfiedSnapshot(); defer { cleanup() }
        let report = MaterializationConformance.check(
            freshConfiguration: KleinConfiguration.base(quant: .int4),
            satisfiedConfiguration: KleinConfiguration.base(quant: .int4, snapshotPath: dir.path))
        XCTAssertTrue(report.passed, report.summary)
    }

    func testBaseFactoryDefaults() {
        let base = KleinConfiguration.base()
        XCTAssertEqual(base.repo, "mlx-community/FLUX.2-klein-base-4B-bf16")
        XCTAssertEqual(base.defaultSteps, 28)
        XCTAssertEqual(base.guidanceScale, 4.0)   // CFG on
        let turbo = KleinConfiguration.turbo()
        XCTAssertEqual(turbo.repo, "mlx-community/FLUX.2-klein-4B-bf16")
        XCTAssertEqual(turbo.defaultSteps, 4)
        XCTAssertEqual(turbo.guidanceScale, 1.0)  // no CFG (distilled)
    }

    func testBaseManifestApacheAndSurfaces() {
        XCTAssertEqual(Klein4BBaseT2IPackage.manifest.license.weightLicense, .apache2)
        let names = Klein4BBaseT2IPackage.manifest.surfaces.map(\.name)
        XCTAssertTrue(names.contains("flux2-klein-4b-base-t2i"))
        XCTAssertTrue(names.contains("flux2-klein-4b-base-edit"))
        // Distinct surfaces from the distilled tier so both co-resist, selectable by PackageID.
        let distilled = Set(Klein4BT2IPackage.manifest.surfaces.map(\.name))
        XCTAssertTrue(Set(names).isDisjoint(with: distilled))
    }

    func testEvictSurfacesLowerFootprintHint() {
        // Non-evict ⇒ nil ⇒ governor uses the static QuantFootprint.
        XCTAssertNil(KleinConfiguration(quant: .int4).residentBytesHint)
        XCTAssertNil(KleinConfiguration(quant: .int4).peakActivationBytesHint)
        // Evict ⇒ the light-tier phys hint (≈5 GB), so the 16 GB tier is admissible.
        let evict = KleinConfiguration.turbo(quant: .int4, evictEncoder: true)
        XCTAssertNotNil(evict.residentBytesHint)
        XCTAssertLessThan(evict.residentBytesHint!, 6_000_000_000)
        XCTAssertNotNil(evict.peakActivationBytesHint)
        XCTAssertLessThan(evict.peakActivationBytesHint!, 6_000_000_000)
    }

    func testGuidanceScaleCodableRoundTrip() throws {
        let cfg = KleinConfiguration.base(quant: .int8)
        let decoded = try JSONDecoder().decode(KleinConfiguration.self, from: JSONEncoder().encode(cfg))
        XCTAssertEqual(decoded.guidanceScale, 4.0)
        XCTAssertEqual(decoded.defaultSteps, 28)
    }
}
