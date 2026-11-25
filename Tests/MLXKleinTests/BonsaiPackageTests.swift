// KleinBonsaiT2IPackage offline: manifest shape, weight sources, the MAT gate and the refusals that need no weights.

import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest

@testable import MLXKlein

final class BonsaiPackageTests: XCTestCase {

    /// A snapshot with the three component folders, and FlowUpscaler's file under `upscaler/`.
    private func satisfiedSnapshot() throws -> (dir: URL, cleanup: () -> Void) {
        let base = FileManager.default.temporaryDirectory.appending(path: "bonsai-mat-\(UUID().uuidString)")
        for sub in ["transformer-packed-mflux", "text_encoder-mlx-4bit", "vae", "upscaler"] {
            try FileManager.default.createDirectory(at: base.appending(path: sub), withIntermediateDirectories: true)
        }
        FileManager.default.createFile(
            atPath: base.appending(path: "upscaler/\(KleinConfiguration.latentUpscalerFile)").path, contents: Data())
        return (base, { try? FileManager.default.removeItem(at: base) })
    }

    func testMATGatePasses() throws {
        let (dir, cleanup) = try satisfiedSnapshot(); defer { cleanup() }
        let report = MaterializationConformance.check(
            freshConfiguration: KleinBonsaiConfiguration(),
            satisfiedConfiguration: KleinBonsaiConfiguration(
                snapshotPath: dir.path, latentUpscalerPath: dir.appending(path: "upscaler").path))
        XCTAssertTrue(report.passed, report.summary)
    }

    func testWeightSourcesArePinnedAndSkipNothingLoadReads() {
        let sources = KleinBonsaiConfiguration().weightSources
        XCTAssertEqual(sources.map(\.role), ["snapshot", "latent-upscaler"])
        XCTAssertEqual(sources[0].repo, "prism-ml/bonsai-image-ternary-4B-mlx-2bit")
        XCTAssertEqual(sources[0].revision, KleinBonsaiConfiguration.measuredRevision)
        for needed in ["transformer-packed-mflux/*", "text_encoder-mlx-4bit/*", "vae/*", "tokenizer/*", "NOTICE.md"] {
            XCTAssertTrue(sources[0].matching!.contains(needed), needed)
        }
        XCTAssertEqual(KleinBonsaiConfiguration(latentUpscaler: false).weightSources.map(\.role), ["snapshot"])
    }

    /// A folder missing one component is not a snapshot: the klein bf16 layout must not pass for it.
    func testAKleinSnapshotIsNotABonsaiSnapshot() throws {
        let base = FileManager.default.temporaryDirectory.appending(path: "bonsai-klein-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: base) }
        for sub in ["transformer", "text_encoder", "vae"] {
            try FileManager.default.createDirectory(at: base.appending(path: sub), withIntermediateDirectories: true)
        }
        let cfg = KleinBonsaiConfiguration(snapshotPath: base.path, latentUpscaler: false)
        XCTAssertEqual(cfg.missingWeightSources(storeRoot: nil).map(\.role), ["snapshot"])
    }

    func testManifestIsTextToImageOnlyWithOneFootprintAndNoAdapters() {
        let manifest = KleinBonsaiT2IPackage.manifest
        XCTAssertEqual(manifest.surfaces.map(\.name), ["flux2-klein-4b-bonsai-t2i"])
        XCTAssertEqual(manifest.surfaces.map(\.capability), [.textToImage])
        XCTAssertNil(manifest.surfaces[0].adapters)
        XCTAssertEqual(manifest.license.weightLicense, .apache2)
        XCTAssertEqual(manifest.provenance.revision, KleinBonsaiConfiguration.measuredRevision)
        XCTAssertEqual(manifest.requirements.footprints.count, 1)
        // The line returns the scalar at 1024² and stays under the 4096² ceiling the klein tiers share.
        let footprint = manifest.requirements.footprints[0]
        let line = footprint.activationScaling!
        XCTAssertEqual(Double(line.baseBytes) + line.bytesPerUnit * 1024 * 1024, Double(footprint.peakActivationBytes),
                       accuracy: 1e6)
        XCTAssertEqual(line.measuredCeiling, Double(KleinLatentUpscaleOption.maxOutputPixels))
        // Lighter than every klein footprint.
        let klein = Klein4BT2IPackage.manifest.requirements.footprints
        XCTAssertLessThan(footprint.residentBytes, klein.map(\.residentBytes).min()!)
    }

    func testWorkloadUnitsFollowTheLatentUpscale() {
        let cfg = KleinBonsaiConfiguration()
        XCTAssertEqual(cfg.workloadUnits(for: T2IRequest(prompt: "p", width: 1024, height: 1024)), 1024 * 1024)
        XCTAssertEqual(cfg.workloadUnits(for: T2IRequest(prompt: "p", width: 1024, height: 1024,
                                                         metaData: ["latentUpscale": .int(2)])), 4096 * 4096)
    }

    func testConfigurationRoundTripsWithoutMachinePaths() throws {
        let cfg = KleinBonsaiConfiguration(snapshotPath: "/tmp/somewhere", defaultSteps: 6, latentUpscaler: false,
                                           latentUpscalerPath: "/tmp/up")
        let decoded = try JSONDecoder().decode(KleinBonsaiConfiguration.self, from: JSONEncoder().encode(cfg))
        XCTAssertEqual(decoded.repo, cfg.repo)
        XCTAssertEqual(decoded.revision, cfg.revision)
        XCTAssertEqual(decoded.defaultSteps, 6)
        XCTAssertFalse(decoded.latentUpscaler)
        XCTAssertNil(decoded.snapshotPath)
        XCTAssertNil(decoded.latentUpscalerPath)
    }

    func testRunBeforeLoadIsNotLoaded() async {
        let pkg = KleinBonsaiT2IPackage(configuration: .init())
        do {
            _ = try await pkg.run(T2IRequest(prompt: "p"))
            XCTFail("expected notLoaded")
        } catch PackageError.notLoaded {
        } catch { XCTFail("unexpected \(error)") }
    }
}
