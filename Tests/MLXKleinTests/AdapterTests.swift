// Typed adapters on both klein packages (contract 1.53.0, AB-A-0192): the ADP-1..4 gate over the
// manifests and the format-3 registry (registry/flux2-klein-4b.json, read by path), and the IC
// references' order. Offline: no weights.

import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest

@testable import MLXKlein

final class AdapterTests: XCTestCase {

    /// The registry's offered adapters, as the engine decodes them (registry-only fields ignored).
    static func registry() throws -> [AdapterDescriptor] {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("registry/flux2-klein-4b.json")
        struct Registry: Decodable { let adapters: [AdapterDescriptor] }
        return try JSONDecoder().decode(Registry.self, from: Data(contentsOf: url)).adapters
    }

    func testADPGateDistilled() throws {
        let report = AdapterConformance.check(manifest: Klein4BT2IPackage.manifest,
                                              configuration: KleinConfiguration.turbo(),
                                              catalog: try Self.registry())
        XCTAssertTrue(report.passed, report.summary)
    }

    func testADPGateBase() throws {
        let report = AdapterConformance.check(manifest: Klein4BBaseT2IPackage.manifest,
                                              configuration: KleinConfiguration.base(),
                                              catalog: try Self.registry())
        XCTAssertTrue(report.passed, report.summary)
    }

    /// Every surface takes adapters per request; only the edit surfaces run IC adapters.
    func testDeclarations() {
        for manifest in [Klein4BT2IPackage.manifest, Klein4BBaseT2IPackage.manifest] {
            for surface in manifest.surfaces {
                let support = try! XCTUnwrap(surface.adapters, surface.name)
                XCTAssertEqual(support.baseModel, "flux2-klein-4b")
                XCTAssertEqual(support.formats, [.bflFused, .diffusersPEFT])
                XCTAssertEqual(support.binding, .perRequest)
                XCTAssertEqual(support.maxCount, 4)
                XCTAssertEqual(support.kinds.contains(.referenceConditioned), surface.capability == .imageEdit,
                               surface.name)
            }
        }
    }

    /// The registry's IC adapters plan onto the edit surface; their conditioning comes back in
    /// slot order whatever order the request used, and leads the references.
    func testICConditioningOrder() throws {
        let catalog = try Self.registry()
        let support = KleinAdapters.support(.imageEdit)
        let skeleton = Image(format: .png, data: Data([1]), width: 1, height: 1)
        let subject = Image(format: .png, data: Data([2]), width: 1, height: 1)
        let ref = AdapterRef(id: "refcontrol-pose", conditioning: [
            ConditioningInput(role: "reference", media: .image(subject)),
            ConditioningInput(role: "pose", media: .image(skeleton)),
        ])
        let plan = try support.plan([ref], capability: .imageEdit, quant: .int8) { id in
            catalog.first { $0.id == id }
        }
        let resolved = plan.map {
            ResolvedAdapter(id: $0.descriptor.id, kind: $0.descriptor.kind, format: $0.descriptor.format,
                            fileURL: URL(fileURLWithPath: "/dev/null"), source: $0.descriptor.source,
                            strength: $0.strength, conditioning: $0.conditioning,
                            footprint: $0.descriptor.resolvedFootprint)
        }
        XCTAssertEqual(try KleinAdapters.conditioningImages(resolved).map(\.data), [Data([1]), Data([2])])
        // An IC adapter can't ride text-to-image.
        XCTAssertThrowsError(try KleinAdapters.support(.textToImage).plan(
            [ref], capability: .textToImage, quant: .int8) { id in catalog.first { $0.id == id } })
    }

    /// The deprecated load-time adapter still builds a specialty package (ModelSheetStudio's pose
    /// tier sets it), and a missing file still fails load() legibly.
    func testDeprecatedLoadTimeLoRA() async {
        var config = KleinConfiguration.turbo(snapshotPath: "/nonexistent-klein-snapshot")
        config.loraPath = "/nonexistent.safetensors"
        config.loraStrength = 0.8
        XCTAssertEqual(config.loraStrength, 0.8)
        do {
            try await Klein4BT2IPackage(configuration: config).load()
            XCTFail("load() passed with no snapshot")
        } catch {
            XCTAssertTrue("\(error)".contains("nonexistent-klein-snapshot"), "\(error)")
        }
    }

    func testTwoICAdaptersAreRefused() {
        func ic(_ id: String) -> ResolvedAdapter {
            ResolvedAdapter(id: id, kind: .referenceConditioned, format: .bflFused,
                            fileURL: URL(fileURLWithPath: "/dev/null"),
                            source: AdapterSource(repo: "o/r", revision: String(repeating: "a", count: 40),
                                                  file: "f.safetensors", sha256: String(repeating: "b", count: 64)),
                            strength: 1, footprint: AdapterFootprint(residentBytes: 1, activationMultiplier: 2))
        }
        XCTAssertThrowsError(try KleinAdapters.conditioningImages([ic("a"), ic("b")]))
    }
}
