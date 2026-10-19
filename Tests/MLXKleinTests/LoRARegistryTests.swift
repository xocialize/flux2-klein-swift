// Registry decode, id lookup, surface filtering, and the load-bearing slot-ordering contract.
// Mirrors the LTX/Turbo LoRARegistryTests. All offline — no weights, no network, no GPU.

import Foundation
import XCTest

import MLXKlein

final class LoRARegistryTests: XCTestCase {

    func testBundledManifestDecodes() throws {
        let reg = try LoRARegistry.bundled()
        XCTAssertEqual(reg.schemaVersion, 1)
        XCTAssertEqual(reg.base, "FLUX.2-klein-4B")
        XCTAssertFalse(reg.adapters.isEmpty)
    }

    func testEntryLookupAndFields() throws {
        let reg = try LoRARegistry.bundled()
        let e = try XCTUnwrap(reg.entry(id: "refcontrol-pose"))
        XCTAssertEqual(e.displayName, "RefControl Pose")
        XCTAssertEqual(e.repo, "xocialize/refcontrol-FLUX.2-klein-4B-pose-lora")
        XCTAssertEqual(e.weightFile, "refcontrol-pose-klein-4b.safetensors")
        XCTAssertEqual(e.trigger, "refcontrol")
        XCTAssertEqual(e.defaultStrength, 1.0)
        XCTAssertEqual(e.surface, "imageEdit")
        XCTAssertEqual(e.kind, "ic")
        XCTAssertTrue(e.isReferenceConditioned)
        XCTAssertEqual(e.verification?.rank, 32)
        XCTAssertEqual(e.verification?.headerDialect, "bfl-fused")
    }

    func testUnknownEntryReturnsNil() throws {
        let reg = try LoRARegistry.bundled()
        XCTAssertNil(reg.entry(id: "does-not-exist"))
    }

    /// The externally-sourced consistency LoRA: plain kind, provenance `source`, no ordered slots.
    func testConsistenceEditEntry() throws {
        let e = try XCTUnwrap(try LoRARegistry.bundled().entry(id: "consistence-edit"))
        XCTAssertEqual(e.kind, "plain")
        XCTAssertFalse(e.isReferenceConditioned)
        XCTAssertEqual(e.repo, "xocialize/consistence-edit-FLUX.2-klein-4B-lora")  // our durable mirror
        XCTAssertEqual(e.weightFile, "f2k_4B_consist_20260314.safetensors")
        XCTAssertEqual(e.defaultStrength, 0.6)
        XCTAssertEqual(e.license, "apache-2.0")
        XCTAssertNotNil(e.source)              // provenance recorded
        XCTAssertTrue(e.slots.isEmpty)         // no load-bearing conditioning slots
        XCTAssertEqual(e.verification?.rank, 128)
        XCTAssertEqual(e.verification?.headerDialect, "diffusers-split")
    }

    func testSlotsAreDeclaredInOrder() throws {
        let e = try XCTUnwrap(try LoRARegistry.bundled().entry(id: "refcontrol-pose"))
        XCTAssertEqual(e.slots.map(\.slot), [1, 2])
        XCTAssertEqual(e.slots.map(\.role), ["pose", "reference"])
        XCTAssertEqual(e.requiredSlots.count, 2)
    }

    /// The core contract: bindings given in ANY order resolve to the manifest's slot order
    /// (pose→reference), since that order is load-bearing for RefControl.
    func testOrderedInputsRespectsDeclaredOrder() throws {
        let e = try XCTUnwrap(try LoRARegistry.bundled().entry(id: "refcontrol-pose"))
        let pose = URL(fileURLWithPath: "/tmp/skeleton.png")
        let ref = URL(fileURLWithPath: "/tmp/subject.png")
        // Supply reference FIRST — output must still be [pose, reference].
        let ordered = try e.orderedInputs(bindings: ["reference": ref, "pose": pose])
        XCTAssertEqual(ordered, [pose, ref])
    }

    func testMissingRequiredSlotThrows() throws {
        let e = try XCTUnwrap(try LoRARegistry.bundled().entry(id: "refcontrol-pose"))
        XCTAssertThrowsError(
            try e.orderedInputs(bindings: ["pose": URL(fileURLWithPath: "/tmp/s.png")])
        ) { err in
            guard case LoRARegistryError.missingSlot(let role, let adapter) = err else {
                return XCTFail("expected missingSlot, got \(err)")
            }
            XCTAssertEqual(role, "reference")
            XCTAssertEqual(adapter, "refcontrol-pose")
        }
    }

    func testUnknownSlotThrows() throws {
        let e = try XCTUnwrap(try LoRARegistry.bundled().entry(id: "refcontrol-pose"))
        XCTAssertThrowsError(
            try e.orderedInputs(bindings: [
                "pose": URL(fileURLWithPath: "/tmp/s.png"),
                "reference": URL(fileURLWithPath: "/tmp/r.png"),
                "bogus": URL(fileURLWithPath: "/tmp/b.png"),
            ])
        ) { err in
            guard case LoRARegistryError.unknownSlot(let role, _) = err else {
                return XCTFail("expected unknownSlot, got \(err)")
            }
            XCTAssertEqual(role, "bogus")
        }
    }

    /// The fal object-removal mirror: rank-16 PEFT-dialect edit LoRA on our own repo.
    func testObjectRemoveEntry() throws {
        let e = try XCTUnwrap(try LoRARegistry.bundled().entry(id: "object-remove"))
        XCTAssertEqual(e.kind, "plain")
        XCTAssertEqual(e.repo, "xocialize/object-remove-FLUX.2-klein-4B-lora")  // durable mirror
        XCTAssertEqual(e.weightFile, "flux-object-remove-lora.safetensors")
        XCTAssertEqual(e.defaultStrength, 1.1)
        XCTAssertEqual(e.license, "apache-2.0")
        XCTAssertEqual(e.verification?.rank, 16)
        XCTAssertEqual(e.trigger, "Remove the highlighted object from the scene")
    }

    /// The fal background-removal mirror (rank-16 PEFT dialect, our repo).
    func testBackgroundRemoveEntry() throws {
        let e = try XCTUnwrap(try LoRARegistry.bundled().entry(id: "background-remove"))
        XCTAssertEqual(e.repo, "xocialize/background-remove-FLUX.2-klein-4B-lora")
        XCTAssertEqual(e.weightFile, "flux-background-remove-lora.safetensors")
        XCTAssertEqual(e.defaultStrength, 1.1)
        XCTAssertEqual(e.license, "apache-2.0")
        XCTAssertEqual(e.trigger, "Remove the background from the image")
        XCTAssertEqual(e.verification?.rank, 16)
    }

    /// The fal green-screen outpaint mirror (rank-16 PEFT dialect, our repo).
    func testOutpaintEntry() throws {
        let e = try XCTUnwrap(try LoRARegistry.bundled().entry(id: "outpaint"))
        XCTAssertEqual(e.repo, "xocialize/outpaint-FLUX.2-klein-4B-lora")
        XCTAssertEqual(e.weightFile, "flux-outpaint-lora.safetensors")
        XCTAssertEqual(e.defaultStrength, 1.1)
        XCTAssertEqual(e.trigger, "Fill the green spaces according to the image")
        XCTAssertEqual(e.verification?.rank, 16)
    }

    /// The fal 2x2 sprite-sheet mirror (rank-16 PEFT dialect, our repo).
    func testSpritesheetEntry() throws {
        let e = try XCTUnwrap(try LoRARegistry.bundled().entry(id: "spritesheet"))
        XCTAssertEqual(e.repo, "xocialize/spritesheet-FLUX.2-klein-4B-lora")
        XCTAssertEqual(e.weightFile, "flux-spritesheet-lora.safetensors")
        XCTAssertEqual(e.defaultStrength, 1.1)
        XCTAssertEqual(e.trigger, "2x2 sprite sheet")
        XCTAssertEqual(e.verification?.rank, 16)
    }

    /// The fal red-zoom mirror (rank-16 PEFT dialect, our repo).
    func testZoomEntry() throws {
        let e = try XCTUnwrap(try LoRARegistry.bundled().entry(id: "zoom"))
        XCTAssertEqual(e.repo, "xocialize/zoom-FLUX.2-klein-4B-lora")
        XCTAssertEqual(e.weightFile, "flux-red-zoom-lora.safetensors")
        XCTAssertEqual(e.defaultStrength, 1.1)
        XCTAssertEqual(e.trigger, "Zoom into the red highlighted area")
        XCTAssertEqual(e.verification?.rank, 16)
    }

    /// The self-trained virtual try-on LoRA: ic with 3 ordered slots (person/top/bottom).
    func testTryOnEntry() throws {
        let e = try XCTUnwrap(try LoRARegistry.bundled().entry(id: "tryon"))
        XCTAssertEqual(e.kind, "ic")
        XCTAssertTrue(e.isReferenceConditioned)
        XCTAssertEqual(e.repo, "xocialize/tryon-FLUX.2-klein-4B-lora")
        XCTAssertEqual(e.trigger, "TRYON")
        XCTAssertEqual(e.slots.map(\.role), ["person", "top", "bottom"])
        XCTAssertEqual(e.requiredSlots.count, 3)
        XCTAssertEqual(e.verification?.rank, 32)
        // ordered-slot binding resolves person→top→bottom regardless of input order
        let ordered = try e.orderedInputs(bindings: [
            "bottom": URL(fileURLWithPath: "/tmp/b.png"),
            "person": URL(fileURLWithPath: "/tmp/p.png"),
            "top": URL(fileURLWithPath: "/tmp/t.png"),
        ])
        XCTAssertEqual(ordered.map(\.lastPathComponent), ["p.png", "t.png", "b.png"])
    }

    func testSurfacedFilter() throws {
        let reg = try LoRARegistry.bundled()
        XCTAssertEqual(reg.surfaced("imageEdit").count,
                       reg.adapters.filter { $0.surface == "imageEdit" }.count)
        XCTAssertTrue(reg.surfaced("textToImage").isEmpty)
        XCTAssertEqual(reg.surfaced(nil).count, reg.adapters.count)
    }

    /// A pre-placed file under the entry's `weightFile` name is used in place — no download.
    func testLoRACacheLocalStoreFirst() async throws {
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("klein-lora-cache-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmp) }

        let e = try XCTUnwrap(try LoRARegistry.bundled().entry(id: "refcontrol-pose"))
        let placed = tmp.appendingPathComponent(e.weightFile)
        try Data("stub".utf8).write(to: placed)

        let resolved = try await LoRACache(directory: tmp).ensure(e)
        XCTAssertEqual(resolved, placed)
    }
}
