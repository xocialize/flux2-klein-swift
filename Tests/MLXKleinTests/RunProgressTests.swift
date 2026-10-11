// RunProgressTests.swift — Klein's end of the V2 run-phase progress plane (AB-A-0139): the core
// reports into its own engine-free `KleinProgress` sink, and the wrapper forwards those events
// into the engine's `RunProgress`. `KleinProgressBridge.forward` IS the wrapper's entire mapping;
// the loop test drives the real KleinPipeline denoise loop through it with a tiny random-init DiT
// on the CPU (no weights, no VAE), so the per-step reports are observed, not just declared.

import Foundation
import Klein
import MLX
import MLXToolKit
import XCTest

@testable import MLXKlein

final class RunProgressTests: XCTestCase {

    private final class Box: @unchecked Sendable {
        private let lock = NSLock()
        private var storage: [RunPhaseReport] = []
        func append(_ r: RunPhaseReport) { lock.lock(); storage.append(r); lock.unlock() }
        var items: [RunPhaseReport] { lock.lock(); defer { lock.unlock() }; return storage }
    }

    /// Bind both task-locals the way `run()` does — the engine's sink outermost, the core's
    /// bridge inside it — and return what the engine plane observed.
    private func forwarded(_ body: () -> Void) -> [RunPhaseReport] {
        let box = Box()
        RunProgress.$sink.withValue({ box.append($0) }) {
            KleinProgress.$sink.withValue(KleinProgressBridge.forward, operation: body)
        }
        return box.items
    }

    // MARK: - forwarding

    func testCoreEventsReachTheEnginePlane() {
        let seen = forwarded {
            KleinProgress.report(.encode)
            KleinProgress.report(.denoise)
            KleinProgress.report(.denoise, step: 1, totalSteps: 4)
            KleinProgress.report(.denoise, step: 4, totalSteps: 4)
            KleinProgress.report(.decode)
        }
        XCTAssertEqual(seen.map(\.phase), [.encode, .denoise, .denoise, .denoise, .decode])
        XCTAssertEqual(seen[2], RunPhaseReport(phase: .denoise, step: 1, totalSteps: 4))
        XCTAssertEqual(seen[3].step, 4)
        XCTAssertNil(seen[0].step)   // phases without steps carry no invented counts
    }

    /// The forward is a raw-value pass-through, which is only correct while the core's names
    /// ARE the canonical constants.
    func testCorePhaseNamesMatchCanonicalRunPhases() {
        XCTAssertEqual(RunPhase(rawValue: KleinProgress.Phase.encode.rawValue), .encode)
        XCTAssertEqual(RunPhase(rawValue: KleinProgress.Phase.denoise.rawValue), .denoise)
        XCTAssertEqual(RunPhase(rawValue: KleinProgress.Phase.decode.rawValue), .decode)
    }

    func testUnboundCoreSinkIsANoOp() {
        // The CLI and the parity gates run with nothing bound — reporting must stay safe there.
        KleinProgress.report(.denoise, step: 1, totalSteps: 4)
        // And a core event outside the wrapper's binding must not reach a bound engine plane.
        let box = Box()
        RunProgress.$sink.withValue({ box.append($0) }) {
            KleinProgress.report(.decode)
        }
        XCTAssertTrue(box.items.isEmpty)
    }

    // MARK: - the real denoise loop

    /// KleinPipeline.generate on a 1+1-layer, one-head DiT at 32² (4 image tokens), 3 steps, no
    /// VAE: the engine plane sees loop entry, then step 1/3 … 3/3, and no decode (vae == nil
    /// returns latents only).
    func testDenoiseLoopReportsEveryStep() {
        _ = Device.defaultDevice()
        Device.setDefault(device: Device(.cpu))
        defer { Device.setDefault(device: Device(.gpu)) }
        let seen: [RunPhaseReport] = Device.withDefaultDevice(Device(.cpu)) {
            let dit = Flux2Transformer(numLayers: 1, numSingleLayers: 1, numHeads: 1,
                                       jointAttentionDim: 16)
            let embeds = MLXRandom.normal([2, 16])
            return forwarded {
                let result = KleinPipeline.generate(
                    transformer: dit, vae: nil, promptEmbeds: embeds, height: 32, width: 32,
                    numInferenceSteps: 3, transformerDtype: .float32)
                XCTAssertNil(result.image)
            }
        }
        XCTAssertEqual(seen, [
            RunPhaseReport(phase: .denoise),
            RunPhaseReport(phase: .denoise, step: 1, totalSteps: 3),
            RunPhaseReport(phase: .denoise, step: 2, totalSteps: 3),
            RunPhaseReport(phase: .denoise, step: 3, totalSteps: 3),
        ])
    }
}
