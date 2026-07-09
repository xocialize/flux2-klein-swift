// CancellationTests.swift — FLUX.2 klein (distilled + base tiers) through the engine's CAN
// gate (offline, no MLX kernels, no weights). CAN-1/2 drive the real run() pre-cancelled:
// the entry checkpoint (`try Task.checkCancellation()` as the FIRST act of run(), before
// notLoaded validation) fires before weights are touched, so a stub configuration suffices.
// CAN-3 is the document of record for the checkpoint cadence:
//   - encode seam — `try Task.checkCancellation()` at the end of encodeAndEvict
//     (Klein4BT2IPackage.swift), after the conditioner is evicted and before the denoise loop.
//   - denoise/step — `if Task.isCancelled { break }` at the top of the denoise loops in
//     KleinPipeline.generate and KleinEditPipeline.generate (Sources/Klein), the shared loops
//     behind t2i AND multi-ref edit on both tiers (non-throwing core API — sanctioned break).
//   - pre-decode seam — a cancelled task skips the monolithic VAE decode (ONE MLX eval, no
//     chunk loop, so no per-chunk decode cadence is claimed).
//   - the wrapper's post-generate `try Task.checkCancellation()` rethrows the
//     CancellationError UNCHANGED (no catch blocks anywhere in Sources — nothing to launder).
// The base tier delegates run() to an inner Klein4BT2IPackage, so one set of checkpoints
// covers both PackageIDs; each still passes the gate independently below.

import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest

@testable import MLXKlein

final class CancellationTests: XCTestCase {

    // MARK: - CAN-1 / CAN-2 — pre-cancelled run() propagation + classification

    func testCANGatePreCancelledRunDistilled() async {
        // Stub config; construction is cheap (C13) and the entry checkpoint throws before
        // validation or weights are touched, so this is offline-safe.
        let package = Klein4BT2IPackage(configuration: KleinConfiguration.turbo())
        let report = await CancellationConformance.checkRun(
            package: package,
            request: T2IRequest(prompt: "probe"))
        XCTAssertTrue(report.passed, report.summary)
    }

    func testCANGatePreCancelledRunBase() async {
        let package = Klein4BBaseT2IPackage(configuration: KleinConfiguration.base(quant: .int4))
        let report = await CancellationConformance.checkRun(
            package: package,
            request: T2IRequest(prompt: "probe"))
        XCTAssertTrue(report.passed, report.summary)
    }

    // MARK: - CAN-3 — checkpoint-cadence declaration (the document of record)

    /// Both tiers share the Klein pipelines: an encode-seam checkpoint after the (optional)
    /// encoder-evict encode, then per-denoise-step Task.isCancelled breaks
    /// (KleinPipeline/KleinEditPipeline denoise loops), then a cancelled-task skip of the
    /// monolithic VAE decode. Only the real per-step denoise cadence is declared; encode and
    /// decode are single forwards (seams, not recurring units).
    private var posture: CancellationConformance.CheckpointPosture {
        .cadence([
            .init(phase: .denoise, unit: .step)
        ])
    }

    func testCANCadenceDeclarationDistilled() {
        // Multi-GB peak activation (11.2 GB declared) implies long runs — the sub-second
        // exemption is not available.
        XCTAssertTrue(CancellationConformance.longRunImplied(by: Klein4BT2IPackage.manifest))
        let report = CancellationConformance.checkCadence(
            manifest: Klein4BT2IPackage.manifest, posture: posture)
        XCTAssertTrue(report.passed, report.summary)
    }

    func testCANCadenceDeclarationBase() {
        XCTAssertTrue(CancellationConformance.longRunImplied(by: Klein4BBaseT2IPackage.manifest))
        let report = CancellationConformance.checkCadence(
            manifest: Klein4BBaseT2IPackage.manifest, posture: posture)
        XCTAssertTrue(report.passed, report.summary)
    }
}
