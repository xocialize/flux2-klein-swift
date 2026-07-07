// MLXEngine `textToImage` + `imageEdit` package for **FLUX.2-klein-base-4B** — the quality tier.
//
// Architecturally identical to the distilled Klein4BT2IPackage (same 5-double + 20-single MMDiT,
// Qwen3-4B 3-layer conditioner, FLUX.2 VAE — verified: transformer config byte-identical, both
// guidance_embeds=false). The base checkpoint is NOT guidance-distilled, so it runs classic
// two-pass classifier-free guidance (guidance 4.0) + negative prompts over ~28 steps — higher
// prompt/edit adherence at the cost of ~2× the forward passes. So this is a thin variant: a
// distinct PackageID + manifest (base surfaces, klein-base provenance) delegating all
// lifecycle/inference to an inner Klein4BT2IPackage whose Configuration carries the base repo +
// guidanceScale 4.0. Select base vs distilled by PackageID.

import Foundation
import MLXToolKit

@InferenceActor
public final class Klein4BBaseT2IPackage: ModelPackage {
    public typealias Configuration = KleinConfiguration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // Base 4B is plain Apache-2.0 (weights) + MIT (port), same as distilled. The 9B base
            // variants are FLUX Non-Commercial and are intentionally NOT wrapped.
            license: LicenseDeclaration(weightLicense: .apache2, portCodeLicense: .mit),
            provenance: Provenance(sourceRepo: "black-forest-labs/FLUX.2-klein-base-4B", revision: "main", tier: 1),
            requirements: RequirementsManifest(
                // Same resident envelope as distilled (identical architecture / weight sizes —
                // delegates to the same inner Klein4BT2IPackage / shared KleinGenerator). Activation
                // is higher: two-pass CFG runs the DiT twice per step. Measured via klein-cli
                // @1024²/28-step base (bf16). phys re-baseline pending in-app.
                footprints: [
                    QuantFootprint(quant: .bf16, residentBytes: 16_000_000_000, peakActivationBytes: 10_000_000_000),
                    QuantFootprint(quant: .int8, residentBytes: 12_000_000_000, peakActivationBytes: 10_000_000_000),
                    QuantFootprint(quant: .int4, residentBytes: 11_000_000_000, peakActivationBytes: 10_000_000_000),
                ],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: nil
            ),
            specialties: [],
            surfaces: [
                T2IContract.descriptor(
                    name: "flux2-klein-4b-base-t2i",
                    summary: "FLUX.2-klein-base-4B text-to-image (Apache-2.0): the QUALITY tier — NOT "
                        + "guidance-distilled, so it runs classic two-pass CFG (guidance 4.0) + "
                        + "negative prompts over ~28 steps for stronger prompt adherence than the "
                        + "4-step distilled tier. Same 4B MMDiT + Qwen3-4B + FLUX.2 VAE.",
                    modes: []
                ),
                IEditContract.descriptor(
                    name: "flux2-klein-4b-base-edit",
                    summary: "FLUX.2-klein-base-4B multi-reference EDITING with CFG: compose from one "
                        + "or more reference images (subject/style/scene) with negative-prompt "
                        + "guidance for tighter instruction adherence than the distilled edit tier. "
                        + "Pass conditioning images in prompt order.",
                    modes: []
                )
            ]
        )
    }

    private let inner: Klein4BT2IPackage

    public nonisolated init(configuration: Configuration) {
        self.inner = Klein4BT2IPackage(configuration: configuration)
    }

    public func load() async throws { try await inner.load() }
    public func unload() async { await inner.unload() }
    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        try await inner.run(request)
    }
}

extension Klein4BBaseT2IPackage {
    /// The author one-liner the engine registers (distinct PackageID from the distilled tier).
    public nonisolated static var registration: PackageRegistration {
        .of(Klein4BBaseT2IPackage.self)
    }
}
