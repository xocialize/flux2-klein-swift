# flux2-klein-swift

Swift/MLX port of Black Forest Labs **[FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)**
(Apache-2.0): a compact rectified-flow **MMDiT** (5 double-stream + 20 single-stream blocks) +
Qwen3-4B 3-layer-tap conditioner + FLUX.2 VAE. Ships MLXEngine's **`textToImage`** capability.

Two products:
- **`Klein`** — the engine-agnostic generator core (DiT + FLUX.2 VAE via the neutral
  [`flux2-vae-mlx-swift`](https://github.com/xocialize/flux2-vae-mlx-swift) package + Qwen3-4B
  encoder + scheduler + pipeline).
- **`MLXKlein`** — the conformant MLXEngine wrappers, both `textToImage` + `imageEdit`, selected by
  `PackageID`:
  - **`Klein4BT2IPackage`** (surfaces `flux2-klein-4b-t2i` / `-edit`) — the **distilled** tier:
    4-step, guidance 1.0 (no CFG), ~6 s @1024². Fast.
  - **`Klein4BBaseT2IPackage`** (surfaces `flux2-klein-4b-base-t2i` / `-base-edit`) — the **base /
    quality** tier: the non-distilled checkpoint run with classic two-pass CFG (guidance 4.0) +
    **negative prompts** over ~28 steps. Stronger adherence on dense scenes / text / fine attributes
    and on subject-preserving edits, at ~2× the forward cost per step (CFG) — plus the edit path
    doubles the sequence, so a base edit is ~4× a base T2I.

> **Status: complete · wrapped · GPU-validated.** Parity vs diffusers goldens (fp32/CPU): DiT
> cosine **≥0.9999995** at the 64×64 production grid (all 5 double + 20 single blocks, norm_out,
> whole-forward) · VAE decode **105–130 dB** (reused flux2-vae) · Qwen3 3-layer-tap encoder cosine
> **0.9999999**. GPU int4 1024²/4-step renders a sharp, prompt-faithful image in **~6 s** (19.2 GB
> peak; DiT **2.35 GB** at int4 — a 16 GB-tier fit). Validated through the real `load → run → decode`
> ModelPackage surface — the fourth public T2I backer alongside Lens, ERNIE-Image and Z-Image.

> **Scope: 4B only.** All FLUX.2-klein **9B** variants are under the FLUX **Non-Commercial** License
> (with a filter-or-review obligation) and are intentionally NOT wrapped here. The 4B (this repo) is
> plain Apache-2.0.

## Consuming it

`.package(url: "https://github.com/xocialize/flux2-klein-swift", from: "0.1.0")`, then import
`MLXKlein` (conformant package) or `Klein` (bare generator).

```swift
import MLXKlein
import MLXToolKit

let package = Klein4BT2IPackage(configuration: .init(
    quant: .int4,                       // ~11 GB pipeline (16 GB Mac); .bf16 ≈ 16 GB
    snapshotPath: "<root>/FLUX.2-klein-4B"))   // or nil → auto-materialize from mlx-community
try await package.load()

// text-to-image
let t2i = try await package.run(T2IRequest(
    prompt: "a red fox in a snowy forest at sunrise, photorealistic",
    width: 1024, height: 1024, seed: 42)) as! T2IResponse   // t2i.image: canonical .png

// multi-reference edit (v0.2.0)
let edit = try await package.run(IEditRequest(
    images: [foxImage],                                     // conditioning images, in prompt order
    prompt: "the fox sitting on a sunny tropical beach",
    width: 1024, height: 1024, seed: 5)) as! IEditResponse
await package.unload()
```

## Run-phase progress (v0.8.1, AB-A-0139)

The packages report into the engine's `RunProgress` plane (contract 1.18.0), so a host shows
`denoise 5/8` instead of an unexplained "running". The phases, in order:

| phase | covers | steps? |
|---|---|---|
| `encode` | Qwen3-4B conditioning (the encoder reload on the evict tier, reported by the wrapper), and the VAE encode of the reference images on the edit path | no |
| `denoise` | the flow-matching Euler loop: once at loop entry with no step, then after each completed step | **yes** — `step i/N`, 1-based |
| `decode` | FLUX.2 VAE decode — one eval, so no per-chunk counts | no |
| `postprocess` | PNG encode (reported by the wrapper, after the model is done) | no |

The core stays engine-free: `Klein` reports into its own `KleinProgress` task-local sink, and
`MLXKlein` forwards those events into `RunProgress` (`KleinProgressBridge`). Nothing bound ⇒ no-op, so the
CLI and the parity gates run unchanged. A host binds a sink around `engine.run` (contract 1.51.0)
or reads `engine.runProgress`.

## Adapters (v0.10.0)

Both packages take **typed adapters per request** (mlx-engine contract 1.53.0, AB-A-0192). A request names adapters by
catalog id (`T2IRequest` / `IEditRequest.adapters`). The engine resolves them against its catalog, fetches and
sha256-checks the files, and hands the package `resolvedAdapters`. The package rank-stacks them onto the resident DiT
(`KleinLoRASwapper`), and changes the stack only when the next request's set differs.

| Surface | `AdapterSupport` |
|---|---|
| `flux2-klein-4b-t2i`, `-base-t2i` | base `flux2-klein-4b`; formats `bfl-fused`, `diffusers-peft`; kinds `plain`; up to 4; strength 0…2; every quant; per request |
| `flux2-klein-4b-edit`, `-base-edit` | the same, plus kind `ic` |

- **IC adapters** (refcontrol-pose, tryon): the conditioning images arrive in slot order and lead the edit's
  references, followed by the request's own images. One IC adapter per request; an IC adapter on text-to-image is
  refused.
- **Every tensor must map.** `KleinLoRA` throws `unknownKeys` for a tensor it can't place, instead of skipping it.
  `KleinLoRA.unmappedKeys(_:)` runs the same check over a file's header.
  - The applicator now also adapts the top-level Linears: the in/out embedders, the timestep embedder, the three
    modulations and the final projection.
  - fal's four plain adapters target 8 of these, and before v0.10.0 they were silently dropped.
  - BFL's `final_layer.adaLN_modulation.1` is refused: diffusers' conversion swaps its scale and shift halves, so its
    delta isn't a rename.
- **The load-time `KleinConfiguration.loraPath` / `loraStrength` are deprecated** (v0.10.1 restored them after 0.10.0
  removed them). A configuration that sets one builds a specialty package around that adapter, as ModelSheetStudio's
  pose tier does. It is applied at load and stays under every request's adapters.
- **IC footprints** (AB-R-0593): ×1.012 (refcontrol-pose, 2 references) and ×1.013 (tryon, 3) over a no-adapter
  text-to-image control, at 1024² int8. Klein's peak is the 1024² VAE decode, so references barely move it. Declared
  as 1.02 each.
- **Measuring:** `klein-cli --pkg-adapter <registry.json> --adapter <id> --file <safetensors> [--cond role=png …]
  [--image png …]` drives the package exactly as the engine does and prints floor, peak and activation. Without
  `--adapter` it is the control arm.

**The registry: `registry/flux2-klein-4b.json`, format 3.1** (ltx-lora-registry v0.3.0, AB-A-0193). This is the source
of truth for the runtime adapter catalog (AB-A-0191).
- Each entry decodes as MLXToolKit's `AdapterDescriptor`, pinned to a commit with a hub-confirmed sha256.
- **Formats** use the engine's constants. fal's `bfl-fused-peft` is `bfl-fused` (the loader strips PEFT's prefix), and
  `diffusers-split` is `diffusers-peft`. The original dialect stays in `verification.headerDialect`.
- **Clearance:** tryon is `perceptual` (an E2E try-on judged in the CLI); the rest are `mechanical`.
- **`object-remove`** sits in the `withdrawn` array, with its truncated pin, until its mirror is replaced (AB-T-0217).
- **Checks:**
  - `python3 -I tools/adapter_registry.py validate [--remote] <this file>`, run from ltx-lora-registry;
  - `AdapterTests` runs the ADP-1..4 gate over both manifests and this file;
  - `LoRAMappingTests` checks every adapter's header (`Tests/KleinTests/Fixtures`) maps completely.

## Latent upscale to 2048² and 4096² (v0.11.0, AB-A-0257)

`metaData["latentUpscale"] = 1 | 2` on either surface (T2I and edit, both tiers): the DiT draws at the request's size
(at most 1024² of area), then [FlowUpscaler](https://huggingface.co/mlx-community/FlowUpscaler-bf16)
(TensorForger, Unlicense; `FlowUpscaler` in flux2-vae-mlx-swift 0.8.0) doubles the latents once or twice, and the
row-band decode draws the larger image. The response is 2× or 4× the draw on each side, up to 4096². It is the picture
the 1024² draw made, with finer detail; Klein asked for 2048² itself draws a different picture (AB-R-0795).

```swift
let r = try await package.run(T2IRequest(prompt: prompt, width: 1024, height: 1024, seed: 44,
                                         metaData: ["latentUpscale": .int(1)])) as! T2IResponse   // 2048×2048
```

- **Seeded:** pass i draws its noise from `seed + i`.
- **Refused:** a draw over 1024² of area with the option, a value other than 1 or 2, and (since v0.11.0) a plain
  draw over 2048² of area, which was never measured (the error points to the option).
- **Weights:** `KleinConfiguration.latentUpscaler` (default on) declares `mlx-community/FlowUpscaler-bf16`
  (118.6 MB) as a second weight source; `latentUpscalerPath` points at a local copy. The upscaler loads on the first
  request that asks for it (237 MB, fp32 compute). Off ⇒ the option is refused and nothing more is downloaded.
- **Every draw decodes in row bands** (plain draws past 1024² since v0.11.0, every size since v0.11.1):
  a native 2048² draw peaks at 6.9 GB over resident instead of the untiled decode's ~42 GB.

**Measured** through this package (`klein-cli --pkg-e2e --upscale 0,1,2`, bf16 distilled, 4 steps, M5 Max, MLX pool
capped at the engine's 2 GB), AB-R-0795's six prompts:

| | 1024² draw | → 2048² | → 4096² |
|---|---|---|---|
| Wall time per request (mean) | 6.75 s | 10.6 s | 33.4 s |
| MLX peak (resident 13.65 GB) | 24.40 GB | 20.75 GB | 39.29 GB |
| Process phys_footprint peak | 25.6 GB | ≤ 25.6 GB | 40.5–40.7 GB |
| DISTS vs the 1024² decode, downscaled (min–max, mean) | — | 0.017–0.035, 0.029 | 0.035–0.056, 0.045 |

The 2048² run peaked below the 1024² one because the 1024² decode was still untiled when this table was measured;
see the next section for the 1024² numbers since. The DISTS band is
the eval's (0.015–0.034 at 2048², 0.031–0.061 at 4096²). `Tools/latent-upscale/drift.py` scores it.

**Footprint.** The upscale and the fp32 decode do not depend on the DiT tier. Each footprint carries an
`ActivationScaling` on output pixels: a line through the tier's scalar at 1024² (11.2 GB distilled, 12.4 GB base) and
27.0 GB at 4096² (4096²'s 25.96 GB over the declared bf16 resident, +4 %). `KleinConfiguration` is
`WorkloadDeclaring` (the draw × 4^passes), so the engine reserves per run and refuses past 4096². A ≤ 1024² run
reserves what it did before. The encoder-evict tier's line runs through its 5.0 GB at 768².

## Row-band decode at every size (v0.11.1, AB-R-1130, AB-R-1134)

`KleinDecode` now takes flux2-vae-mlx-swift's row-band decode at every size, on both surfaces and both tiers. Before,
a draw of 1024² of area or less took the untiled decode, whose temporaries were the run's peak.

Measured through this package (`klein-cli --pkg-e2e`, distilled, 4 steps, seed 42, one run each, M5 Max, MLX pool
capped at 2 GB), v0.11.0 against this change:

| | MLX peak before | MLX peak after | phys_footprint peak before → after |
|---|---|---|---|
| bf16, 1024² | 24.40 GB | 16.24 GB | 25.3 → 18.3 GB |
| int8, 1024² | 21.11 GB | 12.66 GB | 22.9 → 14.9 GB |
| int4, 1024² | 19.36 GB | 10.90 GB | 20.6 → 13.1 GB |
| int4, encoder evicted, 1024² | 13.42 GB | 8.92 GB | 15.1 → 11.2 GB |
| int4, encoder evicted, 768² | 8.92 GB | 8.92 GB | 11.2 → 11.2 GB |
| bf16, 512² | 16.63 GB | 16.24 GB | 18.5 → 18.5 GB |

- **Same picture, not the same bytes:** 61.7–62.8 dB against v0.11.0's output; about 10–12 % of pixels move, by at
  most 5 of 255 levels.
- **Same time:** the decode alone takes 2.2–2.5 s untiled and 1.1–2.2 s in bands at 1024², 0.42–0.57 s against
  0.55–0.56 s at 512², and 0.02 s more at 256².
- **What sets the peak now:** with the encoder resident, the denoise loop. With the encoder evicted, the bf16
  encoder while it encodes (8.92 GB at any size), so a quantized encoder is the next lever for the 16 GB tier.
- **Declared footprints are unchanged** and now over-reserve at 1024² (11.2 GB activation declared; about 2.6 GB
  measured here for bf16). They were set from in-app runs and should be re-declared from one (AB-T-0290).

## Edit references keep their aspect (v0.10.2, AB-A-0213)

Up to 0.10.1, every edit reference (the request's images and IC conditioning images alike) was drawn into an
output-width × output-width square, so a non-square reference was stretched. `KleinReference` now follows diffusers'
`Flux2KleinPipeline`:
- the aspect is kept;
- a reference over the area cap is scaled down to it;
- each side is floored to a multiple of 16, with a center crop of under 16 px;
- a reference is never upscaled.

The cap is **min(1024², output area)**. For outputs of 1024² or larger that is exactly diffusers' rule. For smaller
outputs, a reference never carries more tokens than the output does, so the measured footprints still hold.
- **Measured** (M5 Max, bf16, 1024² output, a 1024×683 reference, 4 steps, same seed): 0.10.1 vs 0.10.2 peaks are
  24,418 vs 24,413 MB MLX and 41.79 vs 41.74 GB process footprint (uncapped cache).
- **The difference you can see:** 0.10.1 draws the subject stretched tall and thin; 0.10.2 keeps its proportions.
- **The CLI:** `klein-cli --edit` uses the same rule.

**The legacy v1 bundle** (`Resources/klein-lora-registry.json`, `MLXKlein.LoRARegistry`, `LoRACache`) still feeds the
CLI's `lora-list`, `lora-info` and `--lora-id` flags, with unpinned downloads. `ADDING-CIVITAI-LORAS.md` covers adding
an adapter.

## Weights

bf16 snapshot on Hugging Face (Apache-2.0), materialized by the engine (`WeightSourcing`) or via
`snapshotPath`: [`mlx-community/FLUX.2-klein-4B-bf16`](https://huggingface.co/mlx-community/FLUX.2-klein-4B-bf16).
int8/int4 are produced at load from the bf16 snapshot. Upstream:
[black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B).

## Architecture notes

- MMDiT: `x_embedder` (128→3072) + `context_embedder` (7680→3072); 5 double-stream blocks (joint
  img+txt attention, per-stream modulation), concat [txt, img], 20 single-stream blocks
  (parallel self-attention + fused SwiGLU MLP), `AdaLayerNormContinuous` + `proj_out`. 4-axis RoPE
  (θ 2000, axes 32×4). Distilled 4-step, guidance 1.0, no negative prompt.
- Text encoder = Qwen3-4B, **hidden layers 9/18/27 concatenated → 7680** context; klein feeds the
  DiT the **full 512-token** padded sequence (causal+padding mask), via the Qwen chat template
  (`add_generation_prompt=True, enable_thinking=False`).
- VAE = FLUX.2 (32 latent channels), reused verbatim from `flux2-vae-mlx-swift`.
- **DiT dtype**: weights load bf16, but the *activation stream runs fp32* — temb is computed in
  fp32 and MLX promotion (fp32 modulation × bf16 stream → fp32) carries it through every block.
  This incidentally shields Klein from the mlx-swift ≤0.31.6 NAX split-K GEMM bug (mlx#3797,
  half-precision-only dispatch window; the single-block `to_out` K=12288/N=3072 would otherwise
  corrupt at M = 512 txt + img tokens ∈ [1366, 4096] — i.e. 512²–896² t2i and 512²-class edits;
  probe-confirmed NaN at the exact production shape in bf16). `Flux2ParallelSelfAttention.
  outProjected` carries a dormant ≤896-row chunk (exact; `KLEIN_NO_CHUNK` env disables) that arms
  automatically if the activation flow ever becomes true bf16. On any mlx-swift bump run
  `klein-cli --nax-probe`; on PASS delete the chunk together with the mage-flow-swift /
  qwen3vl-mlx-swift / boogu-image-swift ones (mlx#3810 ships the fix upstream;
  `boogu-image-swift/tools/check_mlx_swift_3810.sh` checks a tag's vendored mlx).
- **Multi-reference editing** (klein's differentiator, **v0.2.0**): compose a new image from one or
  more reference images via reference-token conditioning — the reference subject is VAE-encoded,
  patchified, bn-normalized, packed, and appended to the target sequence with a 4D-RoPE t-offset
  (`t = 10·(i+1)` per reference). Surface `flux2-klein-4b-edit` (`IEditRequest`, images in prompt
  order). E.g. reference = a fox photo, prompt = "the fox on a tropical beach" → same fox, new
  scene. (Plain path; the KV-cache speed path is a later optimization.)

## Gates

- Offline: `swift test --filter MLXKleinTests` (MAT + conformance, no weights).
- Parity/GPU: `KLEIN_PARITY=1 swift test` (fp32/CPU) and `swift run -c release klein-cli`
  (`--quant 4`, `--pkg-e2e`, …).
- Kernel probe (weights-free, run after any mlx-swift bump):
  `swift run -c release klein-cli --nax-probe` — exercises the single-block `to_out` GEMM shape
  in bf16 across the corruption band; PASS means the mlx#3797 row-chunk is removable.
- Conv-window probe (weights-free, run after any mlx-swift bump): `swift test --filter
  WinogradProbeTests`. PASS with raw conv2d reported exact means the route below is removable.
  Encoder GPU lane: `KLEIN_PARITY=1 KLEIN_SNAPSHOT=<snapshot> swift test -c release -Xswiftc
  -enable-testing --filter E1bVAEEncoderGPULaneTests`.

## GPU numerics: the edit-path VAE encoder (2026-09-24)

mlx's Metal `conv2d` takes a Winograd F(6×6,3×3) path when the conv is 3×3, stride 1, dilation 1,
groups 1, C % 32 == 0, O % 32 == 0, C + O ≥ 256 and N·H·W ≥ 4096. On M5 that path loses about
6.4e-3 relL2 per conv in fp32, because its inner GEMM runs TF32.

`KleinVAEEncoder` has 21 such convs per 1024² reference image, and E1 ran on the CPU lane only.
Every stride-1 3×3 conv is now a `WinogradFreeConv2d`, which routes only the in-window shapes
through `conv3d` with kT = 1. That is the default (`encoder.convRoute`, type `KleinVAEConvRoute`).

Measurements: encode mean against the CPU lane, on a real DIV2K photo at 512² and 1024².

| | Raw conv2d (Winograd) | conv3d route |
|---|---|---|
| relL2, 512² and 1024² | 3.4e-3 · max 5–7e-2 | 1.2e-3 · max 1.6–1.8e-2 |
| GPU fp32 encode time, 1024² / 512² | 322 ms / 81 ms | +150 ms / +33 ms |

- The route's ~1.2e-3 residual is TF32 in the mid-block attention matmuls, not a conv. With
  `MLX_ENABLE_TF32=0` the route is 4.6e-6 at 512².
- At 1024² with TF32 off it is 2.2e-5. The CPU lane's own GroupNorm error grows with group size
  (8e-5 at 1024² against float64), which is why the gate asserts at 512².
- The decoder comes from `flux2-vae-mlx-swift`, which carries the same route in its own release.
- For A/B validation, set `KLEIN_VAE_CONV_ROUTE=winograd` or `encoder.convRoute = .winograd`.

License: port code MIT; model weights Apache-2.0 (Black Forest Labs).
