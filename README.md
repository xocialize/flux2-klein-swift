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

License: port code MIT; model weights Apache-2.0 (Black Forest Labs).
