# flux2-klein-swift — Porting Spec

**Goal:** Swift/MLX port of **Black Forest Labs FLUX.2-klein-4B** (Apache-2.0) serving MLXEngine
**`textToImage`** — the compact rectified-flow MMDiT with unified multi-reference **editing** (the
one capability nothing else in our fleet has). Decision record: [[zimage-flux2klein-t2i-eval]];
second image port after Z-Image ([[z-image-port-active]], now SHIPPED).

**Scope: 4B ONLY.** All 9B klein variants are FLUX Non-Commercial + filter-or-review — skip. The
4B (distilled + base) is plain Apache-2.0.

## Reuse thesis (2 of 3 components already in-house)

- **VAE = `flux2-vae-mlx-swift`** (PROD, parity-locked, shared by Lens + ERNIE) — the FLUX.2 VAE:
  32 latent channels → 128 after 2× pixel-unshuffle. **Net dependency, not a re-port.**
- **Text encoder = Qwen3-4B** (same backbone as Z-Image) but **klein taps 3 hidden layers
  (9/18/27) concatenated → 7680-dim** context (vs Z-Image's single `hidden[-2]`). Extend the
  `Adapted/Qwen3Encoder` from z-image-swift to capture multiple layers.
- **DiT = net-new**, but it's a textbook FLUX MMDiT, **~685 LOC of Python-MLX in mflux** →
  near-mechanical Swift-MLX translation.

## Oracle & references

- **Numeric oracle + reference = mflux** (`../../mflux/src/mflux/models/flux2/`) — first-class
  klein-4B/9B/9b-kv support, runs on-box (Python-MLX). Goldens come from here; no PyTorch rung
  (mflux IS the MLX rung — mlx-swift-integration doctrine).
- **Swift-idiom cross-check = VincentGourbin/flux-2-swift-mlx** (MIT, native Swift MLX) — consult
  for MLX-Swift idioms; do NOT fork (we need a conformant MLXEngine ModelPackage).
- **Weights:** `black-forest-labs/FLUX.2-klein-4B` (~15 GB; single repo, subdirs
  `transformer/ text_encoder/ vae/ tokenizer/`) → `../../weights/FLUX.2-klein-4B` (downloading).

## Architecture (from mflux config + sources — verified 2026-07-06)

**`ModelConfig["flux2-klein-4b"]`:** `num_layers=5` (double-stream), `num_single_layers=20`
(single-stream), `num_attention_heads=24`, `attention_head_dim=128` (inner_dim=3072),
`joint_attention_dim=7680`, `in_channels=128`, `patch_size=1`, `max_sequence_length=512`,
`supports_guidance=True`, `requires_sigma_shift=True`, `axes_dims_rope=(32,32,32,32)`,
`rope_theta=2000`. Text encoder overrides: hidden 2560, intermediate 9728.

**Forward (`Flux2Transformer.__call__`, mflux transformer.py):**
1. Timestep + optional guidance → `Flux2TimestepGuidanceEmbeddings` → `temb`. Timestep
   auto-scaled ×1000 when `max(t)≤1`; same for guidance. (Watch: same time-convention care as
   Z-Image — port the scaling verbatim.)
2. `x_embedder` (128→3072, no bias) on image latent; `context_embedder` (7680→3072) on text.
3. 4-axis RoPE: `pos_embed(img_ids)` + `pos_embed(txt_ids)`, then **concat [txt, img]** rotary.
   Real (cos,sin) form already (mflux `Flux2PosEmbed`) — port directly (θ2000, 4×32 dims).
4. **5 double-stream blocks** (`Flux2TransformerBlock`): separate `LayerNorm(eps 1e-6,
   affine=false)` for img+txt, modulation `(1+scale)·norm + shift`, joint `Flux2Attention`
   (img q/k/v + `added_kv_proj` for txt, RoPE on both), gated residual (`gate_msa`), then
   per-stream FF (mlp_ratio 3.0) gated by `gate_mlp`. Two modulation param-sets (img/txt) from
   `double_stream_modulation_{img,txt}`.
5. **concat [encoder_hidden_states, hidden_states]** → **20 single-stream blocks**
   (`Flux2SingleTransformerBlock`, single modulation set).
6. Strip text tokens, `AdaLayerNormContinuous(temb)` norm_out, `proj_out` (3072→patch·out_ch).

**Modulation (`Flux2Modulation`):** SiLU → Linear(dim→dim·3·sets, no bias) → split to
(shift,scale,gate) per set. Shared machinery with mflux flux1 (`AdaLayerNormContinuous` reused
from `flux.model.flux_transformer`).

**Editing (multi-ref):** reference-image tokens concatenated into the sequence with 4D-RoPE
offsets; `Flux2KVCache` (extract/inject modes) caches reference K/V once (the 9b-kv speedup).
`_blend_trailing_ref_mod_params` blends ref-token modulation. **P6** — port the base T2I path
first (P1-P5), add the edit/KV path after.

## Sampler / pipeline

Distilled: **4 steps**, guidance fixed **1.0**, **no negative prompt** (mflux flux2 README),
Euler on a power schedule `linspace(1,0,steps+1)^(1/rho)`, rho=5; `requires_sigma_shift`. Base
variant (undistilled): ~50-step, guidance >1. Port the mflux pipeline verbatim (dims /16).

## Component map (Swift)

| Component | Source | Swift action |
|---|---|---|
| `Flux2Transformer` | mflux transformer.py (183) | Port isomorphic — MMDiT double+single, keep names. |
| `Flux2PosEmbed` (4-axis RoPE) | pos_embed.py (31) | Port real (cos,sin), θ2000, axes (32,32,32,32); concat txt+img. |
| `Flux2Modulation` / blocks / attention | modulation, transformer_block (62), single (37), attention (98), feed_forward (27) | Port isomorphic; `MLXFast.scaledDotProductAttention`, LayerNorm(affine:false). |
| `AdaLayerNormContinuous` + timestep/guidance embed | mflux flux1 + flux2 | Port. |
| FLUX.2 VAE | **`flux2-vae-mlx-swift` (PROD)** | **Net dep — reuse.** Verify 32-ch/pixel-unshuffle constants. |
| Qwen3 3-layer-tap encoder | mflux flux2 qwen3_text_encoder.py + z-image `Adapted/Qwen3Encoder` | Extend the z-image Qwen3 to capture layers 9/18/27, concat → 7680. |
| `Flux2KVCache` (edit) | flux2_kv_cache.py (118) | P6 — extract/inject for multi-ref editing. |
| Pipeline | mflux flux2 variants/txt2img + edit | Port: 4-step Euler power-schedule, guidance 1.0, dims/16. |
| Engine wrapper | — | `MLXKlein`: `Klein4BT2IPackage` (+ edit surface P6); Apache, split QuantFootprint, WeightSourcing/MAT. |

## Phases & gates (CPU stream for parity; mflux is the oracle)

- **P0** workspace: download, mflux-oracle goldens (`dump_goldens.py`), SPM scaffold.
- **P1** pure math: RoPE (4-axis) + modulation + timestep/guidance embed vs mflux, ≤1e-5.
- **P2** DiT: strict-load bf16; staged parity vs mflux goldens (post-embed, per double block,
  post-concat, per single block, norm_out) cosine ≥0.9999 fp32/CPU. Sub-op goldens.
- **P3** VAE: wire flux2-vae; decode golden PSNR gate + noise-path smoke.
- **P4** encoder: Qwen3 3-layer tap → 7680; token ids exact + features cosine ≥0.999.
- **P5** e2e T2I: 4-step/1024² vs mflux render (injected latents), resolution sweep. GPU lane.
- **P6** multi-ref edit: token-concat + 4D-RoPE offsets (+ optional KV-cache); edit golden.
- **P7** quant int8/int4 (DiT 7.75 GB bf16 → 2.1 GB int4; per-pass cosine + eyeball). GPU gate.
- **P8** wrap (`Klein4BT2IPackage` + edit) C0-C13 + MAT; publish 4B to mlx-community + xocialize.

## Watch-list

- **KV-cache edit path is the differentiator** — but stage it after base T2I works (P6). Don't
  let the edit machinery block the T2I gates.
- Timestep/guidance auto-scale (×1000 when ≤1) — port verbatim; a Z-Image-style silent time bug.
- `joint_attention_dim=7680` = 3×2560 → the encoder MUST tap 3 layers; a single-layer tap
  silently mis-shapes context_embedder input. Gate the encoder output dim explicitly.
- 4B is Apache; **9B is NC — never port/publish it** under this package.
- mflux may quantize/repack; gate against mflux run at bf16 (its own reference dtype).

## Status
- [x] Architecture read (mflux flux2, 685 LOC); config confirmed; spec written
- [x] **P0**: weights (22 GB), diffusers oracle venv, goldens (DiT/encoder/VAE), SPM scaffold
- [x] **P1+P2 GREEN** — full MMDiT ported (Transformer.swift, isomorphic to mflux) + Weights.swift
      (2 remaps only: strip `.timestep_embedder`, `to_out.0`→`to_out`; all bias-free Linears).
      Staged parity vs diffusers fp32/CPU goldens: temb/x_embed/context_embed, all double blocks
      (0,4), all single blocks (0,10,19), norm_out, out_final — every stage cos ≥0.9999997,
      x_embed+context_embed bit-exact. First-run green.
- [x] **P3 GREEN** — flux2-vae-mlx-swift reused verbatim (net dep): klein VAE decode **130 dB**.
- [x] **P4 GREEN** — Qwen3 3-layer-tap encoder (Adapted/Qwen3Encoder, copy-adapt of z-image's,
      extended to capture layers 9/18/27 → 7680). Real-token features cos **0.9999999**.
      NOTE: right-padded; compare only valid tokens (padded positions diverge — causal-only vs
      HF padding-mask — but pipeline packs real tokens, so it's moot). Loader keeps layers 0..26.
- [x] **P5 GREEN (functional)** — pipeline renders a coherent, prompt-faithful lighthouse
      (1024²/4-step, 6.2 s, 24 GB). Isolation gates: DiT cos ≥0.9999995 @64×64 (P2), decode
      105 dB, encoder tap cos 0.9999999. THE BUG (cos 0.53 → coherent): klein feeds the DiT the
      **full 512 padded text tokens** (verified by capturing diffusers' transformer inputs:
      encoder_hidden_states [1,512,7680], txt_ids [1,512,4] pos 0..511, ts0=1.0), NOT the 24
      trimmed real tokens. Fix: KleinTextEncoder pads to 512 + runs a **causal+padding-key mask**
      (padded positions must match diffusers; causal-only diverges — P4). Mask cast to qkv dtype
      (SDPA promotion). Scheduler PORTED exact (mu = linear calculateShift 1.15 @4096; base
      linspace(1, 0.001, N); exponential time-shift; step sigma_next-sigma) — reproduces diffusers
      sigmas exactly; empirical-mu was wrong; TEMP hardcode removed.
      **SECOND bug (lighthouse coherent, fox scrambled): missing CHAT TEMPLATE.** Diffusers encodes
      via apply_chat_template(add_generation_prompt=True, enable_thinking=False) =
      `<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`
      (enable_thinking=False STILL appends an empty think block). Raw tokens survived the lighthouse
      by luck; short prompts scrambled. Fixed via KleinTextEncoder.formatPrompt. DiT gets NO
      attention mask (all 512 text tokens attend); timestep passed /1000 → DiT auto-scales x1000.
      **Prompt-general confirmed: lighthouse + fox both render sharp/coherent.** ~6 s / 24 GB.
- [x] **P7 GREEN** — DiT quant (affine group 64, keepHiPrecision skips embedders/modulation/
      time-embed/norm_out). int4 DiT **2.35 GB** (bf16 7.75 GB); int4 fox render sharp/coherent
      (arguably better than bf16). GPU CLI lane (`--quant 4`).
- [x] **P8 WRAPPER GREEN** — MLXKlein: KleinConfiguration (WeightSourcing, single bf16 snapshot,
      quant-at-load) + Klein4BT2IPackage (surface flux2-klein-4b-t2i), KleinGenerator core.
      Apache/MIT, split QuantFootprint ×3, unload→clearCache, engine pin 0.21.0. Builds clean;
      5/5 MAT+conformance; wrapper e2e (`--pkg-e2e --quant 4`): real load 0.76s/run 6.04s/valid
      1024² PNG/19.2 GB peak/clean unload. klein-4B is all-bf16 → no conversion for publish.
- [x] **PUBLISHED v0.1.0**: xocialize/flux2-klein-swift@v0.1.0 · mlx-community/FLUX.2-klein-4B-bf16
      (24 files, 22.1 GB) + Collection · registry row merged. T2I tier COMPLETE.
- [ ] **P6 multi-ref EDIT (v0.2 — differentiator)**: Flux2KVCache + reference token-concat +
      4D-RoPE t-offsets (mflux variants/edit/flux2_klein_edit.py). 4B only.
