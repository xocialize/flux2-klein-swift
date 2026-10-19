# Adding a CivitAI LoRA to the Klein registry

Checklist for mirroring/referencing a CivitAI LoRA in `klein-lora-registry.json`.
The **license check is the load-bearing step** — do it first, it's the one that trips.

## 1. Verify the license (DO THIS FIRST)

CivitAI licenses are **per-version**, and the machine-readable flags LIE. Check, in order:

1. **The version page License block — authoritative.**
   Open `https://civitai.com/models/<modelId>?modelVersionId=<versionId>` and read the
   **"License:"** field in the right sidebar. It is set **per version** — different versions
   of the same model routinely differ (e.g. one author: Klein-4B = **Apache-2.0**,
   Flux.1-dev weight = **FLUX.1 [dev] Non-Commercial**, Klein-9B = different again).
   The in-app browser (`preview_start` → `get_page_text`) reads this reliably; a plain
   WebFetch summary can grab the wrong version's license — don't trust it alone.

2. **DO NOT trust the CivitAI v1 API `allow*` flags.**
   `allowDerivatives` / `allowNoCredit` / `allowCommercialUse` are **model-level and coarse**
   and do **not** track the per-version license. They said `allowDerivatives=False` on a
   model whose Klein-4B version is Apache-2.0. Ignore them for the license decision.

3. **Strongest signal — the author's own HF repo, if one exists.**
   Authors often link a HuggingFace mirror in the version description. Its declared license
   is an *affirmative publish act* and the best evidence:
   ```
   curl -s https://huggingface.co/api/models/<owner>/<repo> \
     | python3 -c "import sys,json; d=json.load(sys.stdin); print((d.get('cardData') or {}).get('license'), d.get('tags'))"
   ```
   If the author already hosts it on HF under a clear license, that's the license authority.

   **Sourcing rule:** if the LoRA will be referenced by *shipping code*, **mirror it to a
   public `xocialize/...` repo** so availability is under our control — do NOT reference a
   third-party repo (or rely on CivitAI) for anything we ship. Reference the author's repo
   directly only for local/experimental use. Mirror = verbatim file (verify SHA-256 pre- and
   post-upload) + `README.md`/`LICENSE`/`NOTICE` attributing the original author + origin URL
   + author's HF repo. Requires a permissive license (Apache/MIT/etc.); never re-host a
   Non-Commercial or contested-license weight.

4. **The safetensors `__metadata__` almost never carries a license** — only `format`.
   Don't expect the license to be in the file.

5. **Base-model floor.** A LoRA can't out-license its base. FLUX.2-klein-4B is Apache-2.0
   (fine). A LoRA trained on FLUX.1-dev is **Non-Commercial** (BFL license) no matter what
   the author tags it. Confirm the version's `baseModel`.

**Rule:** never *publicly re-host* on a contested/unclear license. If the license is clean
(Apache/permissive, ideally corroborated by the author's HF repo) you may mirror to
`xocialize/...` with attribution; otherwise reference the author's repo, keep it local, or
stop and clarify.

## 2. Get authoritative metadata (fast)

```
# per-version: baseModel, trainedWords, files (name/size/hash), downloadUrl
curl -s https://civitai.com/api/v1/model-versions/<versionId> | python3 -m json.tool
# all versions of a model (ids, names, baseModels)
curl -s https://civitai.com/api/v1/models/<modelId> | python3 -m json.tool
```
Record: exact `weightFile` name, SHA-256 (for provenance), `baseModel`, trained words
(→ `trigger` / `promptTemplate`), suggested strength (→ `defaultStrength`).

## 3. Confirm it loads in our applicator

CivitAI downloads are auth-gated (HTTP 401) — the file must be fetched with the user's
CivitAI account/token, or downloaded in-browser. Then, structural smoke test (no GPU):
```
swift run klein-cli lora-smoke <file.safetensors>
```
Expect `double=60 single=40 total=100` for a full Klein LoRA. `KleinLoRA` handles both
`bfl-fused` and `diffusers-split` dialects; inspect the header to record rank/tensors:
```
python3 - <<'PY'
import struct, json
with open("<file>", "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n))
print("tensors:", len([k for k in h if k!='__metadata__']))
# rank = lora_A/down dim0; dialect = key prefix (transformer.* = diffusers-split)
PY
```

## 4. Add the manifest entry

Append to `klein-lora-registry.json` (schema in `LoRARegistry.swift`). Key fields:
`kind` (`plain` = weights+prompt; `ic` = ordered role-typed `conditioning` slots),
`surface` (`textToImage`/`imageEdit`), `license`, `source` (origin URL for provenance),
`gatedDownload`, `verification` (rank/tensors/dialect/gate). Reference the author's HF
`repo`+`weightFile` if referencing; use `xocialize/...` if mirroring.

## 5. Validate end-to-end

```
swift run klein-cli lora-info <id>            # entry renders, license/source correct
swift run klein-cli --lora-id <id> --edit img.png --quant 4   # (plain edit LoRA)
```
Confirm the output PNG is coherent (not black/garbage → truncated weights). Add a decode
test in `Tests/MLXKleinTests/LoRARegistryTests.swift`.
