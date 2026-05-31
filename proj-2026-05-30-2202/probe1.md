# Step 1 Probe Report

Runtime dir: `/home/khkramer/src/nemotron-january-2026/proj-2026-05-24-from-scratch-runtime/runtime`

Environment: in-repo venv, `torch==2.8.0+cu128`, RTX 5090. All probes were read-only against existing runtime code/artifacts; new tooling only:

- `weights_identity_probe.py`
- `enc_first_parity_probe.py`
- `extraction_cache_api_probe.py`

Re-run commands from `runtime/`:

```bash
HF_HUB_OFFLINE=1 ./.venv/bin/python weights_identity_probe.py
HF_HUB_OFFLINE=1 ./.venv/bin/python enc_first_parity_probe.py --n 200
HF_HUB_OFFLINE=1 ./.venv/bin/python extraction_cache_api_probe.py
```

## (a) Weights Identity

Probe: `weights_identity_probe.py`

`enc_first.ts` named parameters + buffers vs `artifacts/finalize_shared_weights.pt`:

| Check | Result |
|---|---:|
| enc_first tensors found in shared set | 637 / 637 |
| shared tensors covered by enc_first | 637 / 637 |
| direct FQN matches | 0 |
| alias fallback matches (`e.*` -> `encoder.*`) | 637 |
| shape/dtype mismatches | 0 |
| missing FQNs | 0 |
| shared extras | 0 |
| nonzero per-tensor max-abs-diff count | 0 |
| overall max-abs-diff | 0.0 |
| tensor-equal | YES |
| byte-equal | YES |

All per-tensor max-abs-diffs are exactly `0.0`; all 637 tensors are byte-equal after the same alias fallback used by `constants_for_bucket`.

AOTI constant FQN coverage against the same shared set:

| Package | FQNs found | Shared covered | Direct | Alias | Missing |
|---|---:|---:|---:|---:|---:|
| `enc_first_aoti.pt2` | 637 / 637 | 637 / 637 | 637 | 0 | 0 |
| `enc_steady_aoti.pt2` inline | 637 / 637 | 637 / 637 | 0 | 637 | 0 |

Verdict: one shared constants map can serve `enc_first.ts` by alias, `enc_first_aoti.pt2` by direct `encoder.*` FQNs, `enc_steady_aoti.pt2` by alias, and finalize.

## (b) AOTI First-Chunk Parity

Probe: `enc_first_parity_probe.py`

Arm split:

- Baseline chunk 0: shipped `artifacts/enc_first.ts`.
- Candidate chunk 0: `artifacts/enc_first_aoti.pt2`, loaded with `torch._inductor.aoti_load_package`, bound to `finalize_shared_weights.pt` with `loader.load_constants(cmap, False, False, True)`.
- All non-first chunks in both arms: same eager `model.encoder.cache_aware_stream_step`.
- Decode wiring mirrors the shadow: `enc_out.transpose(1, 2).contiguous()` into `ref_greedy_range(..., 0, int(enc_len[0]), ...)`.

Corpus:

- 200 cached `pipecat-ai/stt-benchmark-data` utterances, dataset indices 0..199.
- 4-row b2 subset: first four dataset rows:
  `6744343e-ef8c-3154-50a2-5ad464c723bc`,
  `0a2570b5-668a-ea08-0546-1db4e518482f`,
  `e02b3cc7-0a8d-83d6-85a4-1f3240b18e47`,
  `739f778d-e8e2-be4f-3fe6-015adcac4682`.

Results:

| Check | Result |
|---|---:|
| AOTI first constants bound | 637 / 637 |
| AOTI direct FQN matches | 637 |
| AOTI alias FQN matches | 0 |
| b2 token divergences | 0 / 4 |
| b2 event divergences | 0 / 4 |
| corpus final token-sequence divergences | 0 / 200 |
| corpus event divergences | 2 / 200 |
| divergent final-token sample IDs | none |
| event-divergent sample IDs | `4484a2a0-a854-cd3f-74ef-75c7231b39b8`, `fdca9673-c132-9eb5-d1cf-5fc1cae2be30` |
| WER baseline | 2.918% |
| WER candidate | 2.918% |
| WER delta | +0.0000 pp |
| WER non-empty refs | 199 |

First-chunk TS-vs-AOTI output max-abs over the 200 rows:

| Output | Max abs |
|---|---:|
| `enc_out` | `5.826354e-05` |
| `enc_len` | `0.0` |
| `cache_ch` | `2.203882e-04` |
| `cache_t` | `2.345467e-02` |
| `cache_ch_len` | `0.0` |

Caveat on "event": the probe records a strict chunk-level interim event proxy: each time decode emits new tokens, it stores `(chunk_idx, cumulative_token_prefix)`. The two event divergences did not change final token sequences or WER. For dataset index 24, the same token prefix was emitted one chunk earlier in the AOTI arm, then the event stream realigned. Dataset index 115 appeared as an event divergence in the 200-row run but did not reproduce in a single-utterance rerun; final tokens remained identical.

### Noise-floor / causality evidence (paired review, 2026-05-31)

The Step-1 paired adversarial review (a second independent Codex review + orchestrator review) added three empirical checks that show these interim-event divergences are **greedy-decode boundary-tipping at the system's determinism noise floor, not a robust enc_first regression**:

1. **Reproducibility (orchestrator, 2× identical 130-row runs):** both runs flagged the exact same two sample IDs (`4484a2a0…`=idx24, `fdca9673…`=idx115), 0 token divergences, WER `2.596%` vs `2.596%` (`+0.0000 pp`) both runs. But idx115 reproduces only **in a 130-row run context** — Codex's earlier `--n 1 --start 115` single-utterance rerun did **not** reproduce it. The divergence is therefore **run-context dependent** (kernel autotune/warmup state), not a deterministic property of (AOTI enc_first, utterance 115).
2. **Cross-harness disjointness:** the clean-isolation HF harness (`enc_first_parity_probe.py`, eager steady both arms) flags `{idx24, idx115}`; the production-event bundle harness (`probe_step1.py`, `session_bundle.ts` oracle) flags a **completely disjoint** set `{e018a533, 5dd8def8, 0a4b7986, 81717bc0}`. Each harness's divergent samples are **event-clean and token-clean in the other** (verified in `step1-parity-corpus-rows.csv`: `4484a2a0`/`fdca9673` → `event_match=True, token_match=True`). A causal enc_first regression would flag overlapping utterances across harnesses; zero overlap ⇒ context artifact. (The bundle harness also uses **inline AOTI steady** for non-first chunks, so its 4/1000 conflate enc_first-AOTI with enc_steady-AOTI and are not clean chunk-0 evidence.)
3. **Error non-correlation:** the divergent rows have **near-minimal** first-chunk `enc_out` error (1.4e-6–5.5e-6, *below* the corpus max 6.6e-5), and `enc_len` (frame counts) are byte-equal in every divergent case (`step1-divergent-first-diffs.json`). The flips are greedy-argmax tipping at a token-emission boundary, not a function of AOTI error magnitude.
4. **Metric validity:** the per-chunk `(chunk_idx, prefix)` proxy is **stricter than the production gate**. The real Step-5 event gate is `step6_server_oracle.py`, which compares emitted WebSocket transcript fields (`kind`, `text`, `collector_text`, `is_final`) with **no chunk-index/timing field** — so "same prefix one chunk earlier" is a proxy failure that the server's interim debounce/VAD path need not surface at all.

**Revised verdict (b):** final tokens and WER are **GO** (0 divergence on 200 HF + 1000 bundle utterances). The only divergences are rare, harness-context-dependent, sub-noise-floor interim-timing artifacts that converge to identical finals — they do **not** establish a causal enc_first regression and are **not** the production event gate. The enc_first→shared-constants unify is therefore **GO to BUILD as an opt-in adapter (TS default)**; the default-on flip is deferred to the authoritative Step-5 gate (a NEW full-1000 enc_first shadow + `step6_server_oracle.py` event-for-event), per the plan's explicit opt-in fallback.

## (c) Extraction-Cache API

Probe: `extraction_cache_api_probe.py`

Header inspected:

`runtime/.venv/lib/python3.12/site-packages/torch/include/torch/csrc/inductor/aoti_package/model_package_loader.h`

Relevant header surface:

```text
11:   AOTIModelPackageLoader(
17:   ~AOTIModelPackageLoader();
32:   void load_constants(
37:   std::vector<std::string> get_constant_fqns();
46:   std::string temp_dir_;
```

The only public constructor is package-path based; `temp_dir_` is private. No public constructor accepts an extracted directory or caller-managed temp directory.

Matching PyTorch 2.8.0 source:

`https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/torch/csrc/inductor/aoti_package/model_package_loader.cpp`

Relevant source lines:

```text
47: std::string create_temp_dir() {
51:   std::string temp_dir = "/tmp/XXXXXX";
52:   if (mkdtemp(temp_dir.data()) == nullptr) {
408:   temp_dir_ = create_temp_dir();
```

The source contains no `TMPDIR` mention. This confirms stock Torch 2.8.0 hard-codes `/tmp/XXXXXX` for AOTI extraction and does not expose a public pre-extracted-dir API.

Post-unify residual sizing:

| Artifact group | Size |
|---|---:|
| `enc_first_aoti.pt2` stripped | 3.8 MiB |
| stripped steady buckets, 3 files | 12.3 MiB |
| stripped finalize buckets, 32 files | 124.2 MiB |
| total stripped AOTI extraction payload | 140.3 MiB |
| `finalize_shared_weights.ts` | 2362.9 MiB |
| `finalize_shared_weights.pt` | 2363.0 MiB |
| current pre-unify inline `enc_steady_aoti.pt2` | 2366.5 MiB |

The post-unify large 2.48 GB encoder blob is `torch::jit::load` shared weights, not AOTI extraction. The residual AOTI extraction payload is small enough that a speed cache is not worth a custom loader or Torch patch. The remaining value is `/tmp` leak hygiene.

## DECISIONS

Steps 4-5, enc_first unify:

**GO to BUILD (opt-in adapter, TS default); default-on flip DEFERRED to the Step-5 server-oracle gate.** Revised after paired adversarial review (see §(b) noise-floor evidence). Reasoning:
- Weight identity is clean (637/637 byte-equal) and final tokens + WER are identical on 200 (HF) and 1000 (bundle) utterances — **no transcription regression**.
- The only divergences are rare interim-event differences that are (i) **sub-noise-floor** (run-context dependent; idx115 reproduces only in-context), (ii) **disjoint across the two harnesses** (each harness's diffs are clean in the other ⇒ not causal to enc_first), and (iii) measured by a **per-chunk proxy stricter than the production oracle** (`step6_server_oracle.py` has no chunk-timing field).
- Therefore: **build** the first-encoder adapter + the stripped `enc_first_aoti.pt2` artifact and wire it behind `NEMOTRON_WS_ENC_FIRST_TS` with **TS as the default** (Steps 4-5). Do **not** reduce item 1 to "Step 2 only."
- The **default-on flip is gated at Step 5** by the authoritative gate: a NEW full-1000 enc_first shadow (AOTI first chunk) with 0 final-token divergence **AND** `step6_server_oracle.py` 8/8 event-for-event (incl. first interim) PASS. Given the noise-floor evidence, default-on is *plausible*; if the server oracle shows any real (non-noise) interim regression, ship AOTI enc_first opt-in only (TS stays default) — exactly the plan's Step-5 fallback. A Step-4 lever to try if the flip is marginal: re-export `enc_first_aoti.pt2` with tighter numerics (e.g. `cudnn.allow_tf32=False`) to shrink the ~0.004 `cache_t` diff that feeds chunk 1.

Steps 7-8, extraction cache:

**NO-GO for a speed extraction cache. GO for minimal `/tmp` hygiene.** Torch 2.8.0 exposes no pre-extracted-dir constructor, and post-unify AOTI extraction is only about 140 MiB. Implement startup cleanup of stale owned `/tmp/*/data/aotinductor*` trees if Step 8 proceeds; do not build a custom extraction cache for speed unless later measurements contradict this sizing.
