# Padded-T Finalize Probe Findings

- Continuation verdict: **GO**
- Scope: realistic continuous VAD-stop/debounce finalize, then continued steady chunks
- Finalize variants: exact-T vs zero-padded-to-60 with real processed_signal_length
- Model: `nvidia/nemotron-speech-streaming-en-0.6b`
- Device: `NVIDIA GeForce RTX 5090`
- NeMo: `2.8.0rc0` from `/home/khkramer/src/nemotron-nano-omni/NeMo/nemo/__init__.py`
- Geometry: shift_frames=16, pre_encode_cache_size=9, drop_extra=2, pre_chunks=20, post_chunks=8
- Runtime: 7363.9 ms

## Continuation Verdict

- First finalize tokens/text exact: **yes**
- Post-finalize continuation tokens/text exact: **yes**
- Full captured stream tokens/text exact: **yes**
- Session retained own steady cache after first finalize: **yes**
- Session cold-reset after first finalize: **NO**
- Session adopted fork cache_last_channel_len: **NO**
- Fork cache_last_channel_len diverged as expected: **yes**

## Evidence

The finalize path exercised here is the scheduler batched debounce path: `vad_stop` arms a pending finalize, the probe injects the matching `debounce_expired` event, `_scheduler_drain_once()` batches it through `_scheduler_process_finalize_event_batch()`, and the server resumes via `_continuous_finish_speculative_finalize_locked()`.

| run | final# | real_T | fed_T | session cache before | fork cache out | session cache after | after state | retained own | adopted fork |
|---|---:|---:|---:|---|---|---|---|---|---|
| exact | 0 | 43 | 43 | `[41]` | `[46]` | `[41]` | STREAMING | yes | NO |
| exact | 1 | 43 | 43 | `[57]` | `[62]` | `[57]` | STREAMING | yes | NO |
| padded | 0 | 43 | 60 | `[41]` | `[48]` | `[41]` | STREAMING | yes | NO |
| padded | 1 | 43 | 60 | `[57]` | `[64]` | `[57]` | STREAMING | yes | NO |

- Post-first-final events compared: `['chunk:0020', 'chunk:0021', 'chunk:0022', 'chunk:0023', 'chunk:0024', 'chunk:0025', 'chunk:0026', 'chunk:0027', 'final:0001']`
- First divergence: none.

## Norm / Mode Caveat

- model_eval=True, encoder_eval=True, ConformerConvolution.norm_type=['layer_norm']
- Caveat: the local model is not batch_norm; it reports the norm type above. The probe continued because instance/group norm is absent and layer_norm is per-frame over channels.

## First Probe Context

The initial exact-T vs padded-T finalize probe was byte-exact on tokens/text/encoded_len/real encoder frames for T=42..60. Its only divergence was `cache_last_channel_len` on the disposable finalize fork (`[46]` or `[47]` exact vs `[48]` padded for T<58).

## Blockers

- None.

## Next Step

Proceed to Step 2b: the fork-only cache length divergence is dead for the continued session, so a single B=1 padded T_max finalize bucket is safe to implement behind the planned default-off, fail-closed guard.

## Captured Events

| run | event | kind | cache_len | tokens | text |
|---|---|---|---|---|---|
| exact | chunk:0000 | chunk | `[3]` | `[]` | '' |
| exact | chunk:0001 | chunk | `[5]` | `[]` | '' |
| exact | chunk:0002 | chunk | `[7]` | `[]` | '' |
| exact | chunk:0003 | chunk | `[9]` | `[]` | '' |
| exact | chunk:0004 | chunk | `[11]` | `[]` | '' |
| exact | chunk:0005 | chunk | `[13]` | `[]` | '' |
| exact | chunk:0006 | chunk | `[15]` | `[]` | '' |
| exact | chunk:0007 | chunk | `[17]` | `[]` | '' |
| exact | chunk:0008 | chunk | `[19]` | `[]` | '' |
| exact | chunk:0009 | chunk | `[21]` | `[]` | '' |
| exact | chunk:0010 | chunk | `[23]` | `[]` | '' |
| exact | chunk:0011 | chunk | `[25]` | `[]` | '' |
| exact | chunk:0012 | chunk | `[27]` | `[34, 966, 955]` | "I'm" |
| exact | chunk:0013 | chunk | `[29]` | `[34, 966, 955, 455, 20]` | "I'm trying" |
| exact | chunk:0014 | chunk | `[31]` | `[34, 966, 955, 455, 20, 22]` | "I'm trying to" |
| exact | chunk:0015 | chunk | `[33]` | `[34, 966, 955, 455, 20, 22, 865, 149]` | "I'm trying to create" |
| exact | chunk:0016 | chunk | `[35]` | `[34, 966, 955, 455, 20, 22, 865, 149]` | "I'm trying to create" |
| exact | chunk:0017 | chunk | `[37]` | `[34, 966, 955, 455, 20, 22, 865, 149, 3]` | "I'm trying to create a" |
| exact | chunk:0018 | chunk | `[39]` | `[34, 966, 955, 455, 20, 22, 865, 149, 3]` | "I'm trying to create a" |
| exact | chunk:0019 | chunk | `[41]` | `[34, 966, 955, 455, 20, 22, 865, 149, 3, 19, 113, 42]` | "I'm trying to create a custom" |
| exact | final:0000 | final | `[41]` | `[34, 966, 955, 455, 20, 22, 865, 149, 3, 19, 113, 42, 687, 304, 962]` | "I'm trying to create a custom value." |
| exact | chunk:0020 | chunk | `[43]` | `[34, 966, 955, 455, 20, 22, 865, 149, 3, 19, 113, 42]` | "I'm trying to create a custom" |
| exact | chunk:0021 | chunk | `[45]` | `[34, 966, 955, 455, 20, 22, 865, 149, 3, 19, 113, 42]` | "I'm trying to create a custom" |
| exact | chunk:0022 | chunk | `[47]` | `[34, 966, 955, 455, 20, 22, 865, 149, 3, 19, 113, 42, 687, 613]` | "I'm trying to create a custom values" |
| exact | chunk:0023 | chunk | `[49]` | `[34, 966, 955, 455, 20, 22, 865, 149, 3, 19, 113, 42, 687, 613]` | "I'm trying to create a custom values" |
| exact | chunk:0024 | chunk | `[51]` | `[34, 966, 955, 455, 20, 22, 865, 149, 3, 19, 113, 42, 687, 613, 75]` | "I'm trying to create a custom values for" |
| exact | chunk:0025 | chunk | `[53]` | `[34, 966, 955, 455, 20, 22, 865, 149, 3, 19, 113, 42, 687, 613, 75, 184, 189]` | "I'm trying to create a custom values for my up" |
| exact | chunk:0026 | chunk | `[55]` | `[34, 966, 955, 455, 20, 22, 865, 149, 3, 19, 113, 42, 687, 613, 75, 184, 189]` | "I'm trying to create a custom values for my up" |
| exact | chunk:0027 | chunk | `[57]` | `[34, 966, 955, 455, 20, 22, 865, 149, 3, 19, 113, 42, 687, 613, 75, 184, 189, 954]...(+1)` | "I'm trying to create a custom values for my upcom" |
| exact | final:0001 | final | `[57]` | `[34, 966, 955, 455, 20, 22, 865, 149, 3, 19, 113, 42, 687, 613, 75, 184, 189, 954]...(+3)` | "I'm trying to create a custom values for my upcoming." |
| padded | chunk:0000 | chunk | `[3]` | `[]` | '' |
| padded | chunk:0001 | chunk | `[5]` | `[]` | '' |
| padded | chunk:0002 | chunk | `[7]` | `[]` | '' |
| padded | chunk:0003 | chunk | `[9]` | `[]` | '' |
| padded | chunk:0004 | chunk | `[11]` | `[]` | '' |
| padded | chunk:0005 | chunk | `[13]` | `[]` | '' |
| padded | chunk:0006 | chunk | `[15]` | `[]` | '' |
| padded | chunk:0007 | chunk | `[17]` | `[]` | '' |
| padded | chunk:0008 | chunk | `[19]` | `[]` | '' |
| padded | chunk:0009 | chunk | `[21]` | `[]` | '' |
| padded | chunk:0010 | chunk | `[23]` | `[]` | '' |
| padded | chunk:0011 | chunk | `[25]` | `[]` | '' |
| padded | chunk:0012 | chunk | `[27]` | `[34, 966, 955]` | "I'm" |
| padded | chunk:0013 | chunk | `[29]` | `[34, 966, 955, 455, 20]` | "I'm trying" |
| padded | chunk:0014 | chunk | `[31]` | `[34, 966, 955, 455, 20, 22]` | "I'm trying to" |
| padded | chunk:0015 | chunk | `[33]` | `[34, 966, 955, 455, 20, 22, 865, 149]` | "I'm trying to create" |
| padded | chunk:0016 | chunk | `[35]` | `[34, 966, 955, 455, 20, 22, 865, 149]` | "I'm trying to create" |
| padded | chunk:0017 | chunk | `[37]` | `[34, 966, 955, 455, 20, 22, 865, 149, 3]` | "I'm trying to create a" |
| padded | chunk:0018 | chunk | `[39]` | `[34, 966, 955, 455, 20, 22, 865, 149, 3]` | "I'm trying to create a" |
| padded | chunk:0019 | chunk | `[41]` | `[34, 966, 955, 455, 20, 22, 865, 149, 3, 19, 113, 42]` | "I'm trying to create a custom" |
| padded | final:0000 | final | `[41]` | `[34, 966, 955, 455, 20, 22, 865, 149, 3, 19, 113, 42, 687, 304, 962]` | "I'm trying to create a custom value." |
| padded | chunk:0020 | chunk | `[43]` | `[34, 966, 955, 455, 20, 22, 865, 149, 3, 19, 113, 42]` | "I'm trying to create a custom" |
| padded | chunk:0021 | chunk | `[45]` | `[34, 966, 955, 455, 20, 22, 865, 149, 3, 19, 113, 42]` | "I'm trying to create a custom" |
| padded | chunk:0022 | chunk | `[47]` | `[34, 966, 955, 455, 20, 22, 865, 149, 3, 19, 113, 42, 687, 613]` | "I'm trying to create a custom values" |
| padded | chunk:0023 | chunk | `[49]` | `[34, 966, 955, 455, 20, 22, 865, 149, 3, 19, 113, 42, 687, 613]` | "I'm trying to create a custom values" |
| padded | chunk:0024 | chunk | `[51]` | `[34, 966, 955, 455, 20, 22, 865, 149, 3, 19, 113, 42, 687, 613, 75]` | "I'm trying to create a custom values for" |
| padded | chunk:0025 | chunk | `[53]` | `[34, 966, 955, 455, 20, 22, 865, 149, 3, 19, 113, 42, 687, 613, 75, 184, 189]` | "I'm trying to create a custom values for my up" |
| padded | chunk:0026 | chunk | `[55]` | `[34, 966, 955, 455, 20, 22, 865, 149, 3, 19, 113, 42, 687, 613, 75, 184, 189]` | "I'm trying to create a custom values for my up" |
| padded | chunk:0027 | chunk | `[57]` | `[34, 966, 955, 455, 20, 22, 865, 149, 3, 19, 113, 42, 687, 613, 75, 184, 189, 954]...(+1)` | "I'm trying to create a custom values for my upcom" |
| padded | final:0001 | final | `[57]` | `[34, 966, 955, 455, 20, 22, 865, 149, 3, 19, 113, 42, 687, 613, 75, 184, 189, 954]...(+3)` | "I'm trying to create a custom values for my upcoming." |
