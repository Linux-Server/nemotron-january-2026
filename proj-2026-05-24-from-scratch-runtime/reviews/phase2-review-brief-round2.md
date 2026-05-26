# Phase-2 review — Round 2 charge (adversarial-on-Round-1 + deeper concurrency code)

Round 1 is folded in `reviews/phase2-round1-FOLDED.md` (read it first). Both reviewers converged on 6 blockers.
Round 2 is **adversarial on Round 1 itself** PLUS a **deeper code pass** on the concurrency surface Round 1 only
sampled. Keep foregrounding **G1 (utilization/density)** and **G2 (P50↔P95 TTFT spread)**.

## Part A — attack Round 1 (intellectual honesty)
For each Round-1 blocker and proposed edit, ask: is it **wrong, overstated, or misdirected**? Specifically:
- Is the Step 1a/1b split the right cut, or does it create a false "5090 smoke is cheap" impression when 1a
  actually needs the full correctness+topology+finalize+tail harness?
- Is the hot-bucket finalize collision (Codex R1-M4) actually the primary hazard — or is the **shared STEADY
  loader** (called by every stream every ~160ms, vs finalize once per utterance) the far more acute concurrent-
  `run()` hazard? Re-rank if so. (Check: how often is the single `enc_steady` loader hit vs the per-bucket
  finalize loaders, per stream, at the knee.)
- Is the BW-bound-ceiling prediction (Opus R1-M5) actually established, or is there SM-occupancy headroom that
  could let >2.5× overlap? What in the roofline/0.1b data decides this?
- Does `num_runners>1` actually exist in libtorch 2.8's `AOTIModelPackageLoader` and provide a concurrent runner
  pool? If it does, several Round-1 blockers reduce from "redesign" to "verify" — say so.
- Any Round-1 "required edit" that would over-build Step 1 / waste effort? Cut it.

## Part B — deeper concurrency code pass (Round 1 only sampled this)
Read line-by-line and report concurrency-safety findings with `file:line`:
1. **`SessionState`** (`session_main.cpp:~90-130`) — confirm it is fully per-stream (no shared/static mutable
   state) so N threads can each own one. (Opus R1 found it self-contained + only `static const` strings at
   `:569-570` — verify and look for any counterexample, esp. in the finalize/audio paths.)
2. **The finalize fork/clone path** (`session_main.cpp:~2577-2730`, `finalize_main.cpp` clone_state /
   fork_assert) — does forking allocate fresh per-call, or reuse any shared buffer? Is FORK_ASSERT meaningful
   under concurrency or only serial?
3. **Shared model objects across threads** — `enc_steady` (one loader), the finalize bucket loader map,
   and the TorchScript `joint`/`predict`/`enc_first` modules: which are shared vs must be per-thread? Is
   `torch::jit::Module::forward` safe to call concurrently on a shared module in eval/no-grad, or must each
   thread hold its own handle (the mock did — `microbench.cpp:103`)? Cite the evidence.
4. **The per-label `argmax().item()` sync** (`session_main.cpp:1657`) — confirm it syncs only the calling
   thread's stream (if a per-thread stream is set) and is the GPU-idle window the overlap thesis relies on;
   note any way it could serialize all threads (e.g., default-stream sync).
5. **Audio-fed front + raw-ring + remainder recompute** under concurrency — any shared scratch?

## Part C — anything Round 1 MISSED
Look for sequencing inversions (e.g., should the cheapest decisive micro-checks — does `run()` overlap at all? is
concurrent==serial? does GPU mem stay flat at one copy? — be a pre-Step-1 "kill-shot" gate run BEFORE the full
N-sweep, mirroring the original plan's front-load-cheap-kills pattern?), missing artifacts, or any way the
corrected plan still fails to actually MEASURE G1 or G2.

Write to `proj-2026-05-24-from-scratch-runtime/reviews/codex-phase2-round2.md` (BLOCKER/MAJOR/MINOR/QUESTIONS,
file:line, recommended edits). Mark explicitly where you DISAGREE with or DOWNGRADE a Round-1 finding — that
disagreement is the most valuable output of this round. Round 2 of 5.
