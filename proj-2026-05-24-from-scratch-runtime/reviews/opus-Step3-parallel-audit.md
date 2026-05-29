# Step 3 — independent Opus parallel audit (PAIRED REVIEW)

Written in parallel with Codex (`b8h1l5zpx`) while it implements the residual
Step 3 sub-tasks (lib/admission + lib/scheduler moves, SHA256+JSON extract to
lib/runtime_io/io.{h,cpp}, static-global audit doc).

## Opus pass on the static-global audit (v5 §II step 1.5)

The v5 architecture spec called this out as a potential complexity surface:
"audit `session_main.cpp` for static globals; transfer their ownership to
`SharedRuntime` (which is binary-owned, not library-owned). Library code has
NO statics for resource state."

Empirical finding from grep on `runtime/cpp/lib/session/session.cpp`:

| Category | Count | Disposition |
|---|---:|---|
| `static constexpr` constants | 15 | Stay (value-only, no mutable state). e.g., `BLANK=1024`, `MAX_SYMBOLS=10`, `MODEL_ID`. |
| `static const std::string` (function-local in `Tokenizer::ids_to_text`) | 2 | Stay (function-local immutable). |
| `static` FUNCTION declarations | ~99 | Stay (file-scope linkage; not state). |
| `static` mutable variables at file scope | **0** | None. |
| Function-local `static` mutable variables | **0** | None found in grep `^\s+static [a-zA-Z]` (excluding constexpr/inline/const). |

**Net: lib/session/session.cpp has ZERO mutable static state.**

### Implication for Step 4

The Step 4 spec includes "static-global ownership transfer to SharedRuntime."
The actual transfer work is **MUCH smaller than the v5 architecture feared**
because there's nothing to transfer.

Step 4's SharedRuntime work simplifies to:
- DEFINE SharedRuntime as a clean container for the resources that ARE
  currently owned by stack variables in `session_main_entrypoint()`:
  - `torch::jit::Module bundle` (the session_audio_bundle.ts).
  - `torch::jit::Module enc_first` (the first-encoder TorchScript).
  - The `std::map<std::pair<int64_t, int64_t>, std::unique_ptr<AOTIModelPackageLoader>>`
    returned by `load_finalize_bucket_loaders` (line 1609).
  - The `std::unordered_map<std::string, at::Tensor>` returned by
    `load_shared_constants` (passed in as a const ref).
  - The `Tokenizer` returned by `tokenizer_from_bundle`.
  - The `AudioGeometry` constructed from bundle attributes.
- WIRE SharedRuntime's lifecycle: constructed once in main() (binary-owned),
  passed by const ref into SessionRuntime constructors.
- NO actual code transfer from statics → SharedRuntime is needed (because
  there ARE no statics holding the relevant state).

This is a SIGNIFICANT simplification of the v5 design's risk register
(item 6: "Static-global audit may surface unexpected complexity").
**Risk closed — no complexity surfaced.**

## Opus pass on the file moves (Step 3 sub-tasks 1-3)

The 3 file moves are mechanical and low-risk:
1. `density_admission.{h,cpp}` → `lib/admission/`.
2. `batched_steady_scheduler.{h,cpp}` → `lib/scheduler/`.
3. `steady_batch_primitive.h` → `lib/scheduler/`.

**Verification checklist** (for when Codex's move lands):
- `git mv` should be used (preserves history).
- Update `#include` paths in: `density_main.cpp` (uses density_admission +
  batched_steady_scheduler), `lib/session/session.cpp` (may use
  steady_batch_primitive), `steady_batch_bench.cpp` (uses
  steady_batch_primitive).
- CMakeLists.txt:
  - REMOVE density_admission.cpp + batched_steady_scheduler.cpp from
    density_main's add_executable source list.
  - ADD those .cpp files to nemotron_runtime's add_library source list.
  - (steady_batch_primitive is header-only; just an include path change.)
- The `nemotron_runtime` static library should grow to include:
  - `lib/session/session.cpp`
  - `lib/telemetry/stats_collector.cpp`
  - `lib/runtime_io/picohttpparser.c`
  - `lib/runtime_io/io.cpp` (NEW from Step 3 sub-task 4)
  - `lib/admission/density_admission.cpp` (NEW from Step 3 sub-task 1)
  - `lib/scheduler/batched_steady_scheduler.cpp` (NEW from Step 3 sub-task 2)

## Opus pass on the SHA256+JSON extract (Step 3 sub-task 4)

The plan asks to extract SHA256+JSON helpers from `steady_batch_primitive.h`
to `lib/runtime_io/io.{h,cpp}`. The extraction needs to:
- Identify EXACTLY which helpers are exported (to know what goes in io.h).
- Preserve byte-exact behavior (the SHA256 is used for AOTI bundle
  verification — must produce identical hashes).
- Keep `steady_batch_primitive.h` working by `#include`-ing the new io.h.

Note: `lib/session/session.cpp` ALSO has SHA256 implementations (lines 1141,
1202, 1237 — `rotr`, `sha256_final`, `sha256_tensor_bytes`). These are
DIFFERENT from steady_batch_primitive's — they're for tensor-bytes hashing
(used in tensor comparison/equality checks), not file-bytes hashing.

**Disposition for the duplicate SHA256 code**:
- steady_batch_primitive's `sha256_file` (or equivalent) → lib/runtime_io/io.
- session.cpp's `sha256_tensor_bytes` + `rotr`/`sha256_final` → STAYS in
  session.cpp for now (they're tightly coupled to the tensor-comparison
  test harness). Could be unified in a future cleanup; out of scope for
  Step 3.

This duplication is OK for Step 3 (the scope is just extracting steady's
helpers); future refactor could unify on a single SHA256 module.

## What I CANNOT review until Codex lands

- The actual CMakeLists.txt diff (Codex's specific changes).
- The new `lib/runtime_io/io.h` API surface.
- Whether Codex chose to extract additional helpers (e.g., file_exists,
  directory_exists) into io.h or left them in session.cpp.
- Whether the smoke set passes after the moves.
- Whether any caller missed a #include update (build error sentinel).

When Codex lands, I'll do a focused diff review on:
1. CMakeLists.txt (nemotron_runtime sources updated, executables stripped).
2. lib/runtime_io/io.{h,cpp} (API surface + correctness).
3. session-cpp-static-globals.md (compare against this Opus pass — does it
   confirm "ZERO mutable statics" finding?).
4. The smoke set results (5/5 + N=200 PASS).

## Net

Step 3 is mostly mechanical at this point (file moves + include updates +
CMake updates + a small helper extract). The PAIRED REVIEW pattern produces
this independent audit BEFORE seeing Codex's work, so the fold is genuine
(not "rubber-stamp Codex"). When Codex lands I'll do a side-by-side review
+ fold.
