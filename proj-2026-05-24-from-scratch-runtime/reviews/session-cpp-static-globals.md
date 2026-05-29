# session.cpp Static Global Audit

Date: 2026-05-29
Scope: `runtime/cpp/lib/session/session.cpp`
Step: STEP3B-WS-PLAN v5 Step 3 static-global audit only. No statics are transferred in this step.

## Commands

```bash
grep -nE '^static [^ ]+ [a-z]' runtime/cpp/lib/session/session.cpp
rg -n '^\s*static\b' runtime/cpp/lib/session/session.cpp
```

## Summary

| Category | Count | Disposition |
|---|---:|---|
| `static constexpr` / compile-time constants | 16 | Verified OK, stays. No resource ownership. |
| Function-local immutable strings | 2 | Verified OK, stays. No resource ownership. |
| File-scope static helper functions | 99 | Verified OK, stays private to `session.cpp`. |
| Static resource-state globals | 0 | Nothing to transfer in Step 3. |
| Static rows marked transfer to `SharedRuntime` in Step 4 | 0 | Resource owners to transfer are non-static locals; see below. |

## Non-Static Resource Owners For Step 4

These are not `static`, but they are the resource state Step 4 should move behind `SharedRuntime`:

| Lines | Resource | Current ownership | Step 4 disposition |
|---|---|---|---|
| L3913 | `tokenizer` | automatic local from `tokenizer_from_bundle(bundle)` | Transfer tokenizer ownership/access to `SharedRuntime`. |
| L3929-L3932 | `enc_first`, `enc_first_long_check` | automatic `torch::jit::Module` locals | Transfer model/module ownership to `SharedRuntime` as appropriate for production wrappers. |
| L3933-L3934 | `enc_steady`, `enc_steady_long_check` | automatic `AOTIModelPackageLoader` locals | Transfer AOTI handle ownership to `SharedRuntime`. |
| L3935-L3939 | `joint`, `predict` | automatic `torch::jit::Module` locals | Transfer decoder module ownership to `SharedRuntime`. |
| L3940 | `finalize_loaders` | automatic map returned by `load_finalize_bucket_loaders` | Transfer finalize bucket loader ownership to `SharedRuntime`. |
| L3969-L3977 | `preproc`, `audio_front` | automatic `unique_ptr` locals for audio mode | Transfer preproc/frontend ownership or construction policy to `SharedRuntime`/runtime config. |

## Every Static Match

| Line | Static | Kind | Disposition |
|---|---|---|---|
| L40 | `static constexpr int BLANK = 1024;` | compile-time constant | verified-OK, stays |
| L41 | `static constexpr int MAX_SYMBOLS = 10;` | compile-time constant | verified-OK, stays |
| L42 | `static constexpr int SHIFT = 16;` | compile-time constant | verified-OK, stays |
| L43 | `static constexpr int PRE = 9;` | compile-time constant | verified-OK, stays |
| L44 | `static constexpr int DROP = 2;` | compile-time constant | verified-OK, stays |
| L45 | `static constexpr int RIGHT_CONTEXT = 1;` | compile-time constant | verified-OK, stays |
| L46 | `static constexpr int FINAL_PADDING_FRAMES = 32;` | compile-time constant | verified-OK, stays |
| L47 | `static constexpr int ATT_CONTEXT_LEFT = 70;` | compile-time constant | verified-OK, stays |
| L48 | `static constexpr int ATT_CONTEXT_RIGHT = 1;` | compile-time constant | verified-OK, stays |
| L49 | `static constexpr const char* MODEL_ID = "nvidia/nemotron-speech-streaming-en-0.6b";` | compile-time constant | verified-OK, stays |
| L50 | `static constexpr double ARGMAX_MARGIN_WARNING_THRESHOLD = 1.0e-2;` | compile-time constant | verified-OK, stays |
| L51 | `static constexpr double ARGMAX_MARGIN_UNSAFE_THRESHOLD = 1.0e-3;` | compile-time constant | verified-OK, stays |
| L56 | `static constexpr int64_t EVENT_INTERIM = 0;` | compile-time constant | verified-OK, stays |
| L57 | `static constexpr int64_t EVENT_FINAL = 1;` | compile-time constant | verified-OK, stays |
| L58 | `static constexpr int64_t EVENT_SUPPRESSED = 2;` | compile-time constant | verified-OK, stays |
| L347 | `static std::string utt_attr(int utt, const char* name) {` | helper function | verified-OK, stays |
| L351 | `static std::string utt_chunk_attr(int utt, int chunk, const char* name) {` | helper function | verified-OK, stays |
| L355 | `static std::string prefix_attr(const std::string& prefix, const char* name) {` | helper function | verified-OK, stays |
| L359 | `static std::string prefix_chunk_attr(const std::string& prefix, int chunk, const char* name) {` | helper function | verified-OK, stays |
| L363 | `static std::string stream_turn_prefix(int stream, int turn) {` | helper function | verified-OK, stays |
| L367 | `static std::string stream_end_prefix(int stream) {` | helper function | verified-OK, stays |
| L379 | `static torch::Tensor utt_chunk_tensor(torch::jit::Module& bundle, int utt, int chunk, const char* name) {` | helper function | verified-OK, stays |
| L410 | `static std::vector<double> tensor_to_double_vec(torch::Tensor tensor) {` | helper function | verified-OK, stays |
| L419 | `static std::vector<float> tensor_to_float_vec(torch::Tensor tensor) {` | helper function | verified-OK, stays |
| L428 | `static const char* pass_fail(bool ok) {` | helper function | verified-OK, stays |
| L432 | `static const char* pass_fail_skip(int64_t checks, int64_t pass) {` | helper function | verified-OK, stays |
| L437 | `static const char* mode_name(SessionMode mode) {` | helper function | verified-OK, stays |
| L446 | `static bool tensor_storage_alias(const torch::Tensor& lhs, const torch::Tensor& rhs) {` | helper function | verified-OK, stays |
| L554 | `static std::vector<std::string> split_words(const std::string& text) {` | helper function | verified-OK, stays |
| L562 | `static std::string join_words(const std::vector<std::string>& words) {` | helper function | verified-OK, stays |
| L618 | `static void replace_all(std::string& text, const std::string& needle, const std::string& repl) {` | helper function | verified-OK, stays |
| L631 | `static const std::string marker = "\xE2\x96\x81";` | function-local immutable string | verified-OK, stays |
| L632 | `static const std::string unk_surface = " \xE2\x81\x87 ";` | function-local immutable string | verified-OK, stays |
| L654 | `static std::vector<std::vector<int64_t>> unpack_i64_lists(torch::Tensor flat_tensor,` | helper function | verified-OK, stays |
| L679 | `static std::vector<uint8_t> tensor_to_u8_vec(torch::Tensor tensor) {` | helper function | verified-OK, stays |
| L689 | `static std::vector<std::string> unpack_utf8_strings(torch::Tensor flat_tensor,` | helper function | verified-OK, stays |
| L796 | `static std::string one_text_from_bundle(torch::jit::Module& bundle,` | helper function | verified-OK, stays |
| L809 | `static std::string optional_one_text_from_bundle(torch::jit::Module& bundle,` | helper function | verified-OK, stays |
| L830 | `static bool equal_events(const std::vector<EmittedEvent>& got,` | helper function | verified-OK, stays |
| L887 | `static size_t first_token_diff_index(const std::vector<int64_t>& got,` | helper function | verified-OK, stays |
| L897 | `static int64_t token_or_missing(const std::vector<int64_t>& tokens, size_t index) {` | helper function | verified-OK, stays |
| L945 | `static bool tensor_close_cache(const char* name,` | helper function | verified-OK, stays |
| L979 | `static bool optional_tensor_equal(const char* name, const torch::Tensor& actual, const torch::Tensor& expected) {` | helper function | verified-OK, stays |
| L989 | `static bool float_vec_equal(const char* name,` | helper function | verified-OK, stays |
| L1008 | `static bool int64_equal(const char* name, int64_t actual, int64_t expected, const std::string& label) {` | helper function | verified-OK, stays |
| L1121 | `static void reset_audio_front(SessionState& state, const AudioGeometry& g) {` | helper function | verified-OK, stays |
| L1141 | `static uint32_t rotr(uint32_t x, uint32_t n) {` | pure IO helper | verified-OK, stays |
| L1145 | `static void sha256_transform(Sha256Ctx& ctx, const uint8_t data[64]) {` | pure IO helper | verified-OK, stays |
| L1146 | `static constexpr std::array<uint32_t, 64> k{` | compile-time constant | verified-OK, stays |
| L1191 | `static void sha256_update(Sha256Ctx& ctx, const uint8_t* data, size_t len) {` | pure IO helper | verified-OK, stays |
| L1202 | `static std::string sha256_final(Sha256Ctx& ctx) {` | pure IO helper | verified-OK, stays |
| L1224 | `static std::string sha256_bytes_with_label(const std::string& label,` | pure IO helper | verified-OK, stays |
| L1237 | `static std::string sha256_tensor_bytes(torch::Tensor tensor) {` | pure IO helper | verified-OK, stays |
| L1264 | `static std::string read_text_file(const std::string& path) {` | pure IO helper | verified-OK, stays |
| L1272 | `static size_t skip_ws(const std::string& s, size_t pos) {` | pure IO helper | verified-OK, stays |
| L1277 | `static size_t find_matching_json_delim(const std::string& s, size_t open_pos) {` | pure IO helper | verified-OK, stays |
| L1307 | `static std::string json_value_for_key(const std::string& object, const std::string& key) {` | pure IO helper | verified-OK, stays |
| L1338 | `static std::string json_string_field(const std::string& object, const std::string& key) {` | pure IO helper | verified-OK, stays |
| L1347 | `static int64_t json_int_field(const std::string& object, const std::string& key) {` | pure IO helper | verified-OK, stays |
| L1356 | `static double json_double_field(const std::string& object, const std::string& key) {` | pure IO helper | verified-OK, stays |
| L1365 | `static std::vector<int64_t> json_int_array_field(const std::string& object, const std::string& key) {` | pure IO helper | verified-OK, stays |
| L1379 | `static bool parse_bucket_filename(const std::string& filename, int64_t& drop, int64_t& T) {` | helper function | verified-OK, stays |
| L1469 | `static void require_contract_eq(const char* name, int64_t actual, int64_t expected) {` | helper function | verified-OK, stays |
| L1555 | `static const at::Tensor* resolve_shared_constant(const std::unordered_map<std::string, at::Tensor>& shared_constants,` | helper function | verified-OK, stays |
| L1609 | `static std::map<std::pair<int64_t, int64_t>, std::unique_ptr<AOTIModelPackageLoader>>` | resource-loader helper | function stays; returned loader ownership moves in Step 4 |
| L1648 | `static void observe_margin(MarginStats& stats,` | helper function | verified-OK, stays |
| L1664 | `static void observe_token_margin(MarginStats& stats,` | helper function | verified-OK, stays |
| L1674 | `static void merge_margin_stats(MarginStats& dst, const MarginStats& src) {` | helper function | verified-OK, stays |
| L1686 | `static const TokenMargin* margin_for_token_index(const MarginStats& stats, size_t token_index) {` | helper function | verified-OK, stays |
| L1693 | `static void decode_range(torch::jit::Module& joint,` | helper function | verified-OK, stays |
| L1751 | `static void apply_encoder_outputs(SessionState& state,` | helper function | verified-OK, stays |
| L1793 | `static void observe_first_chunk_drift(torch::jit::Module& bundle,` | helper function | verified-OK, stays |
| L1817 | `static std::vector<at::Tensor> run_steady_encoder(AOTIModelPackageLoader& loader,` | AOTI runner helper | function stays; AOTI handle ownership moves in Step 4 |
| L1834 | `static void warm_stream_encoder_artifacts(torch::jit::Module& bundle,` | helper function | verified-OK, stays |
| L1849 | `static AudioGeometry audio_geometry_from_bundle(torch::jit::Module& bundle) {` | helper function | verified-OK, stays |
| L1892 | `static std::string verify_preproc_manifest(const std::string& dir,` | helper function | verified-OK, stays |
| L1945 | `static AudioCiBundleStats audio_ci_stats_from_bundle(torch::jit::Module& bundle) {` | helper function | verified-OK, stays |
| L1968 | `static bool compare_mel_tensor(const std::string& label,` | helper function | verified-OK, stays |
| L2085 | `static PreprocDeterminismStats run_preproc_determinism_check(const AudioFrontend& audio) {` | helper function | verified-OK, stays |
| L2108 | `static void run_steady_chunk_tensor(SessionState& state,` | helper function | verified-OK, stays |
| L2179 | `static void run_steady_chunk(SessionState& state,` | helper function | verified-OK, stays |
| L2203 | `static void run_steady_chunk_from_audio(SessionState& state,` | helper function | verified-OK, stays |
| L2240 | `static int drain_audio_steady(SessionState& state,` | helper function | verified-OK, stays |
| L2272 | `static void run_steady_chunk_tensor_runtime(SessionState& state,` | helper function | verified-OK, stays |
| L2326 | `static int drain_audio_steady_runtime(SessionState& state,` | helper function | verified-OK, stays |
| L2365 | `static int append_pcm_and_drain_runtime(SessionState& state,` | helper function | verified-OK, stays |
| L2399 | `static int flush_post_stop_audio_runtime(SessionState& state,` | helper function | verified-OK, stays |
| L2435 | `static int vad_start(SessionState& state,` | helper function | verified-OK, stays |
| L2479 | `static int append_audio_and_drain(SessionState& state,` | helper function | verified-OK, stays |
| L2522 | `static FinalizeAudioInputs prepare_finalize_inputs_from_audio(const SessionState& parent,` | helper function | verified-OK, stays |
| L2599 | `static bool verify_finalize_audio_gold(torch::jit::Module& bundle,` | helper function | verified-OK, stays |
| L2642 | `static FinalizeOutcome run_finalize(SessionState& parent,` | helper function | verified-OK, stays |
| L2748 | `static FinalizeOutcome run_finalize_runtime(SessionState& parent,` | helper function | verified-OK, stays |
| L2826 | `static bool equal_one_text_from_bundle(torch::jit::Module& bundle,` | helper function | verified-OK, stays |
| L2838 | `static bool retained_state_matches(SessionState& state,` | helper function | verified-OK, stays |
| L2937 | `static bool cold_reset_state_matches(SessionState& state,` | helper function | verified-OK, stays |
| L2998 | `static bool run_synthetic_word_delta_tests() {` | helper function | verified-OK, stays |
| L3035 | `static bool audio_front_ci_ok(const AudioFrontend& audio) {` | helper function | verified-OK, stays |
| L3042 | `static bool audio_front_unsafe_margin_fail(const AudioFrontend& audio, bool token_exact_ok) {` | helper function | verified-OK, stays |
| L3046 | `static void print_audio_front_summary(const char* label,` | helper function | verified-OK, stays |
| L3094 | `static void print_first_chunk_summary(const char* label, const FirstChunkStats& stats) {` | helper function | verified-OK, stays |
| L3121 | `static bool cache_ownership_ok(const CacheOwnershipStats& stats) {` | helper function | verified-OK, stays |
| L3125 | `static void print_cache_ownership_summary(const char* label, const CacheOwnershipStats& stats) {` | helper function | verified-OK, stays |
| L3135 | `static void print_long_stream_cache_summary(const LongStreamCacheStats& stats) {` | helper function | verified-OK, stays |
| L3148 | `static LongStreamCacheStats run_long_stream_cache_stability_check(` | helper function | verified-OK, stays |
| L3280 | `static void coverage_observe_finalize(CoverageManifest& coverage,` | helper function | verified-OK, stays |
| L3291 | `static std::string set_to_range_string(const std::set<int64_t>& values) {` | helper function | verified-OK, stays |
| L3298 | `static std::vector<float> clip_or_repeat_audio(const std::vector<float>& source, size_t n) {` | helper function | verified-OK, stays |
| L3309 | `static std::vector<float> first_audio_source_from_bundle(torch::jit::Module& bundle, bool multiturn) {` | helper function | verified-OK, stays |
| L3329 | `static size_t audio_needed_for_one_steady_chunk(const AudioFrontend& audio, const SessionState& state) {` | helper function | verified-OK, stays |
| L3339 | `static bool run_real_vad_start_cancel_check(` | helper function | verified-OK, stays |
| L3492 | `static bool run_synthetic_coverage_checks(` | helper function | verified-OK, stays |
| L3585 | `static void print_coverage_manifest(const char* label,` | helper function | verified-OK, stays |
| L3621 | `static bool equal_replay_fingerprint(const ReplayFingerprint& actual,` | helper function | verified-OK, stays |
| L3644 | `static ReplayFingerprint replay_single_row_fingerprint(` | helper function | verified-OK, stays |
| L3712 | `static ReplayFingerprint replay_multiturn_stream_fingerprint(` | helper function | verified-OK, stays |
| L3837 | `static void write_replay_fingerprint_file(const std::string& path,` | helper function | verified-OK, stays |
