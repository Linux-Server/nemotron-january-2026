#include "lib/session/runtime.h"

#include "lib/scheduler/batched_steady_scheduler.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace fs = std::filesystem;

namespace {

using FinalizeBucketKey = std::pair<int64_t, int64_t>;

double unix_now_seconds() {
  using namespace std::chrono;
  return duration<double>(system_clock::now().time_since_epoch()).count();
}

std::string parent_dir(const std::string& path) {
  fs::path p(path);
  if (p.has_parent_path()) return p.parent_path().string();
  return ".";
}

std::string artifact_dir_from_config(const SharedRuntimeConfig& cfg) {
  if (!cfg.steady_artifacts_dir.empty()) return cfg.steady_artifacts_dir;
  if (!cfg.bundle_path.empty()) return parent_dir(cfg.bundle_path);
  return "../artifacts";
}

std::string bundle_path_from_config(const SharedRuntimeConfig& cfg, const std::string& artifact_dir) {
  if (!cfg.bundle_path.empty()) return cfg.bundle_path;
  return (fs::path(artifact_dir) / "session_audio_bundle.ts").string();
}

std::string finalize_buckets_dir_from_config(const SharedRuntimeConfig& cfg, const std::string& artifact_dir) {
  if (!cfg.finalize_buckets_dir.empty()) return cfg.finalize_buckets_dir;
  std::string stripped = (fs::path(artifact_dir) / "stripped_finalize_buckets").string();
  if (directory_exists(stripped)) return stripped;
  return (fs::path(artifact_dir) / "finalize_buckets").string();
}

torch::jit::Module load_module_on_device(const std::string& path, torch::Device device) {
  auto module = torch::jit::load(path);
  module.to(device);
  module.eval();
  return module;
}

bool steady_batch_dir_has_packages(const std::string& dir) {
  for (int bucket : {1, 2, 4}) {
    if (!file_exists((fs::path(dir) / ("enc_steady_aoti_b" + std::to_string(bucket) + ".pt2")).string())) {
      return false;
    }
  }
  return true;
}

std::string resolve_steady_batch_dir(const std::string& artifact_dir, const std::string& configured) {
  if (!configured.empty() && steady_batch_dir_has_packages(configured)) return configured;
  std::vector<std::string> candidates;
  fs::path artifact_path(artifact_dir);
  if (artifact_path.has_parent_path()) {
    candidates.push_back((artifact_path.parent_path() / "steady_b_artifacts").string());
  }
  candidates.push_back("steady_b_artifacts");
  candidates.push_back("../steady_b_artifacts");
  candidates.push_back("runtime/steady_b_artifacts");
  for (const auto& candidate : candidates) {
    if (steady_batch_dir_has_packages(candidate)) return candidate;
  }
  throw std::runtime_error("scheduler_enabled requested but steady batch artifacts were not found");
}

std::map<FinalizeBucketKey, std::unique_ptr<AOTIModelPackageLoader>> load_finalize_loaders_for_runtime(
    const std::string& buckets_dir,
    const std::string& shared_weights,
    const std::string& shared_weights_pt,
    torch::Device device,
    int num_runners,
    BucketManifest* manifest_out,
    std::unordered_map<std::string, at::Tensor>* shared_constants_out) {
  if (num_runners <= 0) throw std::runtime_error("finalize_num_runners must be positive");
  if (!directory_exists(buckets_dir)) throw std::runtime_error("finalize buckets directory missing: " + buckets_dir);
  if (!file_exists(shared_weights)) throw std::runtime_error("finalize shared weights missing: " + shared_weights);

  auto bucket_paths = discover_finalize_buckets(buckets_dir);
  if (bucket_paths.empty()) throw std::runtime_error("no finalize bucket packages found in " + buckets_dir);
  std::string manifest_path = (fs::path(buckets_dir) / "manifest.json").string();
  if (!file_exists(manifest_path)) {
    throw std::runtime_error("finalize bucket manifest is required when buckets are present: " + manifest_path);
  }
  BucketManifest manifest = load_bucket_manifest(manifest_path);
  verify_bucket_manifest(manifest, bucket_paths, buckets_dir, shared_weights_pt);
  auto shared_constants = load_shared_constants(shared_weights, device);

  std::map<FinalizeBucketKey, std::unique_ptr<AOTIModelPackageLoader>> loaders;
  for (const auto& kv : bucket_paths) {
    auto loader = std::make_unique<AOTIModelPackageLoader>(
        kv.second, "model", false, num_runners, device.index());
    auto bucket_constants = constants_for_bucket(shared_constants, *loader, kv.second);
    loader->load_constants(bucket_constants.values, false, false, true);
    loaders.emplace(kv.first, std::move(loader));
  }

  if (manifest_out != nullptr) *manifest_out = std::move(manifest);
  if (shared_constants_out != nullptr) *shared_constants_out = std::move(shared_constants);
  return loaders;
}

std::vector<float> pcm_to_float(const PCMFrame& frame) {
  if (frame.count > 0 && frame.samples == nullptr) {
    throw std::runtime_error("PCMFrame samples is null with non-zero count");
  }
  std::vector<float> out;
  out.reserve(frame.count);
  for (size_t i = 0; i < frame.count; ++i) {
    out.push_back(static_cast<float>(frame.samples[i]) / 32768.0f);
  }
  return out;
}

std::vector<WireEvent> project_events(const std::vector<EmittedEvent>& events,
                                      const std::optional<SessionTiming>& final_timing) {
  std::vector<WireEvent> out;
  out.reserve(events.size());
  for (const auto& event : events) {
    if (event.kind == EVENT_SUPPRESSED) continue;
    WireEvent wire;
    wire.type = "transcript";
    wire.text = event.text;
    if (event.kind == EVENT_INTERIM) {
      wire.is_final = false;
    } else if (event.kind == EVENT_FINAL) {
      wire.is_final = true;
      wire.finalize = true;
      if (final_timing.has_value()) wire.finalize_timing = final_timing->to_wire_json();
    } else {
      continue;
    }
    out.push_back(std::move(wire));
  }
  return out;
}

bool has_final_event(const std::vector<EmittedEvent>& events) {
  return std::any_of(events.begin(), events.end(), [](const EmittedEvent& event) {
    return event.kind == EVENT_FINAL;
  });
}

}  // namespace

struct SharedRuntime::Impl {
  explicit Impl(SharedRuntimeConfig config)
      : cfg(std::move(config)),
        artifact_dir(artifact_dir_from_config(cfg)),
        bundle_path(bundle_path_from_config(cfg, artifact_dir)),
        finalize_buckets_dir(finalize_buckets_dir_from_config(cfg, artifact_dir)),
        device(torch::kCUDA, cfg.device_index) {
    if (cfg.steady_num_runners <= 0) throw std::runtime_error("steady_num_runners must be positive");
    torch::NoGradGuard ng;

    bundle = torch::jit::load(bundle_path);
    verify_session_bundle_meta(bundle, false);
    tokenizer_value = tokenizer_from_bundle(bundle);
    if (cfg.verify_tokenizer) verify_tokenizer_selftest(bundle, tokenizer_value);

    audio_geometry = session_runtime_audio_geometry_from_bundle(bundle);
    std::string preproc_path = (fs::path(artifact_dir) / "preproc.ts").string();
    session_runtime_verify_preproc_manifest(artifact_dir, preproc_path, audio_geometry);
    preproc = load_module_on_device(preproc_path, device);

    enc_first = load_module_on_device((fs::path(artifact_dir) / "enc_first.ts").string(), device);
    enc_first_long_check = load_module_on_device((fs::path(artifact_dir) / "enc_first.ts").string(), device);
    enc_steady = std::make_unique<AOTIModelPackageLoader>(
        (fs::path(artifact_dir) / "enc_steady_aoti.pt2").string(), "model", false, cfg.steady_num_runners,
        device.index());
    enc_steady_long_check = std::make_unique<AOTIModelPackageLoader>(
        (fs::path(artifact_dir) / "enc_steady_aoti.pt2").string(), "model", false, cfg.steady_num_runners,
        device.index());
    joint = load_module_on_device((fs::path(artifact_dir) / "joint_step.ts").string(), device);
    predict = load_module_on_device((fs::path(artifact_dir) / "predict_step.ts").string(), device);

    finalize_loaders = load_finalize_loaders_for_runtime(
        finalize_buckets_dir,
        (fs::path(artifact_dir) / "finalize_shared_weights.ts").string(),
        (fs::path(artifact_dir) / "finalize_shared_weights.pt").string(),
        device,
        cfg.finalize_num_runners,
        &finalize_bucket_manifest,
        &shared_constants);

    if (cfg.scheduler_enabled) {
      std::string batch_dir = resolve_steady_batch_dir(artifact_dir, cfg.steady_artifacts_dir);
      BatchedSteadySchedulerPolicy policy;
      policy.B_max = cfg.b_max;
      policy.window_ms = cfg.batch_window_ms;
      policy.lone_timeout_ms = cfg.batch_lone_timeout_ms;
      policy.queue_capacity = cfg.batch_queue_capacity;
      batched_steady = std::make_unique<BatchedSteadyLoaderSet>(
          batch_dir,
          (fs::path(artifact_dir) / "finalize_shared_weights.ts").string(),
          device,
          cfg.steady_num_runners,
          "shared_runtime_scheduler");
      batched_steady->preload_all();
      scheduler = std::make_unique<BatchedSteadyScheduler>(*batched_steady, device, policy);
      scheduler->warmup_buckets();
      scheduler->start();
    }
  }

  ~Impl() {
    if (scheduler) scheduler->close();
  }

  SharedRuntimeConfig cfg;
  std::string artifact_dir;
  std::string bundle_path;
  std::string finalize_buckets_dir;
  torch::Device device;
  torch::jit::Module bundle;
  Tokenizer tokenizer_value;
  torch::jit::Module enc_first;
  torch::jit::Module enc_first_long_check;
  std::unique_ptr<AOTIModelPackageLoader> enc_steady;
  std::unique_ptr<AOTIModelPackageLoader> enc_steady_long_check;
  torch::jit::Module joint;
  torch::jit::Module predict;
  std::map<FinalizeBucketKey, std::unique_ptr<AOTIModelPackageLoader>> finalize_loaders;
  std::unordered_map<std::string, at::Tensor> shared_constants;
  BucketManifest finalize_bucket_manifest;
  AudioGeometry audio_geometry;
  torch::jit::Module preproc;
  std::unique_ptr<BatchedSteadyLoaderSet> batched_steady;
  std::unique_ptr<BatchedSteadyScheduler> scheduler;
  mutable std::mutex model_mutex;
};

SharedRuntime::SharedRuntime(SharedRuntimeConfig cfg) : impl_(std::make_unique<Impl>(std::move(cfg))) {}

SharedRuntime::~SharedRuntime() = default;

const Tokenizer& SharedRuntime::tokenizer() const {
  return impl_->tokenizer_value;
}

const SharedRuntimeConfig& SharedRuntime::config() const {
  return impl_->cfg;
}

struct SessionRuntime::Impl {
  Impl(const SharedRuntime& shared_in, SessionConfig config)
      : shared(shared_in),
        cfg(std::move(config)),
        audio(nullptr, nullptr) {
    auto& s = *shared.impl_;
    torch::NoGradGuard ng;
    reset_session(state, s.bundle, s.device);
    audio = make_session_runtime_audio_frontend(s.bundle, s.preproc, s.device);
    reset_session_runtime_audio_front(state, *audio);
  }

  std::vector<WireEvent> append_pcm(const PCMFrame& frame) {
    std::vector<float> pcm = pcm_to_float(frame);
    std::vector<EmittedEvent> events;
    auto& s = *shared.impl_;
    {
      std::lock_guard<std::mutex> lock(s.model_mutex);
      session_runtime_append_pcm_and_drain(state,
                                           pcm,
                                           *audio,
                                           s.enc_first,
                                           *s.enc_steady,
                                           s.joint,
                                           s.predict,
                                           s.device,
                                           s.tokenizer_value,
                                           events,
                                           cfg.label + ".append");
    }
    debug_events.insert(debug_events.end(), events.begin(), events.end());
    return project_events(events, std::nullopt);
  }

  void vad_start() {
    pending_timing.reset();
    std::vector<EmittedEvent> events;
    auto& s = *shared.impl_;
    {
      std::lock_guard<std::mutex> lock(s.model_mutex);
      session_runtime_vad_start(state,
                                *audio,
                                s.enc_first,
                                *s.enc_steady,
                                s.joint,
                                s.predict,
                                s.device,
                                s.tokenizer_value,
                                events,
                                cfg.label + ".vad_start");
    }
    debug_events.insert(debug_events.end(), events.begin(), events.end());
  }

  std::vector<WireEvent> vad_stop_and_maybe_finalize() {
    SessionTiming timing;
    double now = unix_now_seconds();
    timing.reason = "debounce_expired";
    timing.vad_stop_ts = now;
    timing.gil_attrib_enabled = cfg.gil_attrib_enabled || shared.impl_->cfg.gil_attrib_enabled;
    pending_timing = timing;
    vad_stop(state);
    if (cfg.finalize_silence_ms == 0) return finalize_with("debounce_expired", FinalizeFinish::SPECULATIVE_KEEP);
    return {};
  }

  std::vector<WireEvent> soft_final(bool finalize_flag) const {
    WireEvent wire;
    wire.type = "transcript";
    wire.text = shared.impl_->tokenizer_value.ids_to_text(state.hyp);
    wire.is_final = true;
    wire.finalize = finalize_flag;
    return {std::move(wire)};
  }

  std::vector<WireEvent> finalize_with(const std::string& reason, FinalizeFinish finish) {
    auto& s = *shared.impl_;
    SessionTiming timing = pending_timing.value_or(SessionTiming{});
    timing.reason = reason;
    timing.debounce_expiry_ts = unix_now_seconds();
    timing.finalize_seq = ++finalize_seq;
    timing.active_sessions_at_emit = cfg.active_sessions_at_emit;
    timing.gil_attrib_enabled = cfg.gil_attrib_enabled || s.cfg.gil_attrib_enabled;

    if (finish == FinalizeFinish::SPECULATIVE_KEEP && state.mode == SessionMode::STREAMING) {
      vad_stop(state);
      if (!timing.vad_stop_ts.has_value()) timing.vad_stop_ts = timing.debounce_expiry_ts;
    }

    std::vector<EmittedEvent> events;
    FinalizeOutcome outcome;
    timing.fork_flush_start_ts = unix_now_seconds();
    auto lock_wait_start = std::chrono::steady_clock::now();
    {
      std::unique_lock<std::mutex> lock(s.model_mutex);
      timing.inference_lock_acquire_wait_ms =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - lock_wait_start).count();
      outcome = session_runtime_finalize(state,
                                         s.bundle,
                                         *audio,
                                         s.finalize_loaders,
                                         s.joint,
                                         s.predict,
                                         s.device,
                                         s.tokenizer_value,
                                         events,
                                         finish,
                                         cfg.label + ".finalize");
    }
    timing.fork_flush_done_ts = unix_now_seconds();
    timing.final_sent_ts = unix_now_seconds();
    timing.was_suppressed = !has_final_event(events);
    last_finalize_tokens = outcome.final_tokens;
    last_timing_value = timing;
    pending_timing.reset();
    debug_events.insert(debug_events.end(), events.begin(), events.end());

    // SessionRuntime deliberately does not call StatsCollector::record here. The WS worker owns
    // the stale-generation/send/drop decision and records last_timing() after that emit decision.
    return project_events(events, timing);
  }

  const SharedRuntime& shared;
  SessionConfig cfg;
  SessionState state;
  RuntimeAudioFrontendPtr audio;
  std::optional<SessionTiming> pending_timing;
  std::optional<SessionTiming> last_timing_value;
  uint64_t finalize_seq = 0;
  std::vector<EmittedEvent> debug_events;
  std::vector<int64_t> last_finalize_tokens;
};

SessionRuntime::SessionRuntime(const SharedRuntime& shared, SessionConfig cfg)
    : impl_(std::make_unique<Impl>(shared, std::move(cfg))) {}

SessionRuntime::~SessionRuntime() = default;

std::vector<WireEvent> SessionRuntime::append_pcm_and_drain(const PCMFrame& frame) {
  return impl_->append_pcm(frame);
}

void SessionRuntime::handle_vad_start() {
  impl_->vad_start();
}

std::vector<WireEvent> SessionRuntime::handle_vad_stop() {
  return impl_->vad_stop_and_maybe_finalize();
}

std::vector<WireEvent> SessionRuntime::reset(bool finalize) {
  if (!finalize) return impl_->soft_final(false);
  return impl_->finalize_with("reset", FinalizeFinish::SPECULATIVE_KEEP);
}

std::vector<WireEvent> SessionRuntime::end(bool finalize) {
  if (!finalize) return impl_->soft_final(false);
  return impl_->finalize_with("end", FinalizeFinish::TRUE_BOUNDARY_COLD_RESET);
}

std::vector<WireEvent> SessionRuntime::finalize_now() {
  return impl_->finalize_with("debounce_expired", FinalizeFinish::SPECULATIVE_KEEP);
}

uint64_t SessionRuntime::generation() const noexcept {
  return impl_->state.generation.load(std::memory_order_acquire);
}

void SessionRuntime::bump_generation() noexcept {
  impl_->state.generation.fetch_add(1, std::memory_order_acq_rel);
}

std::optional<SessionTiming> SessionRuntime::last_timing() const {
  return impl_->last_timing_value;
}

std::vector<EmittedEvent> session_runtime_debug_events(const SessionRuntime& runtime) {
  return runtime.impl_->debug_events;
}

std::vector<int64_t> session_runtime_debug_last_final_tokens(const SessionRuntime& runtime) {
  return runtime.impl_->last_finalize_tokens;
}
