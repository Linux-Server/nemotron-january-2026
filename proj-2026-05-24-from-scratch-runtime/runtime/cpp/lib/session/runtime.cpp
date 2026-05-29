#include "lib/session/runtime.h"

#include "lib/scheduler/batched_steady_scheduler.h"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <functional>
#include <future>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>

namespace fs = std::filesystem;

namespace {

using FinalizeBucketKey = std::pair<int64_t, int64_t>;

class InferenceExecutor;

thread_local InferenceExecutor* current_inference_executor = nullptr;

class InferenceExecutor {
 public:
  explicit InferenceExecutor(int device_index)
      : device_index_(device_index),
        worker_([this]() { worker_loop(); }) {}

  ~InferenceExecutor() {
    close();
  }

  InferenceExecutor(const InferenceExecutor&) = delete;
  InferenceExecutor& operator=(const InferenceExecutor&) = delete;

  template <class F>
  auto run(F&& f) -> std::invoke_result_t<std::decay_t<F>&> {
    if (current_inference_executor == this) {
      throw std::runtime_error("nested inference executor run is not allowed");
    }
    using Fn = std::decay_t<F>;
    using R = std::invoke_result_t<Fn&>;
    auto task = std::make_shared<std::packaged_task<R()>>(Fn(std::forward<F>(f)));
    auto future = task->get_future();
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (closed_) throw std::runtime_error("inference executor is closed");
      tasks_.emplace_back([task]() { (*task)(); });
    }
    cv_.notify_one();
    if constexpr (std::is_void_v<R>) {
      future.get();
    } else {
      return future.get();
    }
  }

  void close() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      closed_ = true;
    }
    cv_.notify_one();
    if (worker_.joinable()) worker_.join();
  }

 private:
  void worker_loop() {
    current_inference_executor = this;
    torch::NoGradGuard no_grad;
    c10::cuda::CUDAGuard device_guard(device_index_);
    for (;;) {
      std::function<void()> task;
      {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [this]() { return closed_ || !tasks_.empty(); });
        if (closed_ && tasks_.empty()) break;
        task = std::move(tasks_.front());
        tasks_.pop_front();
      }
      task();
    }
    current_inference_executor = nullptr;
  }

  int device_index_ = 0;
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<std::function<void()>> tasks_;
  bool closed_ = false;
  std::thread worker_;
};

struct WarmupInput {
  std::string label;
  std::vector<float> audio;
};

double unix_now_seconds() {
  using namespace std::chrono;
  return duration<double>(system_clock::now().time_since_epoch()).count();
}

int validate_finalize_silence_ms(int value) {
  if (value < 0 || value >= 10000) {
    throw std::runtime_error("finalize_silence_ms must be in [0,10000)");
  }
  return value;
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

std::vector<float> tensor_to_float_vector(torch::Tensor tensor) {
  auto flat = tensor.to(torch::kCPU).to(torch::kFloat32).contiguous().reshape({-1});
  std::vector<float> out(static_cast<size_t>(flat.numel()));
  if (!out.empty()) {
    std::memcpy(out.data(), flat.data_ptr<float>(), out.size() * sizeof(float));
  }
  return out;
}

std::optional<WarmupInput> make_bucket_warmup_input(const AudioGeometry& audio_geometry,
                                                    int64_t drop,
                                                    int64_t final_t) {
  int64_t audio_frames = -1;
  if (drop == 0) {
    audio_frames = final_t - FINAL_PADDING_FRAMES - 1;
    if (audio_frames <= 0 || audio_frames >= SHIFT + 1) return std::nullopt;
  } else if (drop == DROP) {
    constexpr int64_t kWarmupSteadyChunks = 2;
    const int64_t final_t_offset =
        PRE + FINAL_PADDING_FRAMES + 1 - kWarmupSteadyChunks * SHIFT;
    const int64_t min_audio_frames = kWarmupSteadyChunks * SHIFT + 1;
    const int64_t next_chunk_audio_frames = (kWarmupSteadyChunks + 1) * SHIFT + 1;
    audio_frames = final_t - final_t_offset;
    if (audio_frames < min_audio_frames || audio_frames >= next_chunk_audio_frames) {
      return std::nullopt;
    }
    const int64_t second_chunk_pending =
        (audio_frames - SHIFT) * audio_geometry.hop_samples;
    if (second_chunk_pending < audio_geometry.preprocess_new_audio_samples) {
      return std::nullopt;
    }
  } else {
    return std::nullopt;
  }

  const int64_t audio_samples = audio_frames * audio_geometry.hop_samples;
  if (audio_samples <= 0) return std::nullopt;
  if (drop == DROP && audio_samples < audio_geometry.preprocess_new_audio_samples) {
    return std::nullopt;
  }
  WarmupInput input;
  input.label = "bucket.drop" + std::to_string(drop) + ".T" + std::to_string(final_t);
  input.audio.assign(static_cast<size_t>(audio_samples), 0.0f);
  return input;
}

std::vector<WarmupInput> make_bucket_warmup_inputs(
    const AudioGeometry& audio_geometry,
    const std::map<FinalizeBucketKey, std::unique_ptr<AOTIModelPackageLoader>>& finalize_loaders) {
  std::vector<WarmupInput> inputs;
  inputs.reserve(finalize_loaders.size());
  for (const auto& kv : finalize_loaders) {
    auto input = make_bucket_warmup_input(audio_geometry, kv.first.first, kv.first.second);
    if (input.has_value()) inputs.push_back(std::move(*input));
  }
  return inputs;
}

std::optional<WarmupInput> make_fixture_warmup_input(torch::jit::Module& bundle) {
  try {
    int64_t rows = scalar_i64(attr_tensor(bundle, "num_utts"));
    int best_utt = -1;
    int best_score = std::numeric_limits<int>::min();
    int64_t best_samples = std::numeric_limits<int64_t>::max();
    for (int64_t utt = 0; utt < rows; ++utt) {
      int64_t final_t = scalar_i64(utt_tensor(bundle, static_cast<int>(utt), "final_T"));
      if (final_t <= 0) continue;
      int64_t steady = scalar_i64(utt_tensor(bundle, static_cast<int>(utt), "num_steady"));
      auto audio = utt_tensor(bundle, static_cast<int>(utt), "audio");
      int64_t samples = audio.numel();
      int score = steady >= 2 ? 2 : (steady >= 1 ? 1 : 0);
      if (score > best_score || (score == best_score && samples < best_samples)) {
        best_utt = static_cast<int>(utt);
        best_score = score;
        best_samples = samples;
      }
    }
    if (best_utt < 0) return std::nullopt;
    WarmupInput input;
    input.label = "utt" + std::to_string(best_utt);
    input.audio = tensor_to_float_vector(utt_tensor(bundle, best_utt, "audio"));
    return input;
  } catch (const std::exception&) {
    return std::nullopt;
  }
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
        device(torch::kCUDA, cfg.device_index),
        inference_executor(cfg.device_index) {
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

    warm_inference_executor();

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
    inference_executor.close();
  }

  void warm_inference_executor() {
    constexpr int kWarmupItersPerInput = 5;
    auto inputs = make_bucket_warmup_inputs(audio_geometry, finalize_loaders);
    if (inputs.empty()) {
      auto fixture = make_fixture_warmup_input(bundle);
      if (fixture.has_value()) inputs.push_back(std::move(*fixture));
    }
    if (inputs.empty()) {
      throw std::runtime_error("unable to build inference executor warmup input");
    }

    int warmed = inference_executor.run([this,
                                         warmup_inputs = std::move(inputs),
                                         warmup_iters = kWarmupItersPerInput]() {
      int completed = 0;
      for (const auto& warmup_input : warmup_inputs) {
        for (int iter = 0; iter < warmup_iters; ++iter) {
          SessionState warm_state;
          reset_session(warm_state, bundle, device);
          auto warm_audio = make_session_runtime_audio_frontend(bundle, preproc, device);
          reset_session_runtime_audio_front(warm_state, *warm_audio);

          std::vector<EmittedEvent> events;
          const std::string label = "inference_warmup." + warmup_input.label +
                                    ".iter" + std::to_string(iter);
          (void)session_runtime_append_pcm_and_drain(warm_state,
                                                     warmup_input.audio,
                                                     *warm_audio,
                                                     enc_first,
                                                     *enc_steady,
                                                     joint,
                                                     predict,
                                                     device,
                                                     tokenizer_value,
                                                     events,
                                                     label + ".append");
          vad_stop(warm_state);
          (void)session_runtime_finalize(warm_state,
                                         bundle,
                                         *warm_audio,
                                         finalize_loaders,
                                         joint,
                                         predict,
                                         device,
                                         tokenizer_value,
                                         events,
                                         FinalizeFinish::SPECULATIVE_KEEP,
                                         label + ".finalize");
          ++completed;
        }
      }
      c10::cuda::device_synchronize();
      return completed;
    });
    std::printf("inference executor warmed: %d iters\n", warmed);
    std::fflush(stdout);
  }

  SharedRuntimeConfig cfg;
  std::string artifact_dir;
  std::string bundle_path;
  std::string finalize_buckets_dir;
  torch::Device device;
  InferenceExecutor inference_executor;
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
        finalize_silence_ms(validate_finalize_silence_ms(cfg.finalize_silence_ms)),
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
    s.inference_executor.run([&]() {
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
    });
    debug_events.insert(debug_events.end(), events.begin(), events.end());
    return project_events(events, std::nullopt);
  }

  void vad_start() {
    vad_state = VadState::SPEAKING;
    vad_deadline_ts.reset();
    pending_timing.reset();
    std::vector<EmittedEvent> events;
    auto& s = *shared.impl_;
    s.inference_executor.run([&]() {
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
    });
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
    if (finalize_silence_ms == 0) {
      return finalize_and_idle("debounce_expired", FinalizeFinish::SPECULATIVE_KEEP);
    }
    vad_state = VadState::PENDING_FINALIZE;
    vad_deadline_ts = now + static_cast<double>(finalize_silence_ms) / 1000.0;
    return {};
  }

  std::vector<WireEvent> poll_timer(double now_unix_ts) {
    if (vad_state != VadState::PENDING_FINALIZE || !vad_deadline_ts.has_value()) return {};
    if (now_unix_ts < *vad_deadline_ts) return {};
    return finalize_and_idle("debounce_expired", FinalizeFinish::SPECULATIVE_KEEP);
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
    auto executor_wait_start = std::chrono::steady_clock::now();
    outcome = s.inference_executor.run([&]() {
      timing.inference_lock_acquire_wait_ms =
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - executor_wait_start).count();
      return session_runtime_finalize(state,
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
    });
    timing.fork_flush_done_ts = unix_now_seconds();
    timing.final_sent_ts = unix_now_seconds();
    timing.was_suppressed = !has_final_event(events);
    last_finalize_tokens = outcome.final_tokens;
    last_timing_value = timing;
    pending_timing.reset();
    debug_events.insert(debug_events.end(), events.begin(), events.end());

    // SessionRuntime leaves stats emission to the WS worker, which owns the stale-generation
    // send/drop decision and records last_timing() after that emit decision.
    return project_events(events, timing);
  }

  std::vector<WireEvent> finalize_and_idle(const std::string& reason, FinalizeFinish finish) {
    auto events = finalize_with(reason, finish);
    clear_vad_state();
    return events;
  }

  void clear_vad_state() {
    vad_state = VadState::IDLE;
    vad_deadline_ts.reset();
  }

  const SharedRuntime& shared;
  SessionConfig cfg;
  int finalize_silence_ms = 0;
  SessionState state;
  RuntimeAudioFrontendPtr audio;
  VadState vad_state = VadState::IDLE;
  std::optional<double> vad_deadline_ts;
  std::optional<SessionTiming> pending_timing;
  std::optional<SessionTiming> last_timing_value;
  uint64_t finalize_seq = 0;
  std::vector<EmittedEvent> debug_events;
  std::vector<int64_t> last_finalize_tokens;
};

SessionRuntime::SessionRuntime(const SharedRuntime& shared, SessionConfig cfg)
    : impl_(std::make_unique<Impl>(shared, std::move(cfg))) {}

SessionRuntime::~SessionRuntime() {
  if (impl_) bump_generation();
}

std::vector<WireEvent> SessionRuntime::append_pcm_and_drain(const PCMFrame& frame) {
  return impl_->append_pcm(frame);
}

void SessionRuntime::handle_vad_start() {
  impl_->vad_start();
}

std::vector<WireEvent> SessionRuntime::handle_vad_stop() {
  return impl_->vad_stop_and_maybe_finalize();
}

std::vector<WireEvent> SessionRuntime::poll_timer(double now_unix_ts) {
  return impl_->poll_timer(now_unix_ts);
}

VadState SessionRuntime::vad_state() const noexcept {
  return impl_->vad_state;
}

std::optional<double> SessionRuntime::vad_deadline_ts() const noexcept {
  return impl_->vad_deadline_ts;
}

std::vector<WireEvent> SessionRuntime::reset(bool finalize) {
  bump_generation();
  if (!finalize) {
    impl_->clear_vad_state();
    return impl_->soft_final(false);
  }
  return impl_->finalize_and_idle("reset", FinalizeFinish::SPECULATIVE_KEEP);
}

std::vector<WireEvent> SessionRuntime::end(bool finalize) {
  bump_generation();
  if (!finalize) {
    impl_->clear_vad_state();
    return impl_->soft_final(false);
  }
  return impl_->finalize_and_idle("end", FinalizeFinish::TRUE_BOUNDARY_COLD_RESET);
}

std::vector<WireEvent> SessionRuntime::finalize_now() {
  return impl_->finalize_and_idle("debounce_expired", FinalizeFinish::SPECULATIVE_KEEP);
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
