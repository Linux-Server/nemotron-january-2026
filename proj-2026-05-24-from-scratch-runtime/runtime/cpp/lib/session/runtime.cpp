#include "lib/session/runtime.h"

#include "lib/scheduler/batched_steady_scheduler.h"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
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
#include <sstream>
#include <thread>
#include <type_traits>
#include <utility>

namespace fs = std::filesystem;

namespace {

using FinalizeBucketKey = std::pair<int64_t, int64_t>;

class InferenceLane;

thread_local InferenceLane* current_inference_lane = nullptr;
std::atomic<int> g_active_shared_scheduler_owners{0};
std::atomic<int> g_active_shared_steady_loader_sets{0};

void runtime_cuda_check(cudaError_t err, const char* expr) {
  if (err == cudaSuccess) return;
  throw std::runtime_error(std::string(expr) + " failed: " + cudaGetErrorString(err));
}

void runtime_cuda_warn(cudaError_t err, const char* expr) noexcept {
  if (err == cudaSuccess) return;
  std::fprintf(stderr, "%s failed during cleanup: %s\n", expr, cudaGetErrorString(err));
}

torch::jit::Module load_module_on_device(const std::string& path, torch::Device device);

int parse_positive_env_int(const char* name, int fallback) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') return fallback;
  errno = 0;
  char* end = nullptr;
  long value = std::strtol(raw, &end, 10);
  if (errno != 0 || end == raw || *end != '\0' ||
      value <= 0 || value > std::numeric_limits<int>::max()) {
    throw std::runtime_error(std::string(name) + " must be a positive integer: " + raw);
  }
  return static_cast<int>(value);
}

size_t gpu_used_bytes() {
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  runtime_cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
  return total_bytes >= free_bytes ? total_bytes - free_bytes : 0;
}

double bytes_to_mib(size_t bytes) {
  return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

class InferenceLane {
 public:
  InferenceLane(int id, torch::Device device, const std::string& artifact_dir)
      : id_(id), device_(device) {
    c10::cuda::CUDAGuard device_guard(device_.index());
    runtime_cuda_check(cudaStreamCreateWithFlags(&raw_stream_, cudaStreamNonBlocking),
                       "cudaStreamCreateWithFlags(inference_lane)");
    stream_.emplace(c10::cuda::getStreamFromExternal(raw_stream_, device_.index()));
    preproc_ = std::make_unique<torch::jit::Module>(
        load_module_on_device((fs::path(artifact_dir) / "preproc.ts").string(), device_));
    joint_ = std::make_unique<torch::jit::Module>(
        load_module_on_device((fs::path(artifact_dir) / "joint_step.ts").string(), device_));
    predict_ = std::make_unique<torch::jit::Module>(
        load_module_on_device((fs::path(artifact_dir) / "predict_step.ts").string(), device_));
    worker_ = std::thread([this]() { worker_loop(); });
  }

  ~InferenceLane() {
    close();
    destroy_stream();
  }

  InferenceLane(const InferenceLane&) = delete;
  InferenceLane& operator=(const InferenceLane&) = delete;

  int id() const noexcept { return id_; }

  c10::cuda::CUDAStream stream() const {
    if (!stream_.has_value()) throw std::runtime_error("inference lane stream has not been initialized");
    return *stream_;
  }

  torch::jit::Module& joint() const { return *joint_; }
  torch::jit::Module& predict() const { return *predict_; }
  torch::jit::Module& preproc() const { return *preproc_; }

  ExecutionContext execution_context() const {
    return {stream(), joint(), predict(), preproc()};
  }

  void synchronize() const {
    runtime_cuda_check(cudaStreamSynchronize(stream().stream()), "cudaStreamSynchronize(inference_lane)");
  }

  template <class F>
  auto run(F&& f) -> std::invoke_result_t<std::decay_t<F>&> {
    if (current_inference_lane == this) {
      throw std::runtime_error("nested inference lane run is not allowed");
    }
    using Fn = std::decay_t<F>;
    using R = std::invoke_result_t<Fn&>;
    auto task = std::make_shared<std::packaged_task<R()>>(Fn(std::forward<F>(f)));
    auto future = task->get_future();
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (closed_) throw std::runtime_error("inference lane is closed");
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
    current_inference_lane = this;
    torch::NoGradGuard no_grad;
    c10::cuda::CUDAGuard device_guard(device_.index());
    for (;;) {
      std::function<void()> task;
      {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [this]() { return closed_ || !tasks_.empty(); });
        if (closed_ && tasks_.empty()) break;
        task = std::move(tasks_.front());
        tasks_.pop_front();
      }
      c10::cuda::CUDAStreamGuard stream_guard(stream());
      task();
    }
    current_inference_lane = nullptr;
  }

  void destroy_stream() noexcept {
    if (raw_stream_ == nullptr) return;
    c10::cuda::CUDAGuard device_guard(device_.index());
    runtime_cuda_warn(cudaStreamSynchronize(raw_stream_), "cudaStreamSynchronize(inference_lane)");
    runtime_cuda_warn(cudaStreamDestroy(raw_stream_), "cudaStreamDestroy(inference_lane)");
    raw_stream_ = nullptr;
    stream_.reset();
  }

  int id_ = 0;
  torch::Device device_;
  cudaStream_t raw_stream_ = nullptr;
  std::optional<c10::cuda::CUDAStream> stream_;
  std::unique_ptr<torch::jit::Module> preproc_;
  std::unique_ptr<torch::jit::Module> joint_;
  std::unique_ptr<torch::jit::Module> predict_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<std::function<void()>> tasks_;
  bool closed_ = false;
  std::thread worker_;
};

class SharedSchedulerOwnership {
 public:
  SharedSchedulerOwnership() = default;
  SharedSchedulerOwnership(const SharedSchedulerOwnership&) = delete;
  SharedSchedulerOwnership& operator=(const SharedSchedulerOwnership&) = delete;

  ~SharedSchedulerOwnership() {
    reset();
  }

  void register_loader_set() {
    int count = g_active_shared_steady_loader_sets.fetch_add(1, std::memory_order_acq_rel) + 1;
    if (count != 1) {
      g_active_shared_steady_loader_sets.fetch_sub(1, std::memory_order_acq_rel);
      throw std::runtime_error("duplicate BatchedSteadyLoaderSet owner detected: active_loader_sets=" +
                               std::to_string(count));
    }
    loader_set_registered_ = true;
  }

  void register_scheduler() {
    int count = g_active_shared_scheduler_owners.fetch_add(1, std::memory_order_acq_rel) + 1;
    if (count != 1) {
      g_active_shared_scheduler_owners.fetch_sub(1, std::memory_order_acq_rel);
      throw std::runtime_error("duplicate BatchedSteadyScheduler owner detected: active_schedulers=" +
                               std::to_string(count));
    }
    scheduler_registered_ = true;
  }

  int active_loader_sets() const {
    return g_active_shared_steady_loader_sets.load(std::memory_order_acquire);
  }

  int active_schedulers() const {
    return g_active_shared_scheduler_owners.load(std::memory_order_acquire);
  }

 private:
  void reset() noexcept {
    if (scheduler_registered_) {
      g_active_shared_scheduler_owners.fetch_sub(1, std::memory_order_acq_rel);
      scheduler_registered_ = false;
    }
    if (loader_set_registered_) {
      g_active_shared_steady_loader_sets.fetch_sub(1, std::memory_order_acq_rel);
      loader_set_registered_ = false;
    }
  }

  bool loader_set_registered_ = false;
  bool scheduler_registered_ = false;
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

struct FinalizeLoaderMemoryRecord {
  int64_t drop = 0;
  int64_t T = 0;
  int num_runners = 0;
  size_t used_before = 0;
  size_t used_after = 0;
  size_t delta = 0;
  size_t cumulative_delta = 0;
};

class FinalizeBucketLoaderPool final : public FinalizeBucketLoaderProvider {
 public:
  FinalizeBucketLoaderPool(const std::string& buckets_dir,
                           const std::string& shared_weights,
                           const std::string& shared_weights_pt,
                           torch::Device device,
                           int num_runners,
                           std::string policy)
      : buckets_dir_(buckets_dir),
        shared_weights_(shared_weights),
        device_(device),
        num_runners_(num_runners),
        policy_(std::move(policy)) {
    if (num_runners_ <= 0) throw std::runtime_error("finalize_num_runners must be positive");
    if (!directory_exists(buckets_dir_)) {
      throw std::runtime_error("finalize buckets directory missing: " + buckets_dir_);
    }
    if (!file_exists(shared_weights_)) {
      throw std::runtime_error("finalize shared weights missing: " + shared_weights_);
    }

    bucket_paths_ = discover_finalize_buckets(buckets_dir_);
    if (bucket_paths_.empty()) throw std::runtime_error("no finalize bucket packages found in " + buckets_dir_);
    std::string manifest_path = (fs::path(buckets_dir_) / "manifest.json").string();
    if (!file_exists(manifest_path)) {
      throw std::runtime_error("finalize bucket manifest is required when buckets are present: " + manifest_path);
    }
    manifest_ = load_bucket_manifest(manifest_path);
    verify_bucket_manifest(manifest_, bucket_paths_, buckets_dir_, shared_weights_pt);
    std::printf("runtime finalize manifest verified: buckets=%zu weights_sha256=%s num_runners=%d policy=%s\n",
                manifest_.buckets.size(),
                manifest_.contract.weights_sha256.c_str(),
                num_runners_,
                policy_.c_str());

    c10::cuda::CUDAGuard device_guard(device_.index());
    runtime_cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize(before finalize shared constants)");
    size_t before = gpu_used_bytes();
    shared_constants_ = load_shared_constants(shared_weights_, device_);
    runtime_cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize(after finalize shared constants)");
    size_t after = gpu_used_bytes();
    shared_delta_bytes_ = after >= before ? after - before : 0;
    std::printf("runtime finalize shared constants loaded: entries=%zu shared_delta_mib=%.3f policy=%s\n",
                shared_constants_.size(),
                bytes_to_mib(shared_delta_bytes_),
                policy_.c_str());
    std::fflush(stdout);
  }

  AOTIModelPackageLoader& get(int64_t drop, int64_t T) override {
    std::lock_guard<std::mutex> lock(mu_);
    auto key = std::make_pair(drop, T);
    auto it = loaders_.find(key);
    if (it != loaders_.end()) return *it->second;
    auto loaded = load_bucket_locked(key);
    return *loaded->second;
  }

  void preload_all() {
    for (const auto& kv : bucket_paths_) {
      (void)get(kv.first.first, kv.first.second);
    }
  }

  std::vector<FinalizeBucketKey> bucket_keys() const {
    std::vector<FinalizeBucketKey> keys;
    keys.reserve(bucket_paths_.size());
    for (const auto& kv : bucket_paths_) keys.push_back(kv.first);
    return keys;
  }

  int num_runners() const noexcept { return num_runners_; }
  size_t total_bucket_count() const noexcept { return bucket_paths_.size(); }

  size_t loaded_bucket_count() const {
    std::lock_guard<std::mutex> lock(mu_);
    return loaders_.size();
  }

  size_t shared_delta_bytes() const noexcept { return shared_delta_bytes_; }

  size_t total_loader_delta_bytes() const {
    std::lock_guard<std::mutex> lock(mu_);
    return total_loader_delta_bytes_;
  }

  std::string memory_json() const {
    std::lock_guard<std::mutex> lock(mu_);
    size_t projected_all = 0;
    if (!records_.empty()) {
      long double mean = static_cast<long double>(total_loader_delta_bytes_) /
                         static_cast<long double>(records_.size());
      projected_all = static_cast<size_t>(mean * static_cast<long double>(bucket_paths_.size()));
    }
    std::ostringstream oss;
    oss << "{\"policy\":\"" << policy_ << "\""
        << ",\"num_runners_per_loaded_bucket\":" << num_runners_
        << ",\"total_manifest_buckets\":" << bucket_paths_.size()
        << ",\"loaded_buckets\":" << loaders_.size()
        << ",\"shared_constants_delta_bytes\":" << shared_delta_bytes_
        << ",\"loader_delta_bytes\":" << total_loader_delta_bytes_
        << ",\"projected_all_buckets_same_runner_delta_bytes\":" << projected_all
        << ",\"records\":[";
    for (size_t i = 0; i < records_.size(); ++i) {
      const auto& r = records_[i];
      if (i > 0) oss << ",";
      oss << "{\"drop\":" << r.drop
          << ",\"T\":" << r.T
          << ",\"num_runners\":" << r.num_runners
          << ",\"used_before_bytes\":" << r.used_before
          << ",\"used_after_bytes\":" << r.used_after
          << ",\"delta_bytes\":" << r.delta
          << ",\"cumulative_delta_bytes\":" << r.cumulative_delta
          << "}";
    }
    oss << "]}";
    return oss.str();
  }

 private:
  std::map<FinalizeBucketKey, std::unique_ptr<AOTIModelPackageLoader>>::iterator load_bucket_locked(
      const FinalizeBucketKey& key) {
    auto path_it = bucket_paths_.find(key);
    if (path_it == bucket_paths_.end()) {
      throw std::runtime_error("runtime finalize missing bucket drop=" +
                               std::to_string(key.first) +
                               " T=" + std::to_string(key.second));
    }

    c10::cuda::CUDAGuard device_guard(device_.index());
    runtime_cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize(before finalize bucket load)");
    size_t before = gpu_used_bytes();
    auto loader = std::make_unique<AOTIModelPackageLoader>(
        path_it->second, "model", false, num_runners_, device_.index());
    auto bucket_constants = constants_for_bucket(shared_constants_, *loader, path_it->second);
    loader->load_constants(bucket_constants.values, false, false, true);
    runtime_cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize(after finalize bucket load)");
    size_t after = gpu_used_bytes();
    size_t delta = after >= before ? after - before : 0;
    total_loader_delta_bytes_ += delta;
    records_.push_back({
        key.first,
        key.second,
        num_runners_,
        before,
        after,
        delta,
        total_loader_delta_bytes_,
    });
    std::printf("runtime finalize bucket loaded: drop=%ld T=%ld constants=%zu direct=%zu alias=%zu "
                "num_runners=%d loader_delta_mib=%.3f cumulative_loader_mib=%.3f policy=%s\n",
                static_cast<long>(key.first),
                static_cast<long>(key.second),
                bucket_constants.values.size(),
                bucket_constants.direct_matches,
                bucket_constants.alias_fallbacks,
                num_runners_,
                bytes_to_mib(delta),
                bytes_to_mib(total_loader_delta_bytes_),
                policy_.c_str());
    std::fflush(stdout);
    auto inserted = loaders_.emplace(key, std::move(loader));
    return inserted.first;
  }

  std::string buckets_dir_;
  std::string shared_weights_;
  torch::Device device_;
  int num_runners_ = 1;
  std::string policy_;
  BucketManifest manifest_;
  std::map<FinalizeBucketKey, std::string> bucket_paths_;
  std::unordered_map<std::string, at::Tensor> shared_constants_;
  std::map<FinalizeBucketKey, std::unique_ptr<AOTIModelPackageLoader>> loaders_;
  mutable std::mutex mu_;
  std::vector<FinalizeLoaderMemoryRecord> records_;
  size_t shared_delta_bytes_ = 0;
  size_t total_loader_delta_bytes_ = 0;
};

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
    const std::vector<FinalizeBucketKey>& finalize_bucket_keys) {
  std::vector<WarmupInput> inputs;
  inputs.reserve(finalize_bucket_keys.size());
  for (const auto& key : finalize_bucket_keys) {
    auto input = make_bucket_warmup_input(audio_geometry, key.first, key.second);
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

    enc_first = load_module_on_device((fs::path(artifact_dir) / "enc_first.ts").string(), device);
    enc_steady = std::make_unique<AOTIModelPackageLoader>(
        (fs::path(artifact_dir) / "enc_steady_aoti.pt2").string(), "model", false, cfg.steady_num_runners,
        device.index());

    finalize_loaders = std::make_unique<FinalizeBucketLoaderPool>(
        finalize_buckets_dir,
        (fs::path(artifact_dir) / "finalize_shared_weights.ts").string(),
        (fs::path(artifact_dir) / "finalize_shared_weights.pt").string(),
        device,
        cfg.finalize_num_runners,
        "ws_shared_finalize_pool");
    finalize_loaders->preload_all();
    std::printf("shared finalize loader pool ready: num_runners=%d loaded_buckets=%zu/%zu "
                "shared_constants_mib=%.3f loader_mib=%.3f memory=%s\n",
                finalize_loaders->num_runners(),
                finalize_loaders->loaded_bucket_count(),
                finalize_loaders->total_bucket_count(),
                bytes_to_mib(finalize_loaders->shared_delta_bytes()),
                bytes_to_mib(finalize_loaders->total_loader_delta_bytes()),
                finalize_loaders->memory_json().c_str());
    std::fflush(stdout);

    build_inference_lanes();
    warm_inference_lanes();

    if (cfg.scheduler_enabled) {
      std::string batch_dir = resolve_steady_batch_dir(artifact_dir, cfg.steady_artifacts_dir);
      scheduler_ownership.register_loader_set();
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
      scheduler_ownership.register_scheduler();
      scheduler = std::make_unique<BatchedSteadyScheduler>(*batched_steady, device, policy);
      scheduler->warmup_buckets();
      scheduler->start();
      std::printf("shared runtime scheduler owner ready: owner=SharedRuntime scheduler_instances=%d "
                  "steady_loader_sets=%d warmup_complete=true dispatcher_started=true\n",
                  scheduler_ownership.active_schedulers(),
                  scheduler_ownership.active_loader_sets());
      std::fflush(stdout);
    } else {
      std::printf("shared runtime scheduler disabled: owner=SharedRuntime scheduler_instances=0 "
                  "steady_loader_sets=0\n");
      std::fflush(stdout);
    }
  }

  ~Impl() {
    if (scheduler) scheduler->close();
    {
      std::lock_guard<std::mutex> lock(lanes_mu);
      lanes_closing = true;
    }
    lanes_cv.notify_all();
    for (auto& lane : lanes) {
      if (lane) lane->close();
    }
  }

  void build_inference_lanes() {
    lane_count = parse_positive_env_int("NEMOTRON_WS_LANES", 1);
    c10::cuda::CUDAGuard device_guard(device.index());
    const size_t used_before_lanes = gpu_used_bytes();
    lanes.reserve(static_cast<size_t>(lane_count));
    for (int lane_id = 0; lane_id < lane_count; ++lane_id) {
      lanes.push_back(std::make_unique<InferenceLane>(lane_id, device, artifact_dir));
      free_lanes.push_back(lane_id);
    }
    runtime_cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize(after inference lanes)");
    const size_t used_after_lanes = gpu_used_bytes();
    lane_delta_bytes = used_after_lanes >= used_before_lanes
                           ? used_after_lanes - used_before_lanes
                           : 0;
    lane_delta_per_lane_bytes = lane_count > 0
                                    ? lane_delta_bytes / static_cast<size_t>(lane_count)
                                    : 0;
    constexpr size_t kBigModuleDuplicationThresholdBytes = 512ull * 1024ull * 1024ull;
    if (lane_delta_per_lane_bytes > kBigModuleDuplicationThresholdBytes) {
      throw std::runtime_error("inference lane memory delta suggests big-module duplication: per_lane_mib=" +
                               std::to_string(bytes_to_mib(lane_delta_per_lane_bytes)));
    }
    std::printf("inference lane pool built: lanes=%d per_lane_mib=%.3f total_lane_mib=%.3f "
                "no_big_module_duplication=true shared_big_modules=enc_first,enc_steady,finalize_loaders\n",
                lane_count,
                bytes_to_mib(lane_delta_per_lane_bytes),
                bytes_to_mib(lane_delta_bytes));
    std::fflush(stdout);
  }

  InferenceLane& acquire_lane(const std::string& label) {
    std::unique_lock<std::mutex> lock(lanes_mu);
    lanes_cv.wait(lock, [this]() { return lanes_closing || !free_lanes.empty(); });
    if (lanes_closing) throw std::runtime_error("inference lane pool is closing");
    int lane_id = free_lanes.front();
    free_lanes.pop_front();
    std::printf("inference lane acquired: lane=%d label=%s\n", lane_id, label.c_str());
    std::fflush(stdout);
    return *lanes.at(static_cast<size_t>(lane_id));
  }

  void release_lane(InferenceLane* lane) noexcept {
    if (lane == nullptr) return;
    {
      std::lock_guard<std::mutex> lock(lanes_mu);
      if (!lanes_closing) free_lanes.push_back(lane->id());
    }
    lanes_cv.notify_one();
  }

  void warm_inference_lanes() {
    constexpr int kWarmupItersPerInput = 5;
    auto inputs = make_bucket_warmup_inputs(audio_geometry, finalize_loaders->bucket_keys());
    const size_t bucket_warmup_inputs = inputs.size();
    if (inputs.empty()) {
      auto fixture = make_fixture_warmup_input(bundle);
      if (fixture.has_value()) inputs.push_back(std::move(*fixture));
    }
    if (inputs.empty()) {
      throw std::runtime_error("unable to build inference lane warmup input");
    }

    int total_warmed = 0;
    for (auto& lane_ptr : lanes) {
      InferenceLane& lane = *lane_ptr;
      int warmed = lane.run([this,
                             &lane,
                             warmup_inputs = inputs,
                             warmup_iters = kWarmupItersPerInput]() {
        int completed = 0;
        for (const auto& warmup_input : warmup_inputs) {
          for (int iter = 0; iter < warmup_iters; ++iter) {
            SessionState warm_state;
            reset_session(warm_state, bundle, device);
            auto warm_audio = make_session_runtime_audio_frontend(bundle, lane.preproc(), device);
            reset_session_runtime_audio_front(warm_state, *warm_audio);

            std::vector<EmittedEvent> events;
            const std::string label = "inference_lane" + std::to_string(lane.id()) +
                                      ".warmup." + warmup_input.label +
                                      ".iter" + std::to_string(iter);
            auto ctx = lane.execution_context();
            {
              std::lock_guard<std::mutex> enc_first_lock(enc_first_mutex);
              (void)session_runtime_append_pcm_and_drain(warm_state,
                                                         warmup_input.audio,
                                                         *warm_audio,
                                                         enc_first,
                                                         *enc_steady,
                                                         ctx,
                                                         device,
                                                         tokenizer_value,
                                                         events,
                                                         label + ".append");
            }
            vad_stop(warm_state);
            (void)session_runtime_finalize(warm_state,
                                           bundle,
                                           *warm_audio,
                                           *finalize_loaders,
                                           ctx,
                                           device,
                                           tokenizer_value,
                                           events,
                                           FinalizeFinish::SPECULATIVE_KEEP,
                                           label + ".finalize");
            ++completed;
          }
        }
        lane.synchronize();
        return completed;
      });
      total_warmed += warmed;
      std::printf("inference lane warmed: lane=%d iters=%d warmup_inputs=%zu finalize_buckets=%zu\n",
                  lane.id(),
                  warmed,
                  inputs.size(),
                  finalize_loaders->total_bucket_count());
      std::fflush(stdout);
    }
    std::printf("inference lane pool warmed: lanes=%d total_iters=%d warmup_inputs=%zu "
                "finalize_bucket_coverage=%zu/%zu per_lane_mib=%.3f\n",
                lane_count,
                total_warmed,
                inputs.size(),
                bucket_warmup_inputs,
                finalize_loaders->total_bucket_count(),
                bytes_to_mib(lane_delta_per_lane_bytes));
    std::fflush(stdout);
  }

  SharedRuntimeConfig cfg;
  std::string artifact_dir;
  std::string bundle_path;
  std::string finalize_buckets_dir;
  torch::Device device;
  torch::jit::Module bundle;
  Tokenizer tokenizer_value;
  torch::jit::Module enc_first;
  std::mutex enc_first_mutex;
  std::unique_ptr<AOTIModelPackageLoader> enc_steady;
  std::unique_ptr<FinalizeBucketLoaderPool> finalize_loaders;
  AudioGeometry audio_geometry;
  int lane_count = 0;
  size_t lane_delta_bytes = 0;
  size_t lane_delta_per_lane_bytes = 0;
  std::vector<std::unique_ptr<InferenceLane>> lanes;
  std::mutex lanes_mu;
  std::condition_variable lanes_cv;
  std::deque<int> free_lanes;
  bool lanes_closing = false;
  SharedSchedulerOwnership scheduler_ownership;
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

bool SharedRuntime::has_scheduler() const noexcept {
  return impl_ && impl_->scheduler != nullptr;
}

BatchedSteadySchedulerTelemetry SharedRuntime::scheduler_telemetry_snapshot() const {
  if (!impl_ || !impl_->scheduler) {
    throw std::runtime_error("SharedRuntime scheduler telemetry requested without scheduler");
  }
  return impl_->scheduler->telemetry_snapshot();
}

struct SessionRuntime::Impl {
  struct LaneLease {
    SharedRuntime::Impl* owner = nullptr;
    InferenceLane* lane = nullptr;

    LaneLease() = default;
    LaneLease(const LaneLease&) = delete;
    LaneLease& operator=(const LaneLease&) = delete;

    ~LaneLease() {
      reset();
    }

    void acquire(SharedRuntime::Impl& owner_in, const std::string& label) {
      reset();
      owner = &owner_in;
      lane = &owner->acquire_lane(label);
    }

    void reset() noexcept {
      if (owner != nullptr && lane != nullptr) {
        owner->release_lane(lane);
      }
      owner = nullptr;
      lane = nullptr;
    }

    InferenceLane& get() const {
      if (lane == nullptr) throw std::runtime_error("session is not bound to an inference lane");
      return *lane;
    }
  };

  Impl(const SharedRuntime& shared_in, SessionConfig config)
      : shared(shared_in),
        cfg(std::move(config)),
        finalize_silence_ms(validate_finalize_silence_ms(cfg.finalize_silence_ms)),
        audio(nullptr, nullptr) {
    auto& s = *shared.impl_;
    lane_lease.acquire(s, cfg.label);
    InferenceLane& lane = lane_lease.get();
    try {
      lane.run([&]() {
        torch::NoGradGuard ng;
        c10::cuda::CUDAGuard device_guard(s.device.index());
        c10::cuda::CUDAStreamGuard stream_guard(lane.stream());
        reset_session(state, s.bundle, s.device);
        audio = make_session_runtime_audio_frontend(s.bundle, lane.preproc(), s.device);
        reset_session_runtime_audio_front(state, *audio);
        lane.synchronize();
      });
    } catch (...) {
      lane_lease.reset();
      throw;
    }
  }

  ~Impl() {
    if (lane_lease.lane != nullptr) {
      try {
        lane().run([&]() { lane().synchronize(); });
      } catch (const std::exception& e) {
        std::fprintf(stderr, "session lane cleanup failed for %s: %s\n", cfg.label.c_str(), e.what());
      }
    }
  }

  InferenceLane& lane() const {
    return lane_lease.get();
  }

  ExecutionContext execution_context() const {
    return lane().execution_context();
  }

  void synchronize_lane_stream() const {
    lane().synchronize();
  }

  std::vector<WireEvent> append_pcm(const PCMFrame& frame) {
    std::vector<float> pcm = pcm_to_float(frame);
    std::vector<EmittedEvent> events;
    auto& s = *shared.impl_;
    lane().run([&]() {
      auto ctx = execution_context();
      if (state.emitted == 0) {
        auto enc_first_wait_start = std::chrono::steady_clock::now();
        std::unique_lock<std::mutex> enc_first_lock(s.enc_first_mutex);
        enc_first_lock_wait_since_finalize_ms +=
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - enc_first_wait_start).count();
        session_runtime_append_pcm_and_drain(state,
                                             pcm,
                                             *audio,
                                             s.enc_first,
                                             *s.enc_steady,
                                             ctx,
                                             s.device,
                                             s.tokenizer_value,
                                             events,
                                             cfg.label + ".append");
      } else {
        session_runtime_append_pcm_and_drain(state,
                                             pcm,
                                             *audio,
                                             s.enc_first,
                                             *s.enc_steady,
                                             ctx,
                                             s.device,
                                             s.tokenizer_value,
                                             events,
                                             cfg.label + ".append");
      }
      synchronize_lane_stream();
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
    lane().run([&]() {
      auto ctx = execution_context();
      if (state.emitted == 0) {
        auto enc_first_wait_start = std::chrono::steady_clock::now();
        std::unique_lock<std::mutex> enc_first_lock(s.enc_first_mutex);
        enc_first_lock_wait_since_finalize_ms +=
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - enc_first_wait_start).count();
        session_runtime_vad_start(state,
                                  *audio,
                                  s.enc_first,
                                  *s.enc_steady,
                                  ctx,
                                  s.device,
                                  s.tokenizer_value,
                                  events,
                                  cfg.label + ".vad_start");
      } else {
        session_runtime_vad_start(state,
                                  *audio,
                                  s.enc_first,
                                  *s.enc_steady,
                                  ctx,
                                  s.device,
                                  s.tokenizer_value,
                                  events,
                                  cfg.label + ".vad_start");
      }
      synchronize_lane_stream();
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
    timing.enc_first_lock_wait_ms = enc_first_lock_wait_since_finalize_ms;

    if (finish == FinalizeFinish::SPECULATIVE_KEEP && state.mode == SessionMode::STREAMING) {
      vad_stop(state);
      if (!timing.vad_stop_ts.has_value()) timing.vad_stop_ts = timing.debounce_expiry_ts;
    }

    std::vector<EmittedEvent> events;
    FinalizeOutcome outcome;
    timing.fork_flush_start_ts = unix_now_seconds();
    auto lane_wait_start = std::chrono::steady_clock::now();
    outcome = lane().run([&]() {
      timing.inference_lock_acquire_wait_ms =
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - lane_wait_start).count();
      auto ctx = execution_context();
      auto result = session_runtime_finalize(state,
                                             s.bundle,
                                             *audio,
                                             *s.finalize_loaders,
                                             ctx,
                                             s.device,
                                             s.tokenizer_value,
                                             events,
                                             finish,
                                             cfg.label + ".finalize");
      synchronize_lane_stream();
      return result;
    });
    timing.fork_flush_done_ts = unix_now_seconds();
    timing.final_sent_ts = unix_now_seconds();
    timing.was_suppressed = !has_final_event(events);
    last_finalize_tokens = outcome.final_tokens;
    last_timing_value = timing;
    pending_timing.reset();
    enc_first_lock_wait_since_finalize_ms = 0.0;
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
  LaneLease lane_lease;
  SessionState state;
  RuntimeAudioFrontendPtr audio;
  VadState vad_state = VadState::IDLE;
  std::optional<double> vad_deadline_ts;
  std::optional<SessionTiming> pending_timing;
  std::optional<SessionTiming> last_timing_value;
  uint64_t finalize_seq = 0;
  double enc_first_lock_wait_since_finalize_ms = 0.0;
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
    impl_->enc_first_lock_wait_since_finalize_ms = 0.0;
    return impl_->soft_final(false);
  }
  return impl_->finalize_and_idle("reset", FinalizeFinish::SPECULATIVE_KEEP);
}

std::vector<WireEvent> SessionRuntime::end(bool finalize) {
  bump_generation();
  if (!finalize) {
    impl_->clear_vad_state();
    impl_->enc_first_lock_wait_since_finalize_ms = 0.0;
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
