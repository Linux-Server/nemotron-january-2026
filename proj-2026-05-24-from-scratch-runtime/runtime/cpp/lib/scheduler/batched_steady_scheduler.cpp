#include "lib/scheduler/batched_steady_scheduler.h"

#include <torch/torch.h>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <pthread.h>
#include <sstream>

#define B2_CUDA_CHECK(expr) BatchedSteadyScheduler::cuda_check((expr), #expr, __FILE__, __LINE__)

namespace {
using Clock = std::chrono::steady_clock;
constexpr size_t kMaxPendingDispatchTimings = 64;
constexpr size_t kMaxPendingDispatches = 64;
constexpr int kDispatchTimingIdlePollMs = 2;
constexpr int kPendingDispatchIdlePollMs = 2;

struct CudaEventOwner {
  cudaEvent_t event = nullptr;

  CudaEventOwner() = default;
  CudaEventOwner(const CudaEventOwner&) = delete;
  CudaEventOwner& operator=(const CudaEventOwner&) = delete;

  ~CudaEventOwner() {
    if (event != nullptr) cudaEventDestroy(event);
  }

  cudaEvent_t release() noexcept {
    cudaEvent_t out = event;
    event = nullptr;
    return out;
  }
};

template <typename Fn>
class ScopeExit {
 public:
  explicit ScopeExit(Fn fn) : fn_(std::move(fn)) {}
  ScopeExit(const ScopeExit&) = delete;
  ScopeExit& operator=(const ScopeExit&) = delete;
  ~ScopeExit() noexcept {
    if (active_) fn_();
  }
  void dismiss() noexcept {
    active_ = false;
  }

 private:
  Fn fn_;
  bool active_ = true;
};

template <typename Fn>
ScopeExit<Fn> make_scope_exit(Fn fn) {
  return ScopeExit<Fn>(std::move(fn));
}

double elapsed_us(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::micro>(end - start).count();
}

std::chrono::milliseconds ms_duration(int ms) {
  return std::chrono::milliseconds(std::max(0, ms));
}

torch::TensorOptions long_options_for(const torch::Tensor& tensor) {
  return torch::TensorOptions().dtype(torch::kLong).device(tensor.device());
}

double timespec_to_us(const timespec& ts) {
  return static_cast<double>(ts.tv_sec) * 1000000.0 + static_cast<double>(ts.tv_nsec) / 1000.0;
}

double current_thread_cpu_us() {
  clockid_t clock_id;
  if (pthread_getcpuclockid(pthread_self(), &clock_id) != 0) {
    clock_id = CLOCK_THREAD_CPUTIME_ID;
  }
  timespec ts{};
  if (clock_gettime(clock_id, &ts) != 0) return -1.0;
  return timespec_to_us(ts);
}

bool aliases_any_raw_output(const at::Tensor& tensor, const std::vector<at::Tensor>& raw) {
  if (!tensor.defined() || !tensor.has_storage()) return false;
  for (const auto& candidate : raw) {
    if (candidate.defined() && candidate.has_storage() && tensor.is_alias_of(candidate)) return true;
  }
  return false;
}

std::vector<at::Tensor> own_row_output_tensors(std::vector<at::Tensor>&& row_tensors,
                                               const std::vector<at::Tensor>& raw) {
  // unpack_prepacked_outputs() uses .contiguous(), but that is a no-op for
  // already-contiguous row views such as B=1 slices. Clone any tensor still
  // aliasing the raw AOTI outputs while dispatcher_stream_ is current, before
  // the per-row completion event is recorded.
  for (auto& tensor : row_tensors) {
    if (aliases_any_raw_output(tensor, raw)) {
      tensor = tensor.clone(at::MemoryFormat::Preserve);
    }
  }
  return std::move(row_tensors);
}
}  // namespace

BatchedSteadyScheduler::BatchedSteadyScheduler(BatchedSteadyLoaderSet& loader_set,
                                               torch::Device device,
                                               BatchedSteadySchedulerPolicy policy)
    : loader_set_(loader_set),
      device_(device),
      policy_(policy),
      dispatch_timing_mode_(dispatch_timing_mode_from_env()),
      dispatcher_stream_(c10::cuda::getStreamFromPool(/*isHighPriority=*/false,
                                                      device.index() >= 0 ? device.index() : 0)) {
  if (policy_.window_ms < 0) throw std::runtime_error("batch steady window_ms must be non-negative");
  if (policy_.lone_timeout_ms < 0) throw std::runtime_error("batch steady lone_timeout_ms must be non-negative");
  if (policy_.B_max != 1 && policy_.B_max != 2 && policy_.B_max != 4) {
    throw std::runtime_error("batch steady B_max must be one of {1,2,4}");
  }
  if (policy_.queue_capacity <= 0) policy_.queue_capacity = 16;
  auto buckets = required_buckets();
  loader_set_.preload_buckets(buckets);
  if (!loader_set_.sealed()) throw std::runtime_error("batch steady loader set failed to seal after preload_buckets()");
  std::printf("B2_SCHEDULER_CONSTRUCT policy={B_max:%d,window_ms:%d,lone_timeout_ms:%d,queue_capacity:%d,use_b2_bucket:%s} "
              "loaded_buckets=%d dispatcher_stream=%p\n",
              policy_.B_max,
              policy_.window_ms,
              policy_.lone_timeout_ms,
              policy_.queue_capacity,
              policy_.use_b2_bucket ? "true" : "false",
              loader_set_.loaded_bucket_count(),
              reinterpret_cast<void*>(dispatcher_stream_.stream()));
}

BatchedSteadyScheduler::~BatchedSteadyScheduler() {
  close();
}

void BatchedSteadyScheduler::cuda_check(cudaError_t err, const char* expr, const char* file, int line) {
  if (err != cudaSuccess) {
    std::ostringstream oss;
    oss << "CUDA error at " << file << ":" << line << " for " << expr
        << ": " << cudaGetErrorString(err);
    throw std::runtime_error(oss.str());
  }
}

BatchedSteadyScheduler::DispatchTimingMode BatchedSteadyScheduler::dispatch_timing_mode_from_env() {
  const char* raw = std::getenv("NEMOTRON_WS_DISPATCH_TIMING");
  if (raw == nullptr || raw[0] == '\0' || std::string(raw) == "sync") return DispatchTimingMode::Sync;
  if (std::string(raw) == "poll") return DispatchTimingMode::Poll;
  if (std::string(raw) == "off") return DispatchTimingMode::Off;
  throw std::runtime_error("NEMOTRON_WS_DISPATCH_TIMING must be one of {sync,poll,off}");
}

std::future<DispatchResult> BatchedSteadyScheduler::enqueue(EnqueueRequest&& request) {
  if (request.producer_event == nullptr) {
    throw std::runtime_error("batch steady enqueue requires a producer-ready CUDA event");
  }
  auto item = std::make_shared<QueueItem>(std::move(request));
  item->enqueue_time = Clock::now();
  auto future = item->promise.get_future();

  std::unique_lock<std::mutex> lock(mutex_);
  cv_capacity_.wait(lock, [&] {
    drain_pending_dispatches_locked(false);
    return closing_ || fault_ || capacity_available_locked();
  });
  if (fault_) std::rethrow_exception(fault_);
  if (closing_) throw std::runtime_error("batch steady enqueue after scheduler close");
  item->sequence = next_sequence_++;
  ++capacity_tokens_in_use_;
  queue_.push_back(item);
  {
    std::lock_guard<std::mutex> telemetry_lock(telemetry_mutex_);
    ++telemetry_.enqueued;
  }
  cv_.notify_one();
  return future;
}

std::optional<std::future<DispatchResult>> BatchedSteadyScheduler::try_enqueue_until(
    EnqueueRequest&& request,
    std::chrono::steady_clock::time_point deadline) {
  if (request.producer_event == nullptr) {
    throw std::runtime_error("batch steady enqueue requires a producer-ready CUDA event");
  }
  auto item = std::make_shared<QueueItem>(std::move(request));
  item->enqueue_time = Clock::now();
  auto future = item->promise.get_future();

  std::unique_lock<std::mutex> lock(mutex_);
  bool admitted = cv_capacity_.wait_until(lock, deadline, [&] {
    drain_pending_dispatches_locked(false);
    return closing_ || fault_ || capacity_available_locked();
  });
  if (!admitted) return std::nullopt;
  if (fault_) std::rethrow_exception(fault_);
  if (closing_) throw std::runtime_error("batch steady enqueue after scheduler close");
  item->sequence = next_sequence_++;
  ++capacity_tokens_in_use_;
  queue_.push_back(item);
  {
    std::lock_guard<std::mutex> telemetry_lock(telemetry_mutex_);
    ++telemetry_.enqueued;
  }
  cv_.notify_one();
  return std::optional<std::future<DispatchResult>>(std::move(future));
}

void BatchedSteadyScheduler::start() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (closed_) throw std::runtime_error("batch steady start after close");
  if (started_) return;
  started_ = true;
  dispatcher_thread_ = std::thread([this] { dispatcher_loop(); });
}

void BatchedSteadyScheduler::close() {
  std::vector<std::shared_ptr<QueueItem>> pending;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) return;
    closing_ = true;
    if (!queue_.empty()) {
      set_pending_exception_locked(&pending);
    }
  }
  for (const auto& item : pending) {
    set_item_exception(item, std::make_exception_ptr(std::runtime_error("batch steady scheduler closed")));
  }
  cv_.notify_all();
  cv_capacity_.notify_all();
  if (dispatcher_thread_.joinable() && dispatcher_thread_.get_id() != std::this_thread::get_id()) {
    dispatcher_thread_.join();
  }
  {
    std::lock_guard<std::mutex> lock(mutex_);
    drain_pending_dispatches_locked(true);
    closed_ = true;
  }
}

int BatchedSteadyScheduler::future_timeout_ms() const {
  int backlog_budget_ms = std::max(policy_.queue_capacity, 1) * 50;
  return std::max(1000, policy_.window_ms + backlog_budget_ms + 200);
}

void BatchedSteadyScheduler::warmup_buckets() {
  c10::cuda::CUDAGuard device_guard(device_.index());
  c10::cuda::CUDAStreamGuard stream_guard(dispatcher_stream_);
  torch::NoGradGuard no_grad;
  for (int bucket : required_buckets()) {
    std::vector<BatchedSteadyInput> ready;
    ready.reserve(static_cast<size_t>(bucket));
    auto chunk = torch::zeros({1, 128, 25}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
    auto cache_ch = torch::zeros({24, 1, 70, 1024}, chunk.options());
    auto cache_t = torch::zeros({24, 1, 1024, 8}, chunk.options());
    auto cache_ch_len = torch::zeros({1}, torch::TensorOptions().dtype(torch::kLong).device(device_));
    for (int row = 0; row < bucket; ++row) {
      ready.push_back({chunk.clone(), cache_ch.clone(), cache_t.clone(), cache_ch_len.clone(),
                       "warmup.B" + std::to_string(bucket) + ".row" + std::to_string(row)});
    }
    auto inputs = pack_into_scratch(ready, bucket);
    auto raw = loader_set_.run_raw_prepacked(inputs, bucket, dispatcher_stream_);
    auto rows = loader_set_.unpack_prepacked_outputs(raw, ready, bucket);
    (void)rows;
    B2_CUDA_CHECK(cudaStreamSynchronize(dispatcher_stream_.stream()));
    {
      std::lock_guard<std::mutex> telemetry_lock(telemetry_mutex_);
      ++telemetry_.warmup_runs;
    }
    std::printf("B2_SCHEDULER_WARMUP bucket=%d rows=%zu\n", bucket, ready.size());
  }
}

std::vector<int> BatchedSteadyScheduler::required_buckets() const {
  return required_buckets_for_policy(policy_);
}

std::vector<int> BatchedSteadyScheduler::required_buckets_for_policy(
    const BatchedSteadySchedulerPolicy& policy) {
  std::vector<int> buckets{1};
  auto add = [&](int bucket) {
    if (std::find(buckets.begin(), buckets.end(), bucket) == buckets.end()) {
      buckets.push_back(bucket);
    }
  };
  if (policy.B_max != 1 && policy.B_max != 2 && policy.B_max != 4) {
    throw std::runtime_error("batch steady B_max must be one of {1,2,4}");
  }
  if (policy.B_max >= 2) add(policy.use_b2_bucket ? 2 : 4);
  if (policy.B_max >= 4) add(4);
  return buckets;
}

int BatchedSteadyScheduler::dispatch_bucket_for_k(int k) const {
  int bucket = BatchedSteadyLoaderSet::bucket_for_k_public(k);
  if (bucket > policy_.B_max) bucket = policy_.B_max;
  bucket = BatchedSteadyLoaderSet::bucket_for_k_public(bucket);
  if (!policy_.use_b2_bucket && bucket == 2) return 4;
  return bucket;
}

void BatchedSteadyScheduler::record_worker_wait(int64_t cycle_id,
                                                int k,
                                                double output_sync_us,
                                                double worker_blocked_us,
                                                double completion_wait_us) {
  std::lock_guard<std::mutex> lock(telemetry_mutex_);
  telemetry_.output_sync_us.push_back(output_sync_us);
  telemetry_.worker_blocked_us.push_back(worker_blocked_us);
  if (completion_wait_us >= 0.0) telemetry_.completion_wait_us.push_back(completion_wait_us);
  if (cycle_id < 0 || k <= 0) return;

  auto expected_it = worker_wait_expected_by_cycle_.find(cycle_id);
  if (expected_it == worker_wait_expected_by_cycle_.end()) {
    expected_it = worker_wait_expected_by_cycle_.emplace(cycle_id, k).first;
  } else {
    expected_it->second = std::max(expected_it->second, k);
  }

  auto& waits = worker_waits_by_cycle_[cycle_id];
  waits.push_back(worker_blocked_us);
  if (static_cast<int>(waits.size()) >= expected_it->second) {
    auto minmax = std::minmax_element(waits.begin(), waits.end());
    telemetry_.per_stream_fairness_spread_us.push_back(*minmax.second - *minmax.first);
    worker_waits_by_cycle_.erase(cycle_id);
    worker_wait_expected_by_cycle_.erase(cycle_id);
  }
}

BatchedSteadySchedulerTelemetry BatchedSteadyScheduler::telemetry_snapshot() const {
  std::lock_guard<std::mutex> lock(telemetry_mutex_);
  return telemetry_;
}

void BatchedSteadyScheduler::dispatcher_loop() {
  const std::thread::id dispatcher_thread_id = std::this_thread::get_id();
  try {
    c10::cuda::CUDAGuard device_guard(device_.index());
    c10::cuda::CUDAStreamGuard stream_guard(dispatcher_stream_);
    torch::NoGradGuard no_grad;
    while (true) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        drain_pending_dispatches_locked(false);
      }
      drain_dispatch_timing_events(false);
      auto batch = gather_batch();
      if (batch.empty()) break;
      dispatch_batch(std::move(batch), dispatcher_thread_id);
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      drain_pending_dispatches_locked(true);
    }
    drain_dispatch_timing_events(true);
  } catch (...) {
    try {
      drain_dispatch_timing_events(true);
    } catch (...) {
    }
    try {
      std::lock_guard<std::mutex> lock(mutex_);
      drain_pending_dispatches_locked(true);
    } catch (...) {
    }
    std::exception_ptr ep = std::current_exception();
    std::vector<std::shared_ptr<QueueItem>> pending;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      fault_ = ep;
      closing_ = true;
      set_pending_exception_locked(&pending);
    }
    for (const auto& item : pending) set_item_exception(item, ep);
    {
      std::lock_guard<std::mutex> telemetry_lock(telemetry_mutex_);
      ++telemetry_.dispatcher_exceptions;
    }
    try {
      std::rethrow_exception(ep);
    } catch (const std::exception& e) {
      std::printf("B2_SCHEDULER_DISPATCHER_FATAL error=%s\n", e.what());
    } catch (...) {
      std::printf("B2_SCHEDULER_DISPATCHER_FATAL error=unknown\n");
    }
    std::fflush(stdout);
    cv_.notify_all();
    cv_capacity_.notify_all();
    return;
  }
}

std::vector<std::shared_ptr<BatchedSteadyScheduler::QueueItem>> BatchedSteadyScheduler::gather_batch() {
  std::vector<std::shared_ptr<QueueItem>> batch;
  std::unique_lock<std::mutex> lock(mutex_);
  auto ready_pred = [&] {
    drain_pending_dispatches_locked(false);
    return closing_ || fault_ || !queue_.empty();
  };
  while (!ready_pred()) {
    bool needs_timed_poll = !pending_dispatches_.empty() ||
                            (dispatch_timing_mode_ == DispatchTimingMode::Poll &&
                             !pending_dispatch_timings_.empty());
    if (needs_timed_poll) {
      lock.unlock();
      if (dispatch_timing_mode_ == DispatchTimingMode::Poll) {
        drain_dispatch_timing_events(false);
      }
      lock.lock();
      if (ready_pred()) break;
      cv_.wait_for(lock,
                   ms_duration(std::min(kDispatchTimingIdlePollMs, kPendingDispatchIdlePollMs)),
                   ready_pred);
    } else {
      cv_.wait(lock, ready_pred);
    }
  }
  if (fault_) std::rethrow_exception(fault_);
  if (closing_ && queue_.empty()) return batch;
  {
    std::lock_guard<std::mutex> telemetry_lock(telemetry_mutex_);
    telemetry_.queue_depth.push_back(static_cast<double>(queue_.size()));
  }

  auto pop_one = [&] {
    auto item = queue_.front();
    queue_.pop_front();
    item->pop_time = Clock::now();
    batch.push_back(item);
  };

  auto first_pop_time = Clock::now();
  (void)first_pop_time;
  pop_one();

  auto gather_start = Clock::now();
  auto window_deadline = gather_start + ms_duration(policy_.window_ms);
  while (static_cast<int>(batch.size()) < policy_.B_max) {
    if (!queue_.empty()) {
      pop_one();
      continue;
    }
    if (closing_) break;

    int wait_ms = static_cast<int>(batch.size()) == 1 ? policy_.lone_timeout_ms : policy_.window_ms;
    if (wait_ms <= 0) break;
    auto deadline = static_cast<int>(batch.size()) == 1
                        ? Clock::now() + ms_duration(wait_ms)
                        : window_deadline;
    auto before_wait = Clock::now();
    bool woke = false;
    while (Clock::now() < deadline) {
      drain_pending_dispatches_locked(false);
      if (closing_ || fault_ || !queue_.empty()) {
        woke = true;
        break;
      }
      auto poll_deadline = std::min(deadline, Clock::now() + ms_duration(kPendingDispatchIdlePollMs));
      woke = cv_.wait_until(lock, poll_deadline, [&] {
        drain_pending_dispatches_locked(false);
        return closing_ || fault_ || !queue_.empty();
      });
      if (woke) break;
    }
    auto after_wait = Clock::now();
    if (fault_) std::rethrow_exception(fault_);
    if (woke) {
      continue;
    }
    double jitter = elapsed_us(deadline, after_wait);
    if (jitter > 0.0) {
      std::lock_guard<std::mutex> telemetry_lock(telemetry_mutex_);
      telemetry_.window_wakeup_jitter_us.push_back(jitter);
    }
    (void)before_wait;
    break;
  }
  return batch;
}

void BatchedSteadyScheduler::dispatch_batch(std::vector<std::shared_ptr<QueueItem>> batch,
                                            std::thread::id dispatcher_thread_id) {
  if (batch.empty()) return;
  int k = static_cast<int>(batch.size());
  bool popped_batch_guard_armed = true;
  auto cleanup_popped_batch = [&](std::exception_ptr ep) noexcept {
    if (!popped_batch_guard_armed) return;
    popped_batch_guard_armed = false;

    CudaEventOwner cleanup_done;
    if (cudaEventCreateWithFlags(&cleanup_done.event, cudaEventDisableTiming) == cudaSuccess &&
        cudaEventRecord(cleanup_done.event, dispatcher_stream_.stream()) == cudaSuccess) {
      (void)cudaEventSynchronize(cleanup_done.event);
    }

    for (const auto& item : batch) {
      if (item->request.producer_event != nullptr) {
        (void)cudaEventDestroy(item->request.producer_event);
        item->request.producer_event = nullptr;
      }
    }

    try {
      std::lock_guard<std::mutex> lock(mutex_);
      release_capacity_tokens_locked(k);
    } catch (...) {
    }

    if (!ep) {
      try {
        ep = std::make_exception_ptr(
            std::runtime_error("batch steady dispatch abandoned before pending record install"));
      } catch (...) {
        ep = std::current_exception();
      }
    }
    for (const auto& item : batch) {
      set_item_exception(item, ep);
    }
    batch.clear();
  };
  auto popped_batch_guard = make_scope_exit([&]() noexcept {
    cleanup_popped_batch(nullptr);
  });

  try {
    assert(dispatcher_stream_.stream() != nullptr);
    assert(dispatcher_thread_id == std::this_thread::get_id());
    auto dispatch_wall_start = Clock::now();
    double dispatch_cpu_start_us = current_thread_cpu_us();
    c10::cuda::CUDAStreamGuard stream_guard(dispatcher_stream_);
    int bucket = dispatch_bucket_for_k(k);

    bool backlog = false;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      backlog = !queue_.empty();
      drain_pending_dispatches_locked(false);
    }

    drain_dispatch_timing_events(false);

    std::vector<BatchedSteadyInput> ready;
    ready.reserve(batch.size());
    for (const auto& item : batch) {
      B2_CUDA_CHECK(cudaStreamWaitEvent(dispatcher_stream_.stream(), item->request.producer_event, 0));
      ready.push_back(item->request.input);
    }
    for (const auto& item : batch) {
      B2_CUDA_CHECK(cudaEventDestroy(item->request.producer_event));
      item->request.producer_event = nullptr;
    }

    auto inputs = pack_into_scratch(ready, bucket);
    auto service_start = Clock::now();
    std::vector<double> gather_waits;
    std::vector<double> service_waits;
    gather_waits.reserve(batch.size());
    service_waits.reserve(batch.size());
    for (const auto& item : batch) {
      gather_waits.push_back(elapsed_us(item->enqueue_time, item->pop_time));
      service_waits.push_back(elapsed_us(item->pop_time, service_start));
    }

    cudaEvent_t ev_start{};
    cudaEvent_t ev_stop{};
    const bool timing_enabled = dispatch_timing_mode_ != DispatchTimingMode::Off;
    if (timing_enabled) {
      B2_CUDA_CHECK(cudaEventCreate(&ev_start));
      B2_CUDA_CHECK(cudaEventCreate(&ev_stop));
      B2_CUDA_CHECK(cudaEventRecord(ev_start, dispatcher_stream_.stream()));
    }
    // Stream-order invariant: scratch reuse is safe because pack_into_scratch()
    // above, run_raw_prepacked(), unpack_prepacked_outputs(), and the next
    // pack_into_scratch() are all issued by this single dispatcher thread onto
    // dispatcher_stream_. The timing sync below is telemetry only; correctness
    // does not rely on it.
    auto raw = loader_set_.run_raw_prepacked(inputs, bucket, dispatcher_stream_);
    double cuda_run_us = -1.0;
    if (timing_enabled) {
      B2_CUDA_CHECK(cudaEventRecord(ev_stop, dispatcher_stream_.stream()));
      if (dispatch_timing_mode_ == DispatchTimingMode::Sync) {
        // This host-sync is only for CUDA elapsed-time telemetry. Step 1a keeps
        // sync as the default so the default dispatcher path remains behaviorally
        // identical while poll/off are opt-in measurement modes.
        B2_CUDA_CHECK(cudaEventSynchronize(ev_stop));
        float elapsed_ms = 0.0f;
        B2_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));
        B2_CUDA_CHECK(cudaEventDestroy(ev_start));
        B2_CUDA_CHECK(cudaEventDestroy(ev_stop));
        ev_start = nullptr;
        ev_stop = nullptr;
        cuda_run_us = static_cast<double>(elapsed_ms) * 1000.0;
      } else {
        pending_dispatch_timings_.push_back({ev_start, ev_stop, k});
        ev_start = nullptr;
        ev_stop = nullptr;
        cap_pending_dispatch_timing_events();
      }
    }

    auto rows = loader_set_.unpack_prepacked_outputs(raw, ready, bucket);
    if (rows.size() != batch.size()) {
      throw std::runtime_error("batch steady dispatch returned wrong row count");
    }

    int64_t cycle_id = 0;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      cycle_id = next_cycle_id_++;
    }
    auto dispatch_wall_end = Clock::now();
    double dispatch_cpu_end_us = current_thread_cpu_us();
    add_dispatch_telemetry(bucket,
                           cycle_id,
                           k,
                           backlog,
                           gather_waits,
                           service_waits,
                           cuda_run_us,
                           0.0,
                           dispatch_wall_start,
                           dispatch_wall_end,
                           dispatch_cpu_start_us,
                           dispatch_cpu_end_us);

    std::vector<DispatchResult> results;
    results.reserve(batch.size());
    for (size_t i = 0; i < batch.size(); ++i) {
      CudaEventOwner completion;
      B2_CUDA_CHECK(cudaEventCreateWithFlags(&completion.event, cudaEventDisableTiming));
      DispatchResult result;
      result.completion.reset(completion.release());
      result.row_tensors = own_row_output_tensors(std::move(rows[i].tensors), raw);
      B2_CUDA_CHECK(cudaEventRecord(result.completion.get(), dispatcher_stream_.stream()));
      result.bucket = bucket;
      result.row = static_cast<int>(i);
      result.k = k;
      result.cycle_id = cycle_id;
      result.gather_wait_us = gather_waits[i];
      result.service_wait_us = service_waits[i];
      result.cuda_run_us = cuda_run_us >= 0.0 ? cuda_run_us : 0.0;
      result.label = rows[i].label;
      results.push_back(std::move(result));
    }

    CudaEventOwner dispatch_done;
    B2_CUDA_CHECK(cudaEventCreateWithFlags(&dispatch_done.event, cudaEventDisableTiming));
    B2_CUDA_CHECK(cudaEventRecord(dispatch_done.event, dispatcher_stream_.stream()));
    {
      PendingDispatch pending;
      pending.dispatch_done = dispatch_done.release();
      pending.input_owners = batch;
      pending.capacity_tokens = k;
      std::lock_guard<std::mutex> lock(mutex_);
      pending_dispatches_.push_back(std::move(pending));
      popped_batch_guard_armed = false;
      popped_batch_guard.dismiss();
      cap_pending_dispatches_locked();
      drain_pending_dispatches_locked(false);
    }

    for (size_t i = 0; i < batch.size(); ++i) {
      batch[i]->promise.set_value(std::move(results[i]));
    }
  } catch (...) {
    cleanup_popped_batch(std::current_exception());
    throw;
  }
}

bool BatchedSteadyScheduler::drain_one_dispatch_timing_event(bool force) {
  if (pending_dispatch_timings_.empty()) return false;
  auto& pending = pending_dispatch_timings_.front();
  if (!force) {
    cudaError_t ready = cudaEventQuery(pending.ev_stop);
    if (ready == cudaErrorNotReady) return false;
    B2_CUDA_CHECK(ready);
  } else {
    B2_CUDA_CHECK(cudaEventSynchronize(pending.ev_stop));
  }
  float elapsed_ms = 0.0f;
  B2_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, pending.ev_start, pending.ev_stop));
  B2_CUDA_CHECK(cudaEventDestroy(pending.ev_start));
  B2_CUDA_CHECK(cudaEventDestroy(pending.ev_stop));
  {
    std::lock_guard<std::mutex> telemetry_lock(telemetry_mutex_);
    add_dispatch_timing_telemetry_locked(pending.k, static_cast<double>(elapsed_ms) * 1000.0, Clock::now());
  }
  pending_dispatch_timings_.pop_front();
  return true;
}

void BatchedSteadyScheduler::drain_dispatch_timing_events(bool force) {
  while (drain_one_dispatch_timing_event(force)) {
  }
}

void BatchedSteadyScheduler::cap_pending_dispatch_timing_events() {
  while (pending_dispatch_timings_.size() > kMaxPendingDispatchTimings) {
    (void)drain_one_dispatch_timing_event(true);
  }
}

bool BatchedSteadyScheduler::capacity_available_locked() const {
  return capacity_tokens_in_use_ < policy_.queue_capacity;
}

void BatchedSteadyScheduler::release_capacity_tokens_locked(int tokens) {
  if (tokens <= 0) return;
  assert(capacity_tokens_in_use_ >= tokens);
  capacity_tokens_in_use_ = std::max(0, capacity_tokens_in_use_ - tokens);
  cv_capacity_.notify_all();
}

bool BatchedSteadyScheduler::drain_one_pending_dispatch_locked(bool force) {
  if (pending_dispatches_.empty()) return false;
  auto& pending = pending_dispatches_.front();
  if (!force) {
    cudaError_t ready = cudaEventQuery(pending.dispatch_done);
    if (ready == cudaErrorNotReady) return false;
    B2_CUDA_CHECK(ready);
  } else {
    B2_CUDA_CHECK(cudaEventSynchronize(pending.dispatch_done));
  }

  PendingDispatch retired = std::move(pending);
  pending_dispatches_.pop_front();
  B2_CUDA_CHECK(cudaEventDestroy(retired.dispatch_done));
  retired.dispatch_done = nullptr;
  release_capacity_tokens_locked(retired.capacity_tokens);
  retired.input_owners.clear();
  return true;
}

void BatchedSteadyScheduler::drain_pending_dispatches_locked(bool force) {
  while (drain_one_pending_dispatch_locked(force)) {
  }
}

void BatchedSteadyScheduler::cap_pending_dispatches_locked() {
  while (pending_dispatches_.size() > kMaxPendingDispatches) {
    (void)drain_one_pending_dispatch_locked(true);
  }
}

std::vector<at::Tensor> BatchedSteadyScheduler::pack_into_scratch(const std::vector<BatchedSteadyInput>& ready,
                                                                  int bucket) {
  if (ready.empty()) throw std::runtime_error("batch steady scratch pack called with no rows");
  auto& scratch = ensure_scratch(bucket, ready.front());
  for (int row = 0; row < bucket; ++row) {
    const auto& src = ready[static_cast<size_t>(row < static_cast<int>(ready.size()) ? row : 0)];
    if (src.chunk.sizes() != ready.front().chunk.sizes()) throw std::runtime_error("batch steady scratch chunk shape mismatch");
    if (src.cache_ch.sizes() != ready.front().cache_ch.sizes()) throw std::runtime_error("batch steady scratch cache_ch shape mismatch");
    if (src.cache_t.sizes() != ready.front().cache_t.sizes()) throw std::runtime_error("batch steady scratch cache_t shape mismatch");
    if (src.cache_ch_len.sizes() != ready.front().cache_ch_len.sizes()) {
      throw std::runtime_error("batch steady scratch cache_ch_len shape mismatch");
    }
    const auto& idx = scratch.row_indices[static_cast<size_t>(row)];
    scratch.chunks.index_copy_(0, idx, src.chunk);
    scratch.cache_ch.index_copy_(1, idx, src.cache_ch);
    scratch.cache_t.index_copy_(1, idx, src.cache_t);
    scratch.cache_ch_len.index_copy_(0, idx, src.cache_ch_len);
    auto len = torch::full({1}, src.chunk.size(2), long_options_for(src.chunk));
    scratch.length.index_copy_(0, idx, len);
  }
  return {
      scratch.chunks,
      scratch.length,
      scratch.cache_ch,
      scratch.cache_t,
      scratch.cache_ch_len,
  };
}

BatchedSteadyScheduler::Scratch& BatchedSteadyScheduler::ensure_scratch(int bucket,
                                                                        const BatchedSteadyInput& first) {
  auto& scratch = scratch_[bucket];
  std::vector<int64_t> chunk_shape(first.chunk.sizes().begin(), first.chunk.sizes().end());
  if (scratch.initialized && scratch.chunk_shape == chunk_shape) return scratch;

  auto chunk_options = first.chunk.options();
  auto long_options = long_options_for(first.chunk);
  scratch.chunk_shape = std::move(chunk_shape);
  scratch.chunks = torch::empty({bucket, first.chunk.size(1), first.chunk.size(2)}, chunk_options);
  scratch.length = torch::empty({bucket}, long_options);
  scratch.cache_ch = torch::empty({first.cache_ch.size(0), bucket, first.cache_ch.size(2), first.cache_ch.size(3)},
                                  first.cache_ch.options());
  scratch.cache_t = torch::empty({first.cache_t.size(0), bucket, first.cache_t.size(2), first.cache_t.size(3)},
                                 first.cache_t.options());
  scratch.cache_ch_len = torch::empty({bucket}, long_options);
  scratch.row_indices.clear();
  scratch.row_indices.reserve(static_cast<size_t>(bucket));
  for (int row = 0; row < bucket; ++row) {
    scratch.row_indices.push_back(torch::full({1}, row, long_options));
  }
  scratch.initialized = true;
  return scratch;
}

void BatchedSteadyScheduler::set_pending_exception_locked(std::vector<std::shared_ptr<QueueItem>>* pending) {
  int released = 0;
  while (!queue_.empty()) {
    pending->push_back(queue_.front());
    queue_.pop_front();
    ++released;
  }
  release_capacity_tokens_locked(released);
}

void BatchedSteadyScheduler::set_item_exception(const std::shared_ptr<QueueItem>& item, std::exception_ptr ep) {
  if (item->request.producer_event != nullptr) {
    cudaEventDestroy(item->request.producer_event);
    item->request.producer_event = nullptr;
  }
  try {
    item->promise.set_exception(ep);
  } catch (const std::future_error&) {
  }
}

void BatchedSteadyScheduler::add_dispatch_telemetry(int bucket,
                                                    int64_t cycle_id,
                                                    int k,
                                                    bool backlog,
                                                    const std::vector<double>& gather_wait_us,
                                                    const std::vector<double>& service_wait_us,
                                                    double cuda_run_us,
                                                    double wakeup_jitter_us,
                                                    Clock::time_point dispatch_wall_start,
                                                    Clock::time_point dispatch_wall_end,
                                                    double dispatch_cpu_start_us,
                                                    double dispatch_cpu_end_us) {
  std::lock_guard<std::mutex> lock(telemetry_mutex_);
  ++telemetry_.dispatch_cycles;
  telemetry_.completed += k;
  if (bucket == 1) ++telemetry_.bucket_b1;
  if (bucket == 2) ++telemetry_.bucket_b2;
  if (bucket == 4) ++telemetry_.bucket_b4;
  if (k == 2 && bucket == 4) ++telemetry_.k2_padded_to_b4;
  if (k == 3 && bucket == 4) ++telemetry_.k3_padded_to_b4;
  if (k == 4) ++telemetry_.k4;
  if (backlog) ++telemetry_.backlog_gt_bmax;
  telemetry_.gather_wait_us.insert(telemetry_.gather_wait_us.end(), gather_wait_us.begin(), gather_wait_us.end());
  telemetry_.service_wait_us.insert(telemetry_.service_wait_us.end(), service_wait_us.begin(), service_wait_us.end());
  if (wakeup_jitter_us > 0.0) telemetry_.window_wakeup_jitter_us.push_back(wakeup_jitter_us);
  if (!dispatcher_measurement_started_) {
    dispatcher_measurement_started_ = true;
    dispatcher_measurement_wall_start_ = dispatch_wall_start;
    dispatcher_measurement_cpu_start_us_ = dispatch_cpu_start_us;
  }
  telemetry_.dispatcher_wall_us = elapsed_us(dispatcher_measurement_wall_start_, dispatch_wall_end);
  if (dispatcher_measurement_cpu_start_us_ >= 0.0 && dispatch_cpu_end_us >= dispatcher_measurement_cpu_start_us_) {
    telemetry_.dispatcher_cpu_us = dispatch_cpu_end_us - dispatcher_measurement_cpu_start_us_;
  }
  worker_wait_expected_by_cycle_[cycle_id] = std::max(worker_wait_expected_by_cycle_[cycle_id], k);
  if (cuda_run_us >= 0.0) add_dispatch_timing_telemetry_locked(k, cuda_run_us, dispatch_wall_end);
}

void BatchedSteadyScheduler::add_dispatch_timing_telemetry_locked(int k,
                                                                  double cuda_run_us,
                                                                  Clock::time_point timing_wall_end) {
  for (int i = 0; i < k; ++i) telemetry_.cuda_run_us.push_back(cuda_run_us);
  telemetry_.dispatcher_stream_run_us += cuda_run_us;
  if (dispatcher_measurement_started_) {
    telemetry_.dispatcher_wall_us =
        std::max(telemetry_.dispatcher_wall_us, elapsed_us(dispatcher_measurement_wall_start_, timing_wall_end));
  }
}

#undef B2_CUDA_CHECK
