#include "batched_steady_scheduler.h"

#include <torch/torch.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <sstream>

#define B2_CUDA_CHECK(expr) BatchedSteadyScheduler::cuda_check((expr), #expr, __FILE__, __LINE__)

namespace {
using Clock = std::chrono::steady_clock;

double elapsed_us(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::micro>(end - start).count();
}

std::chrono::milliseconds ms_duration(int ms) {
  return std::chrono::milliseconds(std::max(0, ms));
}

torch::TensorOptions long_options_for(const torch::Tensor& tensor) {
  return torch::TensorOptions().dtype(torch::kLong).device(tensor.device());
}
}  // namespace

BatchedSteadyScheduler::BatchedSteadyScheduler(BatchedSteadyLoaderSet& loader_set,
                                               torch::Device device,
                                               BatchedSteadySchedulerPolicy policy)
    : loader_set_(loader_set),
      device_(device),
      policy_(policy),
      dispatcher_stream_(c10::cuda::getStreamFromPool(/*isHighPriority=*/false,
                                                      device.index() >= 0 ? device.index() : 0)) {
  if (policy_.window_ms < 0) throw std::runtime_error("batch steady window_ms must be non-negative");
  if (policy_.lone_timeout_ms < 0) throw std::runtime_error("batch steady lone_timeout_ms must be non-negative");
  if (policy_.B_max != 1 && policy_.B_max != 2 && policy_.B_max != 4) {
    throw std::runtime_error("batch steady B_max must be one of {1,2,4}");
  }
  if (policy_.queue_capacity <= 0) policy_.queue_capacity = 16;
  loader_set_.preload_all();
  if (!loader_set_.sealed()) throw std::runtime_error("batch steady loader set failed to seal after preload_all()");
  std::printf("B2_SCHEDULER_CONSTRUCT policy={B_max:%d,window_ms:%d,lone_timeout_ms:%d,queue_capacity:%d} "
              "loaded_buckets=%d dispatcher_stream=%p\n",
              policy_.B_max,
              policy_.window_ms,
              policy_.lone_timeout_ms,
              policy_.queue_capacity,
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

std::future<DispatchResult> BatchedSteadyScheduler::enqueue(EnqueueRequest&& request) {
  if (request.producer_event == nullptr) {
    throw std::runtime_error("batch steady enqueue requires a producer-ready CUDA event");
  }
  auto item = std::make_shared<QueueItem>(std::move(request));
  item->enqueue_time = Clock::now();
  auto future = item->promise.get_future();

  std::unique_lock<std::mutex> lock(mutex_);
  cv_capacity_.wait(lock, [&] {
    return closing_ || fault_ || static_cast<int>(queue_.size()) < policy_.queue_capacity;
  });
  if (fault_) std::rethrow_exception(fault_);
  if (closing_) throw std::runtime_error("batch steady enqueue after scheduler close");
  item->sequence = next_sequence_++;
  queue_.push_back(item);
  {
    std::lock_guard<std::mutex> telemetry_lock(telemetry_mutex_);
    ++telemetry_.enqueued;
  }
  cv_.notify_one();
  return future;
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
      auto ep = std::make_exception_ptr(std::runtime_error("batch steady scheduler closed with pending work"));
      set_pending_exception_locked(ep, &pending);
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
  for (int bucket : {1, 2, 4}) {
    if (bucket > policy_.B_max && policy_.B_max != 4) {
      // The loader set is still preloaded for all buckets, but a B_max=1/2 policy
      // only needs the buckets it can dispatch in this run's measured path.
    }
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

void BatchedSteadyScheduler::record_worker_wait(double output_sync_us, double worker_blocked_us) {
  std::lock_guard<std::mutex> lock(telemetry_mutex_);
  telemetry_.output_sync_us.push_back(output_sync_us);
  telemetry_.worker_blocked_us.push_back(worker_blocked_us);
}

BatchedSteadySchedulerTelemetry BatchedSteadyScheduler::telemetry_snapshot() const {
  std::lock_guard<std::mutex> lock(telemetry_mutex_);
  return telemetry_;
}

void BatchedSteadyScheduler::dispatcher_loop() {
  try {
    c10::cuda::CUDAGuard device_guard(device_.index());
    c10::cuda::CUDAStreamGuard stream_guard(dispatcher_stream_);
    torch::NoGradGuard no_grad;
    while (true) {
      auto batch = gather_batch();
      if (batch.empty()) break;
      dispatch_batch(batch);
    }
  } catch (...) {
    std::exception_ptr ep = std::current_exception();
    std::vector<std::shared_ptr<QueueItem>> pending;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      fault_ = ep;
      closing_ = true;
      set_pending_exception_locked(ep, &pending);
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
    std::exit(1);
  }
}

std::vector<std::shared_ptr<BatchedSteadyScheduler::QueueItem>> BatchedSteadyScheduler::gather_batch() {
  std::vector<std::shared_ptr<QueueItem>> batch;
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [&] { return closing_ || fault_ || !queue_.empty(); });
  if (fault_) std::rethrow_exception(fault_);
  if (closing_ && queue_.empty()) return batch;

  auto pop_one = [&] {
    auto item = queue_.front();
    queue_.pop_front();
    item->pop_time = Clock::now();
    batch.push_back(item);
    cv_capacity_.notify_one();
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
    bool woke = cv_.wait_until(lock, deadline, [&] { return closing_ || fault_ || !queue_.empty(); });
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

void BatchedSteadyScheduler::dispatch_batch(const std::vector<std::shared_ptr<QueueItem>>& batch) {
  if (batch.empty()) return;
  c10::cuda::CUDAStreamGuard stream_guard(dispatcher_stream_);
  int k = static_cast<int>(batch.size());
  int bucket = BatchedSteadyLoaderSet::bucket_for_k_public(k);
  if (bucket > policy_.B_max) bucket = policy_.B_max;
  bucket = BatchedSteadyLoaderSet::bucket_for_k_public(bucket);

  bool backlog = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    backlog = !queue_.empty();
  }

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

  auto service_start = Clock::now();
  auto inputs = pack_into_scratch(ready, bucket);
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
  B2_CUDA_CHECK(cudaEventCreate(&ev_start));
  B2_CUDA_CHECK(cudaEventCreate(&ev_stop));
  B2_CUDA_CHECK(cudaEventRecord(ev_start, dispatcher_stream_.stream()));
  auto raw = loader_set_.run_raw_prepacked(inputs, bucket, dispatcher_stream_);
  B2_CUDA_CHECK(cudaEventRecord(ev_stop, dispatcher_stream_.stream()));
  B2_CUDA_CHECK(cudaEventSynchronize(ev_stop));
  float elapsed_ms = 0.0f;
  B2_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));
  B2_CUDA_CHECK(cudaEventDestroy(ev_start));
  B2_CUDA_CHECK(cudaEventDestroy(ev_stop));
  double cuda_run_us = static_cast<double>(elapsed_ms) * 1000.0;

  auto rows = loader_set_.unpack_prepacked_outputs(raw, ready, bucket);
  if (rows.size() != batch.size()) {
    throw std::runtime_error("batch steady dispatch returned wrong row count");
  }

  int64_t cycle_id = 0;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    cycle_id = next_cycle_id_++;
  }
  add_dispatch_telemetry(bucket, k, backlog, gather_waits, service_waits, cuda_run_us, 0.0);

  for (size_t i = 0; i < batch.size(); ++i) {
    cudaEvent_t completion{};
    B2_CUDA_CHECK(cudaEventCreateWithFlags(&completion, cudaEventDisableTiming));
    B2_CUDA_CHECK(cudaEventRecord(completion, dispatcher_stream_.stream()));
    DispatchResult result;
    result.row_tensors = std::move(rows[i].tensors);
    result.bucket = bucket;
    result.row = static_cast<int>(i);
    result.k = k;
    result.cycle_id = cycle_id;
    result.completion = completion;
    result.gather_wait_us = gather_waits[i];
    result.service_wait_us = service_waits[i];
    result.cuda_run_us = cuda_run_us;
    result.label = rows[i].label;
    batch[i]->promise.set_value(std::move(result));
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

void BatchedSteadyScheduler::set_pending_exception_locked(
    std::exception_ptr ep,
    std::vector<std::shared_ptr<QueueItem>>* pending) {
  (void)ep;
  while (!queue_.empty()) {
    pending->push_back(queue_.front());
    queue_.pop_front();
  }
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
                                                    int k,
                                                    bool backlog,
                                                    const std::vector<double>& gather_wait_us,
                                                    const std::vector<double>& service_wait_us,
                                                    double cuda_run_us,
                                                    double wakeup_jitter_us) {
  std::lock_guard<std::mutex> lock(telemetry_mutex_);
  ++telemetry_.dispatch_cycles;
  telemetry_.completed += k;
  if (bucket == 1) ++telemetry_.bucket_b1;
  if (bucket == 2) ++telemetry_.bucket_b2;
  if (bucket == 4) ++telemetry_.bucket_b4;
  if (k == 3 && bucket == 4) ++telemetry_.k3_padded_to_b4;
  if (k == 4) ++telemetry_.k4;
  if (backlog) ++telemetry_.backlog_gt_bmax;
  telemetry_.gather_wait_us.insert(telemetry_.gather_wait_us.end(), gather_wait_us.begin(), gather_wait_us.end());
  telemetry_.service_wait_us.insert(telemetry_.service_wait_us.end(), service_wait_us.begin(), service_wait_us.end());
  for (int i = 0; i < k; ++i) telemetry_.cuda_run_us.push_back(cuda_run_us);
  if (wakeup_jitter_us > 0.0) telemetry_.window_wakeup_jitter_us.push_back(wakeup_jitter_us);
}

#undef B2_CUDA_CHECK
