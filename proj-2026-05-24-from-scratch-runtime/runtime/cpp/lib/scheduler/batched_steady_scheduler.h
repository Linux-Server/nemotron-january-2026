#pragma once

#include "lib/scheduler/steady_batch_primitive.h"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

struct BatchedSteadySchedulerPolicy {
  int window_ms = 10;
  int lone_timeout_ms = 0;
  int B_max = 4;
  int queue_capacity = 16;
  bool use_b2_bucket = false;
};

struct EnqueueRequest {
  BatchedSteadyInput input;
  c10::cuda::CUDAStream producer;
  cudaEvent_t producer_event = nullptr;
};

struct DispatchResult {
  struct CompletionEvent {
    cudaEvent_t event = nullptr;

    CompletionEvent() = default;
    explicit CompletionEvent(cudaEvent_t e) : event(e) {}
    CompletionEvent(const CompletionEvent&) = delete;
    CompletionEvent& operator=(const CompletionEvent&) = delete;

    CompletionEvent(CompletionEvent&& other) noexcept : event(other.event) {
      other.event = nullptr;
    }

    CompletionEvent& operator=(CompletionEvent&& other) noexcept {
      if (this != &other) {
        reset();
        event = other.event;
        other.event = nullptr;
      }
      return *this;
    }

    ~CompletionEvent() {
      reset();
    }

    cudaEvent_t get() const {
      return event;
    }

    void reset(cudaEvent_t next = nullptr) noexcept {
      if (event != nullptr) cudaEventDestroy(event);
      event = next;
    }
  };

  std::vector<at::Tensor> row_tensors;
  int bucket = 0;
  int row = 0;
  int k = 0;
  int64_t cycle_id = 0;
  CompletionEvent completion;
  double gather_wait_us = 0.0;
  double service_wait_us = 0.0;
  double cuda_run_us = 0.0;
  std::string label;
};

struct BatchedSteadySchedulerTelemetry {
  int64_t enqueued = 0;
  int64_t completed = 0;
  int64_t dispatch_cycles = 0;
  int64_t warmup_runs = 0;
  int64_t bucket_b1 = 0;
  int64_t bucket_b2 = 0;
  int64_t bucket_b4 = 0;
  int64_t k2_padded_to_b4 = 0;
  int64_t k3_padded_to_b4 = 0;
  int64_t k4 = 0;
  int64_t backlog_gt_bmax = 0;
  int64_t dispatcher_exceptions = 0;
  double dispatcher_cpu_us = 0.0;
  double dispatcher_wall_us = 0.0;
  double dispatcher_stream_run_us = 0.0;
  std::vector<double> gather_wait_us;
  std::vector<double> service_wait_us;
  std::vector<double> cuda_run_us;
  std::vector<double> output_sync_us;
  std::vector<double> worker_blocked_us;
  std::vector<double> window_wakeup_jitter_us;
  std::vector<double> queue_depth;
  std::vector<double> per_stream_fairness_spread_us;
};

class BatchedSteadyScheduler {
 public:
  BatchedSteadyScheduler(BatchedSteadyLoaderSet& loader_set,
                         torch::Device device,
                         BatchedSteadySchedulerPolicy policy);
  BatchedSteadyScheduler(const BatchedSteadyScheduler&) = delete;
  BatchedSteadyScheduler& operator=(const BatchedSteadyScheduler&) = delete;
  ~BatchedSteadyScheduler();

  std::future<DispatchResult> enqueue(EnqueueRequest&& request);
  std::optional<std::future<DispatchResult>> try_enqueue_until(
      EnqueueRequest&& request,
      std::chrono::steady_clock::time_point deadline);
  void start();
  void close();
  void warmup_buckets();
  void record_worker_wait(int64_t cycle_id, int k, double output_sync_us, double worker_blocked_us);

  BatchedSteadySchedulerTelemetry telemetry_snapshot() const;
  const BatchedSteadySchedulerPolicy& policy() const { return policy_; }
  int future_timeout_ms() const;

 private:
  using Clock = std::chrono::steady_clock;

  struct QueueItem {
    explicit QueueItem(EnqueueRequest&& r) : request(std::move(r)) {}

    EnqueueRequest request;
    std::promise<DispatchResult> promise;
    Clock::time_point enqueue_time;
    Clock::time_point pop_time;
    int64_t sequence = 0;
  };

  struct Scratch {
    bool initialized = false;
    std::vector<int64_t> chunk_shape;
    torch::Tensor chunks;
    torch::Tensor length;
    torch::Tensor cache_ch;
    torch::Tensor cache_t;
    torch::Tensor cache_ch_len;
    std::vector<torch::Tensor> row_indices;
  };

  void dispatcher_loop();
  std::vector<int> required_buckets() const;
  int dispatch_bucket_for_k(int k) const;
  std::vector<std::shared_ptr<QueueItem>> gather_batch();
  void dispatch_batch(const std::vector<std::shared_ptr<QueueItem>>& batch);
  std::vector<at::Tensor> pack_into_scratch(const std::vector<BatchedSteadyInput>& ready, int bucket);
  Scratch& ensure_scratch(int bucket, const BatchedSteadyInput& first);
  void set_pending_exception_locked(std::vector<std::shared_ptr<QueueItem>>* pending);
  void set_item_exception(const std::shared_ptr<QueueItem>& item, std::exception_ptr ep);
  void add_dispatch_telemetry(int bucket,
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
                              double dispatch_cpu_end_us);
  static void cuda_check(cudaError_t err, const char* expr, const char* file, int line);

  BatchedSteadyLoaderSet& loader_set_;
  torch::Device device_;
  BatchedSteadySchedulerPolicy policy_;
  c10::cuda::CUDAStream dispatcher_stream_;

  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::condition_variable cv_capacity_;
  std::deque<std::shared_ptr<QueueItem>> queue_;
  bool started_ = false;
  bool closing_ = false;
  bool closed_ = false;
  int64_t next_sequence_ = 0;
  int64_t next_cycle_id_ = 0;
  std::exception_ptr fault_;
  std::thread dispatcher_thread_;

  mutable std::mutex telemetry_mutex_;
  BatchedSteadySchedulerTelemetry telemetry_;
  bool dispatcher_measurement_started_ = false;
  Clock::time_point dispatcher_measurement_wall_start_;
  double dispatcher_measurement_cpu_start_us_ = 0.0;
  std::map<int64_t, int> worker_wait_expected_by_cycle_;
  std::map<int64_t, std::vector<double>> worker_waits_by_cycle_;
  std::map<int, Scratch> scratch_;
};
