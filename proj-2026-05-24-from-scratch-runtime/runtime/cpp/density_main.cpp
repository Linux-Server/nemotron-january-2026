// Phase-2 Step-0 density kill-gates.
//
// This file deliberately reuses the validated session implementation by
// compiling session_main.cpp into this translation unit with its standalone
// main symbol renamed. The harness glue below only adds concurrency,
// explicit-stream AOTI calls, timing, and telemetry.
#define main session_main_cpp_entrypoint_disabled
#include "session_main.cpp"
#undef main

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <ctime>
#include <mutex>
#include <numeric>
#include <sys/resource.h>
#include <thread>
#include <unistd.h>

using Clock = std::chrono::steady_clock;

static constexpr int kMinFinalizeP95Samples = 20;

static void cuda_check(cudaError_t err, const char* expr, const char* file, int line) {
  if (err != cudaSuccess) {
    std::ostringstream oss;
    oss << "CUDA error at " << file << ":" << line << " for " << expr
        << ": " << cudaGetErrorString(err);
    throw std::runtime_error(oss.str());
  }
}

#define CUDA_CHECK(expr) cuda_check((expr), #expr, __FILE__, __LINE__)

struct DensityArgs {
  std::string mode = "step0";
  std::string dir = "../artifacts";
  std::vector<int> n_values{1, 2, 4};
  bool n_values_set = false;
  bool target_n_set = false;
  int target_n = 16;
  int workers = 0;
  int num_runners = 0;
  int steady_cases = 32;
  int steady_repeats = 4;
  int correctness_n = 4;
  int correctness_rows = -1;
  int finalize_n = 4;
  std::string finalize_mode = "both";
  std::string stream_mode = "explicit";
  bool mutex_serialize_run = false;
  bool smoke = false;
  bool partial = false;
  bool skip_correctness = false;
  bool skip_steady = false;
  bool skip_finalize = false;
  bool default_stream_control = true;
  bool correctness_default_stream_control = true;
  bool scalar_locality_probe = true;
  bool steady_overlap_probe = true;
  int density_rows = -1;
  int density_sessions_per_worker = 0;
  double density_chunk_period_ms = 160.0;
  bool density_warmup = true;
};

static double elapsed_ms(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

static double elapsed_ms_since(Clock::time_point start) {
  return elapsed_ms(start, Clock::now());
}

static std::vector<int> parse_int_list(const std::string& text) {
  std::vector<int> out;
  size_t pos = 0;
  while (pos < text.size()) {
    size_t comma = text.find(',', pos);
    std::string item = text.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
    if (!item.empty()) out.push_back(std::stoi(item));
    if (comma == std::string::npos) break;
    pos = comma + 1;
  }
  return out;
}

static DensityArgs parse_density_args(int argc, char** argv) {
  DensityArgs args;
  bool dir_set = false;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto need_value = [&](const char* flag) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(std::string(flag) + " requires a value");
      return argv[++i];
    };
    if (arg == "--mode") {
      args.mode = need_value("--mode");
    } else if (arg == "--n-values") {
      args.n_values = parse_int_list(need_value("--n-values"));
      args.n_values_set = true;
    } else if (arg == "--target-n") {
      args.target_n = std::stoi(need_value("--target-n"));
      args.target_n_set = true;
    } else if (arg == "--workers") {
      args.workers = std::stoi(need_value("--workers"));
    } else if (arg == "--num-runners") {
      args.num_runners = std::stoi(need_value("--num-runners"));
    } else if (arg == "--steady-cases") {
      args.steady_cases = std::stoi(need_value("--steady-cases"));
    } else if (arg == "--steady-repeats") {
      args.steady_repeats = std::stoi(need_value("--steady-repeats"));
    } else if (arg == "--correctness-n") {
      args.correctness_n = std::stoi(need_value("--correctness-n"));
    } else if (arg == "--correctness-rows") {
      args.correctness_rows = std::stoi(need_value("--correctness-rows"));
    } else if (arg == "--finalize-n") {
      args.finalize_n = std::stoi(need_value("--finalize-n"));
    } else if (arg == "--finalize-mode") {
      args.finalize_mode = need_value("--finalize-mode");
    } else if (arg == "--stream-mode") {
      args.stream_mode = need_value("--stream-mode");
    } else if (arg == "--mutex-serialize-run") {
      args.mutex_serialize_run = true;
    } else if (arg == "--smoke") {
      args.smoke = true;
      args.partial = true;
    } else if (arg == "--partial") {
      args.partial = true;
    } else if (arg == "--skip-correctness") {
      args.skip_correctness = true;
    } else if (arg == "--skip-steady") {
      args.skip_steady = true;
    } else if (arg == "--skip-finalize") {
      args.skip_finalize = true;
    } else if (arg == "--no-default-stream-control") {
      args.default_stream_control = false;
      args.correctness_default_stream_control = false;
    } else if (arg == "--no-0b-default-stream-control") {
      args.correctness_default_stream_control = false;
    } else if (arg == "--no-scalar-locality-probe") {
      args.scalar_locality_probe = false;
    } else if (arg == "--no-steady-overlap-probe") {
      args.steady_overlap_probe = false;
    } else if (arg == "--density-rows") {
      args.density_rows = std::stoi(need_value("--density-rows"));
    } else if (arg == "--density-sessions-per-worker") {
      args.density_sessions_per_worker = std::stoi(need_value("--density-sessions-per-worker"));
    } else if (arg == "--density-chunk-period-ms") {
      args.density_chunk_period_ms = std::stod(need_value("--density-chunk-period-ms"));
    } else if (arg == "--no-density-warmup") {
      args.density_warmup = false;
    } else if (!dir_set) {
      args.dir = arg;
      dir_set = true;
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  if (args.mode != "step0" && args.mode != "density-sweep") {
    throw std::runtime_error("--mode must be step0 or density-sweep");
  }
  if (args.mode == "density-sweep" && !args.n_values_set) {
    args.n_values = {1, 2, 4, 8, 16};
  }
  if (args.n_values.empty()) throw std::runtime_error("--n-values cannot be empty");
  for (int n : args.n_values) {
    if (n <= 0) throw std::runtime_error("--n-values entries must be positive");
  }
  if (args.target_n > 0 &&
      (args.mode != "density-sweep" || !args.n_values_set || args.target_n_set) &&
      std::find(args.n_values.begin(), args.n_values.end(), args.target_n) == args.n_values.end()) {
    args.n_values.push_back(args.target_n);
  }
  std::sort(args.n_values.begin(), args.n_values.end());
  args.n_values.erase(std::unique(args.n_values.begin(), args.n_values.end()), args.n_values.end());
  if (args.steady_cases <= 0) throw std::runtime_error("--steady-cases must be positive");
  if (args.steady_repeats <= 0) throw std::runtime_error("--steady-repeats must be positive");
  if (args.workers < 0) throw std::runtime_error("--workers must be non-negative");
  if (args.num_runners < 0) throw std::runtime_error("--num-runners must be non-negative");
  if (args.correctness_n <= 0) throw std::runtime_error("--correctness-n must be positive");
  if (args.finalize_n <= 0) throw std::runtime_error("--finalize-n must be positive");
  if (args.stream_mode != "explicit" && args.stream_mode != "default") {
    throw std::runtime_error("--stream-mode must be explicit or default");
  }
  if (args.finalize_mode != "both" && args.finalize_mode != "same" && args.finalize_mode != "mixed") {
    throw std::runtime_error("--finalize-mode must be both, same, or mixed");
  }
  if (args.density_rows == 0 || args.density_rows < -1) {
    throw std::runtime_error("--density-rows must be positive or -1");
  }
  if (args.density_sessions_per_worker < 0) {
    throw std::runtime_error("--density-sessions-per-worker must be non-negative");
  }
  if (args.density_chunk_period_ms <= 0.0) {
    throw std::runtime_error("--density-chunk-period-ms must be positive");
  }
  return args;
}

struct SummaryStats {
  size_t n = 0;
  double p50 = 0.0;
  double p95 = 0.0;
  double p99 = 0.0;
  double mean = 0.0;
  double max = 0.0;
};

static SummaryStats summarize(std::vector<double> values) {
  SummaryStats s;
  s.n = values.size();
  if (values.empty()) return s;
  std::sort(values.begin(), values.end());
  double total = std::accumulate(values.begin(), values.end(), 0.0);
  auto pct = [&](double p) {
    size_t idx = static_cast<size_t>(std::ceil(p * static_cast<double>(values.size())) - 1.0);
    if (idx >= values.size()) idx = values.size() - 1;
    return values[idx];
  };
  s.p50 = pct(0.50);
  s.p95 = pct(0.95);
  s.p99 = pct(0.99);
  s.mean = total / static_cast<double>(values.size());
  s.max = values.back();
  return s;
}

static std::string stats_json(const SummaryStats& s) {
  std::ostringstream oss;
  oss << "{\"n\":" << s.n
      << ",\"p50_ms\":" << s.p50
      << ",\"p95_ms\":" << s.p95
      << ",\"p99_ms\":" << s.p99
      << ",\"p95_minus_p50_ms\":" << (s.p95 - s.p50)
      << ",\"p99_minus_p50_ms\":" << (s.p99 - s.p50)
      << ",\"mean_ms\":" << s.mean
      << ",\"max_ms\":" << s.max
      << "}";
  return oss.str();
}

static std::string value_stats_json(const SummaryStats& s) {
  std::ostringstream oss;
  oss << "{\"n\":" << s.n
      << ",\"p50\":" << s.p50
      << ",\"p95\":" << s.p95
      << ",\"p99\":" << s.p99
      << ",\"p95_minus_p50\":" << (s.p95 - s.p50)
      << ",\"p99_minus_p50\":" << (s.p99 - s.p50)
      << ",\"mean\":" << s.mean
      << ",\"max\":" << s.max
      << "}";
  return oss.str();
}

static std::string timestamp_utc() {
  std::time_t t = std::time(nullptr);
  std::tm tm{};
  gmtime_r(&t, &tm);
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y%m%dT%H%M%SZ", &tm);
  return buf;
}

static std::string sanitize_filename(std::string text) {
  for (char& ch : text) {
    if (!(std::isalnum(static_cast<unsigned char>(ch)) || ch == '-' || ch == '_' || ch == '.')) ch = '_';
  }
  return text;
}

static const char* json_bool(bool value) {
  return value ? "true" : "false";
}

static std::string int_list_json(const std::vector<int>& values) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0) oss << ",";
    oss << values[i];
  }
  oss << "]";
  return oss.str();
}

static void emit_telemetry(const std::string& dir,
                           const std::string& stamp,
                           int num_runners,
                           const std::string& stream_mode,
                           const std::string& topology,
                           const std::string& json) {
  std::string logs_dir = dir + "/logs/" + stamp;
  fs::create_directories(logs_dir);
  std::string path = logs_dir + "/density_num_runners" + std::to_string(num_runners) +
                     "_stream-" + sanitize_filename(stream_mode) +
                     "_topology-" + sanitize_filename(topology) + ".jsonl";
  std::ofstream out(path, std::ios::out | std::ios::app);
  if (!out) throw std::runtime_error("failed to open telemetry log: " + path);
  out << json << "\n";
  if (!out) throw std::runtime_error("failed to write telemetry log: " + path);
  std::printf("DENSITY_TELEMETRY path=%s json=%s\n", path.c_str(), json.c_str());
}

static size_t gpu_used_bytes() {
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
  return total_bytes - free_bytes;
}

static size_t gpu_total_bytes() {
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
  return total_bytes;
}

static Clock::duration ms_duration(double ms) {
  return std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double, std::milli>(ms));
}

static double signed_elapsed_ms(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

static void cleanup_cuda_cache() {
  CUDA_CHECK(cudaDeviceSynchronize());
  c10::cuda::CUDACachingAllocator::emptyCache();
  CUDA_CHECK(cudaDeviceSynchronize());
}

struct MemorySampler {
  std::atomic<bool> stop{false};
  std::thread thread;
  std::atomic<size_t> peak{0};

  ~MemorySampler() {
    stop.store(true);
    if (thread.joinable()) {
      try {
        thread.join();
      } catch (const std::exception&) {
      }
    }
  }

  void start() {
    peak.store(gpu_used_bytes());
    stop.store(false);
    thread = std::thread([this] {
      while (!stop.load(std::memory_order_relaxed)) {
        try {
          size_t used = gpu_used_bytes();
          size_t prev = peak.load(std::memory_order_relaxed);
          while (used > prev && !peak.compare_exchange_weak(prev, used)) {}
        } catch (const std::exception&) {
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
      }
    });
  }

  size_t finish() {
    stop.store(true);
    if (thread.joinable()) thread.join();
    size_t used = gpu_used_bytes();
    size_t prev = peak.load(std::memory_order_relaxed);
    while (used > prev && !peak.compare_exchange_weak(prev, used)) {}
    return peak.load(std::memory_order_relaxed);
  }
};

struct ResourceStats {
  bool gpu_util_available = false;
  int gpu_util_samples = 0;
  double gpu_util_mean_pct = 0.0;
  double gpu_util_p50_pct = 0.0;
  double gpu_util_p95_pct = 0.0;
  double cpu_cores_used = 0.0;
  double cpu_util_pct_of_box = 0.0;
  int cpu_threads = 0;
};

static bool read_nvidia_smi_sample(double* gpu_util_pct, double* mem_used_mib) {
  FILE* pipe = popen("nvidia-smi --id=0 --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null", "r");
  if (pipe == nullptr) return false;
  char buffer[256];
  bool ok = false;
  if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
    std::string text(buffer);
    std::replace(text.begin(), text.end(), ',', ' ');
    std::istringstream iss(text);
    double util = 0.0;
    double mem = 0.0;
    if (iss >> util >> mem) {
      if (gpu_util_pct != nullptr) *gpu_util_pct = util;
      if (mem_used_mib != nullptr) *mem_used_mib = mem;
      ok = true;
    }
  }
  int rc = pclose(pipe);
  (void)rc;
  return ok;
}

static double rusage_seconds(const struct rusage& usage) {
  return static_cast<double>(usage.ru_utime.tv_sec) +
         static_cast<double>(usage.ru_utime.tv_usec) / 1000000.0 +
         static_cast<double>(usage.ru_stime.tv_sec) +
         static_cast<double>(usage.ru_stime.tv_usec) / 1000000.0;
}

struct ResourceSampler {
  std::atomic<bool> stop{false};
  std::thread thread;
  std::mutex mutex;
  std::vector<double> gpu_util_pct;
  Clock::time_point start_wall;
  Clock::time_point end_wall;
  struct rusage start_usage {};
  struct rusage end_usage {};
  int cpu_threads = 0;

  ~ResourceSampler() {
    stop.store(true);
    if (thread.joinable()) {
      try {
        thread.join();
      } catch (const std::exception&) {
      }
    }
  }

  void start() {
    stop.store(false);
    cpu_threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    getrusage(RUSAGE_SELF, &start_usage);
    start_wall = Clock::now();
    thread = std::thread([this] {
      while (!stop.load(std::memory_order_relaxed)) {
        double util = 0.0;
        double mem = 0.0;
        if (read_nvidia_smi_sample(&util, &mem)) {
          std::lock_guard<std::mutex> lock(mutex);
          gpu_util_pct.push_back(util);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
      }
    });
  }

  ResourceStats finish() {
    stop.store(true);
    if (thread.joinable()) thread.join();
    end_wall = Clock::now();
    getrusage(RUSAGE_SELF, &end_usage);
    ResourceStats out;
    out.cpu_threads = cpu_threads;
    double wall_s = std::chrono::duration<double>(end_wall - start_wall).count();
    double cpu_s = rusage_seconds(end_usage) - rusage_seconds(start_usage);
    if (wall_s > 0.0) {
      out.cpu_cores_used = cpu_s / wall_s;
      if (cpu_threads > 0) out.cpu_util_pct_of_box = 100.0 * out.cpu_cores_used / static_cast<double>(cpu_threads);
    }
    std::vector<double> samples;
    {
      std::lock_guard<std::mutex> lock(mutex);
      samples = gpu_util_pct;
    }
    auto gpu = summarize(samples);
    out.gpu_util_available = !samples.empty();
    out.gpu_util_samples = static_cast<int>(samples.size());
    out.gpu_util_mean_pct = gpu.mean;
    out.gpu_util_p50_pct = gpu.p50;
    out.gpu_util_p95_pct = gpu.p95;
    return out;
  }
};

static std::string resource_stats_json(const ResourceStats& stats) {
  std::ostringstream oss;
  oss << "{\"gpu_util_available\":" << json_bool(stats.gpu_util_available)
      << ",\"gpu_util_samples\":" << stats.gpu_util_samples
      << ",\"gpu_util_mean_pct\":" << stats.gpu_util_mean_pct
      << ",\"gpu_util_p50_pct\":" << stats.gpu_util_p50_pct
      << ",\"gpu_util_p95_pct\":" << stats.gpu_util_p95_pct
      << ",\"cpu_cores_used\":" << stats.cpu_cores_used
      << ",\"cpu_util_pct_of_box\":" << stats.cpu_util_pct_of_box
      << ",\"cpu_threads\":" << stats.cpu_threads
      << "}";
  return oss.str();
}

struct StartGate {
  std::mutex mutex;
  std::condition_variable cv;
  int expected = 0;
  int ready = 0;
  bool go = false;
  Clock::time_point start_time;

  explicit StartGate(int n) : expected(n) {}

  void arrive_and_wait() {
    std::unique_lock<std::mutex> lock(mutex);
    ++ready;
    if (ready == expected) cv.notify_all();
    cv.wait(lock, [&] { return go; });
  }

  void wait_until_ready_and_start() {
    std::unique_lock<std::mutex> lock(mutex);
    cv.wait(lock, [&] { return ready == expected; });
    start_time = Clock::now();
    go = true;
    cv.notify_all();
  }

  void wait_until_ready() {
    std::unique_lock<std::mutex> lock(mutex);
    cv.wait(lock, [&] { return ready == expected; });
  }

  void start_now() {
    std::unique_lock<std::mutex> lock(mutex);
    start_time = Clock::now();
    go = true;
    cv.notify_all();
  }
};

struct TimingBuckets {
  std::vector<double> latency_ms;
  std::vector<double> queue_wait_ms;
  std::vector<double> runner_wait_ms;
  std::vector<double> scalar_sync_wait_ms;
  std::vector<double> scalar_sync_pct_of_gpu;
  std::vector<double> steady_gpu_ms;
  std::vector<double> finalize_runner_wait_ms;
  std::vector<double> finalize_gpu_ms;
  std::vector<double> finalize_total_ms;
  std::vector<double> finalize_fork_clone_ms;
  std::vector<double> finalize_aoti_run_cuda_ms;
  std::vector<double> finalize_enc_len_sync_ms;
  std::vector<double> finalize_decode_wall_ms;
  std::vector<double> finalize_decode_item_wait_ms;
  std::vector<double> finalize_decode_tokens;
  std::vector<double> finalize_glue_ms;

  void append(const TimingBuckets& other) {
    auto add = [](std::vector<double>& dst, const std::vector<double>& src) {
      dst.insert(dst.end(), src.begin(), src.end());
    };
    add(latency_ms, other.latency_ms);
    add(queue_wait_ms, other.queue_wait_ms);
    add(runner_wait_ms, other.runner_wait_ms);
    add(scalar_sync_wait_ms, other.scalar_sync_wait_ms);
    add(scalar_sync_pct_of_gpu, other.scalar_sync_pct_of_gpu);
    add(steady_gpu_ms, other.steady_gpu_ms);
    add(finalize_runner_wait_ms, other.finalize_runner_wait_ms);
    add(finalize_gpu_ms, other.finalize_gpu_ms);
    add(finalize_total_ms, other.finalize_total_ms);
    add(finalize_fork_clone_ms, other.finalize_fork_clone_ms);
    add(finalize_aoti_run_cuda_ms, other.finalize_aoti_run_cuda_ms);
    add(finalize_enc_len_sync_ms, other.finalize_enc_len_sync_ms);
    add(finalize_decode_wall_ms, other.finalize_decode_wall_ms);
    add(finalize_decode_item_wait_ms, other.finalize_decode_item_wait_ms);
    add(finalize_decode_tokens, other.finalize_decode_tokens);
    add(finalize_glue_ms, other.finalize_glue_ms);
  }
};

static std::string finalize_phase_stats_json(const TimingBuckets& timings) {
  std::ostringstream oss;
  oss << "{\"fork_clone\":" << stats_json(summarize(timings.finalize_fork_clone_ms))
      << ",\"aoti_run_cuda\":" << stats_json(summarize(timings.finalize_aoti_run_cuda_ms))
      << ",\"enc_len_sync\":" << stats_json(summarize(timings.finalize_enc_len_sync_ms))
      << ",\"decode_wall\":" << stats_json(summarize(timings.finalize_decode_wall_ms))
      << ",\"decode_item_wait\":" << stats_json(summarize(timings.finalize_decode_item_wait_ms))
      << ",\"decode_tokens\":" << value_stats_json(summarize(timings.finalize_decode_tokens))
      << ",\"glue\":" << stats_json(summarize(timings.finalize_glue_ms))
      << "}";
  return oss.str();
}

struct WorkerContext {
  torch::jit::Module bundle;
  torch::jit::Module enc_first;
  torch::jit::Module joint;
  torch::jit::Module predict;
  std::unique_ptr<torch::jit::Module> preproc;
  c10::cuda::CUDAStream stream;

  WorkerContext(torch::jit::Module bundle_in,
                torch::jit::Module enc_first_in,
                torch::jit::Module joint_in,
                torch::jit::Module predict_in,
                std::unique_ptr<torch::jit::Module> preproc_in,
                c10::cuda::CUDAStream stream_in)
      : bundle(std::move(bundle_in)),
        enc_first(std::move(enc_first_in)),
        joint(std::move(joint_in)),
        predict(std::move(predict_in)),
        preproc(std::move(preproc_in)),
        stream(stream_in) {}
};

static torch::jit::Module load_module_on_device(const std::string& path, torch::Device device) {
  auto module = torch::jit::load(path);
  module.to(device);
  module.eval();
  return module;
}

static std::unique_ptr<WorkerContext> make_worker_context(const std::string& dir,
                                                          torch::Device device,
                                                          c10::cuda::CUDAStream stream) {
  auto bundle = torch::jit::load(dir + "/session_bundle.ts");
  verify_session_bundle_meta(bundle, false);
  auto enc_first = load_module_on_device(dir + "/enc_first.ts", device);
  auto joint = load_module_on_device(dir + "/joint_step.ts", device);
  auto predict = load_module_on_device(dir + "/predict_step.ts", device);
  std::unique_ptr<torch::jit::Module> preproc;
  if (file_exists(dir + "/preproc.ts")) {
    preproc = std::make_unique<torch::jit::Module>(load_module_on_device(dir + "/preproc.ts", device));
  }
  return std::make_unique<WorkerContext>(std::move(bundle),
                                         std::move(enc_first),
                                         std::move(joint),
                                         std::move(predict),
                                         std::move(preproc),
                                         stream);
}

static c10::cuda::CUDAStream stream_for_worker(bool explicit_stream, int worker, int device_index = 0) {
  if (!explicit_stream) return c10::cuda::getDefaultCUDAStream(device_index);
  return c10::cuda::getStreamFromPool(/*isHighPriority=*/false, /*device=*/device_index);
}

static uintptr_t stream_handle_value(const c10::cuda::CUDAStream& stream) {
  return reinterpret_cast<uintptr_t>(stream.stream());
}

static std::string stream_handles_json(const std::vector<c10::cuda::CUDAStream>& streams) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < streams.size(); ++i) {
    if (i > 0) oss << ",";
    oss << stream_handle_value(streams[i]);
  }
  oss << "]";
  return oss.str();
}

static std::mutex g_aoti_run_mutex;

static std::vector<at::Tensor> run_aoti_loader(AOTIModelPackageLoader& loader,
                                               const std::vector<at::Tensor>& inputs,
                                               c10::cuda::CUDAStream stream,
                                               bool explicit_stream,
                                               bool mutex_serialize_run) {
  auto invoke = [&]() {
    return explicit_stream
               ? loader.run(inputs, reinterpret_cast<void*>(stream.stream()))
               : loader.run(inputs);
  };
  if (mutex_serialize_run) {
    std::lock_guard<std::mutex> lock(g_aoti_run_mutex);
    return invoke();
  }
  return invoke();
}

static bool strict_events_equal(const std::vector<EmittedEvent>& got,
                                const std::vector<EmittedEvent>& gold,
                                const std::string& label) {
  bool ok = got.size() == gold.size();
  if (!ok) {
    std::printf("    %s strict event count mismatch: got=%zu gold=%zu\n",
                label.c_str(), got.size(), gold.size());
  }
  size_t n = std::min(got.size(), gold.size());
  for (size_t i = 0; i < n; ++i) {
    bool event_ok = got[i].kind == gold[i].kind &&
                    got[i].tokens == gold[i].tokens &&
                    got[i].collector_tokens == gold[i].collector_tokens &&
                    got[i].text == gold[i].text &&
                    got[i].collector_text == gold[i].collector_text;
    if (!event_ok) {
      std::printf("    %s strict event[%zu] mismatch: got_kind=%s gold_kind=%s "
                  "got_tokens=%zu gold_tokens=%zu got_collector=%zu gold_collector=%zu\n",
                  label.c_str(), i, event_kind_name(got[i].kind), event_kind_name(gold[i].kind),
                  got[i].tokens.size(), gold[i].tokens.size(),
                  got[i].collector_tokens.size(), gold[i].collector_tokens.size());
      if (got[i].text != gold[i].text) {
        std::printf("      got text :%s\n", escaped_text(got[i].text).c_str());
        std::printf("      gold text:%s\n", escaped_text(gold[i].text).c_str());
      }
      if (got[i].collector_text != gold[i].collector_text) {
        std::printf("      got collector text :%s\n", escaped_text(got[i].collector_text).c_str());
        std::printf("      gold collector text:%s\n", escaped_text(gold[i].collector_text).c_str());
      }
      std::printf("      got tokens :%s\n", vec_to_string(got[i].tokens).c_str());
      std::printf("      gold tokens:%s\n", vec_to_string(gold[i].tokens).c_str());
      std::printf("      got collector tokens :%s\n", vec_to_string(got[i].collector_tokens).c_str());
      std::printf("      gold collector tokens:%s\n", vec_to_string(gold[i].collector_tokens).c_str());
      ok = false;
      break;
    }
  }
  return ok;
}

static std::vector<at::Tensor> run_steady_encoder_stream(AOTIModelPackageLoader& loader,
                                                         const torch::Tensor& chunk,
                                                         SessionState& state,
                                                         c10::cuda::CUDAStream stream,
                                                         bool explicit_stream,
                                                         bool mutex_serialize_run,
                                                         double* runner_wait_ms) {
  auto device = chunk.device();
  auto L = torch::full({1}, chunk.size(2), torch::dtype(torch::kLong).device(device));
  std::vector<at::Tensor> inputs = {
      chunk.contiguous(),
      L.contiguous(),
      state.clc.contiguous(),
      state.clt.contiguous(),
      state.clcl.contiguous(),
  };
  auto t0 = Clock::now();
  std::vector<at::Tensor> out = run_aoti_loader(loader, inputs, stream, explicit_stream, mutex_serialize_run);
  auto t1 = Clock::now();
  if (runner_wait_ms != nullptr) *runner_wait_ms = elapsed_ms(t0, t1);
  if (out.size() < 5) throw std::runtime_error("steady AOTI encoder returned fewer than 5 outputs");
  return out;
}

static int64_t scalar_i64_timed(torch::Tensor tensor, double* item_wait_ms) {
  auto start = Clock::now();
  int64_t value = tensor.to(torch::kCPU).reshape({-1})[0].item<int64_t>();
  if (item_wait_ms != nullptr) *item_wait_ms += elapsed_ms_since(start);
  return value;
}

static int64_t argmax_item_timed(const torch::Tensor& tensor, double* item_wait_ms) {
  auto start = Clock::now();
  int64_t value = tensor.argmax().item<int64_t>();
  if (item_wait_ms != nullptr) *item_wait_ms += elapsed_ms_since(start);
  return value;
}

static void decode_range_density(torch::jit::Module& joint,
                                 torch::jit::Module& predict,
                                 const torch::Tensor& enc_out,
                                 int64_t enc_len,
                                 torch::Tensor& g,
                                 torch::Tensor& h,
                                 torch::Tensor& c,
                                 std::vector<int64_t>& hyp,
                                 double* item_wait_ms) {
  if (enc_len < 0 || enc_len > enc_out.size(2)) {
    throw std::runtime_error("density enc_len out of range for enc_out: " + std::to_string(enc_len));
  }
  auto f = enc_out.transpose(1, 2).contiguous();
  auto dev = f.device();
  for (int64_t t = 0; t < enc_len; ++t) {
    auto f_t = f.slice(1, t, t + 1);
    for (int n = 0; n < MAX_SYMBOLS; ++n) {
      auto logits = joint.forward({f_t, g}).toTensor();
      auto flat = logits.reshape({-1});
      int64_t k = argmax_item_timed(flat, item_wait_ms);
      if (k == BLANK) break;
      hyp.push_back(k);
      auto y = torch::full({1, 1}, k, torch::dtype(torch::kLong).device(dev));
      auto out = predict.forward({y, h, c}).toTuple();
      g = out->elements()[0].toTensor();
      h = out->elements()[1].toTensor();
      c = out->elements()[2].toTensor();
    }
  }
}

static double apply_encoder_outputs_density(SessionState& state,
                                            const std::vector<at::Tensor>& out,
                                            torch::jit::Module& joint,
                                            torch::jit::Module& predict) {
  if (out.size() < 5) throw std::runtime_error("density encoder returned fewer than 5 outputs");
  double item_wait_ms = 0.0;
  int64_t enc_len = scalar_i64_timed(out[1], &item_wait_ms);
  state.clc = out[2].clone();
  state.clt = out[3].clone();
  state.clcl = out[4].clone();
  decode_range_density(joint, predict, out[0], enc_len, state.g, state.h, state.c, state.hyp, &item_wait_ms);
  return item_wait_ms;
}

static void run_steady_chunk_density(SessionState& state,
                                     torch::jit::Module& bundle,
                                     const std::string& prefix,
                                     int chunk_index,
                                     torch::jit::Module& enc_first,
                                     AOTIModelPackageLoader& enc_steady,
                                     torch::jit::Module& joint,
                                     torch::jit::Module& predict,
                                     torch::Device device,
                                     const Tokenizer& tokenizer,
                                     std::vector<EmittedEvent>& events,
                                     c10::cuda::CUDAStream stream,
                                     bool explicit_stream,
                                     bool mutex_serialize_run,
                                     TimingBuckets* timings,
                                     const std::string& label) {
  c10::cuda::CUDAGuard device_guard(device.index());
  c10::cuda::CUDAStreamGuard stream_guard(stream);
  if (state.mode != SessionMode::STREAMING) throw std::runtime_error("density steady chunk outside STREAMING");

  auto call_start = Clock::now();
  auto new_mel = prefix_chunk_tensor(bundle, prefix, chunk_index, "new_mel").to(device).contiguous();
  int64_t is_first = scalar_i64(prefix_chunk_tensor(bundle, prefix, chunk_index, "is_first"));
  int64_t drop_extra = scalar_i64(prefix_chunk_tensor(bundle, prefix, chunk_index, "drop_extra"));
  int64_t chunk_T = scalar_i64(prefix_chunk_tensor(bundle, prefix, chunk_index, "chunk_T"));
  int64_t emitted_before = scalar_i64(prefix_chunk_tensor(bundle, prefix, chunk_index, "emitted_before"));

  bool expected_first = state.emitted == 0;
  if ((is_first != 0) != expected_first) throw std::runtime_error("density steady first/continuation flag mismatch");
  if (emitted_before != state.emitted) throw std::runtime_error("density steady emitted_before mismatch");
  if (new_mel.size(2) != SHIFT) throw std::runtime_error("density steady new_mel is not SHIFT frames");

  torch::Tensor chunk;
  std::vector<at::Tensor> out;
  double runner_wait = 0.0;
  double gpu_ms = 0.0;
  cudaEvent_t ev_start{};
  cudaEvent_t ev_stop{};
  bool has_aoti_event = false;

  if (expected_first) {
    if (drop_extra != 0 || chunk_T != new_mel.size(2)) throw std::runtime_error("density first steady geometry mismatch");
    chunk = new_mel;
    out = run_first_encoder(enc_first, chunk, state);
  } else {
    if (!state.ring.defined()) throw std::runtime_error("density steady continuation missing mel ring");
    if (drop_extra != DROP || chunk_T != state.ring.size(2) + new_mel.size(2)) {
      throw std::runtime_error("density steady continuation geometry mismatch");
    }
    chunk = torch::cat({state.ring, new_mel}, 2).contiguous();
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));
    CUDA_CHECK(cudaEventRecord(ev_start, stream.stream()));
    out = run_steady_encoder_stream(enc_steady, chunk, state, stream, explicit_stream, mutex_serialize_run, &runner_wait);
    CUDA_CHECK(cudaEventRecord(ev_stop, stream.stream()));
    has_aoti_event = true;
  }

  double scalar_wait_ms = apply_encoder_outputs_density(state, out, joint, predict);

  if (has_aoti_event) {
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    float elapsed = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, ev_start, ev_stop));
    gpu_ms = static_cast<double>(elapsed);
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
  }

  auto cum = state.ring.defined() ? torch::cat({state.ring, new_mel}, 2) : new_mel;
  state.ring = cum.slice(2, std::max<int64_t>(0, cum.size(2) - PRE), cum.size(2)).contiguous();
  state.emitted += new_mel.size(2);
  std::string current_text = tokenizer.ids_to_text(state.hyp);
  if (current_text != state.last_interim_text) {
    emit_event(events,
               EVENT_INTERIM,
               state.hyp,
               state.continuous_emitted_tokens,
               current_text,
               state.continuous_emitted_text);
    state.last_interim_tokens = state.hyp;
    state.last_interim_text = current_text;
  }

  if (timings != nullptr) {
    timings->latency_ms.push_back(elapsed_ms_since(call_start));
    timings->queue_wait_ms.push_back(0.0);
    timings->scalar_sync_wait_ms.push_back(scalar_wait_ms);
    if (!expected_first) {
      timings->steady_gpu_ms.push_back(gpu_ms);
      timings->runner_wait_ms.push_back(std::max(0.0, runner_wait - gpu_ms));
      if (gpu_ms > 0.0) timings->scalar_sync_pct_of_gpu.push_back(100.0 * scalar_wait_ms / gpu_ms);
    }
  }
}

using FinalizeBucketKey = std::pair<int64_t, int64_t>;

struct FinalizeLoaderMemoryRecord {
  int64_t drop = 0;
  int64_t T = 0;
  int num_runners = 0;
  size_t used_before = 0;
  size_t used_after = 0;
  size_t delta = 0;
  size_t cumulative_delta = 0;
};

class FinalizeBucketLoaderPool {
 public:
  FinalizeBucketLoaderPool(const std::string& dir,
                           torch::Device device,
                           int num_runners,
                           std::string policy)
      : dir_(dir),
        device_(device),
        num_runners_(num_runners),
        policy_(std::move(policy)) {
    if (num_runners_ <= 0) throw std::runtime_error("finalize bucket num_runners must be positive");
    buckets_dir_ = dir_ + "/stripped_finalize_buckets";
    if (!directory_exists(buckets_dir_)) buckets_dir_ = dir_ + "/finalize_buckets";
    shared_weights_ = dir_ + "/finalize_shared_weights.ts";
    std::string shared_weights_pt = dir_ + "/finalize_shared_weights.pt";
    if (!directory_exists(buckets_dir_)) throw std::runtime_error("finalize buckets directory missing: " + buckets_dir_);
    if (!file_exists(shared_weights_)) throw std::runtime_error("finalize shared weights missing: " + shared_weights_);

    bucket_paths_ = discover_finalize_buckets(buckets_dir_);
    if (bucket_paths_.empty()) throw std::runtime_error("no finalize bucket packages found in " + buckets_dir_);
    std::string manifest_path = buckets_dir_ + "/manifest.json";
    if (!file_exists(manifest_path)) {
      throw std::runtime_error("finalize bucket manifest is required when buckets are present: " + manifest_path);
    }
    auto manifest = load_bucket_manifest(manifest_path);
    verify_bucket_manifest(manifest, bucket_paths_, buckets_dir_, shared_weights_pt);
    std::printf("density finalize manifest verified: %zu buckets, weights_sha256=%s num_runners=%d policy=%s\n",
                manifest.buckets.size(), manifest.contract.weights_sha256.c_str(), num_runners_,
                policy_.c_str());

    CUDA_CHECK(cudaDeviceSynchronize());
    shared_used_before_ = gpu_used_bytes();
    shared_constants_ = load_shared_constants(shared_weights_, device_);
    CUDA_CHECK(cudaDeviceSynchronize());
    shared_used_after_ = gpu_used_bytes();
    shared_delta_ = shared_used_after_ >= shared_used_before_ ? shared_used_after_ - shared_used_before_ : 0;
    std::printf("density loaded finalize shared constants: %zu entries shared_delta=%.3f GiB policy=%s\n",
                shared_constants_.size(),
                static_cast<double>(shared_delta_) / (1024.0 * 1024.0 * 1024.0),
                policy_.c_str());
  }

  AOTIModelPackageLoader& get(int64_t drop, int64_t T) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto key = std::make_pair(drop, T);
    auto existing = loaders_.find(key);
    if (existing != loaders_.end()) return *existing->second;
    auto loaded = load_bucket_locked(key);
    return *loaded->second;
  }

  void preload(const std::vector<FinalizeBucketKey>& keys) {
    for (const auto& key : keys) {
      (void)get(key.first, key.second);
    }
  }

  void preload_all() {
    std::vector<FinalizeBucketKey> keys;
    keys.reserve(bucket_paths_.size());
    for (const auto& kv : bucket_paths_) keys.push_back(kv.first);
    preload(keys);
  }

  int num_runners() const {
    return num_runners_;
  }

  size_t total_bucket_count() const {
    return bucket_paths_.size();
  }

  size_t loaded_bucket_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return loaders_.size();
  }

  size_t shared_delta() const {
    return shared_delta_;
  }

  size_t total_loader_delta() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return total_loader_delta_;
  }

  size_t projected_all_buckets_same_runners_delta() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (records_.empty()) return 0;
    return static_cast<size_t>((static_cast<long double>(total_loader_delta_) /
                                static_cast<long double>(records_.size())) *
                               static_cast<long double>(bucket_paths_.size()));
  }

  size_t projected_all_buckets_worker_runners_delta(int worker_runners) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (records_.empty() || num_runners_ <= 0) return 0;
    long double mean = static_cast<long double>(total_loader_delta_) /
                       static_cast<long double>(records_.size());
    long double runner_ratio = static_cast<long double>(worker_runners) /
                               static_cast<long double>(num_runners_);
    return static_cast<size_t>(mean * static_cast<long double>(bucket_paths_.size()) * runner_ratio);
  }

  std::string memory_json(int worker_runners) const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t projected_same = 0;
    size_t projected_worker = 0;
    if (!records_.empty()) {
      long double mean = static_cast<long double>(total_loader_delta_) /
                         static_cast<long double>(records_.size());
      projected_same = static_cast<size_t>(mean * static_cast<long double>(bucket_paths_.size()));
      if (num_runners_ > 0) {
        long double runner_ratio = static_cast<long double>(worker_runners) /
                                   static_cast<long double>(num_runners_);
        projected_worker = static_cast<size_t>(mean * static_cast<long double>(bucket_paths_.size()) * runner_ratio);
      }
    }
    std::ostringstream oss;
    oss << "{\"policy\":\"" << policy_ << "\""
        << ",\"num_runners_per_loaded_bucket\":" << num_runners_
        << ",\"worker_runners_requested\":" << worker_runners
        << ",\"total_manifest_buckets\":" << bucket_paths_.size()
        << ",\"loaded_buckets\":" << loaders_.size()
        << ",\"shared_constants_delta_bytes\":" << shared_delta_
        << ",\"loader_delta_bytes\":" << total_loader_delta_
        << ",\"projected_all_buckets_same_runner_cap_delta_bytes\":" << projected_same
        << ",\"projected_old_eager_all_buckets_worker_runner_delta_bytes_linear_estimate\":" << projected_worker
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
      throw std::runtime_error("density no finalize bucket for drop=" + std::to_string(key.first) +
                               " T=" + std::to_string(key.second));
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    size_t before = gpu_used_bytes();
    auto loader = std::make_unique<AOTIModelPackageLoader>(path_it->second, "model", false, num_runners_, -1);
    auto bucket_constants = constants_for_bucket(shared_constants_, *loader, path_it->second);
    loader->load_constants(bucket_constants.values, false, false, true);
    CUDA_CHECK(cudaDeviceSynchronize());
    size_t after = gpu_used_bytes();
    size_t delta = after >= before ? after - before : 0;
    total_loader_delta_ += delta;
    records_.push_back({
        key.first,
        key.second,
        num_runners_,
        before,
        after,
        delta,
        total_loader_delta_,
    });
    std::printf("  density finalize bucket loaded drop=%ld T=%ld constants=%zu direct=%zu alias=%zu "
                "num_runners=%d loader_delta=%.3f GiB cumulative_loader_delta=%.3f GiB policy=%s\n",
                (long)key.first, (long)key.second, bucket_constants.values.size(),
                bucket_constants.direct_matches, bucket_constants.alias_fallbacks, num_runners_,
                static_cast<double>(delta) / (1024.0 * 1024.0 * 1024.0),
                static_cast<double>(total_loader_delta_) / (1024.0 * 1024.0 * 1024.0),
                policy_.c_str());
    auto inserted = loaders_.emplace(key, std::move(loader));
    return inserted.first;
  }

  std::string dir_;
  torch::Device device_;
  int num_runners_ = 1;
  std::string policy_;
  std::string buckets_dir_;
  std::string shared_weights_;
  std::map<FinalizeBucketKey, std::string> bucket_paths_;
  std::unordered_map<std::string, at::Tensor> shared_constants_;
  std::map<FinalizeBucketKey, std::unique_ptr<AOTIModelPackageLoader>> loaders_;
  mutable std::mutex mutex_;
  std::vector<FinalizeLoaderMemoryRecord> records_;
  size_t shared_used_before_ = 0;
  size_t shared_used_after_ = 0;
  size_t shared_delta_ = 0;
  size_t total_loader_delta_ = 0;
};

static int capped_general_finalize_runners(int workers_or_runners) {
  return std::max(1, std::min(workers_or_runners, 2));
}

static std::vector<FinalizeBucketKey> unique_finalize_bucket_keys(const std::vector<FinalizeBucketKey>& keys) {
  std::set<FinalizeBucketKey> seen(keys.begin(), keys.end());
  return std::vector<FinalizeBucketKey>(seen.begin(), seen.end());
}

static FinalizeOutcome run_finalize_density(SessionState& parent,
                                            torch::jit::Module& bundle,
                                            const std::string& prefix,
                                            const std::string& label,
                                            FinalizeBucketLoaderPool& finalize_loaders,
                                            torch::jit::Module& joint,
                                            torch::jit::Module& predict,
                                            torch::Device device,
                                            const Tokenizer& tokenizer,
                                            std::vector<EmittedEvent>& events,
                                            FinalizeFinish finish,
                                            c10::cuda::CUDAStream stream,
                                            bool explicit_stream,
                                            bool mutex_serialize_run,
                                            TimingBuckets* timings) {
  c10::cuda::CUDAGuard device_guard(device.index());
  c10::cuda::CUDAStreamGuard stream_guard(stream);
  auto total_start = Clock::now();

  if (finish == FinalizeFinish::SPECULATIVE_KEEP && parent.mode != SessionMode::PENDING_FINALIZE) {
    throw std::runtime_error("density speculative finalize outside PENDING_FINALIZE");
  }
  if (finish == FinalizeFinish::TRUE_BOUNDARY_COLD_RESET &&
      parent.mode != SessionMode::STREAMING &&
      parent.mode != SessionMode::PENDING_FINALIZE) {
    throw std::runtime_error("density true-boundary finalize outside live state");
  }
  auto fork_clone_start = Clock::now();
  auto snapshot = snapshot_asr(parent);
  parent.mode = SessionMode::FINALIZED;
  snapshot.mode = SessionMode::FINALIZED;
  auto fork = clone_session(parent);
  double fork_clone_ms = elapsed_ms_since(fork_clone_start);

  int64_t drop_extra = scalar_i64(prefix_tensor(bundle, prefix, "final_drop_extra"));
  int64_t final_T = scalar_i64(prefix_tensor(bundle, prefix, "final_T"));
  auto gold = tensor_to_vec(prefix_tensor(bundle, prefix, "gold_tokens"));
  double runner_host_ms = 0.0;
  double gpu_ms = 0.0;
  double enc_len_sync_ms = 0.0;
  double decode_wall_ms = 0.0;
  double decode_item_wait_ms = 0.0;
  double decode_tokens = 0.0;

  if (final_T > 0) {
    auto final_chunk = prefix_tensor(bundle, prefix, "final_chunk_mel").to(device).contiguous();
    if (final_chunk.size(2) != final_T) {
      throw std::runtime_error("density final_chunk_mel T does not match bundle final_T");
    }
    int64_t expected_drop = parent.emitted == 0 ? 0 : DROP;
    if (drop_extra != expected_drop) throw std::runtime_error("density finalize drop_extra does not match parent emitted state");

    auto& finalize_loader = finalize_loaders.get(drop_extra, final_T);

    std::vector<at::Tensor> inputs = {
        final_chunk.contiguous(),
        fork.clc.contiguous(),
        fork.clt.contiguous(),
        fork.clcl.contiguous(),
    };
    cudaEvent_t ev_start{};
    cudaEvent_t ev_stop{};
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));
    CUDA_CHECK(cudaEventRecord(ev_start, stream.stream()));
    auto run_start = Clock::now();
    auto out = run_aoti_loader(finalize_loader, inputs, stream, explicit_stream, mutex_serialize_run);
    runner_host_ms = elapsed_ms_since(run_start);
    CUDA_CHECK(cudaEventRecord(ev_stop, stream.stream()));
    if (out.size() < 2) throw std::runtime_error("density finalize AOTI bucket returned fewer than 2 outputs");
    int64_t enc_len = scalar_i64_timed(out[1], &enc_len_sync_ms);
    if (out.size() >= 5) {
      fork.clc = out[2];
      fork.clt = out[3];
      fork.clcl = out[4];
    }
    size_t hyp_before_decode = fork.hyp.size();
    auto decode_start = Clock::now();
    decode_range_density(joint,
                         predict,
                         out[0],
                         enc_len,
                         fork.g,
                         fork.h,
                         fork.c,
                         fork.hyp,
                         &decode_item_wait_ms);
    decode_wall_ms = elapsed_ms_since(decode_start);
    decode_tokens = static_cast<double>(fork.hyp.size() - hyp_before_decode);
    if (timings != nullptr) {
      timings->scalar_sync_wait_ms.push_back(enc_len_sync_ms + decode_item_wait_ms);
    }
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    float elapsed = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, ev_start, ev_stop));
    gpu_ms = static_cast<double>(elapsed);
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
  }

  FinalizeOutcome outcome;
  outcome.emitted_tokens = fork.hyp.size();
  outcome.final_tokens = fork.hyp;
  outcome.final_text = tokenizer.ids_to_text(fork.hyp);
  outcome.token_ok = equal_tokens(outcome.final_tokens, gold, "final cumulative", label);
  std::string final_text = outcome.final_text;
  std::string delta_text = append_only_delta_text(final_text, parent.continuous_emitted_text);
  auto delta_tokens = append_only_delta_tokens(fork.hyp, parent.continuous_emitted_tokens);
  if (delta_text.empty()) {
    emit_event(events,
               EVENT_SUPPRESSED,
               {},
               parent.continuous_emitted_tokens,
               "",
               parent.continuous_emitted_text);
  } else {
    auto collector_tokens = parent.continuous_emitted_tokens;
    collector_tokens.insert(collector_tokens.end(), delta_tokens.begin(), delta_tokens.end());
    std::string collector_text = append_delta_to_collector(parent.continuous_emitted_text, delta_text);
    emit_event(events,
               EVENT_FINAL,
               delta_tokens,
               collector_tokens,
               delta_text,
               collector_text);
    parent.continuous_emitted_tokens = std::move(collector_tokens);
    parent.continuous_emitted_text = std::move(collector_text);
  }
  outcome.fork_ok = fork_assert_parent_unchanged(parent, snapshot);
  if (finish == FinalizeFinish::SPECULATIVE_KEEP) {
    finish_speculative_finalize(parent);
  } else {
    cold_reset_after_finalize(parent, bundle, device, nullptr);
  }
  if (timings != nullptr) {
    double total_ms = elapsed_ms_since(total_start);
    double glue_ms = total_ms - fork_clone_ms - enc_len_sync_ms - decode_wall_ms;
    if (glue_ms < 0.0) glue_ms = 0.0;
    timings->finalize_runner_wait_ms.push_back(std::max(0.0, runner_host_ms - gpu_ms));
    timings->finalize_gpu_ms.push_back(gpu_ms);
    timings->finalize_total_ms.push_back(total_ms);
    timings->finalize_fork_clone_ms.push_back(fork_clone_ms);
    timings->finalize_aoti_run_cuda_ms.push_back(gpu_ms);
    timings->finalize_enc_len_sync_ms.push_back(enc_len_sync_ms);
    timings->finalize_decode_wall_ms.push_back(decode_wall_ms);
    timings->finalize_decode_item_wait_ms.push_back(decode_item_wait_ms);
    timings->finalize_decode_tokens.push_back(decode_tokens);
    timings->finalize_glue_ms.push_back(glue_ms);
  }
  return outcome;
}

struct RowReplayResult {
  std::vector<int64_t> final_tokens;
  std::vector<EmittedEvent> events;
  bool ok = false;
  std::string error;
};

static RowReplayResult replay_row_density(int utt,
                                          WorkerContext& ctx,
                                          AOTIModelPackageLoader& enc_steady,
                                          FinalizeBucketLoaderPool& finalize_loaders,
                                          torch::Device device,
                                          const Tokenizer& tokenizer,
                                          bool explicit_stream,
                                          bool mutex_serialize_run,
                                          TimingBuckets* timings,
                                          bool check_gold) {
  RowReplayResult result;
  try {
    SessionState session;
    reset_session(session, ctx.bundle, device);
    std::string prefix = "utt" + std::to_string(utt);
    std::string label = "density.utt" + std::to_string(utt);
    int64_t num_steady = scalar_i64(utt_tensor(ctx.bundle, utt, "num_steady"));
    std::vector<EmittedEvent> events;
    for (int chunk = 0; chunk < num_steady; ++chunk) {
      run_steady_chunk_density(session,
                               ctx.bundle,
                               prefix,
                               chunk,
                               ctx.enc_first,
                               enc_steady,
                               ctx.joint,
                               ctx.predict,
                               device,
                               tokenizer,
                               events,
                               ctx.stream,
                               explicit_stream,
                               mutex_serialize_run,
                               timings,
                               label + ".chunk" + std::to_string(chunk));
    }
    bool steady_ok = true;
    if (check_gold) {
      auto steady_gold = tensor_to_vec(utt_tensor(ctx.bundle, utt, "steady_tokens"));
      steady_ok = equal_tokens(session.hyp, steady_gold, "steady cumulative", label);
    }
    session.mode = SessionMode::PENDING_FINALIZE;
    auto finalize = run_finalize_density(session,
                                         ctx.bundle,
                                         prefix,
                                         label,
                                         finalize_loaders,
                                         ctx.joint,
                                         ctx.predict,
                                         device,
                                         tokenizer,
                                         events,
                                         FinalizeFinish::SPECULATIVE_KEEP,
                                         ctx.stream,
                                         explicit_stream,
                                         mutex_serialize_run,
                                         timings);
    bool events_ok = true;
    if (check_gold) {
      events_ok = strict_events_equal(events, gold_events_from_bundle(ctx.bundle, utt), label + ".gold");
    }
    result.final_tokens = std::move(finalize.final_tokens);
    result.events = std::move(events);
    result.ok = steady_ok && finalize.token_ok && finalize.fork_ok && events_ok;
  } catch (const std::exception& e) {
    result.error = e.what();
    result.ok = false;
  }
  return result;
}

struct CorrectnessResult {
  bool ok = false;
  bool identity_ok = false;
  bool scalar_locality_pass = false;
  bool default_stream_control_pass = false;
  bool stream_uniqueness_ok = false;
  bool explicit_stream = true;
  bool mutex_serialize_run = false;
  int rows = 0;
  int workers = 0;
  int num_runners = 0;
  int unique_streams = 0;
  int mismatches = 0;
  double default_stream_penalty = 0.0;
  double throughput_rows_per_s = 0.0;
  double wall_ms = 0.0;
  TimingBuckets timings;
  size_t peak_mem = 0;
  size_t used_before = 0;
  size_t used_after = 0;
  std::vector<uintptr_t> stream_handles;
  std::vector<RowReplayResult> reference;
};

static std::vector<RowReplayResult> build_serial_reference(const DensityArgs& args,
                                                           torch::Device device,
                                                           int rows,
                                                           TimingBuckets* ref_timings) {
  auto stream = stream_for_worker(true, 0);
  auto ctx = make_worker_context(args.dir, device, stream);
  auto tokenizer = tokenizer_from_bundle(ctx->bundle);
  verify_tokenizer_selftest(ctx->bundle, tokenizer);
  AOTIModelPackageLoader enc_steady(args.dir + "/enc_steady_aoti.pt2", "model", false, 1, -1);
  FinalizeBucketLoaderPool finalize_loaders(args.dir, device, 1, "serial_reference_eager_one_runner");
  finalize_loaders.preload_all();
  std::vector<RowReplayResult> refs;
  refs.reserve(rows);
  for (int utt = 0; utt < rows; ++utt) {
    auto result = replay_row_density(utt,
                                     *ctx,
                                     enc_steady,
                                     finalize_loaders,
                                     device,
                                     tokenizer,
                                     true,
                                     false,
                                     ref_timings,
                                     true);
    if (!result.ok) {
      throw std::runtime_error("serial reference failed for utt" + std::to_string(utt) +
                               (result.error.empty() ? "" : ": " + result.error));
    }
    refs.push_back(std::move(result));
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  return refs;
}

static std::string stream_mode_label(bool explicit_stream, bool mutex_serialize_run) {
  std::string mode = explicit_stream ? "explicit" : "default";
  if (mutex_serialize_run) mode += "+mutex";
  return mode;
}

static CorrectnessResult run_correctness_gate_mode(const DensityArgs& args,
                                                   torch::Device device,
                                                   const std::string& stamp,
                                                   int rows,
                                                   const std::vector<RowReplayResult>& reference,
                                                   bool explicit_stream,
                                                   bool mutex_serialize_run,
                                                   int workers,
                                                   int num_runners,
                                                   const std::string& topology) {
  CorrectnessResult result;
  result.rows = rows;
  result.workers = workers;
  result.num_runners = num_runners;
  result.explicit_stream = explicit_stream;
  result.mutex_serialize_run = mutex_serialize_run;
  std::printf("=== DENSITY 0b correctness mode: rows=%d workers=%d num_runners=%d stream_mode=%s topology=%s ===\n",
              rows, workers, num_runners, stream_mode_label(explicit_stream, mutex_serialize_run).c_str(),
              topology.c_str());
  result.used_before = gpu_used_bytes();
  MemorySampler mem;
  mem.start();
  AOTIModelPackageLoader enc_steady(args.dir + "/enc_steady_aoti.pt2", "model", false, num_runners, -1);
  int finalize_num_runners = capped_general_finalize_runners(num_runners);
  FinalizeBucketLoaderPool finalize_loaders(args.dir,
                                            device,
                                            finalize_num_runners,
                                            "0b_general_finalize_eager_capped_runner_pool");
  finalize_loaders.preload_all();
  std::vector<std::unique_ptr<WorkerContext>> contexts;
  contexts.reserve(workers);
  std::vector<c10::cuda::CUDAStream> streams;
  streams.reserve(workers);
  std::set<uintptr_t> stream_ids;
  for (int worker = 0; worker < workers; ++worker) {
    auto stream = stream_for_worker(explicit_stream, worker);
    streams.push_back(stream);
    uintptr_t handle = stream_handle_value(stream);
    stream_ids.insert(handle);
    result.stream_handles.push_back(handle);
    contexts.push_back(make_worker_context(args.dir, device, stream));
  }
  result.unique_streams = static_cast<int>(stream_ids.size());
  result.stream_uniqueness_ok = !explicit_stream || result.unique_streams == workers;
  auto tokenizer = tokenizer_from_bundle(contexts[0]->bundle);

  StartGate gate(workers);
  std::vector<TimingBuckets> worker_timings(workers);
  std::vector<std::string> errors(workers);
  std::vector<int> worker_mismatches(workers, 0);
  std::vector<std::thread> threads;
  threads.reserve(workers);
  for (int worker = 0; worker < workers; ++worker) {
    threads.emplace_back([&, worker] {
      try {
        c10::cuda::CUDAGuard device_guard(device.index());
        gate.arrive_and_wait();
        for (int utt = 0; utt < result.rows; ++utt) {
          auto row = replay_row_density(utt,
                                        *contexts[worker],
                                        enc_steady,
                                        finalize_loaders,
                                        device,
                                        tokenizer,
                                        explicit_stream,
                                        mutex_serialize_run,
                                        &worker_timings[worker],
                                        true);
          bool same = row.ok &&
                      row.final_tokens == reference[utt].final_tokens &&
                      strict_events_equal(row.events,
                                          reference[utt].events,
                                          "density.worker" + std::to_string(worker) + ".utt" + std::to_string(utt));
          if (!same) {
            ++worker_mismatches[worker];
            if (!row.error.empty()) {
              std::printf("  worker%d utt%d error: %s\n", worker, utt, row.error.c_str());
            }
          }
        }
      } catch (const std::exception& e) {
        errors[worker] = e.what();
      }
    });
  }
  gate.wait_until_ready_and_start();
  for (auto& t : threads) t.join();
  auto end_time = Clock::now();
  CUDA_CHECK(cudaDeviceSynchronize());
  result.wall_ms = elapsed_ms(gate.start_time, end_time);
  result.peak_mem = mem.finish();
  result.used_after = gpu_used_bytes();

  for (int worker = 0; worker < workers; ++worker) {
    result.timings.append(worker_timings[worker]);
    result.mismatches += worker_mismatches[worker];
    if (!errors[worker].empty()) {
      std::printf("  worker%d correctness exception: %s\n", worker, errors[worker].c_str());
      ++result.mismatches;
    }
  }
  result.throughput_rows_per_s =
      (static_cast<double>(workers) * static_cast<double>(result.rows)) / (result.wall_ms / 1000.0);
  auto item_pct = summarize(result.timings.scalar_sync_pct_of_gpu);
  result.identity_ok = result.mismatches == 0;
  result.ok = result.identity_ok && result.stream_uniqueness_ok;

  std::ostringstream json;
  json << "{\"check\":\"0b_concurrent_serial_correctness\""
       << ",\"num_runners\":" << num_runners
       << ",\"workers\":" << workers
       << ",\"stream_mode\":\"" << stream_mode_label(explicit_stream, mutex_serialize_run) << "\""
       << ",\"topology\":\"" << topology << "\""
       << ",\"rows\":" << result.rows
       << ",\"mismatches\":" << result.mismatches
       << ",\"identity_pass\":" << json_bool(result.identity_ok)
       << ",\"stream_uniqueness_pass\":" << json_bool(result.stream_uniqueness_ok)
       << ",\"pass\":" << json_bool(result.ok)
       << ",\"throughput_rows_per_s\":" << result.throughput_rows_per_s
       << ",\"wall_ms\":" << result.wall_ms
       << ",\"latency\":" << stats_json(summarize(result.timings.latency_ms))
       << ",\"queue_wait\":" << stats_json(summarize(result.timings.queue_wait_ms))
       << ",\"runner_wait\":" << stats_json(summarize(result.timings.runner_wait_ms))
       << ",\"item_wait\":" << stats_json(summarize(result.timings.scalar_sync_wait_ms))
       << ",\"item_wait_pct_gate\":\"telemetry_only_no_threshold\""
       << ",\"item_wait_pct_of_steady_gpu\":" << stats_json(item_pct)
       << ",\"finalize_wait\":" << stats_json(summarize(result.timings.finalize_runner_wait_ms))
       << ",\"finalize_gpu\":" << stats_json(summarize(result.timings.finalize_gpu_ms))
       << ",\"finalize_loader_memory\":" << finalize_loaders.memory_json(num_runners)
       << ",\"unique_streams\":" << result.unique_streams
       << ",\"stream_handles\":[";
  for (size_t i = 0; i < result.stream_handles.size(); ++i) {
    if (i > 0) json << ",";
    json << result.stream_handles[i];
  }
  json << "]"
       << ",\"peak_gpu_mem_bytes\":" << result.peak_mem
       << ",\"used_before_bytes\":" << result.used_before
       << ",\"used_after_bytes\":" << result.used_after
       << "}";
  emit_telemetry(args.dir,
                 stamp,
                 num_runners,
                 stream_mode_label(explicit_stream, mutex_serialize_run),
                 topology,
                 json.str());

  std::printf("=== DENSITY 0b %s: workers=%d num_runners=%d stream_mode=%s rows=%d mismatches=%d "
              "item_wait_pct_p95=%.2f telemetry_only unique_streams=%d throughput=%.3f rows/s ===\n",
              result.ok ? "PASS" : "FAIL",
              workers,
              num_runners,
              stream_mode_label(explicit_stream, mutex_serialize_run).c_str(),
              result.rows,
              result.mismatches,
              item_pct.p95,
              result.unique_streams,
              result.throughput_rows_per_s);
  return result;
}

struct ScalarLocalityProbeResult {
  bool ran = false;
  bool pass = false;
  bool b_pending_after_item = false;
  double item_wait_ms = 0.0;
  double sentinel_gpu_ms = 0.0;
  double sentinel_sync_wall_ms = 0.0;
  int dim = 4096;
  int iters = 48;
  uintptr_t stream_a = 0;
  uintptr_t stream_b = 0;
};

static ScalarLocalityProbeResult run_scalar_locality_probe(const DensityArgs& args,
                                                           torch::Device device,
                                                           const std::string& stamp) {
  ScalarLocalityProbeResult result;
  result.ran = true;
  c10::cuda::CUDAGuard device_guard(device.index());
  auto stream_a = stream_for_worker(true, 0);
  auto stream_b = stream_for_worker(true, 1);
  result.stream_a = stream_handle_value(stream_a);
  result.stream_b = stream_handle_value(stream_b);
  auto float_opts = torch::TensorOptions().dtype(torch::kFloat32).device(device);
  auto lhs = torch::randn({result.dim, result.dim}, float_opts);
  auto rhs = torch::randn({result.dim, result.dim}, float_opts);
  auto scalar_ready = torch::ones({1}, float_opts);
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t b_start{};
  cudaEvent_t b_stop{};
  CUDA_CHECK(cudaEventCreate(&b_start));
  CUDA_CHECK(cudaEventCreate(&b_stop));
  torch::Tensor sink;
  {
    c10::cuda::CUDAStreamGuard guard_b(stream_b);
    CUDA_CHECK(cudaEventRecord(b_start, stream_b.stream()));
    for (int i = 0; i < result.iters; ++i) {
      sink = torch::matmul(lhs, rhs);
    }
    CUDA_CHECK(cudaEventRecord(b_stop, stream_b.stream()));
  }

  auto item_start = Clock::now();
  {
    c10::cuda::CUDAStreamGuard guard_a(stream_a);
    double value = scalar_ready.item<double>();
    (void)value;
  }
  result.item_wait_ms = elapsed_ms_since(item_start);

  cudaError_t query = cudaEventQuery(b_stop);
  if (query == cudaErrorNotReady) {
    result.b_pending_after_item = true;
  } else {
    CUDA_CHECK(query);
  }
  auto sync_start = Clock::now();
  CUDA_CHECK(cudaEventSynchronize(b_stop));
  result.sentinel_sync_wall_ms = elapsed_ms_since(sync_start);
  float b_elapsed = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&b_elapsed, b_start, b_stop));
  result.sentinel_gpu_ms = static_cast<double>(b_elapsed);
  CUDA_CHECK(cudaEventDestroy(b_start));
  CUDA_CHECK(cudaEventDestroy(b_stop));
  result.pass = result.b_pending_after_item && result.sentinel_gpu_ms > result.item_wait_ms;

  std::ostringstream json;
  json << "{\"check\":\"0b_scalar_locality_sentinel_probe\""
       << ",\"num_runners\":0"
       << ",\"workers\":2"
       << ",\"stream_mode\":\"explicit\""
       << ",\"topology\":\"explicit_stream_item_does_not_drain_unrelated_streams\""
       << ",\"pass\":" << json_bool(result.pass)
       << ",\"stream_a\":" << result.stream_a
       << ",\"stream_b\":" << result.stream_b
       << ",\"b_pending_after_item\":" << json_bool(result.b_pending_after_item)
       << ",\"item_wait_ms\":" << result.item_wait_ms
       << ",\"sentinel_gpu_ms\":" << result.sentinel_gpu_ms
       << ",\"sentinel_sync_wall_ms\":" << result.sentinel_sync_wall_ms
       << ",\"matmul_dim\":" << result.dim
       << ",\"matmul_iters\":" << result.iters
       << "}";
  emit_telemetry(args.dir,
                 stamp,
                 0,
                 "explicit",
                 "scalar_locality_sentinel_probe",
                 json.str());
  std::printf("=== DENSITY 0b SCALAR LOCALITY PROBE %s: item_wait=%.3fms sentinel_gpu=%.3fms "
              "b_pending_after_item=%s ===\n",
              result.pass ? "PASS" : "FAIL",
              result.item_wait_ms,
              result.sentinel_gpu_ms,
              result.b_pending_after_item ? "true" : "false");
  return result;
}

static CorrectnessResult run_correctness_gate(const DensityArgs& args,
                                              torch::Device device,
                                              const std::string& stamp) {
  torch::jit::Module bundle = torch::jit::load(args.dir + "/session_bundle.ts");
  verify_session_bundle_meta(bundle, false);
  int rows_total = static_cast<int>(scalar_i64(attr_tensor(bundle, "num_utts")));
  int rows = args.correctness_rows > 0 ? std::min(args.correctness_rows, rows_total) : rows_total;
  int workers = args.workers > 0 ? args.workers : args.correctness_n;
  int num_runners = args.num_runners > 0 ? args.num_runners : workers;
  bool explicit_stream = args.stream_mode == "explicit";
  std::printf("=== DENSITY 0b correctness: rows=%d/%d workers=%d num_runners=%d per-worker TS handles ===\n",
              rows, rows_total, workers, num_runners);

  TimingBuckets ref_timings;
  auto refs = build_serial_reference(args, device, rows, &ref_timings);
  std::printf("=== DENSITY 0b serial reference PASS: rows=%d ===\n", rows);

  auto primary = run_correctness_gate_mode(args,
                                           device,
                                           stamp,
                                           rows,
                                           refs,
                                           explicit_stream,
                                           args.mutex_serialize_run,
                                           workers,
                                           num_runners,
                                           "one_process_shared_steady_loader_per_thread_session_handles");
  primary.reference = std::move(refs);

  if (args.correctness_default_stream_control && explicit_stream && !args.mutex_serialize_run) {
    cleanup_cuda_cache();
    auto control = run_correctness_gate_mode(args,
                                             device,
                                             stamp,
                                             rows,
                                             primary.reference,
                                             false,
                                             false,
                                             workers,
                                             num_runners,
                                             "negative_default_stream_per_thread_session_handles");
    if (primary.throughput_rows_per_s > 0.0) {
      primary.default_stream_penalty = 1.0 - (control.throughput_rows_per_s / primary.throughput_rows_per_s);
      primary.default_stream_control_pass = primary.default_stream_penalty >= 0.15 && control.identity_ok;
    }
    std::printf("=== DENSITY 0b DEFAULT-STREAM CONTROL %s: penalty=%.1f%% explicit=%.3f rows/s default=%.3f rows/s ===\n",
                primary.default_stream_control_pass ? "PASS" : "FAIL",
                100.0 * primary.default_stream_penalty,
                primary.throughput_rows_per_s,
                control.throughput_rows_per_s);
  } else {
    primary.default_stream_control_pass = !explicit_stream;
  }
  ScalarLocalityProbeResult scalar_probe;
  if (args.scalar_locality_probe && explicit_stream && !args.mutex_serialize_run) {
    scalar_probe = run_scalar_locality_probe(args, device, stamp);
  } else {
    scalar_probe.pass = !explicit_stream || args.mutex_serialize_run;
  }
  primary.scalar_locality_pass = primary.default_stream_control_pass && scalar_probe.pass;
  primary.ok = primary.identity_ok && primary.stream_uniqueness_ok && primary.scalar_locality_pass;

  std::ostringstream json;
  json << "{\"check\":\"0b_scalar_locality_summary\""
       << ",\"num_runners\":" << primary.num_runners
       << ",\"workers\":" << primary.workers
       << ",\"stream_mode\":\"" << stream_mode_label(primary.explicit_stream, primary.mutex_serialize_run) << "\""
       << ",\"identity_pass\":" << json_bool(primary.identity_ok)
       << ",\"stream_uniqueness_pass\":" << json_bool(primary.stream_uniqueness_ok)
       << ",\"default_stream_penalty\":" << primary.default_stream_penalty
       << ",\"default_stream_control_pass\":" << json_bool(primary.default_stream_control_pass)
       << ",\"sentinel_probe_pass\":" << json_bool(scalar_probe.pass)
       << ",\"scalar_locality_pass\":" << json_bool(primary.scalar_locality_pass)
       << ",\"item_wait_pct_gate\":\"telemetry_only_no_threshold\""
       << ",\"pass\":" << json_bool(primary.ok)
       << "}";
  emit_telemetry(args.dir,
                 stamp,
                 primary.num_runners,
                 stream_mode_label(primary.explicit_stream, primary.mutex_serialize_run),
                 "one_process_shared_steady_loader_per_thread_session_handles_scalar_locality_summary",
                 json.str());
  return primary;
}

struct SteadyCase {
  int utt = -1;
  int chunk = -1;
  std::vector<at::Tensor> inputs;
  std::vector<at::Tensor> oracle_outputs;
};

static std::vector<at::Tensor> clone_tensor_vector(const std::vector<at::Tensor>& tensors) {
  std::vector<at::Tensor> out;
  out.reserve(tensors.size());
  for (const auto& tensor : tensors) out.push_back(tensor.clone());
  return out;
}

static bool compare_steady_oracle_outputs(const SteadyCase& steady_case,
                                          const std::vector<at::Tensor>& out,
                                          double atol,
                                          const std::string& label) {
  if (out.size() < 5 || steady_case.oracle_outputs.size() < 5) {
    std::printf("    %s steady oracle output count mismatch: got=%zu oracle=%zu\n",
                label.c_str(), out.size(), steady_case.oracle_outputs.size());
    return false;
  }
  bool ok = true;
  ok = tensor_close("enc_out", out[0], steady_case.oracle_outputs[0], atol, label) && ok;
  ok = tensor_equal("enc_len", out[1].to(steady_case.oracle_outputs[1].device()), steady_case.oracle_outputs[1]) && ok;
  ok = tensor_close("cache_last_channel", out[2], steady_case.oracle_outputs[2], atol, label) && ok;
  ok = tensor_close("cache_last_time", out[3], steady_case.oracle_outputs[3], atol, label) && ok;
  ok = tensor_close("cache_last_channel_len", out[4], steady_case.oracle_outputs[4], atol, label) && ok;
  return ok;
}

static std::vector<SteadyCase> build_steady_cases(const std::string& dir,
                                                  torch::jit::Module& bundle,
                                                  torch::Device device,
                                                  int limit) {
  auto enc_first = load_module_on_device(dir + "/enc_first.ts", device);
  AOTIModelPackageLoader prep_loader(dir + "/enc_steady_aoti.pt2", "model", false, 1, -1);
  double oracle_atol = bundle.hasattr("cache_ci_atol") ? scalar_f64(attr_tensor(bundle, "cache_ci_atol")) : 0.0;
  int rows = static_cast<int>(scalar_i64(attr_tensor(bundle, "num_utts")));
  std::vector<SteadyCase> cases;
  cases.reserve(static_cast<size_t>(limit));
  SessionState state;
  for (int utt = 0; utt < rows && static_cast<int>(cases.size()) < limit; ++utt) {
    reset_session(state, bundle, device);
    std::string prefix = "utt" + std::to_string(utt);
    int64_t num_steady = scalar_i64(utt_tensor(bundle, utt, "num_steady"));
    for (int chunk_index = 0; chunk_index < num_steady && static_cast<int>(cases.size()) < limit; ++chunk_index) {
      auto new_mel = prefix_chunk_tensor(bundle, prefix, chunk_index, "new_mel").to(device).contiguous();
      if (state.emitted == 0) {
        auto out = run_first_encoder(enc_first, new_mel, state);
        state.clc = out[2].clone();
        state.clt = out[3].clone();
        state.clcl = out[4].clone();
      } else {
        auto chunk = torch::cat({state.ring, new_mel}, 2).contiguous();
        auto L = torch::full({1}, chunk.size(2), torch::dtype(torch::kLong).device(device));
        std::vector<at::Tensor> inputs = {
            chunk.contiguous(),
            L.contiguous(),
            state.clc.contiguous(),
            state.clt.contiguous(),
            state.clcl.contiguous(),
        };
        cases.push_back({utt, chunk_index, inputs});
        auto out = prep_loader.run(inputs);
        if (out.size() < 5) throw std::runtime_error("steady case prep AOTI returned fewer than 5 outputs");
        cases.back().oracle_outputs = clone_tensor_vector(out);
        if (!compare_steady_oracle_outputs(cases.back(), out, oracle_atol,
                                           "density.steady_oracle.utt" + std::to_string(utt) +
                                               ".chunk" + std::to_string(chunk_index))) {
          throw std::runtime_error("steady case serial oracle self-compare failed");
        }
        state.clc = out[2].clone();
        state.clt = out[3].clone();
        state.clcl = out[4].clone();
      }
      auto cum = state.ring.defined() ? torch::cat({state.ring, new_mel}, 2) : new_mel;
      state.ring = cum.slice(2, std::max<int64_t>(0, cum.size(2) - PRE), cum.size(2)).contiguous();
      state.emitted += new_mel.size(2);
    }
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  if (cases.empty()) throw std::runtime_error("no steady continuation cases found in session_bundle.ts");
  std::printf("=== DENSITY steady cases prepared: %zu continuation chunks from session_bundle.ts ===\n",
              cases.size());
  return cases;
}

struct SteadyRunResult {
  int num_runners = 0;
  int workers = 0;
  bool explicit_stream = true;
  bool ok = false;
  int errors = 0;
  int oracle_mismatches = 0;
  size_t oracle_checks = 0;
  size_t calls = 0;
  double wall_ms = 0.0;
  double throughput_calls_per_s = 0.0;
  double contention_confounded_overlap_diagnostic = 0.0;
  int unique_streams = 0;
  bool stream_uniqueness_ok = false;
  size_t used_before_loader = 0;
  size_t used_after_loader = 0;
  size_t loader_delta = 0;
  size_t peak_mem = 0;
  size_t used_after_run = 0;
  TimingBuckets timings;
};

static bool run_steady_case(AOTIModelPackageLoader& loader,
                            const SteadyCase& steady_case,
                            c10::cuda::CUDAStream stream,
                            bool explicit_stream,
                            bool mutex_serialize_run,
                            double oracle_atol,
                            bool compare_oracle,
                            TimingBuckets& timings,
                            const std::string& label) {
  c10::cuda::CUDAStreamGuard stream_guard(stream);
  auto call_start = Clock::now();
  cudaEvent_t ev_start{};
  cudaEvent_t ev_stop{};
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));
  CUDA_CHECK(cudaEventRecord(ev_start, stream.stream()));
  auto run_start = Clock::now();
  auto out = run_aoti_loader(loader, steady_case.inputs, stream, explicit_stream, mutex_serialize_run);
  double runner_host_ms = elapsed_ms_since(run_start);
  CUDA_CHECK(cudaEventRecord(ev_stop, stream.stream()));
  if (out.size() < 5) throw std::runtime_error("steady density run returned fewer than 5 outputs");
  CUDA_CHECK(cudaEventSynchronize(ev_stop));
  float elapsed = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed, ev_start, ev_stop));
  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_stop));
  timings.latency_ms.push_back(elapsed_ms_since(call_start));
  timings.queue_wait_ms.push_back(0.0);
  timings.steady_gpu_ms.push_back(static_cast<double>(elapsed));
  timings.runner_wait_ms.push_back(std::max(0.0, runner_host_ms - static_cast<double>(elapsed)));
  if (compare_oracle) return compare_steady_oracle_outputs(steady_case, out, oracle_atol, label);
  return true;
}

static std::vector<std::vector<SteadyCase>> clone_steady_cases_per_worker(const std::vector<SteadyCase>& cases,
                                                                          int workers) {
  std::vector<std::vector<SteadyCase>> out(static_cast<size_t>(workers));
  for (int worker = 0; worker < workers; ++worker) {
    out[worker].reserve(cases.size());
    for (size_t i = 0; i < cases.size(); ++i) {
      const auto& base = cases[(i + static_cast<size_t>(worker)) % cases.size()];
      SteadyCase cloned;
      cloned.utt = base.utt;
      cloned.chunk = base.chunk;
      cloned.inputs = clone_tensor_vector(base.inputs);
      cloned.oracle_outputs = base.oracle_outputs;
      out[worker].push_back(std::move(cloned));
    }
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  return out;
}

static SteadyRunResult run_steady_overlap_once(const DensityArgs& args,
                                               torch::Device device,
                                               const std::string& stamp,
                                               const std::vector<SteadyCase>& cases,
                                               int workers,
                                               int num_runners,
                                               bool explicit_stream,
                                               bool mutex_serialize_run,
                                               double oracle_atol) {
  SteadyRunResult result;
  result.num_runners = num_runners;
  result.workers = workers;
  result.explicit_stream = explicit_stream;
  auto worker_cases = clone_steady_cases_per_worker(cases, workers);
  cleanup_cuda_cache();
  result.used_before_loader = gpu_used_bytes();
  AOTIModelPackageLoader loader(args.dir + "/enc_steady_aoti.pt2", "model", false, num_runners, -1);
  result.used_after_loader = gpu_used_bytes();
  result.loader_delta = result.used_after_loader >= result.used_before_loader
                            ? result.used_after_loader - result.used_before_loader
                            : 0;

  std::vector<c10::cuda::CUDAStream> streams;
  streams.reserve(static_cast<size_t>(workers));
  std::set<uintptr_t> stream_ids;
  for (int worker = 0; worker < workers; ++worker) {
    auto stream = stream_for_worker(explicit_stream, worker);
    streams.push_back(stream);
    stream_ids.insert(stream_handle_value(stream));
  }
  result.unique_streams = static_cast<int>(stream_ids.size());
  result.stream_uniqueness_ok = !explicit_stream || result.unique_streams == workers;

  TimingBuckets warmup_timings;
  for (int worker = 0; worker < workers; ++worker) {
    run_steady_case(loader,
                    worker_cases[worker][0],
                    streams[worker],
                    explicit_stream,
                    mutex_serialize_run,
                    oracle_atol,
                    false,
                    warmup_timings,
                    "density.steady_warmup.worker" + std::to_string(worker));
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  StartGate oracle_gate(workers);
  std::vector<std::string> oracle_errors(workers);
  std::vector<int> oracle_mismatches(workers, 0);
  std::vector<std::thread> oracle_threads;
  oracle_threads.reserve(static_cast<size_t>(workers));
  for (int worker = 0; worker < workers; ++worker) {
    oracle_threads.emplace_back([&, worker] {
      try {
        c10::cuda::CUDAGuard device_guard(device.index());
        TimingBuckets ignored;
        oracle_gate.arrive_and_wait();
        for (size_t i = 0; i < worker_cases[worker].size(); ++i) {
          bool ok = run_steady_case(loader,
                                    worker_cases[worker][i],
                                    streams[worker],
                                    explicit_stream,
                                    mutex_serialize_run,
                                    oracle_atol,
                                    true,
                                    ignored,
                                    "density.steady_oracle.worker" + std::to_string(worker) +
                                        ".case" + std::to_string(i));
          if (!ok) ++oracle_mismatches[worker];
        }
      } catch (const std::exception& e) {
        oracle_errors[worker] = e.what();
      }
    });
  }
  oracle_gate.wait_until_ready_and_start();
  for (auto& thread : oracle_threads) thread.join();
  CUDA_CHECK(cudaDeviceSynchronize());
  for (int worker = 0; worker < workers; ++worker) {
    result.oracle_checks += worker_cases[worker].size();
    result.oracle_mismatches += oracle_mismatches[worker];
    if (!oracle_errors[worker].empty()) {
      ++result.errors;
      ++result.oracle_mismatches;
      std::printf("  steady oracle worker%d exception: %s\n", worker, oracle_errors[worker].c_str());
    }
  }

  MemorySampler mem;
  mem.start();
  StartGate gate(workers);
  std::vector<TimingBuckets> worker_timings(workers);
  std::vector<std::string> errors(workers);
  std::vector<std::thread> threads;
  threads.reserve(static_cast<size_t>(workers));
  for (int worker = 0; worker < workers; ++worker) {
    threads.emplace_back([&, worker] {
      try {
        c10::cuda::CUDAGuard device_guard(device.index());
        gate.arrive_and_wait();
        for (int repeat = 0; repeat < args.steady_repeats; ++repeat) {
          for (const auto& steady_case : worker_cases[worker]) {
            run_steady_case(loader,
                            steady_case,
                            streams[worker],
                            explicit_stream,
                            mutex_serialize_run,
                            oracle_atol,
                            false,
                            worker_timings[worker],
                            "density.steady_timed.worker" + std::to_string(worker));
          }
        }
      } catch (const std::exception& e) {
        errors[worker] = e.what();
      }
    });
  }
  gate.wait_until_ready_and_start();
  for (auto& thread : threads) thread.join();
  auto end_time = Clock::now();
  CUDA_CHECK(cudaDeviceSynchronize());
  result.wall_ms = elapsed_ms(gate.start_time, end_time);
  result.peak_mem = mem.finish();
  result.used_after_run = gpu_used_bytes();

  for (int worker = 0; worker < workers; ++worker) {
    result.timings.append(worker_timings[worker]);
    if (!errors[worker].empty()) {
      ++result.errors;
      std::printf("  steady worker%d exception: %s\n", worker, errors[worker].c_str());
    }
  }
  result.calls = result.timings.latency_ms.size();
  result.throughput_calls_per_s = static_cast<double>(result.calls) / (result.wall_ms / 1000.0);
  double sum_gpu = std::accumulate(result.timings.steady_gpu_ms.begin(),
                                   result.timings.steady_gpu_ms.end(),
                                   0.0);
  result.contention_confounded_overlap_diagnostic = result.wall_ms > 0.0 ? sum_gpu / result.wall_ms : 0.0;
  result.ok = result.errors == 0 && result.oracle_mismatches == 0 && result.stream_uniqueness_ok;

  const std::string stream_mode = stream_mode_label(explicit_stream, mutex_serialize_run);
  const std::string topology = explicit_stream
                                   ? (mutex_serialize_run ? "negative_mutex_serialized_shared_steady_loader_runner_pool"
                                                          : "shared_steady_loader_runner_pool")
                                   : "negative_default_stream_shared_steady_loader";
  std::ostringstream json;
  json << "{\"check\":\"0a_steady_pool_overlap\""
       << ",\"num_runners\":" << num_runners
       << ",\"workers\":" << workers
       << ",\"stream_mode\":\"" << stream_mode << "\""
       << ",\"topology\":\"" << topology << "\""
       << ",\"cases\":" << cases.size()
       << ",\"repeats\":" << args.steady_repeats
       << ",\"calls\":" << result.calls
       << ",\"errors\":" << result.errors
       << ",\"oracle_checks\":" << result.oracle_checks
       << ",\"oracle_mismatches\":" << result.oracle_mismatches
       << ",\"serial_output_oracle_pass\":" << json_bool(result.oracle_mismatches == 0)
       << ",\"stream_uniqueness_pass\":" << json_bool(result.stream_uniqueness_ok)
       << ",\"pass\":" << json_bool(result.ok)
       << ",\"throughput_calls_per_s\":" << result.throughput_calls_per_s
       << ",\"wall_ms\":" << result.wall_ms
       << ",\"latency\":" << stats_json(summarize(result.timings.latency_ms))
       << ",\"queue_wait\":" << stats_json(summarize(result.timings.queue_wait_ms))
       << ",\"runner_wait\":" << stats_json(summarize(result.timings.runner_wait_ms))
       << ",\"item_wait\":" << stats_json(SummaryStats{})
       << ",\"finalize_wait\":" << stats_json(SummaryStats{})
       << ",\"steady_gpu\":" << stats_json(summarize(result.timings.steady_gpu_ms))
       << ",\"contention_confounded_cuda_event_diagnostic\":" << result.contention_confounded_overlap_diagnostic
       << ",\"overlap_estimate_label\":\"contention_confounded_diagnostic_not_overlap_proof\""
       << ",\"unique_streams\":" << result.unique_streams
       << ",\"stream_handles\":" << stream_handles_json(streams)
       << ",\"peak_gpu_mem_bytes\":" << result.peak_mem
       << ",\"used_before_loader_bytes\":" << result.used_before_loader
       << ",\"used_after_loader_bytes\":" << result.used_after_loader
       << ",\"loader_delta_bytes\":" << result.loader_delta
       << ",\"used_after_run_bytes\":" << result.used_after_run
       << "}";
  emit_telemetry(args.dir, stamp, num_runners, stream_mode, topology, json.str());
  std::printf("=== DENSITY 0a %s: num_runners=%d workers=%d stream_mode=%s calls=%zu throughput=%.3f/s "
              "event_diag=%.2f unique_streams=%d oracle_mismatches=%d loader_delta=%.3f GiB peak_mem=%.3f GiB ===\n",
              result.ok ? "RUN" : "ERROR",
              num_runners,
              workers,
              stream_mode.c_str(),
              result.calls,
              result.throughput_calls_per_s,
              result.contention_confounded_overlap_diagnostic,
              result.unique_streams,
              result.oracle_mismatches,
              static_cast<double>(result.loader_delta) / (1024.0 * 1024.0 * 1024.0),
              static_cast<double>(result.peak_mem) / (1024.0 * 1024.0 * 1024.0));
  return result;
}

struct SteadySweepResult {
  std::map<int, SteadyRunResult> explicit_runs;
  std::unique_ptr<SteadyRunResult> num_runners_one_control;
  std::unique_ptr<SteadyRunResult> default_control;
  std::unique_ptr<SteadyRunResult> mutex_control;
  bool overlap_probe_pass = false;
  bool pass = false;
};

struct SteadyOverlapProbeResult {
  bool ran = false;
  bool pass = false;
  bool stream_b_completed_before_stream_a_done = false;
  bool streams_unique = false;
  double stream_a_wall_ms = 0.0;
  double stream_b_gpu_ms = 0.0;
  double stream_b_sync_wall_ms = 0.0;
  int stream_a_repeats = 16;
  int stream_b_matmul_dim = 2048;
  int stream_b_matmul_iters = 8;
  uintptr_t stream_a = 0;
  uintptr_t stream_b = 0;
};

static SteadyOverlapProbeResult run_steady_overlap_probe(const DensityArgs& args,
                                                         torch::Device device,
                                                         const std::string& stamp,
                                                         const std::vector<SteadyCase>& cases,
                                                         double oracle_atol) {
  SteadyOverlapProbeResult result;
  result.ran = true;
  c10::cuda::CUDAGuard device_guard(device.index());
  AOTIModelPackageLoader loader(args.dir + "/enc_steady_aoti.pt2", "model", false, 2, -1);
  auto stream_a = stream_for_worker(true, 0);
  auto stream_b = stream_for_worker(true, 1);
  result.stream_a = stream_handle_value(stream_a);
  result.stream_b = stream_handle_value(stream_b);
  result.streams_unique = result.stream_a != result.stream_b;
  SteadyCase probe_case;
  probe_case.utt = cases[0].utt;
  probe_case.chunk = cases[0].chunk;
  probe_case.inputs = clone_tensor_vector(cases[0].inputs);
  probe_case.oracle_outputs = cases[0].oracle_outputs;
  auto float_opts = torch::TensorOptions().dtype(torch::kFloat32).device(device);
  auto lhs = torch::randn({result.stream_b_matmul_dim, result.stream_b_matmul_dim}, float_opts);
  auto rhs = torch::randn({result.stream_b_matmul_dim, result.stream_b_matmul_dim}, float_opts);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::atomic<bool> a_started{false};
  std::atomic<bool> a_done{false};
  std::string a_error;
  std::thread stream_a_thread([&] {
    try {
      c10::cuda::CUDAGuard thread_guard(device.index());
      TimingBuckets ignored;
      a_started.store(true, std::memory_order_release);
      auto start = Clock::now();
      for (int i = 0; i < result.stream_a_repeats; ++i) {
        run_steady_case(loader,
                        probe_case,
                        stream_a,
                        true,
                        false,
                        oracle_atol,
                        false,
                        ignored,
                        "density.steady_overlap_probe.stream_a");
      }
      result.stream_a_wall_ms = elapsed_ms_since(start);
      a_done.store(true, std::memory_order_release);
    } catch (const std::exception& e) {
      a_error = e.what();
      a_done.store(true, std::memory_order_release);
    }
  });
  while (!a_started.load(std::memory_order_acquire)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(1));

  cudaEvent_t b_start{};
  cudaEvent_t b_stop{};
  CUDA_CHECK(cudaEventCreate(&b_start));
  CUDA_CHECK(cudaEventCreate(&b_stop));
  torch::Tensor sink;
  {
    c10::cuda::CUDAStreamGuard guard_b(stream_b);
    CUDA_CHECK(cudaEventRecord(b_start, stream_b.stream()));
    for (int i = 0; i < result.stream_b_matmul_iters; ++i) {
      sink = torch::matmul(lhs, rhs);
    }
    CUDA_CHECK(cudaEventRecord(b_stop, stream_b.stream()));
  }
  auto sync_start = Clock::now();
  CUDA_CHECK(cudaEventSynchronize(b_stop));
  result.stream_b_sync_wall_ms = elapsed_ms_since(sync_start);
  result.stream_b_completed_before_stream_a_done = !a_done.load(std::memory_order_acquire);
  float b_elapsed = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&b_elapsed, b_start, b_stop));
  result.stream_b_gpu_ms = static_cast<double>(b_elapsed);
  CUDA_CHECK(cudaEventDestroy(b_start));
  CUDA_CHECK(cudaEventDestroy(b_stop));
  if (stream_a_thread.joinable()) stream_a_thread.join();
  if (!a_error.empty()) std::printf("  steady overlap probe stream A exception: %s\n", a_error.c_str());
  result.pass = a_error.empty() && result.streams_unique && result.stream_b_completed_before_stream_a_done;

  std::ostringstream json;
  json << "{\"check\":\"0a_steady_overlap_sentinel_probe\""
       << ",\"num_runners\":2"
       << ",\"workers\":2"
       << ",\"stream_mode\":\"explicit\""
       << ",\"topology\":\"sentinel_stream_b_runs_while_stream_a_encoder_active\""
       << ",\"pass\":" << json_bool(result.pass)
       << ",\"stream_a\":" << result.stream_a
       << ",\"stream_b\":" << result.stream_b
       << ",\"streams_unique\":" << json_bool(result.streams_unique)
       << ",\"stream_b_completed_before_stream_a_done\":" << json_bool(result.stream_b_completed_before_stream_a_done)
       << ",\"stream_a_wall_ms\":" << result.stream_a_wall_ms
       << ",\"stream_b_gpu_ms\":" << result.stream_b_gpu_ms
       << ",\"stream_b_sync_wall_ms\":" << result.stream_b_sync_wall_ms
       << ",\"stream_a_repeats\":" << result.stream_a_repeats
       << ",\"stream_b_matmul_dim\":" << result.stream_b_matmul_dim
       << ",\"stream_b_matmul_iters\":" << result.stream_b_matmul_iters
       << "}";
  emit_telemetry(args.dir, stamp, 2, "explicit", "steady_overlap_sentinel_probe", json.str());
  std::printf("=== DENSITY 0a OVERLAP SENTINEL %s: stream_b_gpu=%.3fms stream_a_wall=%.3fms "
              "b_completed_before_a_done=%s ===\n",
              result.pass ? "PASS" : "FAIL",
              result.stream_b_gpu_ms,
              result.stream_a_wall_ms,
              result.stream_b_completed_before_stream_a_done ? "true" : "false");
  return result;
}

static SteadySweepResult run_steady_sweep(const DensityArgs& args,
                                          torch::Device device,
                                          const std::string& stamp) {
  torch::jit::Module bundle = torch::jit::load(args.dir + "/session_bundle.ts");
  verify_session_bundle_meta(bundle, false);
  auto cases = build_steady_cases(args.dir, bundle, device, args.steady_cases);
  double oracle_atol = bundle.hasattr("cache_ci_atol") ? scalar_f64(attr_tensor(bundle, "cache_ci_atol")) : 0.0;
  cleanup_cuda_cache();

  SteadySweepResult sweep;
  for (int n : args.n_values) {
    int workers = args.workers > 0 ? args.workers : n;
    int num_runners = args.num_runners > 0 ? args.num_runners : n;
    bool explicit_stream = args.stream_mode == "explicit";
    sweep.explicit_runs.emplace(n, run_steady_overlap_once(args,
                                                           device,
                                                           stamp,
                                                           cases,
                                                           workers,
                                                           num_runners,
                                                           explicit_stream,
                                                           args.mutex_serialize_run,
                                                           oracle_atol));
    cleanup_cuda_cache();
  }
  if (args.default_stream_control) {
    int control_n = (args.target_n > 0 &&
                     std::find(args.n_values.begin(), args.n_values.end(), args.target_n) != args.n_values.end())
                        ? args.target_n
                        : (std::find(args.n_values.begin(), args.n_values.end(), 4) != args.n_values.end()
                               ? 4
                               : args.n_values.back());
    int control_workers = args.workers > 0 ? args.workers : control_n;
    int control_runners = args.num_runners > 0 ? args.num_runners : control_workers;
    if (args.stream_mode == "explicit" && !args.mutex_serialize_run && args.num_runners == 0 && args.workers == 0) {
      sweep.num_runners_one_control = std::make_unique<SteadyRunResult>(
          run_steady_overlap_once(args, device, stamp, cases, control_workers, 1, true, false, oracle_atol));
      cleanup_cuda_cache();
      sweep.mutex_control = std::make_unique<SteadyRunResult>(
          run_steady_overlap_once(args, device, stamp, cases, control_workers, control_workers, true, true, oracle_atol));
      cleanup_cuda_cache();
    }
    sweep.default_control = std::make_unique<SteadyRunResult>(
        run_steady_overlap_once(args, device, stamp, cases, control_workers, control_runners, false, false, oracle_atol));
    cleanup_cuda_cache();
  }
  if (args.steady_overlap_probe && args.stream_mode == "explicit" && !args.mutex_serialize_run) {
    auto probe = run_steady_overlap_probe(args, device, stamp, cases, oracle_atol);
    sweep.overlap_probe_pass = probe.pass;
    cleanup_cuda_cache();
  }

  double base = sweep.explicit_runs.count(1) ? sweep.explicit_runs[1].throughput_calls_per_s : 0.0;
  double speedup2 = (base > 0.0 && sweep.explicit_runs.count(2))
                        ? sweep.explicit_runs[2].throughput_calls_per_s / base
                        : 0.0;
  double speedup4 = (base > 0.0 && sweep.explicit_runs.count(4))
                        ? sweep.explicit_runs[4].throughput_calls_per_s / base
                        : 0.0;
  double peak_mem_ratio4 = 0.0;
  if (sweep.explicit_runs.count(1) && sweep.explicit_runs.count(4) && sweep.explicit_runs[1].peak_mem > 0) {
    peak_mem_ratio4 = static_cast<double>(sweep.explicit_runs[4].peak_mem) /
                      static_cast<double>(sweep.explicit_runs[1].peak_mem);
  }
  double target_peak_mem_ratio = 0.0;
  if (args.target_n > 0 &&
      sweep.explicit_runs.count(1) &&
      sweep.explicit_runs.count(args.target_n) &&
      sweep.explicit_runs[1].peak_mem > 0) {
    target_peak_mem_ratio = static_cast<double>(sweep.explicit_runs[args.target_n].peak_mem) /
                            static_cast<double>(sweep.explicit_runs[1].peak_mem);
  }
  double loader_delta_ratio4 = 0.0;
  if (sweep.explicit_runs.count(1) && sweep.explicit_runs.count(4) && sweep.explicit_runs[1].loader_delta > 0) {
    loader_delta_ratio4 = static_cast<double>(sweep.explicit_runs[4].loader_delta) /
                          static_cast<double>(sweep.explicit_runs[1].loader_delta);
  }
  double target_loader_delta_ratio = 0.0;
  if (args.target_n > 0 &&
      sweep.explicit_runs.count(1) &&
      sweep.explicit_runs.count(args.target_n) &&
      sweep.explicit_runs[1].loader_delta > 0) {
    target_loader_delta_ratio = static_cast<double>(sweep.explicit_runs[args.target_n].loader_delta) /
                                static_cast<double>(sweep.explicit_runs[1].loader_delta);
  }
  bool default_control_pass = true;
  double default_penalty = 0.0;
  if (sweep.default_control) {
    double explicit_thr = 0.0;
    for (const auto& kv : sweep.explicit_runs) {
      if (kv.second.workers == sweep.default_control->workers &&
          kv.second.num_runners == sweep.default_control->num_runners) {
        explicit_thr = kv.second.throughput_calls_per_s;
        break;
      }
    }
    if (explicit_thr > 0.0) {
      default_penalty = 1.0 - (sweep.default_control->throughput_calls_per_s / explicit_thr);
      default_control_pass = default_penalty >= 0.15;
    }
  }
  bool loader_delta_flat = false;
  if (target_loader_delta_ratio > 0.0) {
    loader_delta_flat = target_loader_delta_ratio <= 1.15;
  } else if (loader_delta_ratio4 > 0.0) {
    loader_delta_flat = loader_delta_ratio4 <= 1.15;
  }
  bool primary_runs_ok = true;
  for (const auto& kv : sweep.explicit_runs) primary_runs_ok = primary_runs_ok && kv.second.ok;
  sweep.pass = speedup2 >= 1.15 &&
               speedup4 >= 1.30 &&
               loader_delta_flat &&
               default_control_pass &&
               sweep.overlap_probe_pass &&
               primary_runs_ok;

  std::ostringstream summary;
  summary << "{\"check\":\"0a_steady_pool_overlap_summary\""
          << ",\"num_runners\":0"
          << ",\"stream_mode\":\"" << stream_mode_label(args.stream_mode == "explicit", args.mutex_serialize_run) << "\""
          << ",\"topology\":\"shared_steady_loader_runner_pool\""
          << ",\"pass\":" << json_bool(sweep.pass)
          << ",\"speedup_n2\":" << speedup2
          << ",\"speedup_n4\":" << speedup4
          << ",\"memory_gate_metric\":\"loader_delta_used_after_loader_minus_used_before_loader\""
          << ",\"loader_delta_ratio_n4_vs_n1\":" << loader_delta_ratio4
          << ",\"target_n\":" << args.target_n
          << ",\"loader_delta_ratio_target_vs_n1\":" << target_loader_delta_ratio
          << ",\"loader_delta_flat_pass\":" << json_bool(loader_delta_flat)
          << ",\"peak_mem_ratio_n4_vs_n1_diagnostic\":" << peak_mem_ratio4
          << ",\"peak_mem_ratio_target_vs_n1_diagnostic\":" << target_peak_mem_ratio
          << ",\"default_stream_penalty\":" << default_penalty
          << ",\"default_stream_control_pass\":" << json_bool(default_control_pass)
          << ",\"overlap_proof\":\"sentinel_probe\""
          << ",\"overlap_probe_pass\":" << json_bool(sweep.overlap_probe_pass)
          << ",\"primary_runs_pass\":" << json_bool(primary_runs_ok)
          << "}";
  emit_telemetry(args.dir,
                 stamp,
                 0,
                 "explicit",
                 "shared_steady_loader_runner_pool_summary",
                 summary.str());
  std::printf("=== DENSITY 0a SUMMARY %s: speedup@2=%.3fx speedup@4=%.3fx mem_ratio@4=%.3f "
              "loader_delta_ratio@4=%.3f targetN=%d target_loader_delta_ratio=%.3f "
              "default_penalty=%.1f%% overlap_probe=%s ===\n",
              sweep.pass ? "PASS" : "FAIL",
              speedup2,
              speedup4,
              peak_mem_ratio4,
              loader_delta_ratio4,
              args.target_n,
              target_loader_delta_ratio,
              100.0 * default_penalty,
              sweep.overlap_probe_pass ? "PASS" : "FAIL");
  return sweep;
}

struct FinalizeCase {
  int utt = -1;
  int64_t drop = -1;
  int64_t T = -1;
};

static std::vector<FinalizeBucketKey> unique_finalize_bucket_keys_from_cases(const std::vector<FinalizeCase>& cases) {
  std::vector<FinalizeBucketKey> keys;
  keys.reserve(cases.size());
  for (const auto& item : cases) keys.emplace_back(item.drop, item.T);
  return unique_finalize_bucket_keys(keys);
}

static std::vector<FinalizeCase> discover_finalize_cases(torch::jit::Module& bundle, int rows) {
  std::vector<FinalizeCase> cases;
  cases.reserve(static_cast<size_t>(rows));
  for (int utt = 0; utt < rows; ++utt) {
    int64_t drop = scalar_i64(utt_tensor(bundle, utt, "final_drop_extra"));
    int64_t T = scalar_i64(utt_tensor(bundle, utt, "final_T"));
    if (T <= 0) continue;
    cases.push_back({
        utt,
        drop,
        T,
    });
  }
  return cases;
}

static std::vector<FinalizeCase> pick_mixed_finalize_cases(const std::vector<FinalizeCase>& all, int n) {
  std::map<std::pair<int64_t, int64_t>, FinalizeCase> by_bucket;
  for (const auto& item : all) by_bucket.emplace(std::make_pair(item.drop, item.T), item);
  std::vector<FinalizeCase> out;
  for (const auto& kv : by_bucket) {
    out.push_back(kv.second);
    if (static_cast<int>(out.size()) == n) break;
  }
  if (out.empty()) throw std::runtime_error("no finalize cases available");
  while (static_cast<int>(out.size()) < n) out.push_back(out[out.size() % by_bucket.size()]);
  return out;
}

static std::vector<FinalizeCase> pick_same_bucket_finalize_cases(const std::vector<FinalizeCase>& all, int n) {
  if (all.empty()) throw std::runtime_error("no finalize cases available");
  std::map<std::pair<int64_t, int64_t>, std::vector<FinalizeCase>> by_bucket;
  for (const auto& item : all) by_bucket[std::make_pair(item.drop, item.T)].push_back(item);
  auto best = by_bucket.begin();
  for (auto it = by_bucket.begin(); it != by_bucket.end(); ++it) {
    if (it->second.size() > best->second.size()) best = it;
  }
  std::vector<FinalizeCase> out;
  out.reserve(static_cast<size_t>(n));
  for (const auto& item : best->second) {
    out.push_back(item);
    if (static_cast<int>(out.size()) == n) return out;
  }
  while (static_cast<int>(out.size()) < n) out.push_back(best->second[out.size() % best->second.size()]);
  return out;
}

struct FinalizeGateResult {
  bool ok = false;
  bool stream_uniqueness_ok = false;
  int workers = 0;
  int num_runners = 0;
  int steady_num_runners = 0;
  int finalize_num_runners = 0;
  int unique_streams = 0;
  int mismatches = 0;
  double wall_ms = 0.0;
  double throughput_finalize_per_s = 0.0;
  TimingBuckets timings;
  size_t peak_mem = 0;
  std::vector<uintptr_t> stream_handles;
};

static void prepare_finalize_parent(const FinalizeCase& fc,
                                    WorkerContext& ctx,
                                    AOTIModelPackageLoader& enc_steady,
                                    torch::Device device,
                                    const Tokenizer& tokenizer,
                                    bool explicit_stream,
                                    bool mutex_serialize_run,
                                    SessionState& session,
                                    std::vector<EmittedEvent>& events) {
  reset_session(session, ctx.bundle, device);
  std::string prefix = "utt" + std::to_string(fc.utt);
  int64_t num_steady = scalar_i64(utt_tensor(ctx.bundle, fc.utt, "num_steady"));
  TimingBuckets ignored;
  for (int chunk = 0; chunk < num_steady; ++chunk) {
    run_steady_chunk_density(session,
                             ctx.bundle,
                             prefix,
                             chunk,
                             ctx.enc_first,
                             enc_steady,
                             ctx.joint,
                             ctx.predict,
                             device,
                             tokenizer,
                             events,
                             ctx.stream,
                             explicit_stream,
                             mutex_serialize_run,
                             &ignored,
                             "density.finalize_prep.utt" + std::to_string(fc.utt) + ".chunk" + std::to_string(chunk));
  }
  session.mode = SessionMode::PENDING_FINALIZE;
}

static FinalizeGateResult run_finalize_gate_one(const DensityArgs& args,
                                                torch::Device device,
                                                const std::string& stamp,
                                                const std::string& mode,
                                                const std::vector<FinalizeCase>& cases,
                                                const std::vector<RowReplayResult>& reference) {
  FinalizeGateResult result;
  result.workers = static_cast<int>(cases.size());
  result.steady_num_runners = capped_general_finalize_runners(args.num_runners > 0 ? args.num_runners : result.workers);
  bool hot_same_bucket = mode == "same_bucket";
  result.finalize_num_runners = hot_same_bucket ? result.workers : capped_general_finalize_runners(result.workers);
  result.num_runners = result.finalize_num_runners;
  MemorySampler mem;
  mem.start();
  AOTIModelPackageLoader enc_steady(args.dir + "/enc_steady_aoti.pt2", "model", false, result.steady_num_runners, -1);
  FinalizeBucketLoaderPool finalize_loaders(
      args.dir,
      device,
      result.finalize_num_runners,
      hot_same_bucket ? "0c_hot_same_bucket_one_bucket_full_worker_runners"
                      : "0c_mixed_bucket_selected_buckets_capped_runner_pool");
  auto needed_buckets = unique_finalize_bucket_keys_from_cases(cases);
  finalize_loaders.preload(needed_buckets);
  std::vector<std::unique_ptr<WorkerContext>> contexts;
  contexts.reserve(static_cast<size_t>(result.workers));
  std::vector<c10::cuda::CUDAStream> streams;
  streams.reserve(static_cast<size_t>(result.workers));
  std::set<uintptr_t> stream_ids;
  for (int worker = 0; worker < result.workers; ++worker) {
    auto stream = stream_for_worker(true, worker);
    streams.push_back(stream);
    uintptr_t handle = stream_handle_value(stream);
    stream_ids.insert(handle);
    result.stream_handles.push_back(handle);
    contexts.push_back(make_worker_context(args.dir, device, stream));
  }
  result.unique_streams = static_cast<int>(stream_ids.size());
  result.stream_uniqueness_ok = result.unique_streams == result.workers;
  auto tokenizer = tokenizer_from_bundle(contexts[0]->bundle);

  StartGate gate(result.workers);
  std::vector<TimingBuckets> worker_timings(result.workers);
  std::vector<std::string> errors(result.workers);
  std::vector<int> mismatches(result.workers, 0);
  std::vector<std::thread> threads;
  threads.reserve(static_cast<size_t>(result.workers));
  for (int worker = 0; worker < result.workers; ++worker) {
    threads.emplace_back([&, worker] {
      try {
        c10::cuda::CUDAGuard device_guard(device.index());
        SessionState session;
        std::vector<EmittedEvent> events;
        prepare_finalize_parent(cases[worker],
                                *contexts[worker],
                                enc_steady,
                                device,
                                tokenizer,
                                true,
                                args.mutex_serialize_run,
                                session,
                                events);
        gate.arrive_and_wait();
        std::string prefix = "utt" + std::to_string(cases[worker].utt);
        auto outcome = run_finalize_density(session,
                                            contexts[worker]->bundle,
                                            prefix,
                                            "density.finalize_" + mode + ".worker" + std::to_string(worker),
                                            finalize_loaders,
                                            contexts[worker]->joint,
                                            contexts[worker]->predict,
                                            device,
                                            tokenizer,
                                            events,
                                            FinalizeFinish::SPECULATIVE_KEEP,
                                            contexts[worker]->stream,
                                            true,
                                            args.mutex_serialize_run,
                                            &worker_timings[worker]);
        bool same = outcome.token_ok &&
                    outcome.fork_ok &&
                    outcome.final_tokens == reference[cases[worker].utt].final_tokens &&
                    strict_events_equal(events,
                                        reference[cases[worker].utt].events,
                                        "density.finalize_" + mode + ".worker" + std::to_string(worker));
        if (!same) ++mismatches[worker];
      } catch (const std::exception& e) {
        errors[worker] = e.what();
      }
    });
  }
  gate.wait_until_ready_and_start();
  for (auto& t : threads) t.join();
  auto end_time = Clock::now();
  CUDA_CHECK(cudaDeviceSynchronize());
  result.wall_ms = elapsed_ms(gate.start_time, end_time);
  result.peak_mem = mem.finish();

  for (int worker = 0; worker < result.workers; ++worker) {
    result.timings.append(worker_timings[worker]);
    result.mismatches += mismatches[worker];
    if (!errors[worker].empty()) {
      ++result.mismatches;
      std::printf("  finalize %s worker%d exception: %s\n", mode.c_str(), worker, errors[worker].c_str());
    }
  }
  result.throughput_finalize_per_s = static_cast<double>(result.workers) / (result.wall_ms / 1000.0);
  auto finalize_total = summarize(result.timings.finalize_total_ms);
  auto finalize_wait = summarize(result.timings.finalize_runner_wait_ms);
  double wait_pct = finalize_total.p95 > 0.0 ? 100.0 * finalize_wait.p95 / finalize_total.p95 : 0.0;
  result.ok = result.mismatches == 0 && wait_pct <= 25.0 && result.stream_uniqueness_ok;

  std::ostringstream buckets;
  buckets << "[";
  for (size_t i = 0; i < cases.size(); ++i) {
    if (i > 0) buckets << ",";
    buckets << "{\"utt\":" << cases[i].utt << ",\"drop\":" << cases[i].drop << ",\"T\":" << cases[i].T << "}";
  }
  buckets << "]";

  std::ostringstream json;
  json << "{\"check\":\"0c_finalize_concurrency_" << mode << "\""
       << ",\"num_runners\":" << result.num_runners
       << ",\"steady_num_runners\":" << result.steady_num_runners
       << ",\"finalize_num_runners_per_loaded_bucket\":" << result.finalize_num_runners
       << ",\"workers\":" << result.workers
       << ",\"stream_mode\":\"" << stream_mode_label(true, args.mutex_serialize_run) << "\""
       << ",\"topology\":\"shared_finalize_bucket_runner_pool_" << mode << "\""
       << ",\"buckets\":" << buckets.str()
       << ",\"mismatches\":" << result.mismatches
       << ",\"stream_uniqueness_pass\":" << json_bool(result.stream_uniqueness_ok)
       << ",\"unique_streams\":" << result.unique_streams
       << ",\"stream_handles\":" << stream_handles_json(streams)
       << ",\"pass\":" << json_bool(result.ok)
       << ",\"throughput_finalize_per_s\":" << result.throughput_finalize_per_s
       << ",\"wall_ms\":" << result.wall_ms
       << ",\"finalize_wait\":" << stats_json(finalize_wait)
       << ",\"finalize_gpu\":" << stats_json(summarize(result.timings.finalize_gpu_ms))
       << ",\"finalize_total\":" << stats_json(finalize_total)
       << ",\"finalize_phases\":" << finalize_phase_stats_json(result.timings)
       << ",\"finalize_runner_wait_pct_of_total_p95\":" << wait_pct
       << ",\"finalize_loader_memory\":" << finalize_loaders.memory_json(result.workers)
       << ",\"peak_gpu_mem_bytes\":" << result.peak_mem
       << "}";
  emit_telemetry(args.dir,
                 stamp,
                 result.num_runners,
                 stream_mode_label(true, args.mutex_serialize_run),
                 "shared_finalize_bucket_runner_pool_" + mode,
                 json.str());
  std::printf("=== DENSITY 0c %s %s: workers=%d steady_num_runners=%d finalize_num_runners=%d "
              "loaded_buckets=%zu/%zu loader_delta=%.3f GiB unique_streams=%d mismatches=%d "
              "finalize_wait_p95=%.3fms total_p95=%.3fms wait_pct=%.1f%% peak_mem=%.3f GiB ===\n",
              mode.c_str(),
              result.ok ? "PASS" : "FAIL",
              result.workers,
              result.steady_num_runners,
              result.finalize_num_runners,
              finalize_loaders.loaded_bucket_count(),
              finalize_loaders.total_bucket_count(),
              static_cast<double>(finalize_loaders.total_loader_delta()) / (1024.0 * 1024.0 * 1024.0),
              result.unique_streams,
              result.mismatches,
              finalize_wait.p95,
              finalize_total.p95,
              wait_pct,
              static_cast<double>(result.peak_mem) / (1024.0 * 1024.0 * 1024.0));
  return result;
}

static bool run_finalize_gate(const DensityArgs& args,
                              torch::Device device,
                              const std::string& stamp,
                              const CorrectnessResult& correctness) {
  torch::jit::Module bundle = torch::jit::load(args.dir + "/session_bundle.ts");
  verify_session_bundle_meta(bundle, false);
  int rows_total = static_cast<int>(scalar_i64(attr_tensor(bundle, "num_utts")));
  int rows = args.correctness_rows > 0 ? std::min(args.correctness_rows, rows_total) : rows_total;
  auto all_cases = discover_finalize_cases(bundle, rows);
  bool ok = true;
  if (args.finalize_mode == "both" || args.finalize_mode == "mixed") {
    auto mixed = pick_mixed_finalize_cases(all_cases, args.finalize_n);
    auto mixed_result = run_finalize_gate_one(args, device, stamp, "mixed_bucket", mixed, correctness.reference);
    ok = ok && mixed_result.ok;
    cleanup_cuda_cache();
  }
  if (args.finalize_mode == "both" || args.finalize_mode == "same") {
    auto same = pick_same_bucket_finalize_cases(all_cases, args.finalize_n);
    auto same_result = run_finalize_gate_one(args, device, stamp, "same_bucket", same, correctness.reference);
    ok = ok && same_result.ok;
    cleanup_cuda_cache();
  }
  return ok;
}

struct DensitySweepRunResult {
  int n = 0;
  int workers = 0;
  int num_runners = 0;
  int finalize_num_runners = 0;
  int rows_total = 0;
  int requested_sessions = 0;
  int sessions_completed = 0;
  int chunks_completed = 0;
  int finalize_samples = 0;
  int warmup_steady_workers = 0;
  int warmup_finalize_buckets = 0;
  int errors = 0;
  int mismatches = 0;
  int unique_streams = 0;
  bool stream_uniqueness_ok = false;
  bool completed = false;
  bool oom = false;
  bool keepup_ok = false;
  bool ttfs_ok = false;
  bool finalize_p95_valid = false;
  bool correctness_ok = false;
  bool slo_robust = false;
  bool explicit_stream = true;
  double wall_ms = 0.0;
  double offered_audio_ms = 0.0;
  double throughput_realtime_streams = 0.0;
  double throughput_sessions_per_s = 0.0;
  size_t used_before_bytes = 0;
  size_t used_after_loaders_bytes = 0;
  size_t used_after_run_bytes = 0;
  size_t peak_mem_bytes = 0;
  size_t total_mem_bytes = 0;
  TimingBuckets timings;
  std::vector<double> lag_ms;
  std::vector<double> ttfs_ms;
  std::vector<uintptr_t> stream_handles;
  ResourceStats resource_stats;
  std::string finalize_loader_memory_json = "{}";
  std::string error;
};

struct DensityWorkerOutput {
  TimingBuckets timings;
  std::vector<double> lag_ms;
  std::vector<double> ttfs_ms;
  int sessions_completed = 0;
  int chunks_completed = 0;
  int mismatches = 0;
  double offered_audio_ms = 0.0;
  std::string error;
};

static std::vector<std::vector<int>> assign_density_utts(int workers,
                                                         int rows_total,
                                                         int requested_sessions) {
  if (workers <= 0) throw std::runtime_error("density sweep workers must be positive");
  if (rows_total <= 0) throw std::runtime_error("density sweep requires at least one utterance");
  std::vector<std::vector<int>> assigned(static_cast<size_t>(workers));
  for (int i = 0; i < requested_sessions; ++i) {
    assigned[static_cast<size_t>(i % workers)].push_back(i % rows_total);
  }
  return assigned;
}

static std::vector<FinalizeBucketKey> needed_finalize_buckets_for_assignments(torch::jit::Module& bundle,
                                                                              const std::vector<std::vector<int>>& assigned) {
  std::vector<FinalizeBucketKey> keys;
  for (const auto& worker_utts : assigned) {
    for (int utt : worker_utts) {
      int64_t T = scalar_i64(utt_tensor(bundle, utt, "final_T"));
      if (T <= 0) continue;
      int64_t drop = scalar_i64(utt_tensor(bundle, utt, "final_drop_extra"));
      keys.emplace_back(drop, T);
    }
  }
  return unique_finalize_bucket_keys(keys);
}

static std::map<FinalizeBucketKey, FinalizeCase> representative_finalize_cases_for_assignments(
    torch::jit::Module& bundle,
    const std::vector<std::vector<int>>& assigned) {
  std::map<FinalizeBucketKey, FinalizeCase> reps;
  for (const auto& worker_utts : assigned) {
    for (int utt : worker_utts) {
      int64_t T = scalar_i64(utt_tensor(bundle, utt, "final_T"));
      if (T <= 0) continue;
      int64_t drop = scalar_i64(utt_tensor(bundle, utt, "final_drop_extra"));
      FinalizeBucketKey key = std::make_pair(drop, T);
      if (reps.find(key) == reps.end()) {
        reps.emplace(key, FinalizeCase{utt, drop, T});
      }
    }
  }
  return reps;
}

static std::vector<FinalizeBucketKey> finalize_keys_from_representatives(
    const std::map<FinalizeBucketKey, FinalizeCase>& reps) {
  std::vector<FinalizeBucketKey> keys;
  keys.reserve(reps.size());
  for (const auto& kv : reps) keys.push_back(kv.first);
  return keys;
}

static std::vector<std::map<FinalizeBucketKey, FinalizeCase>> representative_finalize_cases_by_worker(
    torch::jit::Module& bundle,
    const std::vector<std::vector<int>>& assigned) {
  std::vector<std::map<FinalizeBucketKey, FinalizeCase>> reps(assigned.size());
  for (size_t worker = 0; worker < assigned.size(); ++worker) {
    for (int utt : assigned[worker]) {
      int64_t T = scalar_i64(utt_tensor(bundle, utt, "final_T"));
      if (T <= 0) continue;
      int64_t drop = scalar_i64(utt_tensor(bundle, utt, "final_drop_extra"));
      FinalizeBucketKey key = std::make_pair(drop, T);
      if (reps[worker].find(key) == reps[worker].end()) {
        reps[worker].emplace(key, FinalizeCase{utt, drop, T});
      }
    }
  }
  return reps;
}

static int count_finalize_samples_for_request(torch::jit::Module& bundle,
                                              int rows_total,
                                              int requested_sessions) {
  int count = 0;
  for (int i = 0; i < requested_sessions; ++i) {
    int utt = i % rows_total;
    if (scalar_i64(utt_tensor(bundle, utt, "final_T")) > 0) ++count;
  }
  return count;
}

static int raise_request_for_valid_finalize_p95(torch::jit::Module& bundle,
                                                int rows_total,
                                                int requested_sessions) {
  int finalize_rows_per_cycle = count_finalize_samples_for_request(bundle, rows_total, rows_total);
  if (finalize_rows_per_cycle <= 0) {
    throw std::runtime_error("density sweep cannot collect finalize p95: no rows with final_T > 0");
  }
  int out = requested_sessions;
  while (count_finalize_samples_for_request(bundle, rows_total, out) < kMinFinalizeP95Samples) {
    ++out;
  }
  return out;
}

static int pick_assigned_steady_warmup_utt(torch::jit::Module& bundle,
                                           const std::vector<int>& worker_utts) {
  for (int utt : worker_utts) {
    if (scalar_i64(utt_tensor(bundle, utt, "num_steady")) >= 2) return utt;
  }
  return -1;
}

static void warm_steady_encoder_once_density(int utt,
                                             WorkerContext& ctx,
                                             AOTIModelPackageLoader& enc_steady,
                                             torch::Device device,
                                             const Tokenizer& tokenizer,
                                             bool explicit_stream,
                                             bool mutex_serialize_run,
                                             const std::string& label) {
  int64_t num_steady = scalar_i64(utt_tensor(ctx.bundle, utt, "num_steady"));
  if (num_steady < 2) {
    throw std::runtime_error("density steady warmup requires an utterance with at least two steady chunks");
  }
  SessionState session;
  reset_session(session, ctx.bundle, device);
  std::vector<EmittedEvent> events;
  std::string prefix = "utt" + std::to_string(utt);
  for (int chunk = 0; chunk < 2; ++chunk) {
    run_steady_chunk_density(session,
                             ctx.bundle,
                             prefix,
                             chunk,
                             ctx.enc_first,
                             enc_steady,
                             ctx.joint,
                             ctx.predict,
                             device,
                             tokenizer,
                             events,
                             ctx.stream,
                             explicit_stream,
                             mutex_serialize_run,
                             nullptr,
                             label + ".chunk" + std::to_string(chunk));
  }
}

static std::string uintptr_list_json(const std::vector<uintptr_t>& values) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0) oss << ",";
    oss << values[i];
  }
  oss << "]";
  return oss.str();
}

static DensitySweepRunResult run_density_sweep_one_impl(const DensityArgs& args,
                                                        torch::Device device,
                                                        const std::string& stamp,
                                                        int n,
                                                        int rows_total,
                                                        const std::vector<RowReplayResult>& reference) {
  DensitySweepRunResult result;
  result.n = n;
  result.workers = n;
  result.num_runners = n;
  result.finalize_num_runners = capped_general_finalize_runners(n);
  result.rows_total = rows_total;
  result.explicit_stream = args.stream_mode == "explicit";
  result.total_mem_bytes = gpu_total_bytes();
  if (n <= 0) throw std::runtime_error("density sweep N must be positive");
  if (!result.explicit_stream) {
    std::printf("=== DENSITY 1a WARNING: --stream-mode=%s is a control, not the Step-1a proven topology ===\n",
                args.stream_mode.c_str());
  }

  if (args.density_sessions_per_worker > 0) {
    result.requested_sessions = args.density_sessions_per_worker * n;
  } else if (args.density_rows > 0) {
    result.requested_sessions = args.density_rows;
  } else {
    result.requested_sessions = rows_total;
  }
  if (result.requested_sessions <= 0) throw std::runtime_error("density sweep requested zero sessions");
  torch::jit::Module assignment_bundle = torch::jit::load(args.dir + "/session_bundle.ts");
  verify_session_bundle_meta(assignment_bundle, false);
  int original_requested_sessions = result.requested_sessions;
  result.requested_sessions = raise_request_for_valid_finalize_p95(assignment_bundle,
                                                                   rows_total,
                                                                   result.requested_sessions);
  if (result.requested_sessions != original_requested_sessions) {
    std::printf("=== DENSITY 1a SAMPLE FLOOR: bumped requested sessions from %d to %d "
                "to collect at least %d finalize samples for valid p95 ===\n",
                original_requested_sessions,
                result.requested_sessions,
                kMinFinalizeP95Samples);
  }
  int reference_rows_needed = std::min(rows_total, result.requested_sessions);
  if (static_cast<int>(reference.size()) < reference_rows_needed) {
    throw std::runtime_error("density sweep serial reference is smaller than the assigned utterance set");
  }
  auto assigned = assign_density_utts(result.workers, rows_total, result.requested_sessions);
  auto bucket_reps = representative_finalize_cases_for_assignments(assignment_bundle, assigned);
  auto needed_buckets = finalize_keys_from_representatives(bucket_reps);
  auto worker_bucket_reps = representative_finalize_cases_by_worker(assignment_bundle, assigned);

  std::printf("=== DENSITY 1a RUN START: N=%d workers=%d steady_num_runners=%d finalize_num_runners=%d "
              "sessions=%d rows_total=%d finalize_samples_requested=%d min_finalize_p95_samples=%d "
              "cadence=%.3fms ===\n",
              n,
              result.workers,
              result.num_runners,
              result.finalize_num_runners,
              result.requested_sessions,
              rows_total,
              count_finalize_samples_for_request(assignment_bundle, rows_total, result.requested_sessions),
              kMinFinalizeP95Samples,
              args.density_chunk_period_ms);

  cleanup_cuda_cache();
  MemorySampler mem;
  mem.start();
  result.used_before_bytes = gpu_used_bytes();
  AOTIModelPackageLoader enc_steady(args.dir + "/enc_steady_aoti.pt2", "model", false, result.num_runners, -1);
  FinalizeBucketLoaderPool finalize_loaders(args.dir,
                                            device,
                                            result.finalize_num_runners,
                                            "1a_density_sweep_capped_finalize_runner_pool");

  finalize_loaders.preload(needed_buckets);
  CUDA_CHECK(cudaDeviceSynchronize());
  result.used_after_loaders_bytes = gpu_used_bytes();

  std::vector<std::unique_ptr<WorkerContext>> contexts;
  contexts.reserve(static_cast<size_t>(result.workers));
  std::vector<c10::cuda::CUDAStream> streams;
  streams.reserve(static_cast<size_t>(result.workers));
  std::set<uintptr_t> stream_ids;
  for (int worker = 0; worker < result.workers; ++worker) {
    auto stream = stream_for_worker(result.explicit_stream, worker);
    streams.push_back(stream);
    uintptr_t handle = stream_handle_value(stream);
    stream_ids.insert(handle);
    result.stream_handles.push_back(handle);
    contexts.push_back(make_worker_context(args.dir, device, stream));
  }
  result.unique_streams = static_cast<int>(stream_ids.size());
  result.stream_uniqueness_ok = !result.explicit_stream || result.unique_streams == result.workers;
  auto tokenizer = tokenizer_from_bundle(contexts[0]->bundle);

  ResourceSampler resources;
  StartGate gate(result.workers);
  std::vector<DensityWorkerOutput> worker_outputs(static_cast<size_t>(result.workers));
  std::vector<int> warmup_steady_done(static_cast<size_t>(result.workers), 0);
  std::vector<int> warmup_finalize_bucket_runs(static_cast<size_t>(result.workers), 0);
  std::atomic<bool> warmup_failed{false};
  std::vector<std::thread> threads;
  threads.reserve(static_cast<size_t>(result.workers));
  for (int worker = 0; worker < result.workers; ++worker) {
    threads.emplace_back([&, worker] {
      auto& out = worker_outputs[static_cast<size_t>(worker)];
      try {
        c10::cuda::CUDAGuard device_guard(device.index());
        if (args.density_warmup) {
          int warm_utt = pick_assigned_steady_warmup_utt(contexts[worker]->bundle,
                                                         assigned[static_cast<size_t>(worker)]);
          if (warm_utt < 0) {
            throw std::runtime_error("density sweep steady warmup failed for worker" +
                                     std::to_string(worker) +
                                     ": no assigned utterance has a steady AOTI continuation");
          }
          warm_steady_encoder_once_density(warm_utt,
                                           *contexts[worker],
                                           enc_steady,
                                           device,
                                           tokenizer,
                                           result.explicit_stream,
                                           args.mutex_serialize_run,
                                           "density.1a.warmup.steady.worker" + std::to_string(worker) +
                                               ".utt" + std::to_string(warm_utt));
          warmup_steady_done[static_cast<size_t>(worker)] = 1;

          for (const auto& kv : worker_bucket_reps[static_cast<size_t>(worker)]) {
            const auto& fc = kv.second;
            SessionState warm_session;
            std::vector<EmittedEvent> warm_events;
            prepare_finalize_parent(fc,
                                    *contexts[worker],
                                    enc_steady,
                                    device,
                                    tokenizer,
                                    result.explicit_stream,
                                    args.mutex_serialize_run,
                                    warm_session,
                                    warm_events);
            std::string warm_label = "density.1a.warmup.worker" + std::to_string(worker) +
                                     ".finalize_bucket.drop" + std::to_string(fc.drop) +
                                     ".T" + std::to_string(fc.T) +
                                     ".utt" + std::to_string(fc.utt);
            auto warm = run_finalize_density(warm_session,
                                             contexts[worker]->bundle,
                                             "utt" + std::to_string(fc.utt),
                                             warm_label,
                                             finalize_loaders,
                                             contexts[worker]->joint,
                                             contexts[worker]->predict,
                                             device,
                                             tokenizer,
                                             warm_events,
                                             FinalizeFinish::SPECULATIVE_KEEP,
                                             contexts[worker]->stream,
                                             result.explicit_stream,
                                             args.mutex_serialize_run,
                                             nullptr);
            if (!warm.token_ok || !warm.fork_ok) {
              throw std::runtime_error("density sweep finalize bucket warmup failed for worker" +
                                       std::to_string(worker) + " drop=" + std::to_string(fc.drop) +
                                       " T=" + std::to_string(fc.T) +
                                       " utt=" + std::to_string(fc.utt));
            }
            ++warmup_finalize_bucket_runs[static_cast<size_t>(worker)];
          }
          CUDA_CHECK(cudaStreamSynchronize(contexts[worker]->stream.stream()));
        }
      } catch (const std::exception& e) {
        warmup_failed.store(true);
        out.error = e.what();
      }

      try {
        c10::cuda::CUDAGuard device_guard(device.index());
        gate.arrive_and_wait();
        if (warmup_failed.load() || !out.error.empty()) return;
        for (int utt : assigned[static_cast<size_t>(worker)]) {
          SessionState session;
          reset_session(session, contexts[worker]->bundle, device);
          std::vector<EmittedEvent> events;
          std::string prefix = "utt" + std::to_string(utt);
          std::string label = "density.1a.N" + std::to_string(n) +
                              ".worker" + std::to_string(worker) +
                              ".utt" + std::to_string(utt);
          int64_t num_steady = scalar_i64(utt_tensor(contexts[worker]->bundle, utt, "num_steady"));
          auto session_start = Clock::now();
          for (int chunk = 0; chunk < num_steady; ++chunk) {
            auto feed_time = session_start + ms_duration(args.density_chunk_period_ms * static_cast<double>(chunk));
            std::this_thread::sleep_until(feed_time);
            run_steady_chunk_density(session,
                                     contexts[worker]->bundle,
                                     prefix,
                                     chunk,
                                     contexts[worker]->enc_first,
                                     enc_steady,
                                     contexts[worker]->joint,
                                     contexts[worker]->predict,
                                     device,
                                     tokenizer,
                                     events,
                                     contexts[worker]->stream,
                                     result.explicit_stream,
                                     args.mutex_serialize_run,
                                     &out.timings,
                                     label + ".chunk" + std::to_string(chunk));
            auto finish = Clock::now();
            auto deadline = session_start + ms_duration(args.density_chunk_period_ms * static_cast<double>(chunk + 1));
            out.lag_ms.push_back(signed_elapsed_ms(deadline, finish));
            ++out.chunks_completed;
          }
          out.offered_audio_ms += args.density_chunk_period_ms * static_cast<double>(num_steady);
          auto vad_deadline = session_start + ms_duration(args.density_chunk_period_ms * static_cast<double>(num_steady));
          std::this_thread::sleep_until(vad_deadline);
          auto ttfs_start = Clock::now();
          vad_stop(session);
          auto finalize = run_finalize_density(session,
                                               contexts[worker]->bundle,
                                               prefix,
                                               label,
                                               finalize_loaders,
                                               contexts[worker]->joint,
                                               contexts[worker]->predict,
                                               device,
                                               tokenizer,
                                               events,
                                               FinalizeFinish::SPECULATIVE_KEEP,
                                               contexts[worker]->stream,
                                               result.explicit_stream,
                                               args.mutex_serialize_run,
                                               &out.timings);
          out.ttfs_ms.push_back(elapsed_ms_since(ttfs_start));
          bool same = finalize.token_ok &&
                      finalize.fork_ok &&
                      finalize.final_tokens == reference[static_cast<size_t>(utt)].final_tokens &&
                      strict_events_equal(events, reference[static_cast<size_t>(utt)].events, label + ".serial_oracle");
          if (!same) ++out.mismatches;
          ++out.sessions_completed;
        }
      } catch (const std::exception& e) {
        out.error = e.what();
      }
    });
  }
  gate.wait_until_ready();
  for (int worker = 0; worker < result.workers; ++worker) {
    result.warmup_steady_workers += warmup_steady_done[static_cast<size_t>(worker)];
    result.warmup_finalize_buckets += warmup_finalize_bucket_runs[static_cast<size_t>(worker)];
  }
  if (args.density_warmup) {
    std::printf("=== DENSITY 1a WARMUP COMPLETE: steady_workers=%d/%d "
                "finalize_bucket_worker_runs=%d unique_loaded_buckets=%zu CUDA_MODULE_LOADING=%s ===\n",
                result.warmup_steady_workers,
                result.workers,
                result.warmup_finalize_buckets,
                needed_buckets.size(),
                std::getenv("CUDA_MODULE_LOADING") ? std::getenv("CUDA_MODULE_LOADING") : "(unset)");
  }
  resources.start();
  gate.start_now();
  for (auto& thread : threads) thread.join();
  auto end_time = Clock::now();
  CUDA_CHECK(cudaDeviceSynchronize());
  result.resource_stats = resources.finish();
  result.wall_ms = elapsed_ms(gate.start_time, end_time);
  result.peak_mem_bytes = mem.finish();
  result.used_after_run_bytes = gpu_used_bytes();
  result.finalize_loader_memory_json = finalize_loaders.memory_json(result.num_runners);

  for (int worker = 0; worker < result.workers; ++worker) {
    const auto& out = worker_outputs[static_cast<size_t>(worker)];
    result.timings.append(out.timings);
    result.lag_ms.insert(result.lag_ms.end(), out.lag_ms.begin(), out.lag_ms.end());
    result.ttfs_ms.insert(result.ttfs_ms.end(), out.ttfs_ms.begin(), out.ttfs_ms.end());
    result.sessions_completed += out.sessions_completed;
    result.chunks_completed += out.chunks_completed;
    result.mismatches += out.mismatches;
    result.offered_audio_ms += out.offered_audio_ms;
    if (!out.error.empty()) {
      ++result.errors;
      std::printf("  density sweep N=%d worker%d exception: %s\n", n, worker, out.error.c_str());
    }
  }

  result.throughput_realtime_streams = result.wall_ms > 0.0 ? result.offered_audio_ms / result.wall_ms : 0.0;
  result.throughput_sessions_per_s = result.wall_ms > 0.0
                                         ? 1000.0 * static_cast<double>(result.sessions_completed) / result.wall_ms
                                         : 0.0;
  result.finalize_samples = static_cast<int>(result.timings.finalize_total_ms.size());
  result.finalize_p95_valid = result.finalize_samples >= kMinFinalizeP95Samples;
  auto lag = summarize(result.lag_ms);
  auto ttfs = summarize(result.ttfs_ms);
  result.keepup_ok = lag.n > 0 && lag.p95 < 500.0;
  result.ttfs_ok = result.finalize_p95_valid && ttfs.n > 0 && ttfs.p95 <= 175.0 && ttfs.p99 <= 250.0;
  result.correctness_ok = result.errors == 0 &&
                          result.mismatches == 0 &&
                          result.sessions_completed == result.requested_sessions &&
                          result.stream_uniqueness_ok;
  result.completed = result.sessions_completed == result.requested_sessions && result.errors == 0;
  result.slo_robust = result.completed && result.correctness_ok && result.keepup_ok && result.ttfs_ok;

  auto steady_gpu = summarize(result.timings.steady_gpu_ms);
  auto finalize_wait = summarize(result.timings.finalize_runner_wait_ms);
  auto finalize_aoti = summarize(result.timings.finalize_aoti_run_cuda_ms);
  auto finalize_total = summarize(result.timings.finalize_total_ms);
  const char* cuda_module_loading = std::getenv("CUDA_MODULE_LOADING");

  std::ostringstream json;
  json << "{\"check\":\"1a_density_sweep_full_session\""
       << ",\"num_runners\":" << result.num_runners
       << ",\"workers\":" << result.workers
       << ",\"steady_num_runners\":" << result.num_runners
       << ",\"finalize_num_runners_per_loaded_bucket\":" << result.finalize_num_runners
       << ",\"cuda_module_loading\":" << json_quote(cuda_module_loading ? cuda_module_loading : "")
       << ",\"stream_mode\":\"" << stream_mode_label(result.explicit_stream, args.mutex_serialize_run) << "\""
       << ",\"topology\":\"shared_steady_loader_per_thread_session_handles_explicit_streams_capped_finalize_pool\""
       << ",\"cadence_ms\":" << args.density_chunk_period_ms
       << ",\"rows_total\":" << rows_total
       << ",\"requested_sessions\":" << result.requested_sessions
       << ",\"sessions_completed\":" << result.sessions_completed
       << ",\"chunks_completed\":" << result.chunks_completed
       << ",\"finalize_samples\":" << result.finalize_samples
       << ",\"min_finalize_p95_samples\":" << kMinFinalizeP95Samples
       << ",\"finalize_p95_valid\":" << json_bool(result.finalize_p95_valid)
       << ",\"warmup\":{\"enabled\":" << json_bool(args.density_warmup)
       << ",\"steady_workers\":" << result.warmup_steady_workers
       << ",\"finalize_bucket_worker_runs\":" << result.warmup_finalize_buckets
       << ",\"finalize_buckets\":" << result.warmup_finalize_buckets
       << ",\"loaded_finalize_buckets\":" << needed_buckets.size()
       << "}"
       << ",\"errors\":" << result.errors
       << ",\"mismatches\":" << result.mismatches
       << ",\"serial_oracle_match_pass\":" << json_bool(result.mismatches == 0 && result.errors == 0)
       << ",\"stream_uniqueness_pass\":" << json_bool(result.stream_uniqueness_ok)
       << ",\"slo_robust\":" << json_bool(result.slo_robust)
       << ",\"keepup_lag_p95_lt_500ms\":" << json_bool(result.keepup_ok)
       << ",\"ttfs_budget_p95_175_p99_250_pass\":" << json_bool(result.ttfs_ok)
       << ",\"ttfs_budget\":{\"p95_ms\":175,\"p99_ms\":250}"
       << ",\"throughput_realtime_streams\":" << result.throughput_realtime_streams
       << ",\"throughput_sessions_per_s\":" << result.throughput_sessions_per_s
       << ",\"wall_ms\":" << result.wall_ms
       << ",\"offered_audio_ms\":" << result.offered_audio_ms
       << ",\"lag\":" << stats_json(lag)
       << ",\"ttfs\":" << stats_json(ttfs)
       << ",\"steady_latency\":" << stats_json(summarize(result.timings.latency_ms))
       << ",\"steady_runner_wait\":" << stats_json(summarize(result.timings.runner_wait_ms))
       << ",\"steady_gpu\":" << stats_json(steady_gpu)
       << ",\"item_wait\":" << stats_json(summarize(result.timings.scalar_sync_wait_ms))
       << ",\"item_wait_pct_of_steady_gpu\":" << stats_json(summarize(result.timings.scalar_sync_pct_of_gpu))
       << ",\"finalize_wait\":" << stats_json(finalize_wait)
       << ",\"finalize_gpu\":" << stats_json(summarize(result.timings.finalize_gpu_ms))
       << ",\"finalize_total\":" << stats_json(finalize_total)
       << ",\"finalize_phases\":" << finalize_phase_stats_json(result.timings)
       << ",\"resource\":" << resource_stats_json(result.resource_stats)
       << ",\"finalize_loader_memory\":" << result.finalize_loader_memory_json
       << ",\"unique_streams\":" << result.unique_streams
       << ",\"stream_handles\":" << uintptr_list_json(result.stream_handles)
       << ",\"peak_gpu_mem_bytes\":" << result.peak_mem_bytes
       << ",\"total_gpu_mem_bytes\":" << result.total_mem_bytes
       << ",\"used_before_bytes\":" << result.used_before_bytes
       << ",\"used_after_loaders_bytes\":" << result.used_after_loaders_bytes
       << ",\"used_after_run_bytes\":" << result.used_after_run_bytes
       << "}";
  emit_telemetry(args.dir,
                 stamp,
                 result.num_runners,
                 stream_mode_label(result.explicit_stream, args.mutex_serialize_run),
                 "1a_full_session_density_sweep",
                 json.str());

  std::printf("=== DENSITY 1a ROW N=%d %s: throughput_rt=%.3f streams sessions/s=%.3f "
              "ttfs_p50/p95/p99=%.3f/%.3f/%.3fms spread=%.3fms lag_p50/p95=%.3f/%.3fms "
              "steady_gpu_p50/p95=%.3f/%.3fms finalize_total_p50/p95=%.3f/%.3fms "
              "finalize_aoti_p50/p95=%.3f/%.3fms finalize_wait_p95=%.3fms finalize_p95_valid=%s "
              "cpu_cores=%.2f/%d gpu_util_mean=%.1f%% peak_mem=%.3fGiB mismatches=%d errors=%d ===\n",
              result.n,
              result.slo_robust ? "SLO_ROBUST" : "NOT_SLO_ROBUST",
              result.throughput_realtime_streams,
              result.throughput_sessions_per_s,
              ttfs.p50,
              ttfs.p95,
              ttfs.p99,
              ttfs.p95 - ttfs.p50,
              lag.p50,
              lag.p95,
              steady_gpu.p50,
              steady_gpu.p95,
              finalize_total.p50,
              finalize_total.p95,
              finalize_aoti.p50,
              finalize_aoti.p95,
              finalize_wait.p95,
              result.finalize_p95_valid ? "true" : "false",
              result.resource_stats.cpu_cores_used,
              result.resource_stats.cpu_threads,
              result.resource_stats.gpu_util_mean_pct,
              static_cast<double>(result.peak_mem_bytes) / (1024.0 * 1024.0 * 1024.0),
              result.mismatches,
              result.errors);
  return result;
}

static DensitySweepRunResult run_density_sweep_one(const DensityArgs& args,
                                                   torch::Device device,
                                                   const std::string& stamp,
                                                   int n,
                                                   int rows_total,
                                                   const std::vector<RowReplayResult>& reference) {
  try {
    return run_density_sweep_one_impl(args, device, stamp, n, rows_total, reference);
  } catch (const std::exception& e) {
    DensitySweepRunResult result;
    result.n = n;
    result.workers = n;
    result.num_runners = n;
    result.finalize_num_runners = capped_general_finalize_runners(std::max(1, n));
    result.rows_total = rows_total;
    result.total_mem_bytes = gpu_total_bytes();
    result.error = e.what();
    std::string lower = result.error;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char ch) {
      return static_cast<char>(std::tolower(ch));
    });
    result.oom = lower.find("out of memory") != std::string::npos ||
                 lower.find("cuda error") != std::string::npos ||
                 lower.find("memory") != std::string::npos;
    try {
      result.peak_mem_bytes = gpu_used_bytes();
      cleanup_cuda_cache();
    } catch (const std::exception&) {
    }
    std::ostringstream json;
    json << "{\"check\":\"1a_density_sweep_full_session\""
         << ",\"num_runners\":" << result.num_runners
         << ",\"workers\":" << result.workers
         << ",\"slo_robust\":false"
         << ",\"oom\":" << json_bool(result.oom)
         << ",\"error\":" << json_quote(result.error)
         << ",\"peak_gpu_mem_bytes\":" << result.peak_mem_bytes
         << ",\"total_gpu_mem_bytes\":" << result.total_mem_bytes
         << "}";
    emit_telemetry(args.dir,
                   stamp,
                   result.num_runners,
                   stream_mode_label(args.stream_mode == "explicit", args.mutex_serialize_run),
                   "1a_full_session_density_sweep_error",
                   json.str());
    std::printf("=== DENSITY 1a ROW N=%d ERROR: oom=%s error=%s ===\n",
                n,
                result.oom ? "true" : "false",
                result.error.c_str());
    return result;
  }
}

struct DensitySweepSummary {
  std::vector<DensitySweepRunResult> runs;
  int knee_n = 0;
  int single_thread_keepup_n = 0;
  double multiplier = 0.0;
  bool pass_to_1b = false;
  bool correctness_at_knee = false;
  std::string binding_slo = "none";
  std::string binding_resource = "not_observed";
};

static const DensitySweepRunResult* find_density_result(const std::vector<DensitySweepRunResult>& runs, int n) {
  for (const auto& run : runs) {
    if (run.n == n) return &run;
  }
  return nullptr;
}

static std::string infer_binding_slo(const std::vector<DensitySweepRunResult>& runs, int knee_n) {
  for (const auto& run : runs) {
    if (run.n <= knee_n) continue;
    if (run.oom) return "memory_oom";
    if (!run.correctness_ok && run.completed) return "correctness";
    if (!run.finalize_p95_valid) return "finalize_p95_invalid";
    if (!run.ttfs_ok) return "ttfs_p95_or_p99";
    if (!run.keepup_ok) return "keepup_lag_p95";
    if (!run.completed) return "runtime_error";
  }
  return "none_observed_in_sweep";
}

static std::string infer_binding_resource(const std::vector<DensitySweepRunResult>& runs, int knee_n) {
  const DensitySweepRunResult* base = find_density_result(runs, 1);
  SummaryStats base_steady_gpu = base != nullptr ? summarize(base->timings.steady_gpu_ms) : SummaryStats{};
  const DensitySweepRunResult* first_bound = nullptr;
  for (const auto& run : runs) {
    if (run.n <= knee_n) continue;
    first_bound = &run;
    break;
  }
  if (first_bound == nullptr) {
    first_bound = runs.empty() ? nullptr : &runs.back();
  }
  if (first_bound == nullptr) return "not_observed";
  if (first_bound->oom) return "memory";
  if (first_bound->total_mem_bytes > 0 &&
      static_cast<double>(first_bound->peak_mem_bytes) / static_cast<double>(first_bound->total_mem_bytes) >= 0.92) {
    return "memory";
  }
  if (first_bound->resource_stats.cpu_threads > 0 &&
      first_bound->resource_stats.cpu_cores_used >= 0.85 * static_cast<double>(first_bound->resource_stats.cpu_threads)) {
    return "CPU cores";
  }
  auto steady_gpu = summarize(first_bound->timings.steady_gpu_ms);
  if (base_steady_gpu.p50 > 0.0 && steady_gpu.p50 / base_steady_gpu.p50 >= 1.50) {
    return "GPU encoder contention";
  }
  if (first_bound->resource_stats.gpu_util_available && first_bound->resource_stats.gpu_util_mean_pct >= 80.0) {
    return "GPU encoder contention";
  }
  if (!first_bound->ttfs_ok) return "finalize/TTFS tail";
  if (!first_bound->keepup_ok) return "mixed_or_unknown_keepup";
  return "not_observed";
}

static void emit_density_sweep_manifest(const DensityArgs& args,
                                        const std::string& stamp,
                                        const DensitySweepSummary& summary,
                                        int rows_total) {
  std::string logs_dir = args.dir + "/logs/" + stamp;
  fs::create_directories(logs_dir);
  std::string path = logs_dir + "/density_sweep_manifest.json";
  std::ofstream out(path, std::ios::out | std::ios::trunc);
  if (!out) throw std::runtime_error("failed to open density sweep manifest: " + path);
  out << "{"
      << "\"stamp\":\"" << stamp << "\""
      << ",\"mode\":\"density-sweep\""
      << ",\"status\":\"" << (summary.pass_to_1b ? "PASS_TO_1B" : "NO_PASS_TO_1B") << "\""
      << ",\"dir\":\"" << args.dir << "\""
      << ",\"n_values\":" << int_list_json(args.n_values)
      << ",\"rows_total\":" << rows_total
      << ",\"density_rows\":" << args.density_rows
      << ",\"density_sessions_per_worker\":" << args.density_sessions_per_worker
      << ",\"density_warmup\":" << json_bool(args.density_warmup)
      << ",\"min_finalize_p95_samples\":" << kMinFinalizeP95Samples
      << ",\"cuda_module_loading\":"
      << json_quote(std::getenv("CUDA_MODULE_LOADING") ? std::getenv("CUDA_MODULE_LOADING") : "")
      << ",\"cadence_ms\":" << args.density_chunk_period_ms
      << ",\"knee_n\":" << summary.knee_n
      << ",\"single_thread_keepup_n\":" << summary.single_thread_keepup_n
      << ",\"multiplier\":" << summary.multiplier
      << ",\"pass_to_1b\":" << json_bool(summary.pass_to_1b)
      << ",\"binding_slo\":\"" << summary.binding_slo << "\""
      << ",\"binding_resource\":\"" << summary.binding_resource << "\""
      << "}\n";
  if (!out) throw std::runtime_error("failed to write density sweep manifest: " + path);
  std::printf("DENSITY_SWEEP_MANIFEST path=%s status=%s\n",
              path.c_str(),
              summary.pass_to_1b ? "PASS_TO_1B" : "NO_PASS_TO_1B");
}

static DensitySweepSummary run_density_sweep(const DensityArgs& args,
                                             torch::Device device,
                                             const std::string& stamp,
                                             int rows_total) {
  int max_n = *std::max_element(args.n_values.begin(), args.n_values.end());
  int max_requested_sessions = rows_total;
  if (args.density_sessions_per_worker > 0) {
    max_requested_sessions = args.density_sessions_per_worker * max_n;
  } else if (args.density_rows > 0) {
    max_requested_sessions = args.density_rows;
  }
  torch::jit::Module reference_sizing_bundle = torch::jit::load(args.dir + "/session_bundle.ts");
  verify_session_bundle_meta(reference_sizing_bundle, false);
  int reference_sessions = raise_request_for_valid_finalize_p95(reference_sizing_bundle,
                                                               rows_total,
                                                               std::max(1, max_requested_sessions));
  int reference_rows = std::min(rows_total, reference_sessions);
  std::printf("=== DENSITY 1a SERIAL ORACLE BUILD: rows=%d/%d ===\n", reference_rows, rows_total);
  TimingBuckets ref_timings;
  auto reference = build_serial_reference(args, device, reference_rows, &ref_timings);
  std::printf("=== DENSITY 1a SERIAL ORACLE PASS: rows=%d/%d ===\n", reference_rows, rows_total);
  cleanup_cuda_cache();

  DensitySweepSummary summary;
  for (int n : args.n_values) {
    auto run = run_density_sweep_one(args, device, stamp, n, rows_total, reference);
    bool stop_after = run.oom;
    summary.runs.push_back(std::move(run));
    cleanup_cuda_cache();
    if (stop_after) {
      std::printf("=== DENSITY 1a STOPPING SWEEP AFTER N=%d: memory/runtime bound hit ===\n", n);
      break;
    }
  }

  for (const auto& run : summary.runs) {
    if (run.slo_robust) summary.knee_n = std::max(summary.knee_n, run.n);
  }
  const auto* n1 = find_density_result(summary.runs, 1);
  summary.single_thread_keepup_n = (n1 != nullptr && n1->slo_robust) ? 1 : 0;
  summary.multiplier = summary.single_thread_keepup_n > 0
                           ? static_cast<double>(summary.knee_n) / static_cast<double>(summary.single_thread_keepup_n)
                           : 0.0;
  const auto* knee = find_density_result(summary.runs, summary.knee_n);
  summary.correctness_at_knee = knee != nullptr && knee->correctness_ok;
  summary.pass_to_1b = summary.multiplier >= 2.0 && summary.correctness_at_knee && summary.knee_n > 0;
  summary.binding_slo = infer_binding_slo(summary.runs, summary.knee_n);
  summary.binding_resource = infer_binding_resource(summary.runs, summary.knee_n);

  std::ostringstream rows_json;
  rows_json << "[";
  for (size_t i = 0; i < summary.runs.size(); ++i) {
    if (i > 0) rows_json << ",";
    const auto& run = summary.runs[i];
    rows_json << "{\"N\":" << run.n
              << ",\"slo_robust\":" << json_bool(run.slo_robust)
              << ",\"throughput_realtime_streams\":" << run.throughput_realtime_streams
              << ",\"finalize_samples\":" << run.finalize_samples
              << ",\"finalize_p95_valid\":" << json_bool(run.finalize_p95_valid)
              << ",\"ttfs\":" << stats_json(summarize(run.ttfs_ms))
              << ",\"lag\":" << stats_json(summarize(run.lag_ms))
              << ",\"steady_gpu\":" << stats_json(summarize(run.timings.steady_gpu_ms))
              << ",\"finalize_total\":" << stats_json(summarize(run.timings.finalize_total_ms))
              << ",\"finalize_phases\":" << finalize_phase_stats_json(run.timings)
              << ",\"peak_gpu_mem_bytes\":" << run.peak_mem_bytes
              << ",\"mismatches\":" << run.mismatches
              << ",\"errors\":" << run.errors
              << ",\"oom\":" << json_bool(run.oom)
              << "}";
  }
  rows_json << "]";

  std::ostringstream json;
  json << "{\"check\":\"1a_density_sweep_summary\""
       << ",\"num_runners\":0"
       << ",\"stream_mode\":\"" << stream_mode_label(args.stream_mode == "explicit", args.mutex_serialize_run) << "\""
       << ",\"topology\":\"shared_steady_loader_per_thread_session_handles_explicit_streams_capped_finalize_pool\""
       << ",\"ttfs_budget\":{\"p95_ms\":175,\"p99_ms\":250}"
       << ",\"keepup_budget\":{\"lag_p95_ms\":500}"
       << ",\"density_warmup\":" << json_bool(args.density_warmup)
       << ",\"min_finalize_p95_samples\":" << kMinFinalizeP95Samples
       << ",\"cuda_module_loading\":"
       << json_quote(std::getenv("CUDA_MODULE_LOADING") ? std::getenv("CUDA_MODULE_LOADING") : "")
       << ",\"rows_total\":" << rows_total
       << ",\"n_values\":" << int_list_json(args.n_values)
       << ",\"knee_n\":" << summary.knee_n
       << ",\"single_thread_keepup_n\":" << summary.single_thread_keepup_n
       << ",\"multiplier\":" << summary.multiplier
       << ",\"pass_to_1b\":" << json_bool(summary.pass_to_1b)
       << ",\"correctness_at_knee\":" << json_bool(summary.correctness_at_knee)
       << ",\"binding_slo\":\"" << summary.binding_slo << "\""
       << ",\"binding_resource\":\"" << summary.binding_resource << "\""
       << ",\"rows\":" << rows_json.str()
       << "}";
  emit_telemetry(args.dir,
                 stamp,
                 0,
                 stream_mode_label(args.stream_mode == "explicit", args.mutex_serialize_run),
                 "1a_full_session_density_sweep_summary",
                 json.str());
  emit_density_sweep_manifest(args, stamp, summary, rows_total);
  std::printf("=== DENSITY 1a SUMMARY %s: knee_N=%d single_thread_keepup_N=%d multiplier=%.3fx "
              "binding_slo=%s binding_resource=%s correctness_at_knee=%s ===\n",
              summary.pass_to_1b ? "PASS_TO_1B" : "NO_PASS_TO_1B",
              summary.knee_n,
              summary.single_thread_keepup_n,
              summary.multiplier,
              summary.binding_slo.c_str(),
              summary.binding_resource.c_str(),
              summary.correctness_at_knee ? "PASS" : "FAIL");
  return summary;
}

static void emit_run_manifest(const DensityArgs& args,
                              const std::string& stamp,
                              const std::string& status,
                              bool canonical_full_run,
                              bool partial,
                              int rows_total,
                              int correctness_rows,
                              const std::string& correctness_status,
                              const std::string& steady_status,
                              const std::string& finalize_status) {
  std::string logs_dir = args.dir + "/logs/" + stamp;
  fs::create_directories(logs_dir);
  std::string path = logs_dir + "/manifest.json";
  std::ofstream out(path, std::ios::out | std::ios::trunc);
  if (!out) throw std::runtime_error("failed to open manifest: " + path);
  out << "{"
      << "\"stamp\":\"" << stamp << "\""
      << ",\"status\":\"" << status << "\""
      << ",\"canonical_full_run\":" << json_bool(canonical_full_run)
      << ",\"partial\":" << json_bool(partial)
      << ",\"smoke\":" << json_bool(args.smoke)
      << ",\"dir\":\"" << args.dir << "\""
      << ",\"n_values\":" << int_list_json(args.n_values)
      << ",\"target_n\":" << args.target_n
      << ",\"workers_override\":" << args.workers
      << ",\"num_runners_override\":" << args.num_runners
      << ",\"stream_mode\":\"" << args.stream_mode << "\""
      << ",\"mutex_serialize_run\":" << json_bool(args.mutex_serialize_run)
      << ",\"steady_cases\":" << args.steady_cases
      << ",\"steady_repeats\":" << args.steady_repeats
      << ",\"correctness_n\":" << args.correctness_n
      << ",\"correctness_rows\":" << correctness_rows
      << ",\"rows_total\":" << rows_total
      << ",\"finalize_n\":" << args.finalize_n
      << ",\"finalize_mode\":\"" << args.finalize_mode << "\""
      << ",\"skip_correctness\":" << json_bool(args.skip_correctness)
      << ",\"skip_steady\":" << json_bool(args.skip_steady)
      << ",\"skip_finalize\":" << json_bool(args.skip_finalize)
      << ",\"default_stream_control\":" << json_bool(args.default_stream_control)
      << ",\"correctness_default_stream_control\":" << json_bool(args.correctness_default_stream_control)
      << ",\"steady_overlap_probe\":" << json_bool(args.steady_overlap_probe)
      << ",\"scalar_locality_probe\":" << json_bool(args.scalar_locality_probe)
      << ",\"correctness_status\":\"" << correctness_status << "\""
      << ",\"steady_status\":\"" << steady_status << "\""
      << ",\"finalize_status\":\"" << finalize_status << "\""
      << "}\n";
  if (!out) throw std::runtime_error("failed to write manifest: " + path);
  std::printf("DENSITY_MANIFEST path=%s status=%s canonical_full_run=%s partial=%s\n",
              path.c_str(),
              status.c_str(),
              canonical_full_run ? "true" : "false",
              partial ? "true" : "false");
}

int main(int argc, char** argv) {
  try {
    DensityArgs args = parse_density_args(argc, argv);
    if (args.mode == "density-sweep") {
      const char* previous_cuda_module_loading = std::getenv("CUDA_MODULE_LOADING");
      if (setenv("CUDA_MODULE_LOADING", "EAGER", 1) != 0) {
        throw std::runtime_error("failed to set CUDA_MODULE_LOADING=EAGER");
      }
      std::printf("=== DENSITY CUDA_MODULE_LOADING=EAGER (was %s) ===\n",
                  previous_cuda_module_loading ? previous_cuda_module_loading : "(unset)");
    }
    torch::NoGradGuard ng;
    auto device = torch::Device(torch::kCUDA, 0);
    c10::cuda::CUDAGuard device_guard(device.index());
    std::string stamp = timestamp_utc();
    torch::jit::Module manifest_bundle = torch::jit::load(args.dir + "/session_bundle.ts");
    verify_session_bundle_meta(manifest_bundle, false);
    int rows_total = static_cast<int>(scalar_i64(attr_tensor(manifest_bundle, "num_utts")));
    int requested_rows = args.correctness_rows > 0 ? std::min(args.correctness_rows, rows_total) : rows_total;
    if (args.mode == "density-sweep") {
      std::printf("=== DENSITY 1a START: dir=%s stamp=%s n_values=",
                  args.dir.c_str(), stamp.c_str());
      for (size_t i = 0; i < args.n_values.size(); ++i) {
        std::printf("%s%d", i == 0 ? "" : ",", args.n_values[i]);
      }
      std::printf(" rows_total=%d density_rows=%d sessions_per_worker=%d cadence=%.3fms stream_mode=%s "
                  "mutex=%s warmup=%s min_finalize_p95_samples=%d CUDA_MODULE_LOADING=%s ===\n",
                  rows_total,
                  args.density_rows,
                  args.density_sessions_per_worker,
                  args.density_chunk_period_ms,
                  args.stream_mode.c_str(),
                  args.mutex_serialize_run ? "true" : "false",
                  args.density_warmup ? "true" : "false",
                  kMinFinalizeP95Samples,
                  std::getenv("CUDA_MODULE_LOADING") ? std::getenv("CUDA_MODULE_LOADING") : "(unset)");
      auto summary = run_density_sweep(args, device, stamp, rows_total);
      return summary.pass_to_1b ? 0 : 1;
    }
    std::printf("=== DENSITY STEP0 START: dir=%s stamp=%s n_values=",
                args.dir.c_str(), stamp.c_str());
    for (size_t i = 0; i < args.n_values.size(); ++i) {
      std::printf("%s%d", i == 0 ? "" : ",", args.n_values[i]);
    }
    std::printf(" target_n=%d correctness_n=%d finalize_n=%d finalize_mode=%s rows=%d/%d stream_mode=%s "
                "workers_override=%d num_runners_override=%d mutex=%s smoke=%s partial=%s ===\n",
                args.target_n,
                args.correctness_n,
                args.finalize_n,
                args.finalize_mode.c_str(),
                requested_rows,
                rows_total,
                args.stream_mode.c_str(),
                args.workers,
                args.num_runners,
                args.mutex_serialize_run ? "true" : "false",
                args.smoke ? "true" : "false",
                args.partial ? "true" : "false");

    CorrectnessResult correctness;
    std::string correctness_status = "SKIP";
    if (!args.skip_correctness) {
      correctness = run_correctness_gate(args, device, stamp);
      if (!correctness.identity_ok) {
        std::printf("=== DENSITY STEP0 STOP-CANDIDATE: 0b token/event identity failed; throughput numbers are not trusted ===\n");
      }
      if (!correctness.ok) {
        std::printf("=== DENSITY STEP0 CONTINUE_UNTRUSTED: 0b identity/scalar-locality/topology gate failed ===\n");
      }
      correctness_status = correctness.ok ? "PASS" : (correctness.identity_ok ? "FAIL" : "STOP_CANDIDATE");
      cleanup_cuda_cache();
    } else {
      std::printf("=== DENSITY 0b SKIP: throughput numbers will be correctness-untrusted ===\n");
    }

    bool steady_ok = true;
    std::string steady_status = "SKIP";
    if (!args.skip_steady) {
      auto steady = run_steady_sweep(args, device, stamp);
      steady_ok = steady.pass;
      steady_status = steady_ok ? "PASS" : "FAIL";
      cleanup_cuda_cache();
    }

    bool finalize_ok = true;
    std::string finalize_status = "SKIP";
    if (!args.skip_finalize) {
      if (args.skip_correctness) {
        throw std::runtime_error("0c finalize gate needs serial references; do not combine --skip-correctness with finalize");
      }
      finalize_ok = run_finalize_gate(args, device, stamp, correctness);
      finalize_status = finalize_ok ? "PASS" : "FAIL";
    }

    bool no_skips = !args.skip_correctness && !args.skip_steady && !args.skip_finalize;
    bool rows_full = requested_rows == rows_total;
    bool has_n1 = std::find(args.n_values.begin(), args.n_values.end(), 1) != args.n_values.end();
    bool has_n2 = std::find(args.n_values.begin(), args.n_values.end(), 2) != args.n_values.end();
    bool has_n4 = std::find(args.n_values.begin(), args.n_values.end(), 4) != args.n_values.end();
    bool has_target = args.target_n <= 0 ||
                      std::find(args.n_values.begin(), args.n_values.end(), args.target_n) != args.n_values.end();
    bool canonical_full_run = no_skips &&
                              rows_full &&
                              !args.smoke &&
                              !args.partial &&
                              args.workers == 0 &&
                              args.num_runners == 0 &&
                              args.finalize_mode == "both" &&
                              args.stream_mode == "explicit" &&
                              !args.mutex_serialize_run &&
                              args.default_stream_control &&
                              args.correctness_default_stream_control &&
                              args.scalar_locality_probe &&
                              args.steady_overlap_probe &&
                              has_n1 &&
                              has_n2 &&
                              has_n4 &&
                              has_target;
    bool partial = args.partial || args.smoke || !no_skips || !rows_full || !canonical_full_run;
    bool gate_all = no_skips && correctness.ok && steady_ok && finalize_ok;
    std::string final_status = partial ? "PARTIAL_DIAGNOSTIC" : (gate_all ? "PASS" : "FAIL");
    emit_run_manifest(args,
                      stamp,
                      final_status,
                      canonical_full_run,
                      partial,
                      rows_total,
                      requested_rows,
                      correctness_status,
                      steady_status,
                      finalize_status);
    std::printf("=== DENSITY STEP0 %s: correctness=%s steady=%s finalize=%s canonical_full_run=%s partial=%s stamp=%s ===\n",
                final_status.c_str(),
                correctness_status.c_str(),
                steady_status.c_str(),
                finalize_status.c_str(),
                canonical_full_run ? "true" : "false",
                partial ? "true" : "false",
                stamp.c_str());
    return gate_all ? 0 : 1;
  } catch (const std::exception& e) {
    std::printf("DENSITY setup failed: %s\n", e.what());
    return 2;
  }
}
