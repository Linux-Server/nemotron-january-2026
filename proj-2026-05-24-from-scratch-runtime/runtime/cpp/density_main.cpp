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
#include <thread>

using Clock = std::chrono::steady_clock;

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
  std::string dir = "../artifacts";
  std::vector<int> n_values{1, 2, 4};
  int target_n = 16;
  int workers = 0;
  int num_runners = 0;
  int steady_cases = 32;
  int steady_repeats = 4;
  int correctness_n = 4;
  int correctness_rows = -1;
  int finalize_n = 4;
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
    if (arg == "--n-values") {
      args.n_values = parse_int_list(need_value("--n-values"));
    } else if (arg == "--target-n") {
      args.target_n = std::stoi(need_value("--target-n"));
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
    } else if (!dir_set) {
      args.dir = arg;
      dir_set = true;
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  if (args.n_values.empty()) throw std::runtime_error("--n-values cannot be empty");
  if (args.target_n > 0 &&
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

static void cleanup_cuda_cache() {
  CUDA_CHECK(cudaDeviceSynchronize());
  c10::cuda::CUDACachingAllocator::emptyCache();
  CUDA_CHECK(cudaDeviceSynchronize());
}

struct MemorySampler {
  std::atomic<bool> stop{false};
  std::thread thread;
  std::atomic<size_t> peak{0};

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
  }
};

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

static FinalizeOutcome run_finalize_density(SessionState& parent,
                                            torch::jit::Module& bundle,
                                            const std::string& prefix,
                                            const std::string& label,
                                            std::map<std::pair<int64_t, int64_t>, std::unique_ptr<AOTIModelPackageLoader>>& finalize_loaders,
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
  auto snapshot = snapshot_asr(parent);
  parent.mode = SessionMode::FINALIZED;
  snapshot.mode = SessionMode::FINALIZED;
  auto fork = clone_session(parent);

  int64_t drop_extra = scalar_i64(prefix_tensor(bundle, prefix, "final_drop_extra"));
  int64_t final_T = scalar_i64(prefix_tensor(bundle, prefix, "final_T"));
  auto gold = tensor_to_vec(prefix_tensor(bundle, prefix, "gold_tokens"));
  double runner_host_ms = 0.0;
  double gpu_ms = 0.0;

  if (final_T > 0) {
    auto final_chunk = prefix_tensor(bundle, prefix, "final_chunk_mel").to(device).contiguous();
    if (final_chunk.size(2) != final_T) {
      throw std::runtime_error("density final_chunk_mel T does not match bundle final_T");
    }
    int64_t expected_drop = parent.emitted == 0 ? 0 : DROP;
    if (drop_extra != expected_drop) throw std::runtime_error("density finalize drop_extra does not match parent emitted state");

    auto loader_it = finalize_loaders.find(std::make_pair(drop_extra, final_T));
    if (loader_it == finalize_loaders.end()) {
      throw std::runtime_error("density no finalize bucket for drop=" + std::to_string(drop_extra) +
                               " T=" + std::to_string(final_T));
    }

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
    auto out = run_aoti_loader(*loader_it->second, inputs, stream, explicit_stream, mutex_serialize_run);
    runner_host_ms = elapsed_ms_since(run_start);
    CUDA_CHECK(cudaEventRecord(ev_stop, stream.stream()));
    if (out.size() < 2) throw std::runtime_error("density finalize AOTI bucket returned fewer than 2 outputs");
    double scalar_wait_ms = 0.0;
    int64_t enc_len = scalar_i64_timed(out[1], &scalar_wait_ms);
    if (out.size() >= 5) {
      fork.clc = out[2];
      fork.clt = out[3];
      fork.clcl = out[4];
    }
    decode_range_density(joint, predict, out[0], enc_len, fork.g, fork.h, fork.c, fork.hyp, &scalar_wait_ms);
    if (timings != nullptr) timings->scalar_sync_wait_ms.push_back(scalar_wait_ms);
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
    timings->finalize_runner_wait_ms.push_back(std::max(0.0, runner_host_ms - gpu_ms));
    timings->finalize_gpu_ms.push_back(gpu_ms);
    timings->finalize_total_ms.push_back(elapsed_ms_since(total_start));
  }
  return outcome;
}

static std::map<std::pair<int64_t, int64_t>, std::unique_ptr<AOTIModelPackageLoader>>
load_finalize_bucket_loaders_density(const std::string& dir, torch::Device device, int num_runners) {
  std::string buckets_dir = dir + "/stripped_finalize_buckets";
  if (!directory_exists(buckets_dir)) buckets_dir = dir + "/finalize_buckets";
  std::string shared_weights = dir + "/finalize_shared_weights.ts";
  std::string shared_weights_pt = dir + "/finalize_shared_weights.pt";
  if (!directory_exists(buckets_dir)) throw std::runtime_error("finalize buckets directory missing: " + buckets_dir);
  if (!file_exists(shared_weights)) throw std::runtime_error("finalize shared weights missing: " + shared_weights);

  auto bucket_paths = discover_finalize_buckets(buckets_dir);
  if (bucket_paths.empty()) throw std::runtime_error("no finalize bucket packages found in " + buckets_dir);
  std::string manifest_path = buckets_dir + "/manifest.json";
  if (!file_exists(manifest_path)) {
    throw std::runtime_error("finalize bucket manifest is required when buckets are present: " + manifest_path);
  }
  auto manifest = load_bucket_manifest(manifest_path);
  verify_bucket_manifest(manifest, bucket_paths, buckets_dir, shared_weights_pt);
  std::printf("density finalize manifest verified: %zu buckets, weights_sha256=%s num_runners=%d\n",
              manifest.buckets.size(), manifest.contract.weights_sha256.c_str(), num_runners);

  auto shared_constants = load_shared_constants(shared_weights, device);
  std::printf("density loaded finalize shared constants: %zu entries\n", shared_constants.size());

  std::map<std::pair<int64_t, int64_t>, std::unique_ptr<AOTIModelPackageLoader>> loaders;
  for (const auto& kv : bucket_paths) {
    int64_t drop = kv.first.first;
    int64_t T = kv.first.second;
    const std::string& pkg = kv.second;
    auto loader = std::make_unique<AOTIModelPackageLoader>(pkg, "model", false, num_runners, -1);
    auto bucket_constants = constants_for_bucket(shared_constants, *loader, pkg);
    loader->load_constants(bucket_constants.values, false, false, true);
    std::printf("  density finalize bucket drop=%ld T=%ld constants=%zu direct=%zu alias=%zu num_runners=%d\n",
                (long)drop, (long)T, bucket_constants.values.size(),
                bucket_constants.direct_matches, bucket_constants.alias_fallbacks, num_runners);
    loaders.emplace(kv.first, std::move(loader));
  }
  return loaders;
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
                                          std::map<std::pair<int64_t, int64_t>, std::unique_ptr<AOTIModelPackageLoader>>& finalize_loaders,
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
  auto finalize_loaders = load_finalize_bucket_loaders_density(args.dir, device, 1);
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
  auto finalize_loaders = load_finalize_bucket_loaders_density(args.dir, device, num_runners);
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
  result.num_runners = args.num_runners > 0 ? args.num_runners : result.workers;
  MemorySampler mem;
  mem.start();
  AOTIModelPackageLoader enc_steady(args.dir + "/enc_steady_aoti.pt2", "model", false, result.num_runners, -1);
  auto finalize_loaders = load_finalize_bucket_loaders_density(args.dir, device, result.num_runners);
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
       << ",\"finalize_runner_wait_pct_of_total_p95\":" << wait_pct
       << ",\"peak_gpu_mem_bytes\":" << result.peak_mem
       << "}";
  emit_telemetry(args.dir,
                 stamp,
                 result.num_runners,
                 stream_mode_label(true, args.mutex_serialize_run),
                 "shared_finalize_bucket_runner_pool_" + mode,
                 json.str());
  std::printf("=== DENSITY 0c %s %s: workers=%d num_runners=%d unique_streams=%d mismatches=%d "
              "finalize_wait_p95=%.3fms total_p95=%.3fms wait_pct=%.1f%% ===\n",
              mode.c_str(),
              result.ok ? "PASS" : "FAIL",
              result.workers,
              result.num_runners,
              result.unique_streams,
              result.mismatches,
              finalize_wait.p95,
              finalize_total.p95,
              wait_pct);
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
  auto same = pick_same_bucket_finalize_cases(all_cases, args.finalize_n);
  auto mixed = pick_mixed_finalize_cases(all_cases, args.finalize_n);
  auto same_result = run_finalize_gate_one(args, device, stamp, "same_bucket", same, correctness.reference);
  cleanup_cuda_cache();
  auto mixed_result = run_finalize_gate_one(args, device, stamp, "mixed_bucket", mixed, correctness.reference);
  return same_result.ok && mixed_result.ok;
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
    torch::NoGradGuard ng;
    DensityArgs args = parse_density_args(argc, argv);
    auto device = torch::Device(torch::kCUDA, 0);
    c10::cuda::CUDAGuard device_guard(device.index());
    std::string stamp = timestamp_utc();
    torch::jit::Module manifest_bundle = torch::jit::load(args.dir + "/session_bundle.ts");
    verify_session_bundle_meta(manifest_bundle, false);
    int rows_total = static_cast<int>(scalar_i64(attr_tensor(manifest_bundle, "num_utts")));
    int requested_rows = args.correctness_rows > 0 ? std::min(args.correctness_rows, rows_total) : rows_total;
    std::printf("=== DENSITY STEP0 START: dir=%s stamp=%s n_values=",
                args.dir.c_str(), stamp.c_str());
    for (size_t i = 0; i < args.n_values.size(); ++i) {
      std::printf("%s%d", i == 0 ? "" : ",", args.n_values[i]);
    }
    std::printf(" target_n=%d correctness_n=%d finalize_n=%d rows=%d/%d stream_mode=%s "
                "workers_override=%d num_runners_override=%d mutex=%s smoke=%s partial=%s ===\n",
                args.target_n,
                args.correctness_n,
                args.finalize_n,
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
