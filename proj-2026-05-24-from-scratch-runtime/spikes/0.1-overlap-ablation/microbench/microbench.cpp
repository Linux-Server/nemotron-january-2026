// 0.1b native launch-overlap microbench (C++ / libtorch) — compile-targeted; needs a GPU box + libtorch SDK to build+run.
//
// Gate: does GIL-free MULTI-THREAD intake/dispatch raise sustainable L40S streams/box above the Python single-thread
// ceiling (~16-20/box), reclaiming the 40-65% idle GPU? GATE = >=1.5x (~28/box). Spec: ../0.1b-microbench-spec.md.
//
// Model: M dispatcher lanes, each = its OWN OS thread + CUDA stream + static IO buffers + a captured CUDA graph of the
// steady encoder (from export_encoder.py). Each lane replays ONLY its own graph (no cross-thread graph sharing => safe).
// N simulated streams emit a chunk every 160ms into a shared intake queue; the M lanes pull+dispatch. The A/B:
//   M=1   -> mimics Python's single-thread intake wall;   M=cores -> the native thesis.
// "Mock decode" approximates COST (a small GPU op + a host-side stall for the eager .item()-loop CPU time), NOT math.
// CALIBRATE --decode-* so the M=1 baseline reproduces the measured ~16-20/box BEFORE trusting M=cores (see README).
//
// Build: CMakeLists.txt (libtorch 2.8.0+cu128 cxx11-abi). Run examples:
//   ./microbench --module artifacts/encoder_steady_b1.ts --lanes 1  --streams N --duration-s 30   # baseline
//   ./microbench --module artifacts/encoder_steady_b1.ts --lanes $(nproc) --streams N --duration-s 30  # thesis
// Sweep N to find the max N where p95 chunk latency stays in SLO budget; that N (× procs) = streams/box.
//
// NOTE: libtorch C++ API signatures are version-sensitive (CUDAGraph::capture_begin pool arg, getStreamFromPool,
// nvml). Verify against the installed 2.8.0 headers at build; the structure below targets that API.

#include <torch/script.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <nvml.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t){ return std::chrono::duration<double,std::milli>(clk::now()-t).count(); }

struct Args {
  std::string module = "artifacts/encoder_steady_b1.ts";
  int lanes = 1;            // = dispatcher threads (each owns one lane); A/B: 1 vs nproc
  int streams = 8;
  int duration_s = 30;
  int decode_gpu_iters = 0;    // mock-decode GPU cost: extra dummy-GEMM iters per chunk (calibrate)
  int decode_host_us = 10000;  // mock-decode host cost: the GIL-held decode stand-in. CALIBRATED to the measured
                               // per-chunk thread-busy ~10.4ms (decode 8.2ms + glue) — proj-2026-05-24-0859/gil-attribution.md
  int chunk_ms = 160;          // steady cadence
  int slo_p95_ms = 300;        // SLO budget for the chunk intake->done proxy (tune to the vad_stop->final budget)
};

static Args parse_args(int argc, char** argv){
  Args a; auto eq=[&](const char* k,const char* v){return std::strcmp(k,v)==0;};
  for(int i=1;i<argc;i++){
    if(eq(argv[i],"--module")&&i+1<argc) a.module=argv[++i];
    else if(eq(argv[i],"--lanes")&&i+1<argc) a.lanes=std::atoi(argv[++i]);
    else if(eq(argv[i],"--streams")&&i+1<argc) a.streams=std::atoi(argv[++i]);
    else if(eq(argv[i],"--duration-s")&&i+1<argc) a.duration_s=std::atoi(argv[++i]);
    else if(eq(argv[i],"--decode-gpu-iters")&&i+1<argc) a.decode_gpu_iters=std::atoi(argv[++i]);
    else if(eq(argv[i],"--decode-host-us")&&i+1<argc) a.decode_host_us=std::atoi(argv[++i]);
    else if(eq(argv[i],"--chunk-ms")&&i+1<argc) a.chunk_ms=std::atoi(argv[++i]);
    else if(eq(argv[i],"--slo-p95-ms")&&i+1<argc) a.slo_p95_ms=std::atoi(argv[++i]);
  }
  return a;
}

// Steady B=1 proto inputs — shapes from artifacts/shapes.json (confirmed by export_encoder.py):
//   processed [1,128,25] f32, length [1] i64, clc [24,1,70,1024] f32, clt [24,1,1024,8] f32, clcl [1] i64.
// Values are irrelevant for a throughput bench (we measure launch/overlap, not correctness) -> zeros.
static std::vector<torch::Tensor> make_proto(){
  auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  auto i64 = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
  std::vector<torch::Tensor> p;
  p.push_back(torch::zeros({1,128,25}, f32));            // processed_signal
  p.push_back(torch::full({1}, 25, i64));                // processed_signal_length
  p.push_back(torch::zeros({24,1,70,1024}, f32));        // cache_last_channel
  p.push_back(torch::zeros({24,1,1024,8}, f32));         // cache_last_time
  p.push_back(torch::zeros({1}, i64));                   // cache_last_channel_len
  return p;
}

struct Lane {
  c10::cuda::CUDAStream stream;
  at::cuda::CUDAGraph graph;
  std::vector<torch::Tensor> static_in;
  std::vector<torch::jit::IValue> ivals;
  torch::jit::Module mod;          // each lane gets its own module handle (forward is stateless post-load)
  torch::Tensor mock_a, mock_b;    // mock-decode dummy GEMM operands
  at::cuda::CUDAEvent done_evt;    // per-lane completion event (lane is serial: dispatch -> host decode -> wait)
  Lane(c10::cuda::CUDAStream s, torch::jit::Module m): stream(s), mod(std::move(m)) {}

  void prepare(const std::vector<torch::Tensor>& proto){
    for(auto& t: proto){ static_in.push_back(t.clone()); }
    for(auto& t: static_in) ivals.emplace_back(t);
    auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    mock_a = torch::randn({256,256}, f32); mock_b = torch::randn({256,256}, f32);
  }
  void capture(){
    c10::cuda::CUDAStreamGuard g(stream);
    for(int i=0;i<3;i++){ auto o = mod.forward(ivals); (void)o; }   // warmup (pre-capture)
    stream.synchronize();
    graph.capture_begin();                 // NB: 2.8 signature may want a pool handle; verify at build
    mod.forward(ivals);
    graph.capture_end();
  }
  void process(const std::vector<torch::Tensor>& inputs, int decode_gpu_iters){
    c10::cuda::CUDAStreamGuard g(stream);
    for(size_t i=0;i<static_in.size();++i) static_in[i].copy_(inputs[i], /*non_blocking=*/true);
    graph.replay();                        // encoder dispatched ASYNC on this lane's stream (pipelines; NOT blocked on)
    for(int i=0;i<decode_gpu_iters;i++){ auto c = torch::matmul(mock_a, mock_b); mock_a = c; } // mock decode GPU cost
    done_evt.record(stream);               // mark this chunk's GPU work; the worker host-decodes, THEN waits on it
  }
  void wait(){ done_evt.synchronize(); }   // ~0 if the GPU is keeping up; GROWS when the GPU saturates (catches the
                                           // GPU-bound knee). Per-lane streams still overlap on the GPU across lanes.
  void finish(){ stream.synchronize(); }
};

struct Chunk { int stream_id; clk::time_point ready; };
struct IntakeQueue {
  std::queue<Chunk> q; std::mutex m; std::condition_variable cv; std::atomic<bool> stop{false};
  void push(Chunk c){ {std::lock_guard<std::mutex> l(m); q.push(c);} cv.notify_one(); }
  bool pop(Chunk& out){ std::unique_lock<std::mutex> l(m); cv.wait(l,[&]{return !q.empty()||stop;});
                        if(q.empty()) return false; out=q.front(); q.pop(); return true; }
};

static void print_pct(std::vector<double>& v, const char* label){
  if(v.empty()){ std::printf("%s: (no samples)\n", label); return; }
  std::sort(v.begin(), v.end());
  auto pct=[&](double p){ return v[std::min(v.size()-1,(size_t)(p*v.size())) ]; };
  std::printf("%s: n=%zu p50=%.1f p95=%.1f p99=%.1f max=%.1f ms\n",
              label, v.size(), pct(0.50), pct(0.95), pct(0.99), v.back());
}

int main(int argc, char** argv){
  Args a = parse_args(argc, argv);
  torch::NoGradGuard ng;
  auto proto = make_proto();

  // One lane per dispatcher thread (no cross-thread graph sharing). Each loads its own module handle.
  // unique_ptr: CUDAGraph is non-movable (holds an incomplete-type intrusive_ptr), so Lane must never be moved/copied.
  std::vector<std::unique_ptr<Lane>> lanes;
  for(int i=0;i<a.lanes;i++){
    torch::jit::Module m = torch::jit::load(a.module); m.to(torch::kCUDA); m.eval();
    auto lane = std::make_unique<Lane>(c10::cuda::getStreamFromPool(false), std::move(m));
    lane->prepare(proto);
    lane->capture();
    lanes.push_back(std::move(lane));
  }

  IntakeQueue iq;
  std::atomic<bool> run{true};
  std::mutex lat_m; std::vector<double> latencies;   // chunk intake->done (the SLO proxy)

  // Dispatcher threads: thread i owns lane i; pulls chunks, processes on its lane, records latency.
  std::vector<std::thread> workers;
  for(int i=0;i<a.lanes;i++) workers.emplace_back([&,i]{
    Chunk c; Lane& lane = *lanes[i];
    while(iq.pop(c)){
      lane.process(proto, a.decode_gpu_iters);                                    // dispatch encoder async + record event
      if(a.decode_host_us>0) std::this_thread::sleep_for(std::chrono::microseconds(a.decode_host_us)); // GIL-held decode
      lane.wait();                                                                // wait GPU (≈0 unless GPU-saturated)
      double lat = ms_since(c.ready);
      { std::lock_guard<std::mutex> l(lat_m); latencies.push_back(lat); }
    }
  });

  // NVML GPU-util sampler.
  std::vector<unsigned int> util_samples;
  std::thread util_thread;
  bool nvml_ok = (nvmlInit_v2()==NVML_SUCCESS);
  nvmlDevice_t dev{};
  if(nvml_ok && nvmlDeviceGetHandleByIndex_v2(0,&dev)==NVML_SUCCESS){
    util_thread = std::thread([&]{
      while(run){ nvmlUtilization_t u{}; if(nvmlDeviceGetUtilizationRates(dev,&u)==NVML_SUCCESS) util_samples.push_back(u.gpu);
                  std::this_thread::sleep_for(std::chrono::milliseconds(100)); }
    });
  }

  // Stream generators: N streams, a chunk every chunk_ms. TODO: periodic finalize path.
  std::vector<std::thread> gens;
  for(int s=0;s<a.streams;s++) gens.emplace_back([&,s]{
    while(run){ iq.push({s, clk::now()}); std::this_thread::sleep_for(std::chrono::milliseconds(a.chunk_ms)); }
  });

  std::this_thread::sleep_for(std::chrono::seconds(a.duration_s));
  run=false; iq.stop=true; iq.cv.notify_all();
  for(auto& g: gens) g.join();
  for(auto& w: workers) w.join();
  for(auto& l: lanes) l->finish();   // drain pipelined GPU work
  if(util_thread.joinable()) util_thread.join();

  std::printf("=== lanes=%d streams=%d decode_host_us=%d decode_gpu_iters=%d ===\n",
              a.lanes, a.streams, a.decode_host_us, a.decode_gpu_iters);
  print_pct(latencies, "chunk_latency");
  if(!util_samples.empty()){
    double avg=0; for(auto u:util_samples) avg+=u; avg/=util_samples.size();
    std::printf("gpu_util avg=%.0f%% (n=%zu)\n", avg, util_samples.size());
  }
  // Read the SLO knee externally: the max --streams where chunk_latency p95 <= --slo-p95-ms; (knee_Mcores/knee_M1)
  // and absolute streams/box on L40S vs the >=1.5x (~28/box) gate. Calibrate decode_* so M=1 reproduces ~16-20/box.
  if(nvml_ok) nvmlShutdown();
  return 0;
}
