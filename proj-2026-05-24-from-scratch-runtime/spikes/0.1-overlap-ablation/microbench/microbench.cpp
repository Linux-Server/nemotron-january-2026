// 0.1b native launch-overlap microbench (C++ / libtorch) — SKELETON, build-ready, needs iteration on a GPU box.
//
// Question (gate): does GIL-free MULTI-THREAD intake/dispatch raise sustainable L40S streams/box above the Python
// single-thread-intake ceiling (~16-20/box), reclaiming the 40-65% idle GPU? GATE = >=1.5x (~28/box).
//
// Design: K "lanes", each = its own OS thread + CUDA stream + static IO buffers + a captured CUDA graph of the steady
// encoder (loaded from export_encoder.py). N simulated streams emit a chunk every 160ms; an INTAKE pool of M threads
// dispatches ready chunks onto lanes (M=1 mimics Python single-intake; M=cores is the native thesis). A "mock decode"
// approximates the real decode COST (calibrate vs Python model_wall<=35ms + decode host time), NOT its math.
//
// Build: see CMakeLists.txt (needs libtorch 2.8.0+cu128 C++ SDK). Run: ./microbench --module artifacts/encoder_steady_b1.ts
//   --intake-threads {1|cores} --streams N --lanes K --duration-s 30 --decode-gpu-us .. --decode-host-us ..
// Sweep N at fixed (lanes, intake-threads) to find the sustainable-N at the SLO (p95 chunk latency in budget).
//
// STATUS: structurally complete for the core mechanic (graph capture + replay from threads + M-thread intake). The
// SLO-sweep harness, NVML GPU-util sampling, finalize path, and MPS/multi-proc variants are marked TODO. Do NOT trust
// numbers until calibrated against the Python model_wall + a single-thread baseline that reproduces ~16-20/box.

#include <torch/script.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <cstdio>
#include <string>

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t){ return std::chrono::duration<double,std::milli>(clk::now()-t).count(); }

struct Args { std::string module="artifacts/encoder_steady_b1.ts"; int intake_threads=1; int streams=8; int lanes=3;
              int duration_s=30; int decode_gpu_us=300; int decode_host_us=200; }; // calibrate decode_* vs Python!

// One lane: dedicated stream + static buffers + captured graph. Replay is NOT thread-safe across lanes sharing a graph,
// so each lane owns its own (per-lane pool = the memory cost the density story must account for — spike 0.11).
struct Lane {
  c10::cuda::CUDAStream stream;
  at::cuda::CUDAGraph graph;
  std::vector<torch::Tensor> static_in;     // [processed, length, clc, clt, clcl]
  std::vector<torch::jit::IValue> ivals;    // view over static_in for module.forward
  torch::jit::Module* mod;
  bool captured=false;
  explicit Lane(c10::cuda::CUDAStream s): stream(s) {}

  void prepare(torch::jit::Module& m, const std::vector<torch::Tensor>& proto){
    mod=&m;
    for(auto& t: proto) static_in.push_back(t.clone());     // static buffers we copy inputs into each replay
    for(auto& t: static_in) ivals.emplace_back(t);
  }
  void capture(){
    c10::cuda::CUDAStreamGuard g(stream);
    // warmup (required before capture)
    for(int i=0;i<3;i++){ auto out = mod->forward(ivals); (void)out; }
    stream.synchronize();
    graph.capture_begin();
    mod->forward(ivals);          // recorded into the graph; outputs live in the static graph pool
    graph.capture_end();
    captured=true;
  }
  void replay(const std::vector<torch::Tensor>& inputs){
    c10::cuda::CUDAStreamGuard g(stream);
    for(size_t i=0;i<static_in.size();++i) static_in[i].copy_(inputs[i], /*non_blocking=*/true);
    graph.replay();
    // mock decode (cost stand-in): a tiny GPU op + a host-side stall approximating the eager decode's host cost.
    // TODO: replace with a calibrated kernel; the host stall models the .item()-loop CPU time that the GIL serializes.
  }
};

// ---- a minimal MPMC chunk queue (the "intake") ----
struct Chunk { int stream_id; clk::time_point ready; };
struct IntakeQueue {
  std::queue<Chunk> q; std::mutex m; std::condition_variable cv; std::atomic<bool> stop{false};
  void push(Chunk c){ {std::lock_guard<std::mutex> l(m); q.push(c);} cv.notify_one(); }
  bool pop(Chunk& out){ std::unique_lock<std::mutex> l(m); cv.wait(l,[&]{return !q.empty()||stop;});
                        if(q.empty()) return false; out=q.front(); q.pop(); return true; }
};

int main(int argc, char** argv){
  Args a; // TODO: parse argv into a (module, intake_threads, streams, lanes, duration_s, decode_*).
  (void)argc; (void)argv;

  torch::jit::Module mod = torch::jit::load(a.module);
  mod.to(torch::kCUDA); mod.eval();
  torch::NoGradGuard ng;

  // Build proto inputs at the steady B=1 shape (TODO: read from shapes.json instead of hardcoding).
  auto dev = torch::kCUDA;
  // proto = [processed [1,128,25], length [1], clc, clt, clcl] — shapes from shapes.json.
  std::vector<torch::Tensor> proto; // TODO: fill from shapes.json (clc/clt/clcl come from get_initial_cache_state).

  // Lanes (each its own stream + captured graph).
  std::vector<Lane> lanes;
  for(int i=0;i<a.lanes;i++){ lanes.emplace_back(c10::cuda::getStreamFromPool(/*high_priority=*/false));
                              lanes.back().prepare(mod, proto); lanes.back().capture(); }

  // Latency record (chunk intake->done) for the SLO knee.
  std::mutex lat_m; std::vector<double> latencies;
  std::atomic<int> next_lane{0};

  IntakeQueue iq;
  std::atomic<bool> run{true};

  // INTAKE THREADS — the variable under test. M=1 reproduces the Python single-asyncio-intake wall; M=cores is the
  // native thesis. Each intake thread pops ready chunks and dispatches onto a lane (round-robin here; real scheduler
  // would pin per session — fine for the launch-overlap question).
  std::vector<std::thread> intake;
  for(int t=0;t<a.intake_threads;t++) intake.emplace_back([&]{
    Chunk c;
    while(iq.pop(c)){
      auto t0 = c.ready;
      int li = next_lane.fetch_add(1) % lanes.size();
      lanes[li].replay(proto);                          // dispatch (graph replay) + mock decode
      lanes[li].stream.synchronize();                   // completion (TODO: CUDA-event dependency instead of sync)
      std::this_thread::sleep_for(std::chrono::microseconds(a.decode_host_us)); // mock decode host cost (GIL-held in Py)
      {std::lock_guard<std::mutex> l(lat_m); latencies.push_back(ms_since(t0));}
    }
  });

  // STREAM GENERATORS — N streams, each a chunk every 160ms (steady cadence). TODO: add periodic finalize.
  std::vector<std::thread> gens;
  for(int s=0;s<a.streams;s++) gens.emplace_back([&,s]{
    while(run){ iq.push({s, clk::now()}); std::this_thread::sleep_for(std::chrono::milliseconds(160)); }
  });

  std::this_thread::sleep_for(std::chrono::seconds(a.duration_s));
  run=false; iq.stop=true; iq.cv.notify_all();
  for(auto& g: gens) g.join(); for(auto& t: intake) t.join();

  // Report p50/p95/p99 chunk latency. SLO knee = max N where p95 stays in budget (the vad_stop->final proxy).
  // TODO: sort latencies, print percentiles; sample NVML GPU util during the run; sweep N externally to find the knee.
  std::printf("collected %zu samples (intake_threads=%d lanes=%d streams=%d)\n",
              latencies.size(), a.intake_threads, a.lanes, a.streams);
  // GATE: knee(M=cores) / knee(M=1) and absolute streams/box on L40S >= 1.5x baseline (~28/box) => GO.
  return 0;
}
