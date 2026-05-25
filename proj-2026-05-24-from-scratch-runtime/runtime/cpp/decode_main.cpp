// 1.2a — C++ RNNT greedy decode (the verified reference, translated to libtorch C++). Loads the exported
// joint_step.ts + predict_step.ts + a self-contained cpp_bundle.ts (init constants + the real_1 fixture enc/gold as
// named buffers), runs the greedy loop, and self-checks BYTE-EXACT vs the gold y_sequence. Proves the decode is
// byte-exact in C++ (B1b). Build: CMakeLists.txt (manual-link libtorch, no nvcc — same as the 0.1b microbench).
#include <torch/script.h>
#include <cstdio>
#include <vector>

static constexpr int BLANK = 1024;
static constexpr int MAX_SYMBOLS = 10;

int main(int argc, char** argv) {
  std::string dir = argc > 1 ? argv[1] : "../artifacts";
  torch::NoGradGuard ng;
  auto joint   = torch::jit::load(dir + "/joint_step.ts");   joint.to(torch::kCUDA);   joint.eval();
  auto predict = torch::jit::load(dir + "/predict_step.ts"); predict.to(torch::kCUDA); predict.eval();
  auto bundle  = torch::jit::load(dir + "/cpp_bundle.ts");   bundle.to(torch::kCUDA);

  auto g    = bundle.attr("sos_g").toTensor();          // [1,1,640]
  auto h    = bundle.attr("init_h").toTensor();         // [2,1,640]
  auto c    = bundle.attr("init_c").toTensor();
  auto enc  = bundle.attr("enc").toTensor();            // [1,1024,T]
  int  T    = bundle.attr("enc_len").toTensor().item<int64_t>();
  auto gold = bundle.attr("gold").toTensor().to(torch::kCPU);

  auto f = enc.transpose(1, 2).contiguous();            // [1,T,1024]
  std::vector<int64_t> hyp;
  for (int t = 0; t < T; ++t) {
    auto f_t = f.slice(1, t, t + 1);                    // [1,1,1024]
    for (int n = 0; n < MAX_SYMBOLS; ++n) {
      auto logits = joint.forward({f_t, g}).toTensor(); // [1,1,1,1025]
      int64_t k = logits.reshape({-1}).argmax().item<int64_t>();
      if (k == BLANK) break;
      hyp.push_back(k);
      auto y = torch::full({1, 1}, k, torch::dtype(torch::kLong).device(torch::kCUDA));
      auto out = predict.forward({y, h, c}).toTuple();
      g = out->elements()[0].toTensor();
      h = out->elements()[1].toTensor();
      c = out->elements()[2].toTensor();
    }
  }

  // compare
  bool ok = ((int)hyp.size() == gold.size(0));
  for (int i = 0; ok && i < (int)hyp.size(); ++i) ok = (hyp[i] == gold[i].item<int64_t>());
  std::printf("C++ decode: %zu tokens vs gold %ld  -> %s\n", hyp.size(), (long)gold.size(0), ok ? "BYTE-EXACT PASS" : "FAIL");
  if (!ok) {
    std::printf("  got :"); for (auto k : hyp) std::printf(" %ld", (long)k); std::printf("\n  gold:");
    for (int i = 0; i < gold.size(0); ++i) std::printf(" %ld", (long)gold[i].item<int64_t>()); std::printf("\n");
  }
  return ok ? 0 : 1;
}
