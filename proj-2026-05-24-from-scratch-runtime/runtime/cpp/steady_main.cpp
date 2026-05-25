// 1.2b — C++ STREAMING steady pipeline: the verified Python streaming loop, ported. Cache-aware encoder chunk loop
// (first chunk drop=0 @ enc_first.ts; steady = cat(ring[9], new[16]) drop=2 @ enc_steady.ts; ring=last 9), threading
// the encoder cache + carrying decoder state, decode per chunk. Self-checks TOKEN-exact (T1) vs gold (== NeMo
// streaming). Build: CMakeLists.txt (manual-link libtorch). Note: encoder is token-exact (T1), ~1e-5 drift vs eager
// (T2a refinement = torch.export/dynamic).
#include <torch/script.h>
#include <cstdio>
#include <vector>

static constexpr int BLANK = 1024, MAX_SYMBOLS = 10, SHIFT = 16, PRE = 9;

int main(int argc, char** argv) {
  std::string dir = argc > 1 ? argv[1] : "../artifacts";
  torch::NoGradGuard ng; auto CU = torch::kCUDA;
  auto enc_first  = torch::jit::load(dir + "/enc_first.ts");  enc_first.to(CU);  enc_first.eval();
  auto enc_steady = torch::jit::load(dir + "/enc_steady.ts"); enc_steady.to(CU); enc_steady.eval();
  auto joint   = torch::jit::load(dir + "/joint_step.ts");    joint.to(CU);   joint.eval();
  auto predict = torch::jit::load(dir + "/predict_step.ts");  predict.to(CU); predict.eval();
  auto init    = torch::jit::load(dir + "/cpp_bundle.ts");    init.to(CU);
  auto sb      = torch::jit::load(dir + "/stream_bundle.ts"); sb.to(CU);

  auto mel  = sb.attr("mel").toTensor();                 // [1,128,Tm]
  auto gold = sb.attr("gold").toTensor().to(torch::kCPU);
  auto clc  = sb.attr("clc0").toTensor().clone();
  auto clt  = sb.attr("clt0").toTensor().clone();
  auto clcl = sb.attr("clcl0").toTensor().clone();
  int Tm = mel.size(2);

  auto g = init.attr("sos_g").toTensor(), h = init.attr("init_h").toTensor(), c = init.attr("init_c").toTensor();
  torch::Tensor ring;  // last PRE mel frames
  std::vector<int64_t> hyp;
  int emitted = 0, nchunks = 0;
  for (int pos = 0; pos < Tm; pos += SHIFT, ++nchunks) {
    auto new_mel = mel.slice(2, pos, std::min(pos + SHIFT, Tm));
    torch::Tensor chunk; torch::jit::Module* mod;
    if (emitted == 0) { chunk = new_mel; mod = &enc_first; }
    else { chunk = torch::cat({ring, new_mel}, 2); mod = &enc_steady; }
    auto L = torch::full({1}, chunk.size(2), torch::dtype(torch::kLong).device(CU));
    auto out = mod->forward({chunk, L, clc, clt, clcl}).toTuple();
    auto eo = out->elements()[0].toTensor();             // [1,1024,To]
    int To = out->elements()[1].toTensor().item<int64_t>();
    clc = out->elements()[2].toTensor(); clt = out->elements()[3].toTensor(); clcl = out->elements()[4].toTensor();

    auto f = eo.transpose(1, 2).contiguous();
    for (int t = 0; t < To; ++t) {
      auto f_t = f.slice(1, t, t + 1);
      for (int n = 0; n < MAX_SYMBOLS; ++n) {
        int64_t k = joint.forward({f_t, g}).toTensor().reshape({-1}).argmax().item<int64_t>();
        if (k == BLANK) break;
        hyp.push_back(k);
        auto y = torch::full({1, 1}, k, torch::dtype(torch::kLong).device(CU));
        auto o = predict.forward({y, h, c}).toTuple();
        g = o->elements()[0].toTensor(); h = o->elements()[1].toTensor(); c = o->elements()[2].toTensor();
      }
    }
    auto cum = ring.defined() ? torch::cat({ring, new_mel}, 2) : new_mel;
    ring = cum.slice(2, std::max<int64_t>(0, cum.size(2) - PRE), cum.size(2));
    emitted += new_mel.size(2);
  }

  bool ok = ((int)hyp.size() == gold.size(0));
  for (int i = 0; ok && i < (int)hyp.size(); ++i) ok = (hyp[i] == gold[i].item<int64_t>());
  std::printf("C++ STREAMING: chunks=%d -> %zu tokens vs gold %ld -> %s (T1 token-exact)\n",
              nchunks, hyp.size(), (long)gold.size(0), ok ? "PASS" : "FAIL");
  if (!ok) { std::printf("  got :"); for (auto k : hyp) std::printf(" %ld", (long)k);
             std::printf("\n  gold:"); for (int i=0;i<gold.size(0);++i) std::printf(" %ld",(long)gold[i].item<int64_t>()); std::printf("\n"); }
  return ok ? 0 : 1;
}
