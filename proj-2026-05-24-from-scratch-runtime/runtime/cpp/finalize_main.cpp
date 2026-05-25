// 1.3b — C++ continuous-finalize compute path.
//
// Phase A: load finalize_bundle.ts, fork/clone parent encoder + decoder state, decode-continuation over the eager
// finalize enc_out, and assert token-exactness vs finalize_ref and NeMo stream+finalize oracle. Also proves the parent
// state is byte-identical after the fork decode.
//
// Phase B: if enc_finalize_aoti.pt2 exists, run the finalize keep_all_outputs AOTI encoder on contiguous cloned inputs,
// then decode and assert the same gold tokens.
#include <torch/script.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <vector>

using torch::inductor::AOTIModelPackageLoader;

static constexpr int BLANK = 1024;
static constexpr int MAX_SYMBOLS = 10;
static constexpr int SHIFT = 16;
static constexpr int PRE = 9;
static constexpr int DROP = 2;

struct ParentState {
  torch::Tensor clc;
  torch::Tensor clt;
  torch::Tensor clcl;
  torch::Tensor g;
  torch::Tensor h;
  torch::Tensor c;
};

static bool file_exists(const std::string& path) {
  struct stat st;
  return stat(path.c_str(), &st) == 0;
}

static std::string row_attr(int row, const char* name) {
  return "row" + std::to_string(row) + "_" + std::string(name);
}

static torch::Tensor attr_tensor(torch::jit::Module& module, const std::string& name) {
  return module.attr(name).toTensor();
}

static torch::Tensor row_tensor(torch::jit::Module& bundle, int row, const char* name) {
  return attr_tensor(bundle, row_attr(row, name));
}

static int64_t scalar_i64(torch::Tensor tensor) {
  return tensor.to(torch::kCPU).reshape({-1})[0].item<int64_t>();
}

static std::vector<int64_t> tensor_to_vec(torch::Tensor tensor) {
  auto flat = tensor.to(torch::kCPU).to(torch::kLong).contiguous().view({-1});
  std::vector<int64_t> out;
  out.reserve(flat.numel());
  for (int64_t i = 0; i < flat.numel(); ++i) out.push_back(flat[i].item<int64_t>());
  return out;
}

static std::string vec_to_string(const std::vector<int64_t>& values) {
  std::ostringstream oss;
  for (auto value : values) oss << ' ' << value;
  return oss.str();
}

static bool equal_tokens(const std::vector<int64_t>& got,
                         const std::vector<int64_t>& gold,
                         const char* label,
                         int row) {
  bool ok = got == gold;
  if (!ok) {
    std::printf("    row%d %s token mismatch: got_len=%zu gold_len=%zu\n", row, label, got.size(), gold.size());
    std::printf("      got :%s\n", vec_to_string(got).c_str());
    std::printf("      gold:%s\n", vec_to_string(gold).c_str());
  }
  return ok;
}

static bool tensor_equal(const char* name, const torch::Tensor& actual, const torch::Tensor& expected) {
  bool meta_ok = actual.scalar_type() == expected.scalar_type() &&
                 actual.sizes().vec() == expected.sizes().vec();
  bool eq = meta_ok && at::equal(actual, expected);
  if (!eq) {
    std::printf("    FORK_ASSERT %s mismatch: dtype %d/%d sizes",
                name, (int)actual.scalar_type(), (int)expected.scalar_type());
    for (auto s : actual.sizes()) std::printf(" %ld", (long)s);
    std::printf(" vs");
    for (auto s : expected.sizes()) std::printf(" %ld", (long)s);
    std::printf("\n");
  }
  return eq;
}

static ParentState clone_state(const ParentState& state) {
  return {
      state.clc.clone(),
      state.clt.clone(),
      state.clcl.clone(),
      state.g.clone(),
      state.h.clone(),
      state.c.clone(),
  };
}

static bool fork_assert_parent_unchanged(const ParentState& parent, const ParentState& snapshot) {
  bool ok = true;
  ok = tensor_equal("cache_last_channel", parent.clc, snapshot.clc) && ok;
  ok = tensor_equal("cache_last_time", parent.clt, snapshot.clt) && ok;
  ok = tensor_equal("cache_last_channel_len", parent.clcl, snapshot.clcl) && ok;
  ok = tensor_equal("pred_out", parent.g, snapshot.g) && ok;
  ok = tensor_equal("decoder_state.h", parent.h, snapshot.h) && ok;
  ok = tensor_equal("decoder_state.c", parent.c, snapshot.c) && ok;
  return ok;
}

static void decode_range(torch::jit::Module& joint,
                         torch::jit::Module& predict,
                         const torch::Tensor& enc_out,
                         int64_t enc_len,
                         torch::Tensor& g,
                         torch::Tensor& h,
                         torch::Tensor& c,
                         std::vector<int64_t>& hyp) {
  if (enc_len < 0 || enc_len > enc_out.size(2)) {
    throw std::runtime_error("enc_len out of range for enc_out");
  }
  auto f = enc_out.transpose(1, 2).contiguous();  // [1,T,1024]
  auto dev = f.device();
  for (int64_t t = 0; t < enc_len; ++t) {
    auto f_t = f.slice(1, t, t + 1);
    for (int n = 0; n < MAX_SYMBOLS; ++n) {
      auto logits = joint.forward({f_t, g}).toTensor();
      int64_t k = logits.reshape({-1}).argmax().item<int64_t>();
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

static ParentState load_parent(torch::jit::Module& bundle, int row, torch::Device device) {
  return {
      row_tensor(bundle, row, "cache_last_channel").to(device),
      row_tensor(bundle, row, "cache_last_time").to(device),
      row_tensor(bundle, row, "cache_last_channel_len").to(device),
      row_tensor(bundle, row, "pre_final_pred_out").to(device),
      row_tensor(bundle, row, "pre_final_h").to(device),
      row_tensor(bundle, row, "pre_final_c").to(device),
  };
}

static std::vector<int64_t> run_decode_from_eager(torch::jit::Module& bundle,
                                                  int row,
                                                  ParentState& fork,
                                                  torch::jit::Module& joint,
                                                  torch::jit::Module& predict,
                                                  torch::Device device) {
  std::vector<int64_t> hyp = tensor_to_vec(row_tensor(bundle, row, "pre_final_tokens"));
  auto enc_out = row_tensor(bundle, row, "eager_enc_out").to(device);
  int64_t enc_len = scalar_i64(row_tensor(bundle, row, "eager_enc_len"));
  decode_range(joint, predict, enc_out, enc_len, fork.g, fork.h, fork.c, hyp);
  return hyp;
}

static std::vector<int64_t> run_decode_from_aoti(AOTIModelPackageLoader& loader,
                                                 torch::jit::Module& bundle,
                                                 int row,
                                                 ParentState& fork,
                                                 torch::jit::Module& joint,
                                                 torch::jit::Module& predict,
                                                 torch::Device device) {
  std::vector<at::Tensor> inputs = {
      row_tensor(bundle, row, "chunk_mel").to(device).contiguous(),
      fork.clc.contiguous(),
      fork.clt.contiguous(),
      fork.clcl.contiguous(),
  };
  auto out = loader.run(inputs);
  if (out.size() < 2) throw std::runtime_error("enc_finalize_aoti.pt2 returned fewer than 2 outputs");

  std::vector<int64_t> hyp = tensor_to_vec(row_tensor(bundle, row, "pre_final_tokens"));
  int64_t enc_len = scalar_i64(out[1]);
  decode_range(joint, predict, out[0], enc_len, fork.g, fork.h, fork.c, hyp);
  return hyp;
}

int main(int argc, char** argv) {
  std::string dir = argc > 1 ? argv[1] : "../artifacts";
  torch::NoGradGuard ng;
  auto device = torch::Device(torch::kCUDA);

  auto joint = torch::jit::load(dir + "/joint_step.ts");
  joint.to(device);
  joint.eval();
  auto predict = torch::jit::load(dir + "/predict_step.ts");
  predict.to(device);
  predict.eval();
  auto bundle = torch::jit::load(dir + "/finalize_bundle.ts");

  auto meta = attr_tensor(bundle, "meta").to(torch::kCPU).to(torch::kLong).contiguous();
  int64_t rows = meta[0].item<int64_t>();
  if (meta[1].item<int64_t>() != BLANK || meta[2].item<int64_t>() != MAX_SYMBOLS ||
      meta[3].item<int64_t>() != SHIFT || meta[4].item<int64_t>() != PRE ||
      meta[5].item<int64_t>() != DROP) {
    std::printf("METADATA MISMATCH: bundle blank/msym/shift/pre/drop = %ld/%ld/%ld/%ld/%ld\n",
                meta[1].item<int64_t>(), meta[2].item<int64_t>(), meta[3].item<int64_t>(),
                meta[4].item<int64_t>(), meta[5].item<int64_t>());
    return 2;
  }
  int64_t num_rows = scalar_i64(attr_tensor(bundle, "num_rows"));
  if (num_rows != rows) {
    std::printf("num_rows mismatch: meta=%ld buffer=%ld\n", (long)rows, (long)num_rows);
    return 2;
  }

  std::printf("=== Phase A: eager finalize enc_out decode-continuation (%ld rows) ===\n", (long)rows);
  bool phase_a_ok = true;
  for (int row = 0; row < rows; ++row) {
    int64_t drop_extra = scalar_i64(row_tensor(bundle, row, "drop_extra"));
    bool emitted_gt0 = row_tensor(bundle, row, "emitted_gt0").to(torch::kCPU).item<bool>();
    int64_t enc_len = scalar_i64(row_tensor(bundle, row, "eager_enc_len"));
    auto parent = load_parent(bundle, row, device);
    auto snapshot = clone_state(parent);
    auto fork = clone_state(parent);

    std::vector<int64_t> got;
    bool row_ok = true;
    try {
      got = run_decode_from_eager(bundle, row, fork, joint, predict, device);
    } catch (const std::exception& e) {
      std::printf("  row%d Phase A decode threw: %s\n", row, e.what());
      row_ok = false;
    }
    auto gold_ref = tensor_to_vec(row_tensor(bundle, row, "finalize_ref_final_tokens"));
    auto gold_nemo = tensor_to_vec(row_tensor(bundle, row, "nemo_stream_finalize_tokens"));
    bool ref_ok = row_ok && equal_tokens(got, gold_ref, "finalize_ref", row);
    bool nemo_ok = row_ok && equal_tokens(got, gold_nemo, "nemo_stream_finalize", row);
    bool fork_ok = fork_assert_parent_unchanged(parent, snapshot);
    bool meta_ok = ((drop_extra != 0) == emitted_gt0);
    if (!meta_ok) {
      std::printf("    row%d metadata mismatch: drop_extra=%ld emitted_gt0=%d\n",
                  row, (long)drop_extra, (int)emitted_gt0);
    }
    bool all = ref_ok && nemo_ok && fork_ok && meta_ok;
    phase_a_ok = phase_a_ok && all;
    std::printf("  row%d drop=%ld emitted_gt0=%d enc_len=%ld tokens=%zu: "
                "finalize_ref=%s nemo=%s FORK_ASSERT=%s\n",
                row, (long)drop_extra, (int)emitted_gt0, (long)enc_len, got.size(),
                ref_ok ? "PASS" : "FAIL", nemo_ok ? "PASS" : "FAIL",
                fork_ok ? "PASS" : "FAIL");
  }

  std::string aoti_pkg = dir + "/enc_finalize_aoti.pt2";
  bool phase_b_present = file_exists(aoti_pkg);
  bool phase_b_ok = true;
  if (!phase_b_present) {
    std::printf("PHASE B skipped (enc_finalize_aoti.pt2 not present)\n");
  } else {
    std::printf("=== Phase B: AOTI finalize encoder + decode-continuation (%s) ===\n", aoti_pkg.c_str());
    AOTIModelPackageLoader loader(aoti_pkg, "model", /*run_single_threaded=*/false,
                                  /*num_runners=*/1, /*device_index=*/-1);
    for (int row = 0; row < rows; ++row) {
      auto parent = load_parent(bundle, row, device);
      auto snapshot = clone_state(parent);
      auto fork = clone_state(parent);
      std::vector<int64_t> got;
      bool row_ok = true;
      try {
        got = run_decode_from_aoti(loader, bundle, row, fork, joint, predict, device);
      } catch (const std::exception& e) {
        std::printf("  row%d Phase B threw: %s\n", row, e.what());
        row_ok = false;
      }
      auto gold_ref = tensor_to_vec(row_tensor(bundle, row, "finalize_ref_final_tokens"));
      auto gold_nemo = tensor_to_vec(row_tensor(bundle, row, "nemo_stream_finalize_tokens"));
      bool ref_ok = row_ok && equal_tokens(got, gold_ref, "finalize_ref", row);
      bool nemo_ok = row_ok && equal_tokens(got, gold_nemo, "nemo_stream_finalize", row);
      bool fork_ok = fork_assert_parent_unchanged(parent, snapshot);
      bool all = ref_ok && nemo_ok && fork_ok;
      phase_b_ok = phase_b_ok && all;
      std::printf("  row%d tokens=%zu: finalize_ref=%s nemo=%s FORK_ASSERT=%s\n",
                  row, got.size(), ref_ok ? "PASS" : "FAIL", nemo_ok ? "PASS" : "FAIL",
                  fork_ok ? "PASS" : "FAIL");
    }
  }

  std::printf("=== FINALIZE %s: PhaseA=%s PhaseB=%s ===\n",
              (phase_a_ok && phase_b_ok) ? "PASS" : "FAIL",
              phase_a_ok ? "PASS" : "FAIL",
              phase_b_present ? (phase_b_ok ? "PASS" : "FAIL") : "SKIPPED");
  return (phase_a_ok && phase_b_ok) ? 0 : 1;
}
