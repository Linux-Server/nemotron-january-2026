// 1.3b — C++ continuous-finalize compute path.
//
// Phase A: load finalize_bundle.ts, fork/clone parent encoder + decoder state, decode-continuation over the eager
// finalize enc_out, and assert token-exactness vs finalize_ref and NeMo stream+finalize oracle. Also proves the parent
// state is byte-identical after the fork decode.
//
// Phase B: if per-exact-T finalize AOTI buckets + shared weights exist, wire one CUDA weight set into every bucket,
// route each row by (drop_extra, chunk T), run on contiguous cloned inputs, then decode and assert the same gold tokens.
#include <torch/script.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <vector>

using torch::inductor::AOTIModelPackageLoader;
namespace fs = std::filesystem;

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

static bool directory_exists(const std::string& path) {
  struct stat st;
  return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
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

static bool parse_bucket_filename(const std::string& filename, int64_t& drop, int64_t& T) {
  const std::string prefix = "enc_finalize_d";
  const std::string mid = "_T";
  const std::string suffix = ".pt2";
  if (filename.rfind(prefix, 0) != 0) return false;
  if (filename.size() <= prefix.size() + suffix.size()) return false;
  if (filename.compare(filename.size() - suffix.size(), suffix.size(), suffix) != 0) return false;

  size_t tpos = filename.find(mid, prefix.size());
  if (tpos == std::string::npos) return false;
  std::string drop_s = filename.substr(prefix.size(), tpos - prefix.size());
  std::string T_s = filename.substr(tpos + mid.size(), filename.size() - suffix.size() - (tpos + mid.size()));
  if (drop_s.empty() || T_s.empty()) return false;

  try {
    size_t n = 0;
    long long d = std::stoll(drop_s, &n);
    if (n != drop_s.size()) return false;
    n = 0;
    long long t = std::stoll(T_s, &n);
    if (n != T_s.size()) return false;
    if (d < 0 || t <= 0) return false;
    drop = d;
    T = t;
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

static std::map<std::pair<int64_t, int64_t>, std::string> discover_finalize_buckets(const std::string& buckets_dir) {
  std::map<std::pair<int64_t, int64_t>, std::string> buckets;
  for (const auto& entry : fs::directory_iterator(buckets_dir)) {
    if (!entry.is_regular_file()) continue;
    int64_t drop = 0;
    int64_t T = 0;
    std::string filename = entry.path().filename().string();
    if (!parse_bucket_filename(filename, drop, T)) continue;
    auto key = std::make_pair(drop, T);
    auto path = entry.path().string();
    auto inserted = buckets.emplace(key, path);
    if (!inserted.second) {
      throw std::runtime_error("duplicate finalize bucket for (drop,T)=(" + std::to_string(drop) + "," +
                               std::to_string(T) + "): " + inserted.first->second + " and " + path);
    }
  }
  return buckets;
}

static std::unordered_map<std::string, at::Tensor> load_shared_constants(const std::string& weights_path,
                                                                         torch::Device device) {
  auto weights_module = torch::jit::load(weights_path);
  auto weights = weights_module.attr("weights").toGenericDict();
  std::unordered_map<std::string, at::Tensor> constants;
  constants.reserve(weights.size());
  for (const auto& item : weights) {
    if (!item.key().isString()) throw std::runtime_error("finalize_shared_weights.ts has a non-string key");
    if (!item.value().isTensor()) throw std::runtime_error("finalize_shared_weights.ts has a non-tensor value");
    constants.emplace(item.key().toStringRef(), item.value().toTensor().to(device));
  }

  // Exporters have used both self.e and self.encoder for the wrapped module. Keep both names pointing at the same
  // CUDA tensors so the bucket packages can share one weight set across either FQN convention.
  std::vector<std::pair<std::string, at::Tensor>> aliases;
  aliases.reserve(constants.size());
  for (const auto& kv : constants) {
    if (kv.first.rfind("e.", 0) == 0) {
      aliases.emplace_back("encoder." + kv.first.substr(2), kv.second);
    } else if (kv.first.rfind("encoder.", 0) == 0) {
      aliases.emplace_back("e." + kv.first.substr(8), kv.second);
    }
  }
  for (auto& alias : aliases) {
    if (constants.find(alias.first) == constants.end()) constants.emplace(std::move(alias));
  }
  return constants;
}

static std::unordered_map<std::string, at::Tensor> constants_for_bucket(
    const std::unordered_map<std::string, at::Tensor>& shared_constants,
    AOTIModelPackageLoader& loader,
    const std::string& pkg) {
  auto fqns = loader.get_constant_fqns();
  std::unordered_map<std::string, at::Tensor> bucket_constants;
  bucket_constants.reserve(fqns.size());
  std::vector<std::string> missing;
  for (const auto& fqn : fqns) {
    auto it = shared_constants.find(fqn);
    if (it == shared_constants.end()) {
      missing.push_back(fqn);
    } else {
      bucket_constants.emplace(fqn, it->second);
    }
  }
  if (!missing.empty()) {
    std::ostringstream oss;
    oss << "bucket " << pkg << " missing " << missing.size() << " shared weights; first missing:";
    for (size_t i = 0; i < std::min<size_t>(missing.size(), 5); ++i) oss << ' ' << missing[i];
    throw std::runtime_error(oss.str());
  }
  return bucket_constants;
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
  if (out.size() < 2) throw std::runtime_error("finalize AOTI bucket returned fewer than 2 outputs");

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

  // Prefer the stripped buckets (the deployable form: tiny wrapper .so, weights supplied via load_constants); fall back
  // to the unstripped finalize_buckets/ if present. (1.3b-enc-scale moved the buckets to stripped_finalize_buckets/.)
  std::string buckets_dir = dir + "/stripped_finalize_buckets";
  if (!directory_exists(buckets_dir)) buckets_dir = dir + "/finalize_buckets";
  std::string shared_weights = dir + "/finalize_shared_weights.ts";
  bool phase_b_present = false;
  bool phase_b_ok = true;
  if (!directory_exists(buckets_dir) || !file_exists(shared_weights)) {
    std::printf("PHASE B skipped (finalize buckets not present)\n");
  } else {
    auto bucket_paths = discover_finalize_buckets(buckets_dir);
    if (bucket_paths.empty()) {
      std::printf("PHASE B skipped (finalize buckets not present)\n");
    } else {
      phase_b_present = true;
      std::printf("=== Phase B: AOTI finalize encoder buckets + decode-continuation (%zu buckets) ===\n",
                  bucket_paths.size());
      try {
        auto shared_constants = load_shared_constants(shared_weights, device);
        std::printf("  loaded shared constants: %zu FQN entries (aliases included) from %s\n",
                    shared_constants.size(), shared_weights.c_str());

        std::map<std::pair<int64_t, int64_t>, std::unique_ptr<AOTIModelPackageLoader>> loaders;
        for (const auto& kv : bucket_paths) {
          int64_t drop = kv.first.first;
          int64_t T = kv.first.second;
          const std::string& pkg = kv.second;
          auto loader = std::make_unique<AOTIModelPackageLoader>(pkg, "model", /*run_single_threaded=*/false,
                                                                 /*num_runners=*/1, /*device_index=*/-1);
          auto bucket_constants = constants_for_bucket(shared_constants, *loader, pkg);
          // C++ signature: (constants_map, use_inactive, check_full_update, user_managed).
          loader->load_constants(bucket_constants, /*use_inactive=*/false,
                                 /*check_full_update=*/false, /*user_managed=*/true);
          std::printf("  bucket drop=%ld T=%ld: loaded %zu constants -> %s\n",
                      (long)drop, (long)T, bucket_constants.size(), pkg.c_str());
          loaders.emplace(kv.first, std::move(loader));
        }

        for (int row = 0; row < rows; ++row) {
          int64_t drop_extra = scalar_i64(row_tensor(bundle, row, "drop_extra"));
          auto chunk_mel = row_tensor(bundle, row, "chunk_mel");
          int64_t chunk_T = chunk_mel.size(chunk_mel.dim() - 1);
          auto loader_it = loaders.find(std::make_pair(drop_extra, chunk_T));
          if (loader_it == loaders.end()) {
            // FAIL-CLOSED (enc-scale review B2): no bucket -> no finalize encoder -> dropped final transcript.
            // Until a validated eager fallback exists, treat as a hard failure, not a skip.
            std::printf("  row%d drop=%ld T=%ld: no bucket for (drop,T) -> FAIL (no validated fallback)\n",
                        row, (long)drop_extra, (long)chunk_T);
            phase_b_ok = false;
            continue;
          }

          auto parent = load_parent(bundle, row, device);
          auto snapshot = clone_state(parent);
          auto fork = clone_state(parent);
          std::vector<int64_t> got;
          bool row_ok = true;
          try {
            got = run_decode_from_aoti(*loader_it->second, bundle, row, fork, joint, predict, device);
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
          std::printf("  row%d drop=%ld T=%ld tokens=%zu: finalize_ref=%s nemo=%s FORK_ASSERT=%s\n",
                      row, (long)drop_extra, (long)chunk_T, got.size(),
                      ref_ok ? "PASS" : "FAIL", nemo_ok ? "PASS" : "FAIL",
                      fork_ok ? "PASS" : "FAIL");
        }
      } catch (const std::exception& e) {
        std::printf("  Phase B setup threw: %s\n", e.what());
        phase_b_ok = false;
      }
    }
  }

  std::printf("=== FINALIZE %s: PhaseA=%s PhaseB=%s ===\n",
              (phase_a_ok && phase_b_ok) ? "PASS" : "FAIL",
              phase_a_ok ? "PASS" : "FAIL",
              phase_b_present ? (phase_b_ok ? "PASS" : "FAIL") : "SKIPPED");
  return (phase_a_ok && phase_b_ok) ? 0 : 1;
}
