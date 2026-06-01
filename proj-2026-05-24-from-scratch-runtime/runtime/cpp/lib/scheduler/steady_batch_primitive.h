#pragma once

#include <torch/script.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include "lib/runtime_io/io.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using torch::inductor::AOTIModelPackageLoader;

namespace bsteady_detail {
namespace fs = std::filesystem;

struct BucketConstants {
  std::unordered_map<std::string, at::Tensor> values;
  size_t direct_matches = 0;
  size_t alias_fallbacks = 0;
};

struct ManifestBucket {
  int B = 0;
  std::string package;
  std::string package_sha256;
  std::string ep_sha256;
  std::string shared_weight_sha256;
  int64_t package_bytes = 0;
  int64_t ep_bytes = 0;
  int64_t shared_weight_bytes = 0;
};

using runtime_io::directory_exists;
using runtime_io::file_exists;
using runtime_io::find_matching_json_delim;
using runtime_io::json_int_field;
using runtime_io::json_string_field;
using runtime_io::json_value_for_key;
using runtime_io::read_text_file;
using runtime_io::sha256_file;
using runtime_io::skip_ws;

static inline int64_t file_size_bytes(const std::string& path) {
  std::error_code ec;
  auto size = fs::file_size(path, ec);
  if (ec) throw std::runtime_error("could not stat file size: " + path);
  if (size > static_cast<uintmax_t>(std::numeric_limits<int64_t>::max())) {
    throw std::runtime_error("file too large to validate size: " + path);
  }
  return static_cast<int64_t>(size);
}

static inline std::vector<ManifestBucket> load_manifest_buckets(const std::string& manifest_path) {
  std::string text = read_text_file(manifest_path);
  size_t top_buckets = text.rfind("\"buckets\"");
  if (top_buckets == std::string::npos) throw std::runtime_error("steady manifest missing top-level buckets");
  std::string buckets_arr = json_value_for_key(text.substr(top_buckets), "buckets");
  if (buckets_arr.empty() || buckets_arr.front() != '[') throw std::runtime_error("steady manifest buckets is not an array");
  std::vector<ManifestBucket> buckets;
  size_t pos = 1;
  while (pos + 1 < buckets_arr.size()) {
    pos = skip_ws(buckets_arr, pos);
    if (pos >= buckets_arr.size() || buckets_arr[pos] == ']') break;
    if (buckets_arr[pos] == ',') {
      ++pos;
      continue;
    }
    if (buckets_arr[pos] != '{') throw std::runtime_error("steady manifest bucket entry is not an object");
    size_t end = find_matching_json_delim(buckets_arr, pos);
    std::string obj = buckets_arr.substr(pos, end - pos + 1);
    ManifestBucket b;
    b.B = static_cast<int>(json_int_field(obj, "B"));
    b.package = json_string_field(obj, "package");
    b.package_sha256 = json_string_field(obj, "package_sha256");
    b.ep_sha256 = json_string_field(obj, "ep_sha256", false);
    b.shared_weight_sha256 = json_string_field(obj, "shared_weight_sha256");
    std::string byte_sizes = json_value_for_key(obj, "byte_sizes", false);
    if (!byte_sizes.empty()) {
      b.package_bytes = json_int_field(byte_sizes, "package");
      b.ep_bytes = json_int_field(byte_sizes, "ep");
      b.shared_weight_bytes = json_int_field(byte_sizes, "shared_weight");
    }
    buckets.push_back(std::move(b));
    pos = end + 1;
  }
  return buckets;
}

static inline std::unordered_map<std::string, at::Tensor> load_shared_constants(const std::string& weights_path,
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
  return constants;
}

static inline const at::Tensor* resolve_shared_constant(
    const std::unordered_map<std::string, at::Tensor>& shared_constants,
    const std::string& fqn,
    bool& used_alias) {
  auto it = shared_constants.find(fqn);
  if (it != shared_constants.end()) {
    used_alias = false;
    return &it->second;
  }

  std::string alt;
  if (fqn.rfind("encoder.", 0) == 0) {
    alt = "e." + fqn.substr(8);
  } else if (fqn.rfind("e.", 0) == 0) {
    alt = "encoder." + fqn.substr(2);
  } else {
    return nullptr;
  }
  it = shared_constants.find(alt);
  if (it == shared_constants.end()) return nullptr;
  used_alias = true;
  return &it->second;
}

static inline BucketConstants constants_for_bucket(
    const std::unordered_map<std::string, at::Tensor>& shared_constants,
    AOTIModelPackageLoader& loader,
    const std::string& pkg) {
  auto fqns = loader.get_constant_fqns();
  BucketConstants bucket_constants;
  bucket_constants.values.reserve(fqns.size());
  std::vector<std::string> missing;
  for (const auto& fqn : fqns) {
    bool used_alias = false;
    const at::Tensor* tensor = resolve_shared_constant(shared_constants, fqn, used_alias);
    if (tensor == nullptr) {
      missing.push_back(fqn);
    } else {
      if (used_alias) {
        ++bucket_constants.alias_fallbacks;
      } else {
        ++bucket_constants.direct_matches;
      }
      bucket_constants.values.emplace(fqn, *tensor);
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
}  // namespace bsteady_detail

struct BatchedSteadyInput {
  torch::Tensor chunk;       // [1, 128, 25]
  torch::Tensor cache_ch;    // [24, 1, 70, 1024]
  torch::Tensor cache_t;     // [24, 1, 1024, 8]
  torch::Tensor cache_ch_len;  // [1]
  std::string label;
};

struct BatchedSteadyOutput {
  std::vector<at::Tensor> tensors;  // enc_out, enc_len, cache_ch, cache_t, cache_ch_len, all row-shaped
  int bucket = 0;
  int row = 0;
  std::string label;
};

class BatchedSteadyLoaderSet {
 public:
  BatchedSteadyLoaderSet(std::string package_dir,
                         std::string shared_weights_ts,
                         torch::Device device,
                         int num_runners,
                         std::string policy)
      : package_dir_(std::move(package_dir)),
        shared_weights_ts_(std::move(shared_weights_ts)),
        device_(device),
        num_runners_(num_runners),
        policy_(std::move(policy)) {
    if (num_runners_ <= 0) throw std::runtime_error("batched steady num_runners must be positive");
    if (!bsteady_detail::directory_exists(package_dir_)) {
      throw std::runtime_error("batched steady package directory missing: " + package_dir_);
    }
    if (!bsteady_detail::file_exists(shared_weights_ts_)) {
      throw std::runtime_error("batched steady shared weights missing: " + shared_weights_ts_);
    }
    verify_manifest();
    shared_constants_ = bsteady_detail::load_shared_constants(shared_weights_ts_, device_);
    std::printf("density loaded batched steady shared constants: %zu entries policy=%s\n",
                shared_constants_.size(),
                policy_.c_str());
  }

  void preload_all() {
    if (sealed_) return;
    for (int bucket : kBuckets) {
      (void)load_bucket(bucket);
    }
    sealed_ = true;
  }

  void preload_buckets(const std::vector<int>& buckets) {
    if (sealed_) return;
    if (buckets.empty()) throw std::runtime_error("batched steady preload_buckets requires at least one bucket");
    for (int bucket : buckets) {
      if (std::find(kBuckets.begin(), kBuckets.end(), bucket) == kBuckets.end()) {
        throw std::runtime_error("batched steady invalid preload bucket B=" + std::to_string(bucket));
      }
      (void)load_bucket(bucket);
    }
    sealed_ = true;
  }

  std::vector<BatchedSteadyOutput> run(const std::vector<BatchedSteadyInput>& ready,
                                       c10::cuda::CUDAStream stream) {
    c10::cuda::CUDAStreamGuard stream_guard(stream);
    if (ready.empty()) throw std::runtime_error("batched steady run called with no ready rows");
    if (ready.size() > 4) {
      throw std::runtime_error("batched steady supports at most K=4 rows in B1, got K=" +
                               std::to_string(ready.size()));
    }
    int bucket = bucket_for_k(static_cast<int>(ready.size()));
    auto inputs = pack_inputs(ready, bucket);
    return run_prepacked(inputs, ready, bucket, stream);
  }

  std::vector<BatchedSteadyOutput> run_prepacked(const std::vector<at::Tensor>& inputs,
                                                 const std::vector<BatchedSteadyInput>& ready,
                                                 int bucket,
                                                 c10::cuda::CUDAStream stream) {
    c10::cuda::CUDAStreamGuard stream_guard(stream);
    if (ready.empty()) throw std::runtime_error("batched steady prepacked run called with no ready rows");
    auto out = run_raw_prepacked(inputs, bucket, stream);
    return unpack_prepacked_outputs(out, ready, bucket);
  }

  std::vector<at::Tensor> run_raw_prepacked(const std::vector<at::Tensor>& inputs,
                                            int bucket,
                                            c10::cuda::CUDAStream stream) {
    c10::cuda::CUDAStreamGuard stream_guard(stream);
    auto& loader = get(bucket);
    auto out = loader.run(inputs, reinterpret_cast<void*>(stream.stream()));
    if (out.size() < 5) throw std::runtime_error("batched steady AOTI returned fewer than 5 outputs");
    return out;
  }

  std::vector<BatchedSteadyOutput> unpack_prepacked_outputs(const std::vector<at::Tensor>& out,
                                                            const std::vector<BatchedSteadyInput>& ready,
                                                            int bucket) {
    return unpack_outputs(out, ready, bucket);
  }

  int loaded_bucket_count() const {
    return static_cast<int>(loaders_.size());
  }

  const std::string& package_dir() const {
    return package_dir_;
  }

  const std::string& shared_weights_ts() const {
    return shared_weights_ts_;
  }

  bool sealed() const {
    return sealed_;
  }

  static int bucket_for_k_public(int k) {
    return bucket_for_k(k);
  }

 private:
  inline static constexpr std::array<int, 3> kBuckets = {1, 2, 4};

  static int bucket_for_k(int k) {
    if (k <= 0) throw std::runtime_error("batched steady bucket_for_k requires K>0");
    for (int bucket : kBuckets) {
      if (k <= bucket) return bucket;
    }
    throw std::runtime_error("batched steady K exceeds largest bucket: " + std::to_string(k));
  }

  std::string package_path(int bucket) const {
    return (bsteady_detail::fs::path(package_dir_) /
            ("enc_steady_aoti_b" + std::to_string(bucket) + ".pt2")).string();
  }

  void verify_manifest() const {
    const std::string manifest_path = (bsteady_detail::fs::path(package_dir_) / "MANIFEST.json").string();
    if (!bsteady_detail::file_exists(manifest_path)) {
      throw std::runtime_error("batched steady MANIFEST.json is required: " + manifest_path);
    }
    auto buckets = bsteady_detail::load_manifest_buckets(manifest_path);
    std::set<int> seen;
    int64_t shared_size = bsteady_detail::file_size_bytes(shared_weights_ts_);
    std::string shared_sha = bsteady_detail::sha256_file(shared_weights_ts_);
    size_t ep_verified = 0;
    for (const auto& entry : buckets) {
      if (!seen.emplace(entry.B).second) {
        throw std::runtime_error("batched steady manifest duplicate B=" + std::to_string(entry.B));
      }
      std::string expected = "enc_steady_aoti_b" + std::to_string(entry.B) + ".pt2";
      if (entry.package != expected) {
        throw std::runtime_error("batched steady manifest package mismatch for B=" + std::to_string(entry.B) +
                                 ": got " + entry.package + " expected " + expected);
      }
      std::string expected_ep = "enc_steady_t2a_b" + std::to_string(entry.B) + ".pt2";
      std::string path = (bsteady_detail::fs::path(package_dir_) / entry.package).string();
      if (!bsteady_detail::file_exists(path)) throw std::runtime_error("batched steady package missing: " + path);
      if (entry.package_bytes > 0 && bsteady_detail::file_size_bytes(path) != entry.package_bytes) {
        throw std::runtime_error("batched steady package size mismatch for " + entry.package);
      }
      if (entry.shared_weight_bytes > 0 && shared_size != entry.shared_weight_bytes) {
        throw std::runtime_error("batched steady shared weight size mismatch for B=" +
                                 std::to_string(entry.B));
      }
      std::string package_sha = bsteady_detail::sha256_file(path);
      if (package_sha != entry.package_sha256) {
        throw std::runtime_error("batched steady package sha256 mismatch for " + entry.package +
                                 ": manifest=" + entry.package_sha256 + " actual=" + package_sha);
      }
      if (shared_sha != entry.shared_weight_sha256) {
        throw std::runtime_error("batched steady shared weight sha256 mismatch for B=" +
                                 std::to_string(entry.B) + ": manifest=" +
                                 entry.shared_weight_sha256 + " actual=" + shared_sha);
      }
      std::string ep_path = (bsteady_detail::fs::path(package_dir_) / expected_ep).string();
      if (!bsteady_detail::file_exists(ep_path)) {
        std::printf("density batched steady EP sha skipped: B=%d path=%s reason=missing\n",
                    entry.B,
                    ep_path.c_str());
      } else {
        if (entry.ep_sha256.empty()) {
          throw std::runtime_error("batched steady manifest missing ep_sha256 for B=" + std::to_string(entry.B));
        }
        if (entry.ep_bytes > 0 && bsteady_detail::file_size_bytes(ep_path) != entry.ep_bytes) {
          throw std::runtime_error("batched steady EP size mismatch for " + expected_ep);
        }
        std::string ep_sha = bsteady_detail::sha256_file(ep_path);
        if (ep_sha != entry.ep_sha256) {
          throw std::runtime_error("batched steady EP sha256 mismatch for " + expected_ep +
                                   ": manifest=" + entry.ep_sha256 + " actual=" + ep_sha);
        }
        ++ep_verified;
      }
    }
    for (int bucket : kBuckets) {
      if (seen.find(bucket) == seen.end()) {
        throw std::runtime_error("batched steady manifest missing B=" + std::to_string(bucket));
      }
    }
    std::printf("density batched steady manifest verified: buckets=%zu ep_verified=%zu shared_weight_sha256=%s\n",
                buckets.size(),
                ep_verified,
                buckets.empty() ? "" : buckets.front().shared_weight_sha256.c_str());
    std::fflush(stdout);
  }

  AOTIModelPackageLoader& load_bucket(int bucket) {
    auto existing = loaders_.find(bucket);
    if (existing != loaders_.end()) return *existing->second;
    auto path = package_path(bucket);
    if (!bsteady_detail::file_exists(path)) throw std::runtime_error("missing batched steady package: " + path);
    std::printf("  density batched steady bucket load begin B=%d package=%s policy=%s\n",
                bucket,
                path.c_str(),
                policy_.c_str());
    std::fflush(stdout);
    auto loader = std::make_unique<AOTIModelPackageLoader>(
        path, "model", /*run_single_threaded=*/false, num_runners_, device_.index());
    auto bucket_constants = bsteady_detail::constants_for_bucket(shared_constants_, *loader, path);
    loader->load_constants(bucket_constants.values, false, false, true);
    std::printf("  density batched steady bucket loaded B=%d package=%s constants=%zu direct=%zu alias=%zu "
                "num_runners=%d policy=%s\n",
                bucket,
                path.c_str(),
                bucket_constants.values.size(),
                bucket_constants.direct_matches,
                bucket_constants.alias_fallbacks,
                num_runners_,
                policy_.c_str());
    std::fflush(stdout);
    auto inserted = loaders_.emplace(bucket, std::move(loader));
    return *inserted.first->second;
  }

  AOTIModelPackageLoader& get(int bucket) {
    if (!sealed_) {
      throw std::runtime_error("batched steady loader get() before preload_all() sealed the loader set");
    }
    auto existing = loaders_.find(bucket);
    if (existing == loaders_.end()) {
      throw std::runtime_error("batched steady loader requested unpreloaded bucket B=" + std::to_string(bucket));
    }
    return *existing->second;
  }

  static void verify_row_shapes(const BatchedSteadyInput& row, const BatchedSteadyInput& first) {
    if (row.chunk.sizes() != first.chunk.sizes()) throw std::runtime_error("batched steady chunk shape mismatch");
    if (row.cache_ch.sizes() != first.cache_ch.sizes()) throw std::runtime_error("batched steady cache_ch shape mismatch");
    if (row.cache_t.sizes() != first.cache_t.sizes()) throw std::runtime_error("batched steady cache_t shape mismatch");
    if (row.cache_ch_len.sizes() != first.cache_ch_len.sizes()) {
      throw std::runtime_error("batched steady cache_ch_len shape mismatch");
    }
    if (row.chunk.device() != first.chunk.device()) throw std::runtime_error("batched steady row device mismatch");
  }

  static std::vector<at::Tensor> pack_inputs(const std::vector<BatchedSteadyInput>& ready, int bucket) {
    const auto& first = ready.front();
    std::vector<at::Tensor> chunks;
    std::vector<at::Tensor> cache_ch;
    std::vector<at::Tensor> cache_t;
    std::vector<at::Tensor> cache_ch_len;
    chunks.reserve(static_cast<size_t>(bucket));
    cache_ch.reserve(static_cast<size_t>(bucket));
    cache_t.reserve(static_cast<size_t>(bucket));
    cache_ch_len.reserve(static_cast<size_t>(bucket));
    for (int row = 0; row < bucket; ++row) {
      const auto& src = ready[static_cast<size_t>(row < static_cast<int>(ready.size()) ? row : 0)];
      verify_row_shapes(src, first);
      chunks.push_back(src.chunk.contiguous());
      cache_ch.push_back(src.cache_ch.contiguous());
      cache_t.push_back(src.cache_t.contiguous());
      cache_ch_len.push_back(src.cache_ch_len.contiguous());
    }
    auto device = first.chunk.device();
    auto length = torch::full({bucket},
                              first.chunk.size(2),
                              torch::TensorOptions().dtype(torch::kLong).device(device));
    return {
        torch::cat(chunks, 0).contiguous(),
        length.contiguous(),
        torch::cat(cache_ch, 1).contiguous(),
        torch::cat(cache_t, 1).contiguous(),
        torch::cat(cache_ch_len, 0).contiguous(),
    };
  }

  static std::vector<BatchedSteadyOutput> unpack_outputs(const std::vector<at::Tensor>& out,
                                                         const std::vector<BatchedSteadyInput>& ready,
                                                         int bucket) {
    std::vector<BatchedSteadyOutput> rows;
    rows.reserve(ready.size());
    for (int64_t row = 0; row < static_cast<int64_t>(ready.size()); ++row) {
      BatchedSteadyOutput item;
      item.bucket = bucket;
      item.row = static_cast<int>(row);
      item.label = ready[static_cast<size_t>(row)].label;
      item.tensors = {
          out[0].select(0, row).unsqueeze(0).contiguous(),
          out[1].select(0, row).reshape({1}).contiguous(),
          out[2].select(1, row).unsqueeze(1).contiguous(),
          out[3].select(1, row).unsqueeze(1).contiguous(),
          out[4].select(0, row).reshape({1}).contiguous(),
      };
      rows.push_back(std::move(item));
    }
    return rows;
  }

  std::string package_dir_;
  std::string shared_weights_ts_;
  torch::Device device_;
  int num_runners_ = 1;
  std::string policy_;
  bool sealed_ = false;
  std::unordered_map<std::string, at::Tensor> shared_constants_;
  std::map<int, std::unique_ptr<AOTIModelPackageLoader>> loaders_;
};
