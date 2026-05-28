#pragma once

#include <array>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

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
    if (!directory_exists(package_dir_)) {
      throw std::runtime_error("batched steady package directory missing: " + package_dir_);
    }
    if (!file_exists(shared_weights_ts_)) {
      throw std::runtime_error("batched steady shared weights missing: " + shared_weights_ts_);
    }
    shared_constants_ = load_shared_constants(shared_weights_ts_, device_);
    std::printf("density loaded batched steady shared constants: %zu entries policy=%s\n",
                shared_constants_.size(),
                policy_.c_str());
  }

  void preload_all() {
    for (int bucket : kBuckets) {
      (void)get(bucket);
    }
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
    auto& loader = get(bucket);
    auto inputs = pack_inputs(ready, bucket);
    auto out = loader.run(inputs, reinterpret_cast<void*>(stream.stream()));
    if (out.size() < 5) throw std::runtime_error("batched steady AOTI returned fewer than 5 outputs");
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
    return (fs::path(package_dir_) / ("enc_steady_aoti_b" + std::to_string(bucket) + ".pt2")).string();
  }

  AOTIModelPackageLoader& get(int bucket) {
    auto existing = loaders_.find(bucket);
    if (existing != loaders_.end()) return *existing->second;
    auto path = package_path(bucket);
    if (!file_exists(path)) throw std::runtime_error("missing batched steady package: " + path);
    auto loader = std::make_unique<AOTIModelPackageLoader>(
        path, "model", /*run_single_threaded=*/false, num_runners_, device_.index());
    auto bucket_constants = constants_for_bucket(shared_constants_, *loader, path);
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
    auto inserted = loaders_.emplace(bucket, std::move(loader));
    return *inserted.first->second;
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
      // Padding duplicates row 0's valid tensors. Pad rows are never unpacked,
      // and using a real row avoids introducing out-of-distribution cache values.
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
  std::unordered_map<std::string, at::Tensor> shared_constants_;
  std::map<int, std::unique_ptr<AOTIModelPackageLoader>> loaders_;
};
