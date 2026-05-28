#pragma once

#include <torch/script.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <utility>
#include <vector>

using torch::inductor::AOTIModelPackageLoader;

namespace bsteady_detail {
namespace fs = std::filesystem;

struct Sha256Ctx {
  std::array<uint8_t, 64> data{};
  uint32_t datalen = 0;
  uint64_t bitlen = 0;
  std::array<uint32_t, 8> state{
      0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
      0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
};

struct BucketConstants {
  std::unordered_map<std::string, at::Tensor> values;
  size_t direct_matches = 0;
  size_t alias_fallbacks = 0;
};

struct ManifestBucket {
  int B = 0;
  std::string package;
  std::string package_sha256;
  std::string shared_weight_sha256;
};

static inline bool file_exists(const std::string& path) {
  struct stat st;
  return stat(path.c_str(), &st) == 0;
}

static inline bool directory_exists(const std::string& path) {
  struct stat st;
  return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

static inline uint32_t rotr(uint32_t x, uint32_t n) {
  return (x >> n) | (x << (32U - n));
}

static inline void sha256_transform(Sha256Ctx& ctx, const uint8_t data[64]) {
  static constexpr std::array<uint32_t, 64> k{
      0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
      0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
      0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
      0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
      0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
      0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
      0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
      0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};

  std::array<uint32_t, 64> m{};
  for (uint32_t i = 0, j = 0; i < 16; ++i, j += 4) {
    m[i] = (static_cast<uint32_t>(data[j]) << 24) |
           (static_cast<uint32_t>(data[j + 1]) << 16) |
           (static_cast<uint32_t>(data[j + 2]) << 8) |
           (static_cast<uint32_t>(data[j + 3]));
  }
  for (uint32_t i = 16; i < 64; ++i) {
    uint32_t s0 = rotr(m[i - 15], 7) ^ rotr(m[i - 15], 18) ^ (m[i - 15] >> 3);
    uint32_t s1 = rotr(m[i - 2], 17) ^ rotr(m[i - 2], 19) ^ (m[i - 2] >> 10);
    m[i] = m[i - 16] + s0 + m[i - 7] + s1;
  }

  uint32_t a = ctx.state[0], b = ctx.state[1], c = ctx.state[2], d = ctx.state[3];
  uint32_t e = ctx.state[4], f = ctx.state[5], g = ctx.state[6], h = ctx.state[7];
  for (uint32_t i = 0; i < 64; ++i) {
    uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
    uint32_t ch = (e & f) ^ ((~e) & g);
    uint32_t temp1 = h + s1 + ch + k[i] + m[i];
    uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
    uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
    uint32_t temp2 = s0 + maj;
    h = g;
    g = f;
    f = e;
    e = d + temp1;
    d = c;
    c = b;
    b = a;
    a = temp1 + temp2;
  }
  ctx.state[0] += a; ctx.state[1] += b; ctx.state[2] += c; ctx.state[3] += d;
  ctx.state[4] += e; ctx.state[5] += f; ctx.state[6] += g; ctx.state[7] += h;
}

static inline void sha256_update(Sha256Ctx& ctx, const uint8_t* data, size_t len) {
  for (size_t i = 0; i < len; ++i) {
    ctx.data[ctx.datalen++] = data[i];
    if (ctx.datalen == 64) {
      sha256_transform(ctx, ctx.data.data());
      ctx.bitlen += 512;
      ctx.datalen = 0;
    }
  }
}

static inline std::string sha256_final(Sha256Ctx& ctx) {
  uint32_t i = ctx.datalen;
  uint64_t total_bits = ctx.bitlen + static_cast<uint64_t>(ctx.datalen) * 8U;

  ctx.data[i++] = 0x80U;
  if (i > 56) {
    while (i < 64) ctx.data[i++] = 0;
    sha256_transform(ctx, ctx.data.data());
    i = 0;
  }
  while (i < 56) ctx.data[i++] = 0;
  for (int shift = 56; shift >= 0; shift -= 8) {
    ctx.data[i++] = static_cast<uint8_t>((total_bits >> shift) & 0xffU);
  }
  sha256_transform(ctx, ctx.data.data());

  std::ostringstream oss;
  oss << std::hex << std::setfill('0');
  for (uint32_t word : ctx.state) oss << std::setw(8) << word;
  return oss.str();
}

static inline std::string sha256_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open for sha256: " + path);
  Sha256Ctx ctx;
  std::array<char, 1024 * 1024> buffer{};
  while (f) {
    f.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    std::streamsize got = f.gcount();
    if (got > 0) {
      sha256_update(ctx, reinterpret_cast<const uint8_t*>(buffer.data()), static_cast<size_t>(got));
    }
  }
  return sha256_final(ctx);
}

static inline std::string read_text_file(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open manifest: " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

static inline size_t skip_ws(const std::string& s, size_t pos) {
  while (pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos]))) ++pos;
  return pos;
}

static inline size_t find_matching_json_delim(const std::string& s, size_t open_pos) {
  char open = s.at(open_pos);
  char close = open == '{' ? '}' : ']';
  int depth = 0;
  bool in_string = false;
  bool escape = false;
  for (size_t i = open_pos; i < s.size(); ++i) {
    char ch = s[i];
    if (in_string) {
      if (escape) {
        escape = false;
      } else if (ch == '\\') {
        escape = true;
      } else if (ch == '"') {
        in_string = false;
      }
      continue;
    }
    if (ch == '"') {
      in_string = true;
    } else if (ch == open) {
      ++depth;
    } else if (ch == close) {
      --depth;
      if (depth == 0) return i;
    }
  }
  throw std::runtime_error("unterminated JSON object/array in manifest");
}

static inline std::string json_value_for_key(const std::string& object,
                                             const std::string& key,
                                             bool required = true) {
  std::string needle = "\"" + key + "\"";
  size_t key_pos = object.find(needle);
  if (key_pos == std::string::npos) {
    if (required) throw std::runtime_error("manifest missing key: " + key);
    return "";
  }
  size_t colon = object.find(':', key_pos + needle.size());
  if (colon == std::string::npos) throw std::runtime_error("manifest key has no colon: " + key);
  size_t start = skip_ws(object, colon + 1);
  if (start >= object.size()) throw std::runtime_error("manifest key has no value: " + key);

  size_t end = start;
  if (object[start] == '{' || object[start] == '[') {
    end = find_matching_json_delim(object, start) + 1;
  } else if (object[start] == '"') {
    bool escape = false;
    for (end = start + 1; end < object.size(); ++end) {
      char ch = object[end];
      if (escape) {
        escape = false;
      } else if (ch == '\\') {
        escape = true;
      } else if (ch == '"') {
        ++end;
        break;
      }
    }
  } else {
    while (end < object.size() && object[end] != ',' && object[end] != '}' && object[end] != ']') ++end;
  }
  return object.substr(start, end - start);
}

static inline std::string json_string_field(const std::string& object,
                                            const std::string& key,
                                            bool required = true) {
  std::string value = json_value_for_key(object, key, required);
  if (value.empty() && !required) return "";
  value = value.substr(skip_ws(value, 0));
  if (value.size() < 2 || value.front() != '"' || value.back() != '"') {
    throw std::runtime_error("manifest key is not a string: " + key);
  }
  return value.substr(1, value.size() - 2);
}

static inline int64_t json_int_field(const std::string& object, const std::string& key) {
  std::string value = json_value_for_key(object, key);
  size_t n = 0;
  long long out = std::stoll(value, &n);
  n = skip_ws(value, n);
  if (n != value.size()) throw std::runtime_error("manifest key is not an integer: " + key);
  return out;
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
    b.shared_weight_sha256 = json_string_field(obj, "shared_weight_sha256");
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
    std::string shared_sha = bsteady_detail::sha256_file(shared_weights_ts_);
    for (const auto& entry : buckets) {
      if (!seen.emplace(entry.B).second) {
        throw std::runtime_error("batched steady manifest duplicate B=" + std::to_string(entry.B));
      }
      std::string expected = "enc_steady_aoti_b" + std::to_string(entry.B) + ".pt2";
      if (entry.package != expected) {
        throw std::runtime_error("batched steady manifest package mismatch for B=" + std::to_string(entry.B) +
                                 ": got " + entry.package + " expected " + expected);
      }
      std::string path = (bsteady_detail::fs::path(package_dir_) / entry.package).string();
      if (!bsteady_detail::file_exists(path)) throw std::runtime_error("batched steady package missing: " + path);
      std::string actual = bsteady_detail::sha256_file(path);
      if (actual != entry.package_sha256) {
        throw std::runtime_error("batched steady package sha256 mismatch for " + entry.package +
                                 ": manifest=" + entry.package_sha256 + " actual=" + actual);
      }
      if (entry.shared_weight_sha256 != shared_sha) {
        throw std::runtime_error("batched steady shared weight sha256 mismatch for B=" +
                                 std::to_string(entry.B) + ": manifest=" + entry.shared_weight_sha256 +
                                 " actual=" + shared_sha);
      }
    }
    for (int bucket : kBuckets) {
      if (seen.find(bucket) == seen.end()) {
        throw std::runtime_error("batched steady manifest missing B=" + std::to_string(bucket));
      }
    }
    std::printf("density batched steady manifest verified: buckets=%zu shared_weight_sha256=%s\n",
                buckets.size(),
                shared_sha.c_str());
  }

  AOTIModelPackageLoader& load_bucket(int bucket) {
    auto existing = loaders_.find(bucket);
    if (existing != loaders_.end()) return *existing->second;
    auto path = package_path(bucket);
    if (!bsteady_detail::file_exists(path)) throw std::runtime_error("missing batched steady package: " + path);
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
