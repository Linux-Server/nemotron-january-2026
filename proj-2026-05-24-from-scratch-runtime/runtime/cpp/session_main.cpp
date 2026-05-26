// 1.4 Phase-1 EXIT GATE: single-stream MEL-level session composition.
//
// Replays artifacts/session_bundle.ts, runs the production C++ steady path
// (first chunk TorchScript, steady chunks AOTIModelPackageLoader), then forks
// the session state for the AOTI finalize bucket selected by (drop_extra, T).
// The emitted cumulative tokens must exactly equal finalize_ref gold tokens.
// The ordered interim/final/suppressed event stream is checked at the same
// WORD/TEXT level as finalize_ref._continuous_append_only_delta.
#include <torch/script.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <cstdint>
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
namespace fs = std::filesystem;

static constexpr int BLANK = 1024;
static constexpr int MAX_SYMBOLS = 10;
static constexpr int SHIFT = 16;
static constexpr int PRE = 9;
static constexpr int DROP = 2;
static constexpr int RIGHT_CONTEXT = 1;
static constexpr int FINAL_PADDING_FRAMES = 32;
static constexpr int ATT_CONTEXT_LEFT = 70;
static constexpr int ATT_CONTEXT_RIGHT = 1;
static constexpr const char* MODEL_ID = "nvidia/nemotron-speech-streaming-en-0.6b";

enum class SessionMode { STREAMING, PENDING_FINALIZE, FINALIZED };

static constexpr int64_t EVENT_INTERIM = 0;
static constexpr int64_t EVENT_FINAL = 1;
static constexpr int64_t EVENT_SUPPRESSED = 2;

struct EmittedEvent {
  int64_t kind = -1;
  std::vector<int64_t> tokens;
  std::vector<int64_t> collector_tokens;
  std::string text;
  std::string collector_text;
};

struct Tokenizer {
  std::vector<std::string> pieces;

  std::string ids_to_text(const std::vector<int64_t>& ids) const;
};

struct SessionState {
  torch::Tensor clc;
  torch::Tensor clt;
  torch::Tensor clcl;
  torch::Tensor g;
  torch::Tensor h;
  torch::Tensor c;
  torch::Tensor ring;
  int64_t emitted = 0;
  std::vector<int64_t> hyp;
  std::vector<int64_t> last_interim_tokens;
  std::vector<int64_t> continuous_emitted_tokens;
  std::string last_interim_text;
  std::string continuous_emitted_text;
  SessionMode mode = SessionMode::STREAMING;
};

struct AsrSnapshot {
  torch::Tensor clc;
  torch::Tensor clt;
  torch::Tensor clcl;
  torch::Tensor g;
  torch::Tensor h;
  torch::Tensor c;
  torch::Tensor ring;
  int64_t emitted = 0;
  std::vector<int64_t> hyp;
};

struct ManifestContract {
  std::string model_id;
  std::vector<int64_t> att_context;
  int64_t right_context = -1;
  int64_t shift = -1;
  int64_t pre_encode_cache = -1;
  int64_t drop_extra = -1;
  int64_t final_padding_frames = -1;
  int64_t blank = -1;
  int64_t max_symbols = -1;
  std::string weights_sha256;
};

struct ManifestBucket {
  int64_t drop = -1;
  int64_t T = -1;
  std::string pkg;
  std::string pkg_sha256;
};

struct BucketManifest {
  ManifestContract contract;
  std::vector<ManifestBucket> buckets;
};

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

static bool file_exists(const std::string& path) {
  struct stat st;
  return stat(path.c_str(), &st) == 0;
}

static bool directory_exists(const std::string& path) {
  struct stat st;
  return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

static std::string utt_attr(int utt, const char* name) {
  return "utt" + std::to_string(utt) + "_" + std::string(name);
}

static std::string utt_chunk_attr(int utt, int chunk, const char* name) {
  return "utt" + std::to_string(utt) + "_chunk" + std::to_string(chunk) + "_" + std::string(name);
}

static torch::Tensor attr_tensor(torch::jit::Module& module, const std::string& name) {
  return module.attr(name).toTensor();
}

static torch::Tensor utt_tensor(torch::jit::Module& bundle, int utt, const char* name) {
  return attr_tensor(bundle, utt_attr(utt, name));
}

static torch::Tensor utt_chunk_tensor(torch::jit::Module& bundle, int utt, int chunk, const char* name) {
  return attr_tensor(bundle, utt_chunk_attr(utt, chunk, name));
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

static std::string escaped_text(const std::string& text) {
  std::ostringstream oss;
  oss << '"';
  for (unsigned char ch : text) {
    if (ch == '\\') {
      oss << "\\\\";
    } else if (ch == '"') {
      oss << "\\\"";
    } else if (ch == '\n') {
      oss << "\\n";
    } else if (ch == '\r') {
      oss << "\\r";
    } else if (ch == '\t') {
      oss << "\\t";
    } else if (ch < 0x20 || ch == 0x7f) {
      oss << "\\x" << std::hex << std::setw(2) << std::setfill('0')
          << static_cast<int>(ch) << std::dec << std::setfill(' ');
    } else {
      oss << static_cast<char>(ch);
    }
  }
  oss << '"';
  return oss.str();
}

static const char* event_kind_name(int64_t kind) {
  if (kind == EVENT_INTERIM) return "interim";
  if (kind == EVENT_FINAL) return "final";
  if (kind == EVENT_SUPPRESSED) return "suppressed";
  return "unknown";
}

static std::vector<int64_t> append_only_delta_tokens(const std::vector<int64_t>& final_tokens,
                                                     const std::vector<int64_t>& emitted_tokens) {
  // Token-id port of finalize_ref._continuous_append_only_delta:
  // common-prefix append, deletion/correction suppression, then suffix/prefix overlap trim.
  size_t common = 0;
  size_t pair_count = std::min(emitted_tokens.size(), final_tokens.size());
  while (common < pair_count && emitted_tokens[common] == final_tokens[common]) {
    ++common;
  }

  std::vector<int64_t> delta_tokens;
  if (common == emitted_tokens.size()) {
    delta_tokens.assign(final_tokens.begin() + static_cast<std::ptrdiff_t>(common),
                        final_tokens.end());
  } else if (final_tokens.size() <= emitted_tokens.size()) {
    delta_tokens.clear();
  } else {
    delta_tokens.assign(final_tokens.begin() + static_cast<std::ptrdiff_t>(emitted_tokens.size()),
                        final_tokens.end());
    size_t max_overlap = std::min(emitted_tokens.size(), delta_tokens.size());
    for (size_t overlap = max_overlap; overlap > 0; --overlap) {
      bool matches = true;
      size_t emitted_start = emitted_tokens.size() - overlap;
      for (size_t i = 0; i < overlap; ++i) {
        if (emitted_tokens[emitted_start + i] != delta_tokens[i]) {
          matches = false;
          break;
        }
      }
      if (matches) {
        delta_tokens.erase(delta_tokens.begin(),
                           delta_tokens.begin() + static_cast<std::ptrdiff_t>(overlap));
        break;
      }
    }
  }
  return delta_tokens;
}

static std::vector<std::string> split_words(const std::string& text) {
  std::istringstream iss(text);
  std::vector<std::string> words;
  std::string word;
  while (iss >> word) words.push_back(word);
  return words;
}

static std::string join_words(const std::vector<std::string>& words) {
  std::ostringstream oss;
  for (size_t i = 0; i < words.size(); ++i) {
    if (i > 0) oss << ' ';
    oss << words[i];
  }
  return oss.str();
}

static std::string append_only_delta_text(const std::string& final_text,
                                          const std::string& emitted_text) {
  auto final_words = split_words(final_text);
  auto emitted_words = split_words(emitted_text);

  size_t common = 0;
  size_t pair_count = std::min(emitted_words.size(), final_words.size());
  while (common < pair_count && emitted_words[common] == final_words[common]) {
    ++common;
  }

  std::vector<std::string> delta_words;
  if (common == emitted_words.size()) {
    delta_words.assign(final_words.begin() + static_cast<std::ptrdiff_t>(common),
                       final_words.end());
  } else if (final_words.size() <= emitted_words.size()) {
    delta_words.clear();
  } else {
    delta_words.assign(final_words.begin() + static_cast<std::ptrdiff_t>(emitted_words.size()),
                       final_words.end());
    size_t max_overlap = std::min(emitted_words.size(), delta_words.size());
    for (size_t overlap = max_overlap; overlap > 0; --overlap) {
      bool matches = true;
      size_t emitted_start = emitted_words.size() - overlap;
      for (size_t i = 0; i < overlap; ++i) {
        if (emitted_words[emitted_start + i] != delta_words[i]) {
          matches = false;
          break;
        }
      }
      if (matches) {
        delta_words.erase(delta_words.begin(),
                          delta_words.begin() + static_cast<std::ptrdiff_t>(overlap));
        break;
      }
    }
  }
  return join_words(delta_words);
}

static std::string append_delta_to_collector(const std::string& collector,
                                             const std::string& delta) {
  if (delta.empty()) return collector;
  if (collector.empty()) return delta;
  return collector + " " + delta;
}

static void replace_all(std::string& text, const std::string& needle, const std::string& repl) {
  if (needle.empty()) return;
  size_t pos = 0;
  while ((pos = text.find(needle, pos)) != std::string::npos) {
    text.replace(pos, needle.size(), repl);
    pos += repl.size();
  }
}

std::string Tokenizer::ids_to_text(const std::vector<int64_t>& ids) const {
  if (ids.empty()) return "";
  std::string text;
  bool strip_dummy_prefix = false;
  static const std::string marker = "\xE2\x96\x81";
  static const std::string unk_surface = " \xE2\x81\x87 ";

  for (size_t i = 0; i < ids.size(); ++i) {
    int64_t id = ids[i];
    if (id < 0 || id >= static_cast<int64_t>(pieces.size())) {
      throw std::runtime_error("token id out of tokenizer piece range: " + std::to_string(id));
    }
    const std::string& piece = pieces[static_cast<size_t>(id)];
    if (i == 0 && piece.rfind(marker, 0) == 0) strip_dummy_prefix = true;
    if (piece == "<unk>") {
      text += unk_surface;
    } else {
      text += piece;
    }
  }
  replace_all(text, marker, " ");
  if (strip_dummy_prefix && !text.empty() && text[0] == ' ') {
    text.erase(text.begin());
  }
  return text;
}

static std::vector<std::vector<int64_t>> unpack_i64_lists(torch::Tensor flat_tensor,
                                                          torch::Tensor offsets_tensor,
                                                          const char* label,
                                                          int utt) {
  auto flat = tensor_to_vec(flat_tensor);
  auto offsets = tensor_to_vec(offsets_tensor);
  if (offsets.empty()) {
    throw std::runtime_error(std::string(label) + " offsets empty for utt" + std::to_string(utt));
  }
  if (offsets.front() != 0 || offsets.back() != static_cast<int64_t>(flat.size())) {
    throw std::runtime_error(std::string(label) + " offsets do not cover flat payload for utt" + std::to_string(utt));
  }
  std::vector<std::vector<int64_t>> out;
  out.reserve(offsets.size() - 1);
  for (size_t i = 0; i + 1 < offsets.size(); ++i) {
    int64_t start = offsets[i];
    int64_t end = offsets[i + 1];
    if (start < 0 || end < start || end > static_cast<int64_t>(flat.size())) {
      throw std::runtime_error(std::string(label) + " invalid offsets for utt" + std::to_string(utt));
    }
    out.emplace_back(flat.begin() + start, flat.begin() + end);
  }
  return out;
}

static std::vector<uint8_t> tensor_to_u8_vec(torch::Tensor tensor) {
  auto flat = tensor.to(torch::kCPU).to(torch::kUInt8).contiguous().view({-1});
  std::vector<uint8_t> out;
  out.reserve(flat.numel());
  for (int64_t i = 0; i < flat.numel(); ++i) {
    out.push_back(flat[i].item<uint8_t>());
  }
  return out;
}

static std::vector<std::string> unpack_utf8_strings(torch::Tensor flat_tensor,
                                                    torch::Tensor offsets_tensor,
                                                    const char* label,
                                                    int utt) {
  auto flat = tensor_to_u8_vec(flat_tensor);
  auto offsets = tensor_to_vec(offsets_tensor);
  if (offsets.empty()) {
    throw std::runtime_error(std::string(label) + " offsets empty for utt" + std::to_string(utt));
  }
  if (offsets.front() != 0 || offsets.back() != static_cast<int64_t>(flat.size())) {
    throw std::runtime_error(std::string(label) + " offsets do not cover flat payload for utt" + std::to_string(utt));
  }
  std::vector<std::string> out;
  out.reserve(offsets.size() - 1);
  for (size_t i = 0; i + 1 < offsets.size(); ++i) {
    int64_t start = offsets[i];
    int64_t end = offsets[i + 1];
    if (start < 0 || end < start || end > static_cast<int64_t>(flat.size())) {
      throw std::runtime_error(std::string(label) + " invalid offsets for utt" + std::to_string(utt));
    }
    out.emplace_back(reinterpret_cast<const char*>(flat.data() + start),
                     static_cast<size_t>(end - start));
  }
  return out;
}

static Tokenizer tokenizer_from_bundle(torch::jit::Module& bundle) {
  Tokenizer tokenizer;
  tokenizer.pieces = unpack_utf8_strings(
      attr_tensor(bundle, "token_piece_bytes"),
      attr_tensor(bundle, "token_piece_offsets"),
      "token_piece",
      -1);
  if (tokenizer.pieces.empty()) throw std::runtime_error("tokenizer piece table is empty");
  return tokenizer;
}

static void verify_tokenizer_selftest(torch::jit::Module& bundle, const Tokenizer& tokenizer) {
  auto sequences = unpack_i64_lists(
      attr_tensor(bundle, "detok_selftest_tokens"),
      attr_tensor(bundle, "detok_selftest_token_offsets"),
      "detok_selftest_tokens",
      -1);
  auto texts = unpack_utf8_strings(
      attr_tensor(bundle, "detok_selftest_text_bytes"),
      attr_tensor(bundle, "detok_selftest_text_offsets"),
      "detok_selftest_text",
      -1);
  if (sequences.size() != texts.size()) {
    throw std::runtime_error("detok selftest sequence/text count mismatch");
  }
  for (size_t i = 0; i < sequences.size(); ++i) {
    std::string got = tokenizer.ids_to_text(sequences[i]);
    if (got != texts[i]) {
      std::ostringstream oss;
      oss << "detok selftest failed at sequence " << i
          << " tokens=" << vec_to_string(sequences[i])
          << " got=" << escaped_text(got)
          << " gold=" << escaped_text(texts[i]);
      throw std::runtime_error(oss.str());
    }
  }
  std::printf("tokenizer detok selftest PASS: pieces=%zu sequences=%zu\n",
              tokenizer.pieces.size(), sequences.size());
}

static std::vector<EmittedEvent> gold_events_from_bundle(torch::jit::Module& bundle, int utt) {
  auto kinds = tensor_to_vec(utt_tensor(bundle, utt, "event_kinds"));
  auto tokens = unpack_i64_lists(
      utt_tensor(bundle, utt, "event_tokens"),
      utt_tensor(bundle, utt, "event_token_offsets"),
      "event_tokens",
      utt);
  auto collectors = unpack_i64_lists(
      utt_tensor(bundle, utt, "event_collector_tokens"),
      utt_tensor(bundle, utt, "event_collector_token_offsets"),
      "event_collector_tokens",
      utt);
  auto texts = unpack_utf8_strings(
      utt_tensor(bundle, utt, "event_text_bytes"),
      utt_tensor(bundle, utt, "event_text_offsets"),
      "event_text",
      utt);
  auto collector_texts = unpack_utf8_strings(
      utt_tensor(bundle, utt, "event_collector_text_bytes"),
      utt_tensor(bundle, utt, "event_collector_text_offsets"),
      "event_collector_text",
      utt);
  if (tokens.size() != kinds.size() || collectors.size() != kinds.size() ||
      texts.size() != kinds.size() || collector_texts.size() != kinds.size()) {
    throw std::runtime_error("event payload count mismatch for utt" + std::to_string(utt));
  }
  std::vector<EmittedEvent> events;
  events.reserve(kinds.size());
  for (size_t i = 0; i < kinds.size(); ++i) {
    events.push_back({kinds[i], tokens[i], collectors[i], texts[i], collector_texts[i]});
  }
  return events;
}

static void emit_event(std::vector<EmittedEvent>& events,
                       int64_t kind,
                       const std::vector<int64_t>& tokens,
                       const std::vector<int64_t>& collector_tokens,
                       const std::string& text,
                       const std::string& collector_text) {
  events.push_back({kind, tokens, collector_tokens, text, collector_text});
}

static bool equal_events(const std::vector<EmittedEvent>& got,
                         const std::vector<EmittedEvent>& gold,
                         int utt) {
  bool ok = got.size() == gold.size();
  if (!ok) {
    std::printf("    utt%d event count mismatch: got=%zu gold=%zu\n",
                utt, got.size(), gold.size());
  }
  size_t n = std::min(got.size(), gold.size());
  for (size_t i = 0; i < n; ++i) {
    bool event_ok = got[i].kind == gold[i].kind &&
                    got[i].text == gold[i].text &&
                    got[i].collector_text == gold[i].collector_text;
    if (!event_ok) {
      std::printf("    utt%d event[%zu] mismatch: got_kind=%s gold_kind=%s\n",
                  utt, i, event_kind_name(got[i].kind), event_kind_name(gold[i].kind));
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

static bool equal_tokens(const std::vector<int64_t>& got,
                         const std::vector<int64_t>& gold,
                         const char* label,
                         int utt) {
  bool ok = got == gold;
  if (!ok) {
    std::printf("    utt%d %s token mismatch: got_len=%zu gold_len=%zu\n",
                utt, label, got.size(), gold.size());
    std::printf("      got :%s\n", vec_to_string(got).c_str());
    std::printf("      gold:%s\n", vec_to_string(gold).c_str());
    size_t n = std::min(got.size(), gold.size());
    for (size_t i = 0; i < n; ++i) {
      if (got[i] != gold[i]) {
        std::printf("      first_diff=%zu got=%ld gold=%ld\n",
                    i, (long)got[i], (long)gold[i]);
        break;
      }
    }
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

static bool optional_tensor_equal(const char* name, const torch::Tensor& actual, const torch::Tensor& expected) {
  if (actual.defined() != expected.defined()) {
    std::printf("    FORK_ASSERT %s defined mismatch: %d/%d\n",
                name, (int)actual.defined(), (int)expected.defined());
    return false;
  }
  if (!actual.defined()) return true;
  return tensor_equal(name, actual, expected);
}

static SessionState clone_session(const SessionState& state) {
  SessionState out;
  out.clc = state.clc.clone();
  out.clt = state.clt.clone();
  out.clcl = state.clcl.clone();
  out.g = state.g.clone();
  out.h = state.h.clone();
  out.c = state.c.clone();
  out.ring = state.ring.defined() ? state.ring.clone() : torch::Tensor();
  out.emitted = state.emitted;
  out.hyp = state.hyp;
  out.last_interim_tokens = state.last_interim_tokens;
  out.continuous_emitted_tokens = state.continuous_emitted_tokens;
  out.last_interim_text = state.last_interim_text;
  out.continuous_emitted_text = state.continuous_emitted_text;
  out.mode = state.mode;
  return out;
}

static AsrSnapshot snapshot_asr(const SessionState& state) {
  return {
      state.clc.clone(),
      state.clt.clone(),
      state.clcl.clone(),
      state.g.clone(),
      state.h.clone(),
      state.c.clone(),
      state.ring.defined() ? state.ring.clone() : torch::Tensor(),
      state.emitted,
      state.hyp,
  };
}

static bool fork_assert_parent_unchanged(const SessionState& parent, const AsrSnapshot& snapshot) {
  bool ok = true;
  ok = tensor_equal("cache_last_channel", parent.clc, snapshot.clc) && ok;
  ok = tensor_equal("cache_last_time", parent.clt, snapshot.clt) && ok;
  ok = tensor_equal("cache_last_channel_len", parent.clcl, snapshot.clcl) && ok;
  ok = tensor_equal("pred_out", parent.g, snapshot.g) && ok;
  ok = tensor_equal("decoder_state.h", parent.h, snapshot.h) && ok;
  ok = tensor_equal("decoder_state.c", parent.c, snapshot.c) && ok;
  ok = optional_tensor_equal("mel_frame_ring", parent.ring, snapshot.ring) && ok;
  if (parent.emitted != snapshot.emitted) {
    std::printf("    FORK_ASSERT emitted_frames mismatch: %ld/%ld\n",
                (long)parent.emitted, (long)snapshot.emitted);
    ok = false;
  }
  if (parent.hyp != snapshot.hyp) {
    std::printf("    FORK_ASSERT hyp_tokens mismatch: parent=%zu snapshot=%zu\n",
                parent.hyp.size(), snapshot.hyp.size());
    ok = false;
  }
  return ok;
}

static void reset_session(SessionState& state, torch::jit::Module& bundle, torch::Device device) {
  state.clc = attr_tensor(bundle, "init_clc").to(device).clone();
  state.clt = attr_tensor(bundle, "init_clt").to(device).clone();
  state.clcl = attr_tensor(bundle, "init_clcl").to(device).clone();
  state.g = attr_tensor(bundle, "init_g").to(device).clone();
  state.h = attr_tensor(bundle, "init_h").to(device).clone();
  state.c = attr_tensor(bundle, "init_c").to(device).clone();
  state.ring = torch::Tensor();
  state.emitted = 0;
  state.hyp.clear();
  state.last_interim_tokens.clear();
  state.continuous_emitted_tokens.clear();
  state.last_interim_text.clear();
  state.continuous_emitted_text.clear();
  state.mode = SessionMode::STREAMING;
}

static uint32_t rotr(uint32_t x, uint32_t n) {
  return (x >> n) | (x << (32U - n));
}

static void sha256_transform(Sha256Ctx& ctx, const uint8_t data[64]) {
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

static void sha256_update(Sha256Ctx& ctx, const uint8_t* data, size_t len) {
  for (size_t i = 0; i < len; ++i) {
    ctx.data[ctx.datalen++] = data[i];
    if (ctx.datalen == 64) {
      sha256_transform(ctx, ctx.data.data());
      ctx.bitlen += 512;
      ctx.datalen = 0;
    }
  }
}

static std::string sha256_final(Sha256Ctx& ctx) {
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

static std::string sha256_file(const std::string& path) {
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

static std::string read_text_file(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open manifest: " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

static size_t skip_ws(const std::string& s, size_t pos) {
  while (pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos]))) ++pos;
  return pos;
}

static size_t find_matching_json_delim(const std::string& s, size_t open_pos) {
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

static std::string json_value_for_key(const std::string& object, const std::string& key) {
  std::string needle = "\"" + key + "\"";
  size_t key_pos = object.find(needle);
  if (key_pos == std::string::npos) throw std::runtime_error("manifest missing key: " + key);
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

static std::string json_string_field(const std::string& object, const std::string& key) {
  std::string value = json_value_for_key(object, key);
  value = value.substr(skip_ws(value, 0));
  if (value.size() < 2 || value.front() != '"' || value.back() != '"') {
    throw std::runtime_error("manifest key is not a string: " + key);
  }
  return value.substr(1, value.size() - 2);
}

static int64_t json_int_field(const std::string& object, const std::string& key) {
  std::string value = json_value_for_key(object, key);
  size_t n = 0;
  long long out = std::stoll(value, &n);
  n = skip_ws(value, n);
  if (n != value.size()) throw std::runtime_error("manifest key is not an integer: " + key);
  return out;
}

static std::vector<int64_t> json_int_array_field(const std::string& object, const std::string& key) {
  std::string value = json_value_for_key(object, key);
  if (value.empty() || value.front() != '[' || value.back() != ']') {
    throw std::runtime_error("manifest key is not an array: " + key);
  }
  std::vector<int64_t> out;
  std::regex num_re("-?\\d+");
  for (auto it = std::sregex_iterator(value.begin(), value.end(), num_re);
       it != std::sregex_iterator(); ++it) {
    out.push_back(std::stoll((*it)[0].str()));
  }
  return out;
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

static BucketManifest load_bucket_manifest(const std::string& manifest_path) {
  std::string text = read_text_file(manifest_path);
  std::string contract_obj = json_value_for_key(text, "CONTRACT");
  std::string buckets_arr = json_value_for_key(text, "buckets");
  if (contract_obj.empty() || contract_obj.front() != '{') throw std::runtime_error("manifest CONTRACT is not an object");
  if (buckets_arr.empty() || buckets_arr.front() != '[') throw std::runtime_error("manifest buckets is not an array");

  BucketManifest manifest;
  manifest.contract.model_id = json_string_field(contract_obj, "model_id");
  manifest.contract.att_context = json_int_array_field(contract_obj, "att_context");
  manifest.contract.right_context = json_int_field(contract_obj, "right_context");
  manifest.contract.shift = json_int_field(contract_obj, "shift");
  manifest.contract.pre_encode_cache = json_int_field(contract_obj, "pre_encode_cache");
  manifest.contract.drop_extra = json_int_field(contract_obj, "drop_extra");
  manifest.contract.final_padding_frames = json_int_field(contract_obj, "final_padding_frames");
  manifest.contract.blank = json_int_field(contract_obj, "blank");
  manifest.contract.max_symbols = json_int_field(contract_obj, "max_symbols");
  manifest.contract.weights_sha256 = json_string_field(contract_obj, "weights_sha256");

  size_t pos = 1;
  while (pos + 1 < buckets_arr.size()) {
    pos = skip_ws(buckets_arr, pos);
    if (pos >= buckets_arr.size() || buckets_arr[pos] == ']') break;
    if (buckets_arr[pos] == ',') {
      ++pos;
      continue;
    }
    if (buckets_arr[pos] != '{') throw std::runtime_error("manifest bucket entry is not an object");
    size_t end = find_matching_json_delim(buckets_arr, pos);
    std::string obj = buckets_arr.substr(pos, end - pos + 1);
    ManifestBucket b;
    b.drop = json_int_field(obj, "drop");
    b.T = json_int_field(obj, "T");
    b.pkg = json_string_field(obj, "pkg");
    b.pkg_sha256 = json_string_field(obj, "pkg_sha256");
    manifest.buckets.push_back(std::move(b));
    pos = end + 1;
  }
  return manifest;
}

static void require_contract_eq(const char* name, int64_t actual, int64_t expected) {
  if (actual != expected) {
    throw std::runtime_error(std::string("manifest CONTRACT mismatch for ") + name +
                             ": got " + std::to_string(actual) +
                             " expected " + std::to_string(expected));
  }
}

static void verify_bucket_manifest(const BucketManifest& manifest,
                                   const std::map<std::pair<int64_t, int64_t>, std::string>& discovered,
                                   const std::string& buckets_dir,
                                   const std::string& shared_weights_pt) {
  const auto& c = manifest.contract;
  if (c.model_id != MODEL_ID) {
    throw std::runtime_error("manifest CONTRACT model_id mismatch: " + c.model_id);
  }
  if (c.att_context.size() != 2 || c.att_context[0] != ATT_CONTEXT_LEFT || c.att_context[1] != ATT_CONTEXT_RIGHT) {
    throw std::runtime_error("manifest CONTRACT att_context mismatch");
  }
  require_contract_eq("right_context", c.right_context, RIGHT_CONTEXT);
  require_contract_eq("shift", c.shift, SHIFT);
  require_contract_eq("pre_encode_cache", c.pre_encode_cache, PRE);
  require_contract_eq("drop_extra", c.drop_extra, DROP);
  require_contract_eq("final_padding_frames", c.final_padding_frames, FINAL_PADDING_FRAMES);
  require_contract_eq("blank", c.blank, BLANK);
  require_contract_eq("max_symbols", c.max_symbols, MAX_SYMBOLS);

  if (!file_exists(shared_weights_pt)) {
    throw std::runtime_error("manifest requires shared weights .pt but file is missing: " + shared_weights_pt);
  }
  std::string weights_sha = sha256_file(shared_weights_pt);
  if (weights_sha != c.weights_sha256) {
    throw std::runtime_error("shared weights sha256 mismatch: manifest=" + c.weights_sha256 + " actual=" + weights_sha);
  }

  std::set<std::pair<int64_t, int64_t>> manifest_keys;
  std::set<std::string> manifest_pkgs;
  for (const auto& b : manifest.buckets) {
    if (!manifest_keys.emplace(b.drop, b.T).second) {
      throw std::runtime_error("duplicate manifest bucket key drop=" + std::to_string(b.drop) +
                               " T=" + std::to_string(b.T));
    }
    if (!manifest_pkgs.emplace(b.pkg).second) throw std::runtime_error("duplicate manifest pkg: " + b.pkg);

    int64_t parsed_drop = 0;
    int64_t parsed_T = 0;
    if (!parse_bucket_filename(b.pkg, parsed_drop, parsed_T) || parsed_drop != b.drop || parsed_T != b.T) {
      throw std::runtime_error("manifest pkg filename does not match drop/T: " + b.pkg);
    }

    auto found = discovered.find(std::make_pair(b.drop, b.T));
    if (found == discovered.end()) {
      throw std::runtime_error("manifest bucket missing from directory: " + b.pkg);
    }
    fs::path expected_path = fs::path(buckets_dir) / b.pkg;
    if (fs::path(found->second).filename() != expected_path.filename()) {
      throw std::runtime_error("manifest/discovered pkg name mismatch for " + b.pkg);
    }
    std::string actual_sha = sha256_file(expected_path.string());
    if (actual_sha != b.pkg_sha256) {
      throw std::runtime_error("bucket sha256 mismatch for " + b.pkg +
                               ": manifest=" + b.pkg_sha256 + " actual=" + actual_sha);
    }
  }

  for (const auto& kv : discovered) {
    if (manifest_keys.find(kv.first) == manifest_keys.end()) {
      throw std::runtime_error("bucket file is not listed in manifest: " + kv.second);
    }
  }
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
  return constants;
}

static const at::Tensor* resolve_shared_constant(const std::unordered_map<std::string, at::Tensor>& shared_constants,
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

static BucketConstants constants_for_bucket(
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

static std::map<std::pair<int64_t, int64_t>, std::unique_ptr<AOTIModelPackageLoader>>
load_finalize_bucket_loaders(const std::string& dir, torch::Device device) {
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
  std::printf("finalize manifest verified: %zu buckets, weights_sha256=%s\n",
              manifest.buckets.size(), manifest.contract.weights_sha256.c_str());

  auto shared_constants = load_shared_constants(shared_weights, device);
  std::printf("loaded finalize shared constants: %zu entries\n", shared_constants.size());

  std::map<std::pair<int64_t, int64_t>, std::unique_ptr<AOTIModelPackageLoader>> loaders;
  for (const auto& kv : bucket_paths) {
    int64_t drop = kv.first.first;
    int64_t T = kv.first.second;
    const std::string& pkg = kv.second;
    auto loader = std::make_unique<AOTIModelPackageLoader>(pkg, "model", false, 1, -1);
    auto bucket_constants = constants_for_bucket(shared_constants, *loader, pkg);
    loader->load_constants(bucket_constants.values, false, false, true);
    std::printf("  finalize bucket drop=%ld T=%ld constants=%zu direct=%zu alias=%zu\n",
                (long)drop, (long)T, bucket_constants.values.size(),
                bucket_constants.direct_matches, bucket_constants.alias_fallbacks);
    loaders.emplace(kv.first, std::move(loader));
  }
  return loaders;
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
    throw std::runtime_error("enc_len out of range for enc_out: " + std::to_string(enc_len));
  }
  auto f = enc_out.transpose(1, 2).contiguous();
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

static void apply_encoder_outputs(SessionState& state,
                                  const std::vector<at::Tensor>& out,
                                  torch::jit::Module& joint,
                                  torch::jit::Module& predict) {
  if (out.size() < 5) throw std::runtime_error("encoder returned fewer than 5 outputs");
  int64_t enc_len = scalar_i64(out[1]);
  state.clc = out[2];
  state.clt = out[3];
  state.clcl = out[4];
  decode_range(joint, predict, out[0], enc_len, state.g, state.h, state.c, state.hyp);
}

static std::vector<at::Tensor> run_first_encoder(torch::jit::Module& enc_first,
                                                 const torch::Tensor& chunk,
                                                 SessionState& state) {
  auto device = chunk.device();
  auto L = torch::full({1}, chunk.size(2), torch::dtype(torch::kLong).device(device));
  auto tuple = enc_first.forward({chunk.contiguous(), L.contiguous(),
                                  state.clc.contiguous(), state.clt.contiguous(),
                                  state.clcl.contiguous()}).toTuple();
  std::vector<at::Tensor> out;
  out.reserve(5);
  for (int i = 0; i < 5; ++i) out.push_back(tuple->elements()[i].toTensor());
  return out;
}

static std::vector<at::Tensor> run_steady_encoder(AOTIModelPackageLoader& loader,
                                                  const torch::Tensor& chunk,
                                                  SessionState& state) {
  auto device = chunk.device();
  auto L = torch::full({1}, chunk.size(2), torch::dtype(torch::kLong).device(device));
  std::vector<at::Tensor> inputs = {
      chunk.contiguous(),
      L.contiguous(),
      state.clc.contiguous(),
      state.clt.contiguous(),
      state.clcl.contiguous(),
  };
  auto out = loader.run(inputs);
  if (out.size() < 5) throw std::runtime_error("steady AOTI encoder returned fewer than 5 outputs");
  return out;
}

static void run_steady_chunk(SessionState& state,
                             torch::jit::Module& bundle,
                             int utt,
                             int chunk_index,
                             torch::jit::Module& enc_first,
                             AOTIModelPackageLoader& enc_steady,
                             torch::jit::Module& joint,
                             torch::jit::Module& predict,
                             torch::Device device,
                             const Tokenizer& tokenizer,
                             std::vector<EmittedEvent>& events) {
  if (state.mode != SessionMode::STREAMING) throw std::runtime_error("steady chunk outside STREAMING");

  auto new_mel = utt_chunk_tensor(bundle, utt, chunk_index, "new_mel").to(device).contiguous();
  int64_t is_first = scalar_i64(utt_chunk_tensor(bundle, utt, chunk_index, "is_first"));
  int64_t drop_extra = scalar_i64(utt_chunk_tensor(bundle, utt, chunk_index, "drop_extra"));
  int64_t chunk_T = scalar_i64(utt_chunk_tensor(bundle, utt, chunk_index, "chunk_T"));
  int64_t emitted_before = scalar_i64(utt_chunk_tensor(bundle, utt, chunk_index, "emitted_before"));

  bool expected_first = state.emitted == 0;
  if ((is_first != 0) != expected_first) throw std::runtime_error("steady first/continuation flag mismatch");
  if (emitted_before != state.emitted) throw std::runtime_error("steady emitted_before mismatch");
  if (new_mel.size(2) != SHIFT) throw std::runtime_error("steady new_mel is not SHIFT frames");

  torch::Tensor chunk;
  std::vector<at::Tensor> out;
  if (expected_first) {
    if (drop_extra != 0 || chunk_T != new_mel.size(2)) throw std::runtime_error("first steady geometry mismatch");
    chunk = new_mel;
    out = run_first_encoder(enc_first, chunk, state);
  } else {
    if (!state.ring.defined()) throw std::runtime_error("steady continuation missing mel ring");
    if (drop_extra != DROP || chunk_T != state.ring.size(2) + new_mel.size(2)) {
      throw std::runtime_error("steady continuation geometry mismatch");
    }
    chunk = torch::cat({state.ring, new_mel}, 2).contiguous();
    out = run_steady_encoder(enc_steady, chunk, state);
  }

  apply_encoder_outputs(state, out, joint, predict);

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
}

struct FinalizeOutcome {
  bool token_ok = false;
  bool fork_ok = false;
  size_t emitted_tokens = 0;
};

static FinalizeOutcome run_finalize(SessionState& parent,
                                    torch::jit::Module& bundle,
                                    int utt,
                                    std::map<std::pair<int64_t, int64_t>, std::unique_ptr<AOTIModelPackageLoader>>& finalize_loaders,
                                    torch::jit::Module& joint,
                                    torch::jit::Module& predict,
                                    torch::Device device,
                                    const Tokenizer& tokenizer,
                                    std::vector<EmittedEvent>& events) {
  if (parent.mode != SessionMode::PENDING_FINALIZE) throw std::runtime_error("finalize outside PENDING_FINALIZE");
  auto snapshot = snapshot_asr(parent);
  parent.mode = SessionMode::FINALIZED;
  auto fork = clone_session(parent);

  int64_t drop_extra = scalar_i64(utt_tensor(bundle, utt, "final_drop_extra"));
  int64_t final_T = scalar_i64(utt_tensor(bundle, utt, "final_T"));
  auto gold = tensor_to_vec(utt_tensor(bundle, utt, "gold_tokens"));

  if (final_T > 0) {
    auto final_chunk = utt_tensor(bundle, utt, "final_chunk_mel").to(device).contiguous();
    if (final_chunk.size(2) != final_T) {
      throw std::runtime_error("final_chunk_mel T does not match bundle final_T");
    }
    int64_t expected_drop = parent.emitted == 0 ? 0 : DROP;
    if (drop_extra != expected_drop) throw std::runtime_error("finalize drop_extra does not match parent emitted state");

    auto loader_it = finalize_loaders.find(std::make_pair(drop_extra, final_T));
    if (loader_it == finalize_loaders.end()) {
      throw std::runtime_error("no finalize bucket for drop=" + std::to_string(drop_extra) +
                               " T=" + std::to_string(final_T));
    }

    std::vector<at::Tensor> inputs = {
        final_chunk.contiguous(),
        fork.clc.contiguous(),
        fork.clt.contiguous(),
        fork.clcl.contiguous(),
    };
    auto out = loader_it->second->run(inputs);
    if (out.size() < 2) throw std::runtime_error("finalize AOTI bucket returned fewer than 2 outputs");
    int64_t enc_len = scalar_i64(out[1]);
    if (out.size() >= 5) {
      fork.clc = out[2];
      fork.clt = out[3];
      fork.clcl = out[4];
    }
    decode_range(joint, predict, out[0], enc_len, fork.g, fork.h, fork.c, fork.hyp);
  }

  FinalizeOutcome outcome;
  outcome.emitted_tokens = fork.hyp.size();
  outcome.token_ok = equal_tokens(fork.hyp, gold, "final cumulative", utt);
  std::string final_text = tokenizer.ids_to_text(fork.hyp);
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
  parent.mode = SessionMode::STREAMING;
  return outcome;
}

static bool run_synthetic_word_delta_tests() {
  struct Case {
    const char* name;
    std::string emitted;
    std::string final_text;
    std::string expected_delta;
    std::string expected_collector;
    int64_t expected_kind;
  };
  std::vector<Case> cases = {
      {"New->Newark", "I live in New", "I live in Newark", "", "I live in New", EVENT_SUPPRESSED},
      {"play->playing", "we play", "we playing", "", "we play", EVENT_SUPPRESSED},
      {"duplicate final", "hello world", "hello world", "", "hello world", EVENT_SUPPRESSED},
      {"shorter final", "hello world today", "hello world", "", "hello world today", EVENT_SUPPRESSED},
      {"clean append", "hello world", "hello world today", "today", "hello world today", EVENT_FINAL},
      {"overlap trim", "alpha beta", "alpha gamma beta delta", "delta", "alpha beta delta", EVENT_FINAL},
  };

  bool all = true;
  for (const auto& c : cases) {
    std::string delta = append_only_delta_text(c.final_text, c.emitted);
    std::string collector = append_delta_to_collector(c.emitted, delta);
    int64_t kind = delta.empty() ? EVENT_SUPPRESSED : EVENT_FINAL;
    bool ok = delta == c.expected_delta &&
              collector == c.expected_collector &&
              kind == c.expected_kind;
    std::printf("SYNTHETIC %-16s %s delta=%s collector=%s\n",
                c.name,
                ok ? "PASS" : "FAIL",
                escaped_text(delta).c_str(),
                escaped_text(collector).c_str());
    all = ok && all;
  }
  std::printf("=== SYNTHETIC word-delta %s ===\n", all ? "PASS" : "FAIL");
  return all;
}

static void verify_session_bundle_meta(torch::jit::Module& bundle) {
  auto meta = attr_tensor(bundle, "meta").to(torch::kCPU).to(torch::kLong).contiguous();
  if (meta.numel() < 8) throw std::runtime_error("session bundle meta is too short");
  int64_t rows = meta[0].item<int64_t>();
  int64_t num_utts = scalar_i64(attr_tensor(bundle, "num_utts"));
  if (num_utts != rows) throw std::runtime_error("session bundle num_utts/meta mismatch");
  if (meta[1].item<int64_t>() != BLANK || meta[2].item<int64_t>() != MAX_SYMBOLS ||
      meta[3].item<int64_t>() != SHIFT || meta[4].item<int64_t>() != PRE ||
      meta[5].item<int64_t>() != DROP || meta[6].item<int64_t>() != FINAL_PADDING_FRAMES ||
      meta[7].item<int64_t>() != RIGHT_CONTEXT) {
    std::ostringstream oss;
    oss << "session bundle metadata mismatch: blank/max_symbols/shift/pre/drop/final_pad/right="
        << meta[1].item<int64_t>() << "/" << meta[2].item<int64_t>() << "/"
        << meta[3].item<int64_t>() << "/" << meta[4].item<int64_t>() << "/"
        << meta[5].item<int64_t>() << "/" << meta[6].item<int64_t>() << "/"
        << meta[7].item<int64_t>();
    throw std::runtime_error(oss.str());
  }
}

int main(int argc, char** argv) {
  std::string dir = "../artifacts";
  bool check_events = true;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--tokens-only" || arg == "--skip-events") {
      check_events = false;
    } else {
      dir = arg;
    }
  }
  try {
    torch::NoGradGuard ng;
    auto device = torch::Device(torch::kCUDA);

    auto bundle = torch::jit::load(dir + "/session_bundle.ts");
    verify_session_bundle_meta(bundle);
    auto tokenizer = tokenizer_from_bundle(bundle);
    verify_tokenizer_selftest(bundle, tokenizer);
    bool synthetic_ok = run_synthetic_word_delta_tests();
    int64_t rows = scalar_i64(attr_tensor(bundle, "num_utts"));

    auto enc_first = torch::jit::load(dir + "/enc_first.ts");
    enc_first.to(device);
    enc_first.eval();
    AOTIModelPackageLoader enc_steady(dir + "/enc_steady_aoti.pt2", "model", false, 1, -1);
    auto joint = torch::jit::load(dir + "/joint_step.ts");
    joint.to(device);
    joint.eval();
    auto predict = torch::jit::load(dir + "/predict_step.ts");
    predict.to(device);
    predict.eval();
    auto finalize_loaders = load_finalize_bucket_loaders(dir, device);

    std::printf("=== SESSION single-stream replay: %ld utterances (events=%s) ===\n",
                (long)rows, check_events ? "check" : "skip");
    SessionState session;
    int steady_pass = 0;
    int final_pass = 0;
    int event_pass = 0;
    int fork_pass = 0;

    for (int utt = 0; utt < rows; ++utt) {
      reset_session(session, bundle, device);
      int64_t sample_index = scalar_i64(utt_tensor(bundle, utt, "sample_index"));
      int64_t num_steady = scalar_i64(utt_tensor(bundle, utt, "num_steady"));
      int64_t final_drop = scalar_i64(utt_tensor(bundle, utt, "final_drop_extra"));
      int64_t final_T = scalar_i64(utt_tensor(bundle, utt, "final_T"));
      std::vector<EmittedEvent> gold_events;
      if (check_events) gold_events = gold_events_from_bundle(bundle, utt);
      std::vector<EmittedEvent> events;

      bool row_ok = true;
      try {
        for (int chunk = 0; chunk < num_steady; ++chunk) {
          run_steady_chunk(session, bundle, utt, chunk, enc_first, enc_steady, joint, predict, device, tokenizer, events);
        }
      } catch (const std::exception& e) {
        std::printf("  utt%d sample=%ld steady threw: %s\n", utt, (long)sample_index, e.what());
        row_ok = false;
      }

      auto steady_gold = tensor_to_vec(utt_tensor(bundle, utt, "steady_tokens"));
      bool steady_ok = row_ok && equal_tokens(session.hyp, steady_gold, "steady cumulative", utt);
      if (steady_ok) ++steady_pass;

      session.mode = SessionMode::PENDING_FINALIZE;
      FinalizeOutcome finalize;
      try {
        finalize = run_finalize(session, bundle, utt, finalize_loaders, joint, predict, device, tokenizer, events);
      } catch (const std::exception& e) {
        std::printf("  utt%d sample=%ld finalize threw: %s\n", utt, (long)sample_index, e.what());
        finalize.token_ok = false;
        finalize.fork_ok = false;
      }

      bool events_ok = true;
      if (check_events) {
        events_ok = row_ok && finalize.token_ok && equal_events(events, gold_events, utt);
      }
      if (finalize.token_ok) ++final_pass;
      if (check_events && events_ok) ++event_pass;
      if (finalize.fork_ok) ++fork_pass;
      auto gold = tensor_to_vec(utt_tensor(bundle, utt, "gold_tokens"));
      if (check_events) {
        std::printf("  utt%d sample=%ld steady_chunks=%ld final(drop=%ld,T=%ld) "
                    "steady=%s final=%s events=%s FORK_ASSERT=%s tokens=%zu/%zu events=%zu/%zu\n",
                    utt, (long)sample_index, (long)num_steady, (long)final_drop, (long)final_T,
                    steady_ok ? "PASS" : "FAIL",
                    finalize.token_ok ? "PASS" : "FAIL",
                    events_ok ? "PASS" : "FAIL",
                    finalize.fork_ok ? "PASS" : "FAIL",
                    finalize.emitted_tokens, gold.size(),
                    events.size(), gold_events.size());
      } else {
        std::printf("  utt%d sample=%ld steady_chunks=%ld final(drop=%ld,T=%ld) "
                    "steady=%s final=%s events=SKIP FORK_ASSERT=%s tokens=%zu/%zu\n",
                    utt, (long)sample_index, (long)num_steady, (long)final_drop, (long)final_T,
                    steady_ok ? "PASS" : "FAIL",
                    finalize.token_ok ? "PASS" : "FAIL",
                    finalize.fork_ok ? "PASS" : "FAIL",
                    finalize.emitted_tokens, gold.size());
      }
    }

    bool all = synthetic_ok && steady_pass == rows && final_pass == rows && fork_pass == rows &&
               (!check_events || event_pass == rows);
    if (check_events) {
      std::printf("=== SESSION %s: synthetic=%s steady=%d/%ld final_token_exact=%d/%ld event_text_exact=%d/%ld FORK_ASSERT=%d/%ld ===\n",
                  all ? "PASS" : "FAIL",
                  synthetic_ok ? "PASS" : "FAIL",
                  steady_pass, (long)rows,
                  final_pass, (long)rows,
                  event_pass, (long)rows,
                  fork_pass, (long)rows);
    } else {
      std::printf("=== SESSION %s: synthetic=%s steady=%d/%ld final_token_exact=%d/%ld event_text_exact=SKIP FORK_ASSERT=%d/%ld ===\n",
                  all ? "PASS" : "FAIL",
                  synthetic_ok ? "PASS" : "FAIL",
                  steady_pass, (long)rows,
                  final_pass, (long)rows,
                  fork_pass, (long)rows);
    }
    return all ? 0 : 1;
  } catch (const std::exception& e) {
    std::printf("SESSION setup failed: %s\n", e.what());
    return 2;
  }
}
