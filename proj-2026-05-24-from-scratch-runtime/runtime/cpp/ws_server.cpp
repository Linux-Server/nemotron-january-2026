#include "lib/admission/density_admission.h"
#include "lib/runtime_io/json.hpp"
#include "lib/scheduler/batched_steady_scheduler.h"
#include "lib/session/runtime.h"
#include "lib/telemetry/stats_collector.h"
#include "lib/ws/framing.h"
#include "lib/ws/handshake.h"
#include "lib/ws/routes.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <deque>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

namespace {

constexpr int kAdminWorkers = 2;
constexpr size_t kAdminQueueDepth = 16;
constexpr int kDefaultAdmissionBacklogCap = 12;
constexpr int kDefaultBatchMax = 4;
constexpr int kDefaultBatchWindowMs = 10;
constexpr int kDefaultBatchLoneTimeoutMs = 0;
constexpr int kDefaultBatchQueueCapacity = 16;
constexpr size_t kDefaultStatsWindow = 2048;
constexpr size_t kDefaultWsMaxMessageSize = ws_framing::kMaxMessageSize;
constexpr int kDefaultWsPingIntervalSec = 60;
constexpr int kDefaultWsPongTimeoutSec = 30;
constexpr int kDefaultShutdownDrainSec = 30;
constexpr int kDefaultFinalizeSilenceMs = 0;

std::string json_quote(const std::string& value) {
  return nlohmann::json(value).dump();
}

const char* json_bool(bool value) {
  return value ? "true" : "false";
}

std::string http_reason(int status) {
  switch (status) {
    case 200:
      return "OK";
    case 400:
      return "Bad Request";
    case 404:
      return "Not Found";
    case 431:
      return "Request Header Fields Too Large";
    case 503:
      return "Service Unavailable";
    default:
      return "Error";
  }
}

std::string build_json_response(int status, const std::string& body) {
  std::ostringstream oss;
  oss << "HTTP/1.1 " << status << " " << http_reason(status) << "\r\n"
      << "Content-Type: application/json\r\n"
      << "Content-Length: " << body.size() << "\r\n"
      << "Connection: close\r\n"
      << "\r\n"
      << body;
  return oss.str();
}

bool send_all(int fd, const void* data, size_t size) {
  const char* p = static_cast<const char*>(data);
  size_t sent = 0;
  while (sent < size) {
    ssize_t n = ::send(fd, p + sent, size - sent, MSG_NOSIGNAL);
    if (n > 0) {
      sent += static_cast<size_t>(n);
      continue;
    }
    if (n < 0 && errno == EINTR) continue;
    return false;
  }
  return true;
}

bool send_all(int fd, const std::string& data) {
  return send_all(fd, data.data(), data.size());
}

bool send_all(int fd, const std::vector<uint8_t>& data) {
  return send_all(fd, data.data(), data.size());
}

void close_fd(int* fd) {
  if (fd != nullptr && *fd >= 0) {
    ::close(*fd);
    *fd = -1;
  }
}

class UniqueFd {
 public:
  UniqueFd() = default;
  explicit UniqueFd(int fd) : fd_(fd) {}
  UniqueFd(const UniqueFd&) = delete;
  UniqueFd& operator=(const UniqueFd&) = delete;
  UniqueFd(UniqueFd&& other) noexcept : fd_(other.fd_) {
    other.fd_ = -1;
  }
  UniqueFd& operator=(UniqueFd&& other) noexcept {
    if (this != &other) {
      reset();
      fd_ = other.fd_;
      other.fd_ = -1;
    }
    return *this;
  }
  ~UniqueFd() {
    reset();
  }
  int get() const { return fd_; }
  int release() {
    int fd = fd_;
    fd_ = -1;
    return fd;
  }
  void reset(int next = -1) {
    if (fd_ >= 0) ::close(fd_);
    fd_ = next;
  }

 private:
  int fd_ = -1;
};

int parse_int_strict(const std::string& text, const char* label) {
  if (text.empty()) throw std::runtime_error(std::string(label) + " requires an integer");
  errno = 0;
  char* end = nullptr;
  long value = std::strtol(text.c_str(), &end, 10);
  if (errno != 0 || end == text.c_str() || *end != '\0' ||
      value < std::numeric_limits<int>::min() ||
      value > std::numeric_limits<int>::max()) {
    throw std::runtime_error(std::string(label) + " must be an integer: " + text);
  }
  return static_cast<int>(value);
}

uint64_t parse_u64_strict(const std::string& text, const char* label) {
  if (text.empty()) throw std::runtime_error(std::string(label) + " requires an integer");
  if (!std::all_of(text.begin(), text.end(), [](unsigned char ch) { return std::isdigit(ch); })) {
    throw std::runtime_error(std::string(label) + " must be a non-negative integer: " + text);
  }
  errno = 0;
  char* end = nullptr;
  unsigned long long value = std::strtoull(text.c_str(), &end, 10);
  if (errno != 0 || end == text.c_str() || *end != '\0') {
    throw std::runtime_error(std::string(label) + " must be a non-negative integer: " + text);
  }
  return static_cast<uint64_t>(value);
}

int read_env_int(const char* name, int fallback) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') return fallback;
  return parse_int_strict(raw, name);
}

std::optional<uint64_t> read_env_u64_optional(const char* name) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') return std::nullopt;
  return parse_u64_strict(raw, name);
}

uint64_t read_env_u64(const char* name, uint64_t fallback) {
  auto value = read_env_u64_optional(name);
  return value.value_or(fallback);
}

size_t read_env_size_t(const char* name, size_t fallback) {
  uint64_t value = read_env_u64(name, fallback);
  if (value == 0 || value > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    const char* raw = std::getenv(name);
    throw std::runtime_error(std::string("invalid positive integer env var ") + name + "=" +
                             (raw != nullptr ? raw : std::to_string(value)));
  }
  return static_cast<size_t>(value);
}

bool read_env_enabled(const char* name, bool fallback) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') return fallback;
  return std::string(raw) != "0";
}

bool file_exists(const fs::path& path) {
  std::error_code ec;
  return fs::is_regular_file(path, ec);
}

bool dir_exists(const fs::path& path) {
  std::error_code ec;
  return fs::is_directory(path, ec);
}

std::string weakly_canonical_string(const fs::path& path) {
  std::error_code ec;
  fs::path out = fs::weakly_canonical(path, ec);
  if (ec) return path.lexically_normal().string();
  return out.string();
}

bool artifact_dir_valid(const fs::path& dir) {
  return file_exists(dir / "session_audio_bundle.ts") &&
         file_exists(dir / "preproc.ts") &&
         file_exists(dir / "enc_first.ts") &&
         file_exists(dir / "enc_steady_aoti.pt2") &&
         file_exists(dir / "finalize_shared_weights.ts");
}

std::vector<fs::path> ancestor_bases(const fs::path& start) {
  std::vector<fs::path> out;
  fs::path cur = start;
  for (int i = 0; i < 8 && !cur.empty(); ++i) {
    out.push_back(cur);
    fs::path parent = cur.parent_path();
    if (parent == cur) break;
    cur = parent;
  }
  return out;
}

std::string resolve_artifact_dir(const std::string& argv0) {
  std::vector<fs::path> candidates;
  const char* env = std::getenv("NEMOTRON_ARTIFACT_DIR");
  if (env != nullptr && env[0] != '\0') candidates.emplace_back(env);

  std::error_code ec;
  fs::path cwd = fs::current_path(ec);
  if (!ec) {
    for (const auto& base : ancestor_bases(cwd)) {
      candidates.push_back(base / "artifacts");
      candidates.push_back(base / "runtime" / "artifacts");
    }
  }
  if (!argv0.empty()) {
    fs::path exe = fs::absolute(argv0, ec);
    if (!ec) {
      fs::path base = exe.has_parent_path() ? exe.parent_path() : exe;
      for (const auto& ancestor : ancestor_bases(base)) {
        candidates.push_back(ancestor / "artifacts");
        candidates.push_back(ancestor / "runtime" / "artifacts");
      }
    }
  }
  candidates.emplace_back("../artifacts");
  candidates.emplace_back("artifacts");
  candidates.emplace_back("runtime/artifacts");

  for (const auto& candidate : candidates) {
    if (artifact_dir_valid(candidate)) return weakly_canonical_string(candidate);
  }
  throw std::runtime_error("could not resolve runtime artifacts directory; set NEMOTRON_ARTIFACT_DIR");
}

bool steady_batch_dir_valid(const fs::path& dir) {
  return file_exists(dir / "MANIFEST.json") &&
         file_exists(dir / "enc_steady_aoti_b1.pt2") &&
         file_exists(dir / "enc_steady_aoti_b2.pt2") &&
         file_exists(dir / "enc_steady_aoti_b4.pt2");
}

std::string resolve_steady_batch_dir(const std::string& configured,
                                     bool explicit_path,
                                     const std::string& artifact_dir,
                                     const std::string& argv0,
                                     bool required) {
  std::vector<fs::path> candidates;
  if (!configured.empty()) candidates.emplace_back(configured);

  if (!explicit_path) {
    fs::path artifact_path(artifact_dir);
    if (artifact_path.has_parent_path()) {
      candidates.push_back(artifact_path.parent_path() / "steady_b_artifacts");
    }

    std::error_code ec;
    fs::path cwd = fs::current_path(ec);
    if (!ec) {
      for (const auto& base : ancestor_bases(cwd)) {
        candidates.push_back(base / "steady_b_artifacts");
        candidates.push_back(base / "runtime" / "steady_b_artifacts");
      }
    }
    if (!argv0.empty()) {
      fs::path exe = fs::absolute(argv0, ec);
      if (!ec) {
        fs::path base = exe.has_parent_path() ? exe.parent_path() : exe;
        for (const auto& ancestor : ancestor_bases(base)) {
          candidates.push_back(ancestor / "steady_b_artifacts");
          candidates.push_back(ancestor / "runtime" / "steady_b_artifacts");
        }
      }
    }
    candidates.emplace_back("steady_b_artifacts");
    candidates.emplace_back("../steady_b_artifacts");
    candidates.emplace_back("runtime/steady_b_artifacts");
  }

  for (const auto& candidate : candidates) {
    if (steady_batch_dir_valid(candidate)) return weakly_canonical_string(candidate);
  }

  if (required) {
    fs::path first = configured.empty() ? fs::path("./steady_b_artifacts") : fs::path(configured);
    if (!file_exists(first / "MANIFEST.json")) {
      throw std::runtime_error("steady batch MANIFEST.json missing: " +
                               (first / "MANIFEST.json").string());
    }
    throw std::runtime_error("steady batch artifacts missing enc_steady_aoti_b{1,2,4}.pt2 under: " +
                             first.string());
  }
  return configured.empty() ? "./steady_b_artifacts" : configured;
}

struct Summary {
  size_t n = 0;
  double p50 = 0.0;
  double p95 = 0.0;
  double p99 = 0.0;
  double mean = 0.0;
  double max = 0.0;
};

Summary summarize(std::vector<double> values) {
  Summary out;
  out.n = values.size();
  if (values.empty()) return out;
  std::sort(values.begin(), values.end());
  auto percentile = [&](double p) {
    size_t idx = static_cast<size_t>(std::llround(p * static_cast<double>(values.size() - 1)));
    if (idx >= values.size()) idx = values.size() - 1;
    return values[idx];
  };
  out.p50 = percentile(0.50);
  out.p95 = percentile(0.95);
  out.p99 = percentile(0.99);
  out.max = values.back();
  out.mean = std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
  return out;
}

std::string value_stats_json(const Summary& s) {
  std::ostringstream oss;
  oss << std::setprecision(17);
  oss << "{\"n\":" << s.n
      << ",\"p50\":" << s.p50
      << ",\"p95\":" << s.p95
      << ",\"p99\":" << s.p99
      << ",\"p95_minus_p50\":" << (s.p95 - s.p50)
      << ",\"p99_minus_p50\":" << (s.p99 - s.p50)
      << ",\"mean\":" << s.mean
      << ",\"max\":" << s.max
      << "}";
  return oss.str();
}

std::string scheduler_us_stats_json(const std::vector<double>& values) {
  Summary s = summarize(values);
  std::ostringstream oss;
  oss << std::setprecision(17);
  oss << "{\"n\":" << s.n
      << ",\"p50_us\":" << s.p50
      << ",\"p95_us\":" << s.p95
      << ",\"p99_us\":" << s.p99
      << ",\"p95_minus_p50_us\":" << (s.p95 - s.p50)
      << ",\"p99_minus_p50_us\":" << (s.p99 - s.p50)
      << ",\"mean_us\":" << s.mean
      << ",\"max_us\":" << s.max
      << "}";
  return oss.str();
}

double pct_clamped(double numerator, double denominator) {
  if (denominator <= 0.0) return 0.0;
  double pct = 100.0 * numerator / denominator;
  if (pct < 0.0) return 0.0;
  if (pct > 100.0) return 100.0;
  return pct;
}

std::string scheduler_telemetry_json(const BatchedSteadySchedulerTelemetry& telemetry) {
  std::ostringstream oss;
  oss << std::setprecision(17);
  oss << "{\"counts\":{\"enqueued\":" << telemetry.enqueued
      << ",\"completed\":" << telemetry.completed
      << ",\"dispatch_cycles\":" << telemetry.dispatch_cycles
      << ",\"warmup_runs\":" << telemetry.warmup_runs
      << ",\"B1\":" << telemetry.bucket_b1
      << ",\"B2\":" << telemetry.bucket_b2
      << ",\"B4\":" << telemetry.bucket_b4
      << ",\"K2_padded_to_B4\":" << telemetry.k2_padded_to_b4
      << ",\"K3_padded_to_B4\":" << telemetry.k3_padded_to_b4
      << ",\"K4\":" << telemetry.k4
      << ",\"backlog_gt_bmax\":" << telemetry.backlog_gt_bmax
      << ",\"dispatcher_exceptions\":" << telemetry.dispatcher_exceptions
      << "}"
      << ",\"dispatcher_cpu_pct\":" << pct_clamped(telemetry.dispatcher_cpu_us, telemetry.dispatcher_wall_us)
      << ",\"dispatcher_cpu_us\":" << telemetry.dispatcher_cpu_us
      << ",\"dispatcher_wall_us\":" << telemetry.dispatcher_wall_us
      << ",\"dispatcher_stream_util_pct\":"
      << pct_clamped(telemetry.dispatcher_stream_run_us, telemetry.dispatcher_wall_us)
      << ",\"dispatcher_stream_run_us\":" << telemetry.dispatcher_stream_run_us
      << ",\"timers\":{\"gather_wait_us\":" << scheduler_us_stats_json(telemetry.gather_wait_us)
      << ",\"service_wait_us\":" << scheduler_us_stats_json(telemetry.service_wait_us)
      << ",\"cuda_run_us\":" << scheduler_us_stats_json(telemetry.cuda_run_us)
      << ",\"output_sync_us\":" << scheduler_us_stats_json(telemetry.output_sync_us)
      << ",\"worker_blocked_us\":" << scheduler_us_stats_json(telemetry.worker_blocked_us)
      << ",\"window_wakeup_jitter_us\":" << scheduler_us_stats_json(telemetry.window_wakeup_jitter_us)
      << "}"
      << ",\"queue_depth\":" << value_stats_json(summarize(telemetry.queue_depth))
      << ",\"per_stream_fairness_spread_us\":"
      << scheduler_us_stats_json(telemetry.per_stream_fairness_spread_us)
      << "}";
  return oss.str();
}

std::string python_admission_json(const DensityAdmission& admission) {
  AdmissionTelemetry telemetry = admission.telemetry_snapshot();
  uint64_t backlog_count = telemetry.active_count + telemetry.backlog_count;
  std::ostringstream oss;
  oss << "{\"enabled\":true"
      << ",\"attempted\":" << telemetry.offered
      << ",\"admitted\":" << telemetry.admitted
      << ",\"rejected\":" << telemetry.shed_close_count
      << ",\"max_backlog\":" << telemetry.backlog_peak
      << ",\"max_ready_age_ms\":0"
      << ",\"signal\":{"
      << "\"queued_events\":" << telemetry.backlog_count
      << ",\"ready_count\":0"
      << ",\"backlog_count\":" << backlog_count
      << ",\"oldest_ready_age_ms\":0"
      << ",\"oldest_ready_session_id\":null"
      << "}}";
  return oss.str();
}

struct ServerConfig {
  int port = -1;
  bool port_set = false;
  uint64_t admission_active_cap = 0;
  bool admission_active_cap_set = false;
  uint64_t admission_backlog_cap = kDefaultAdmissionBacklogCap;
  bool admission_backlog_cap_set = false;
  std::string steady_batch_dir = "./steady_b_artifacts";
  std::string effective_steady_batch_dir = "./steady_b_artifacts";
  bool steady_batch_dir_explicit = false;
  std::string process_label;
  bool selftest_and_exit = false;
  bool print_config = false;
  std::string artifact_dir;
  std::string argv0;

  bool scheduler_enabled = false;
  int batch_b_max = kDefaultBatchMax;
  int batch_window_ms = kDefaultBatchWindowMs;
  int batch_lone_timeout_ms = kDefaultBatchLoneTimeoutMs;
  int batch_queue_capacity = kDefaultBatchQueueCapacity;
  int device_index = 0;
  int steady_num_runners = 1;
  int finalize_num_runners = 1;

  bool stats_enabled = true;
  size_t stats_window = kDefaultStatsWindow;
  size_t ws_max_message_size = kDefaultWsMaxMessageSize;
  int ws_ping_interval_sec = kDefaultWsPingIntervalSec;
  int ws_pong_timeout_sec = kDefaultWsPongTimeoutSec;
  int shutdown_drain_sec = kDefaultShutdownDrainSec;
  int finalize_silence_ms = kDefaultFinalizeSilenceMs;

  int selftest_ws_close_delay_ms = 0;
  bool selftest_lightweight_runtime = false;
};

void populate_env_config(ServerConfig* cfg) {
  cfg->scheduler_enabled = read_env_int("NEMOTRON_DENSITY_BATCH_STEADY", 0) != 0;
  cfg->batch_b_max = read_env_int("NEMOTRON_DENSITY_BATCH_MAX", kDefaultBatchMax);
  cfg->batch_window_ms = read_env_int("NEMOTRON_DENSITY_BATCH_WINDOW_MS", kDefaultBatchWindowMs);
  cfg->batch_lone_timeout_ms =
      read_env_int("NEMOTRON_DENSITY_BATCH_LONE_TIMEOUT_MS", kDefaultBatchLoneTimeoutMs);
  cfg->batch_queue_capacity =
      read_env_int("NEMOTRON_DENSITY_BATCH_QUEUE_CAPACITY", kDefaultBatchQueueCapacity);
  cfg->device_index = read_env_int("NEMOTRON_DENSITY_DEVICE_INDEX", 0);
  cfg->steady_num_runners = read_env_int("NEMOTRON_DENSITY_STEADY_RUNNERS", 1);
  cfg->finalize_num_runners = read_env_int("NEMOTRON_DENSITY_FINALIZE_RUNNERS", 1);
  cfg->stats_enabled = read_env_enabled("NEMOTRON_STATS_ENABLED", true);
  cfg->stats_window = read_env_size_t("NEMOTRON_STATS_WINDOW", kDefaultStatsWindow);
  cfg->ws_max_message_size =
      read_env_size_t("NEMOTRON_WS_MAX_MESSAGE_SIZE", kDefaultWsMaxMessageSize);
  cfg->ws_ping_interval_sec = read_env_int("NEMOTRON_WS_PING_INTERVAL_SEC", kDefaultWsPingIntervalSec);
  cfg->ws_pong_timeout_sec = read_env_int("NEMOTRON_WS_PONG_TIMEOUT_SEC", kDefaultWsPongTimeoutSec);
  cfg->shutdown_drain_sec = read_env_int("NEMOTRON_SHUTDOWN_DRAIN_SEC", kDefaultShutdownDrainSec);
  cfg->finalize_silence_ms = read_env_int("NEMOTRON_FINALIZE_SILENCE_MS", kDefaultFinalizeSilenceMs);

  if (!cfg->admission_active_cap_set) {
    auto env_active = read_env_u64_optional("NEMOTRON_DENSITY_ADMISSION_ACTIVE_CAP");
    if (env_active.has_value()) {
      cfg->admission_active_cap = *env_active;
      cfg->admission_active_cap_set = true;
    }
  }
  if (!cfg->admission_backlog_cap_set) {
    cfg->admission_backlog_cap = read_env_u64("NEMOTRON_DENSITY_ADMISSION_BACKLOG_CAP",
                                              kDefaultAdmissionBacklogCap);
  }
}

void validate_config(ServerConfig* cfg, bool require_port_and_admission) {
  populate_env_config(cfg);
  cfg->artifact_dir = resolve_artifact_dir(cfg->argv0);
  cfg->effective_steady_batch_dir = resolve_steady_batch_dir(cfg->steady_batch_dir,
                                                             cfg->steady_batch_dir_explicit,
                                                             cfg->artifact_dir,
                                                             cfg->argv0,
                                                             cfg->scheduler_enabled);
  if (require_port_and_admission && !cfg->port_set) {
    throw std::runtime_error("warning: --port is required; no compiled default is provided");
  }
  if (cfg->port_set && (cfg->port < 0 || cfg->port > 65535)) {
    throw std::runtime_error("--port must be in [0, 65535]");
  }
  if (require_port_and_admission && !cfg->admission_active_cap_set) {
    throw std::runtime_error("--admission-active-cap or NEMOTRON_DENSITY_ADMISSION_ACTIVE_CAP is required");
  }
  if (cfg->admission_active_cap_set && cfg->admission_active_cap == 0) {
    throw std::runtime_error("--admission-active-cap must be positive");
  }
  if (cfg->batch_b_max != 1 && cfg->batch_b_max != 2 && cfg->batch_b_max != 4) {
    throw std::runtime_error("NEMOTRON_DENSITY_BATCH_MAX must be one of 1, 2, 4");
  }
  if (cfg->batch_window_ms < 0 || cfg->batch_lone_timeout_ms < 0 ||
      cfg->batch_queue_capacity <= 0) {
    throw std::runtime_error("batch scheduler timing/capacity env vars must be non-negative, with capacity > 0");
  }
  if (cfg->steady_num_runners <= 0 || cfg->finalize_num_runners <= 0) {
    throw std::runtime_error("runner counts must be positive");
  }
}

std::string config_table(const ServerConfig& cfg) {
  std::ostringstream oss;
  oss << "[runtime]\n"
      << "  scheduler_enabled = " << json_bool(cfg.scheduler_enabled) << "\n"
      << "  steady_batch_dir = " << cfg.effective_steady_batch_dir << "\n"
      << "\n[admission]\n"
      << "  active_cap = " << cfg.admission_active_cap << "\n"
      << "  backlog_cap = " << cfg.admission_backlog_cap << "\n"
      << "\n[stats]\n"
      << "  enabled = " << json_bool(cfg.stats_enabled) << "\n"
      << "  window_size = " << cfg.stats_window << "\n"
      << "\n[ws]\n"
      << "  max_message_size = " << cfg.ws_max_message_size << "\n"
      << "  ping_interval_sec = " << cfg.ws_ping_interval_sec << "\n"
      << "  pong_timeout_sec = " << cfg.ws_pong_timeout_sec << "\n"
      << "\n[shutdown]\n"
      << "  drain_sec = " << cfg.shutdown_drain_sec << "\n";
  return oss.str();
}

ServerConfig parse_args(int argc, char** argv) {
  ServerConfig cfg;
  if (argc > 0 && argv[0] != nullptr) cfg.argv0 = argv[0];
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto need_value = [&](const char* flag) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(std::string(flag) + " requires a value");
      return argv[++i];
    };
    if (arg == "--port") {
      cfg.port = parse_int_strict(need_value("--port"), "--port");
      cfg.port_set = true;
    } else if (arg == "--admission-active-cap") {
      cfg.admission_active_cap = parse_u64_strict(need_value("--admission-active-cap"),
                                                  "--admission-active-cap");
      cfg.admission_active_cap_set = true;
    } else if (arg == "--admission-backlog-cap") {
      cfg.admission_backlog_cap = parse_u64_strict(need_value("--admission-backlog-cap"),
                                                   "--admission-backlog-cap");
      cfg.admission_backlog_cap_set = true;
    } else if (arg == "--steady-batch-dir") {
      cfg.steady_batch_dir = need_value("--steady-batch-dir");
      cfg.steady_batch_dir_explicit = true;
    } else if (arg == "--process-label") {
      cfg.process_label = need_value("--process-label");
    } else if (arg == "--selftest-and-exit") {
      cfg.selftest_and_exit = true;
    } else if (arg == "--print-config") {
      cfg.print_config = true;
    } else if (arg == "--help" || arg == "-h") {
      std::cout
          << "usage: ws_server --port <int> --admission-active-cap <int> [options]\n"
          << "options:\n"
          << "  --admission-backlog-cap <int>   default 12 or env override\n"
          << "  --steady-batch-dir <path>       default ./steady_b_artifacts\n"
          << "  --process-label <str>\n"
          << "  --print-config\n"
          << "  --selftest-and-exit\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  return cfg;
}

struct ReadHttpOutcome {
  ws_handshake::ParseResult result = ws_handshake::ParseResult::NEED_MORE;
  ws_handshake::HttpRequest request;
};

ReadHttpOutcome read_http_request_from_socket(int fd) {
  ReadHttpOutcome out;
  std::string buffer;
  buffer.reserve(ws_handshake::kMaxHttpHeaderBytes);
  auto deadline = Clock::now() + std::chrono::milliseconds(ws_handshake::kHttpReadTimeoutMs);

  for (;;) {
    out.result = ws_handshake::parse_http_request(buffer, out.request);
    if (out.result != ws_handshake::ParseResult::NEED_MORE) return out;

    auto now = Clock::now();
    if (now >= deadline) {
      out.result = ws_handshake::ParseResult::MALFORMED;
      return out;
    }
    int timeout_ms = static_cast<int>(
        std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count());
    if (timeout_ms < 0) timeout_ms = 0;
    pollfd pfd{};
    pfd.fd = fd;
    pfd.events = POLLIN;
    int pr = ::poll(&pfd, 1, timeout_ms);
    if (pr == 0) {
      out.result = ws_handshake::ParseResult::MALFORMED;
      return out;
    }
    if (pr < 0) {
      if (errno == EINTR) continue;
      out.result = ws_handshake::ParseResult::MALFORMED;
      return out;
    }
    char chunk[1024];
    ssize_t n = ::recv(fd, chunk, sizeof(chunk), 0);
    if (n > 0) {
      buffer.append(chunk, static_cast<size_t>(n));
      continue;
    }
    if (n < 0 && errno == EINTR) continue;
    out.result = ws_handshake::ParseResult::MALFORMED;
    return out;
  }
}

std::optional<size_t> parse_last_query(const ws_routes::Route& route, std::string* error_body) {
  auto it = route.query_params.find("last");
  if (it == route.query_params.end() || it->second.empty()) return std::nullopt;
  const std::string& raw = it->second;
  bool digits = std::all_of(raw.begin(), raw.end(), [](unsigned char ch) { return std::isdigit(ch); });
  if (!digits) {
    *error_body = "{\"error\":" + json_quote("invalid 'last': '" + raw + "'") + "}";
    return std::nullopt;
  }
  uint64_t parsed = 0;
  try {
    parsed = parse_u64_strict(raw, "last");
  } catch (const std::exception&) {
    *error_body = "{\"error\":" + json_quote("invalid 'last': '" + raw + "'") + "}";
    return std::nullopt;
  }
  if (parsed == 0 || parsed > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    *error_body = "{\"error\":" + json_quote("invalid 'last': '" + raw + "'") + "}";
    return std::nullopt;
  }
  return static_cast<size_t>(parsed);
}

struct ServerState {
  explicit ServerState(ServerConfig c) : cfg(std::move(c)) {}

  ServerConfig cfg;
  std::unique_ptr<SharedRuntime> shared_runtime;
  std::unique_ptr<DensityAdmission> admission;
  std::unique_ptr<StatsCollector> stats;
  std::unique_ptr<BatchedSteadyLoaderSet> scheduler_loader;
  std::unique_ptr<BatchedSteadyScheduler> scheduler;
  std::atomic<bool> model_loaded{false};
  std::atomic<uint64_t> next_stream_id{1};
};

std::string health_json(const ServerState& state) {
  bool loaded = state.model_loaded.load(std::memory_order_acquire);
  std::ostringstream oss;
  oss << "{\"status\":\"" << (loaded ? "healthy" : "loading") << "\""
      << ",\"model_loaded\":" << json_bool(loaded);
  if (state.admission) {
    oss << ",\"admission\":" << python_admission_json(*state.admission);
  }
  if (!state.cfg.process_label.empty()) {
    oss << ",\"pid\":" << static_cast<long long>(::getpid())
        << ",\"process_label\":" << json_quote(state.cfg.process_label);
  }
  oss << "}";
  return oss.str();
}

struct AdminJob {
  int fd = -1;
  ws_routes::Route route;
};

class AdminHandlerPool {
 public:
  explicit AdminHandlerPool(std::shared_ptr<ServerState> state) : state_(std::move(state)) {}
  AdminHandlerPool(const AdminHandlerPool&) = delete;
  AdminHandlerPool& operator=(const AdminHandlerPool&) = delete;

  void start() {
    for (int i = 0; i < kAdminWorkers; ++i) {
      workers_.emplace_back([this] { worker_loop(); });
    }
  }

  bool try_enqueue(AdminJob job) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopping_ || queue_.size() >= kAdminQueueDepth) return false;
    queue_.push_back(std::move(job));
    cv_.notify_one();
    return true;
  }

  void stop() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    cv_.notify_all();
    for (auto& worker : workers_) {
      if (worker.joinable()) worker.join();
    }
    workers_.clear();
  }

 private:
  void worker_loop() {
    for (;;) {
      AdminJob job;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&] { return stopping_ || !queue_.empty(); });
        if (queue_.empty()) {
          if (stopping_) break;
          continue;
        }
        job = std::move(queue_.front());
        queue_.pop_front();
      }
      handle_job(std::move(job));
    }
  }

  void handle_job(AdminJob job) {
    UniqueFd fd(job.fd);
    std::string body;
    int status = 200;
    try {
      switch (job.route.kind) {
        case ws_routes::RouteKind::HEALTH:
          body = health_json(*state_);
          break;
        case ws_routes::RouteKind::STATS: {
          std::string error_body;
          std::optional<size_t> last = parse_last_query(job.route, &error_body);
          if (!error_body.empty()) {
            status = 400;
            body = std::move(error_body);
          } else {
            body = state_->stats->snapshot_json(last);
          }
          break;
        }
        case ws_routes::RouteKind::SCHEDULER_TELEMETRY:
          if (!state_->scheduler) {
            status = 404;
            body = "{\"error\":\"no scheduler\"}";
          } else {
            body = scheduler_telemetry_json(state_->scheduler->telemetry_snapshot());
          }
          break;
        default:
          status = 404;
          body = "{\"error\":\"not_found\"}";
          break;
      }
    } catch (const std::exception& e) {
      status = 503;
      body = "{\"error\":" + json_quote(e.what()) + "}";
    }
    (void)send_all(fd.get(), build_json_response(status, body));
  }

  std::shared_ptr<ServerState> state_;
  std::vector<std::thread> workers_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<AdminJob> queue_;
  bool stopping_ = false;
};

class AdmissionCloseGuard {
 public:
  AdmissionCloseGuard(DensityAdmission* admission, std::string stream_id)
      : admission_(admission), stream_id_(std::move(stream_id)) {}
  AdmissionCloseGuard(const AdmissionCloseGuard&) = delete;
  AdmissionCloseGuard& operator=(const AdmissionCloseGuard&) = delete;
  ~AdmissionCloseGuard() {
    if (admission_ != nullptr) admission_->on_close(stream_id_);
  }
  void dismiss() { admission_ = nullptr; }

 private:
  DensityAdmission* admission_ = nullptr;
  std::string stream_id_;
};

void ws_worker(int fd, std::shared_ptr<ServerState> state) {
  UniqueFd conn(fd);
  std::string stream_id = "ws-" + std::to_string(::getpid()) + "-" +
                          std::to_string(state->next_stream_id.fetch_add(1));

  // Python prepares the websocket before admission rejection, then closes with WS-1013.
  // The v5 pre-handshake HTTP-503 path is an architectural extension deferred past Step 7.
  AdmitResult admit = state->admission->try_admit(stream_id);
  if (admit.shed()) {
    (void)send_all(conn.get(), ws_framing::write_close_frame(1013, "admission_backpressure"));
    return;
  }

  AdmissionCloseGuard guard(state->admission.get(), stream_id);
  (void)send_all(conn.get(), ws_framing::write_frame(ws_framing::Opcode::TEXT, "{\"type\":\"ready\"}"));
  if (state->cfg.selftest_ws_close_delay_ms > 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(state->cfg.selftest_ws_close_delay_ms));
  }
  (void)send_all(conn.get(), ws_framing::write_close_frame(1011, "server-not-ready"));
}

class WsServer {
 public:
  explicit WsServer(ServerConfig cfg) : state_(std::make_shared<ServerState>(std::move(cfg))) {}
  WsServer(const WsServer&) = delete;
  WsServer& operator=(const WsServer&) = delete;
  ~WsServer() {
    stop();
  }

  void start() {
    if (running_.load()) return;
    construct_runtime();
    listen_fd_.reset(create_listener(state_->cfg.port));
    state_->cfg.port = bound_port(listen_fd_.get());
    admin_pool_ = std::make_unique<AdminHandlerPool>(state_);
    admin_pool_->start();
    running_.store(true, std::memory_order_release);
    accept_thread_ = std::thread([this] { accept_loop(); });
  }

  void stop() {
    if (!running_.exchange(false)) return;
    if (listen_fd_.get() >= 0) {
      (void)::shutdown(listen_fd_.get(), SHUT_RDWR);
    }
    listen_fd_.reset();
    if (accept_thread_.joinable()) accept_thread_.join();
    if (admin_pool_) {
      admin_pool_->stop();
      admin_pool_.reset();
    }
    {
      std::lock_guard<std::mutex> lock(ws_threads_mutex_);
      for (auto& worker : ws_threads_) {
        if (worker.joinable()) worker.join();
      }
      ws_threads_.clear();
    }
    state_.reset();
  }

  int port() const {
    return state_->cfg.port;
  }

 private:
  void construct_runtime() {
    auto stats = std::make_unique<StatsCollector>(state_->cfg.stats_window, state_->cfg.stats_enabled);

    SharedRuntimeConfig shared_cfg;
    shared_cfg.bundle_path = (fs::path(state_->cfg.artifact_dir) / "session_audio_bundle.ts").string();
    shared_cfg.steady_artifacts_dir = state_->cfg.artifact_dir;
    std::string stripped = (fs::path(state_->cfg.artifact_dir) / "stripped_finalize_buckets").string();
    shared_cfg.finalize_buckets_dir = dir_exists(stripped)
                                          ? stripped
                                          : (fs::path(state_->cfg.artifact_dir) / "finalize_buckets").string();
    shared_cfg.b_max = state_->cfg.batch_b_max;
    shared_cfg.batch_window_ms = state_->cfg.batch_window_ms;
    shared_cfg.batch_lone_timeout_ms = state_->cfg.batch_lone_timeout_ms;
    shared_cfg.batch_queue_capacity = state_->cfg.batch_queue_capacity;
    shared_cfg.device_index = state_->cfg.device_index;
    shared_cfg.steady_num_runners = state_->cfg.steady_num_runners;
    shared_cfg.finalize_num_runners = state_->cfg.finalize_num_runners;
    // Step 7 owns only the server skeleton and telemetry route. SessionRuntime scheduler
    // lifecycle integration is Step 9, so ws_server keeps a telemetry-visible scheduler here.
    shared_cfg.scheduler_enabled = false;

    state_->admission = std::make_unique<DensityAdmission>(state_->cfg.admission_active_cap,
                                                           state_->cfg.admission_backlog_cap);
    stats->set_admission(state_->admission.get());
    state_->stats = std::move(stats);
    if (!state_->cfg.selftest_lightweight_runtime) {
      state_->shared_runtime = std::make_unique<SharedRuntime>(shared_cfg);
    }

    if (state_->cfg.scheduler_enabled) {
      if (state_->cfg.selftest_lightweight_runtime) {
        throw std::runtime_error("selftest lightweight runtime cannot construct scheduler telemetry");
      }
      torch::Device device(torch::kCUDA, state_->cfg.device_index);
      state_->scheduler_loader = std::make_unique<BatchedSteadyLoaderSet>(
          state_->cfg.effective_steady_batch_dir,
          (fs::path(state_->cfg.artifact_dir) / "finalize_shared_weights.ts").string(),
          device,
          state_->cfg.steady_num_runners,
          "ws_server");
      BatchedSteadySchedulerPolicy policy;
      policy.B_max = state_->cfg.batch_b_max;
      policy.window_ms = state_->cfg.batch_window_ms;
      policy.lone_timeout_ms = state_->cfg.batch_lone_timeout_ms;
      policy.queue_capacity = state_->cfg.batch_queue_capacity;
      state_->scheduler =
          std::make_unique<BatchedSteadyScheduler>(*state_->scheduler_loader, device, policy);
      state_->scheduler->warmup_buckets();
      state_->scheduler->start();
    }

    state_->model_loaded.store(true, std::memory_order_release);
  }

  int create_listener(int port) {
    UniqueFd fd(::socket(AF_INET, SOCK_STREAM, 0));
    if (fd.get() < 0) throw std::runtime_error(std::string("socket failed: ") + std::strerror(errno));
    int yes = 1;
    if (::setsockopt(fd.get(), SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes)) != 0) {
      throw std::runtime_error(std::string("setsockopt SO_REUSEADDR failed: ") + std::strerror(errno));
    }
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(static_cast<uint16_t>(port));
    if (::bind(fd.get(), reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
      throw std::runtime_error(std::string("bind failed: ") + std::strerror(errno));
    }
    if (::listen(fd.get(), 128) != 0) {
      throw std::runtime_error(std::string("listen failed: ") + std::strerror(errno));
    }
    return fd.release();
  }

  int bound_port(int fd) const {
    sockaddr_in addr{};
    socklen_t len = sizeof(addr);
    if (::getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len) != 0) {
      throw std::runtime_error(std::string("getsockname failed: ") + std::strerror(errno));
    }
    return ntohs(addr.sin_port);
  }

  void accept_loop() {
    while (running_.load(std::memory_order_acquire)) {
      sockaddr_in peer{};
      socklen_t peer_len = sizeof(peer);
      int fd = ::accept(listen_fd_.get(), reinterpret_cast<sockaddr*>(&peer), &peer_len);
      if (fd < 0) {
        if (errno == EINTR) continue;
        if (!running_.load(std::memory_order_acquire)) break;
        continue;
      }
      handle_accepted(fd);
    }
  }

  void handle_accepted(int fd) {
    UniqueFd conn(fd);
    ReadHttpOutcome read = read_http_request_from_socket(conn.get());
    if (read.result == ws_handshake::ParseResult::MALFORMED) {
      (void)send_all(conn.get(), build_json_response(400, "{\"error\":\"bad_request\"}"));
      return;
    }
    if (read.result == ws_handshake::ParseResult::OVERSIZE_HEADERS) {
      (void)send_all(conn.get(), build_json_response(431, "{\"error\":\"headers_too_large\"}"));
      return;
    }
    if (read.result != ws_handshake::ParseResult::OK) {
      (void)send_all(conn.get(), build_json_response(400, "{\"error\":\"bad_request\"}"));
      return;
    }

    ws_routes::Route route = ws_routes::dispatch(read.request);
    if (route.kind == ws_routes::RouteKind::HEALTH ||
        route.kind == ws_routes::RouteKind::STATS ||
        route.kind == ws_routes::RouteKind::SCHEDULER_TELEMETRY) {
      AdminJob job;
      job.fd = conn.release();
      job.route = std::move(route);
      if (!admin_pool_->try_enqueue(std::move(job))) {
        UniqueFd rejected(job.fd);
        (void)send_all(rejected.get(), build_json_response(503, "{\"error\":\"admin_queue_full\"}"));
      }
      return;
    }

    if (route.kind == ws_routes::RouteKind::WEBSOCKET) {
      auto key_it = read.request.headers.find("sec-websocket-key");
      std::string accept_key = ws_handshake::compute_accept_key(key_it->second);
      if (!send_all(conn.get(), ws_handshake::build_handshake_response(accept_key))) return;
      std::lock_guard<std::mutex> lock(ws_threads_mutex_);
      ws_threads_.emplace_back([state = state_, fd = conn.release()] { ws_worker(fd, state); });
      return;
    }

    if (route.kind == ws_routes::RouteKind::BAD_REQUEST) {
      (void)send_all(conn.get(), build_json_response(400, "{\"error\":\"bad_request\"}"));
    } else {
      (void)send_all(conn.get(), build_json_response(404, "{\"error\":\"not_found\"}"));
    }
  }

  std::shared_ptr<ServerState> state_;
  std::atomic<bool> running_{false};
  UniqueFd listen_fd_;
  std::unique_ptr<AdminHandlerPool> admin_pool_;
  std::thread accept_thread_;
  std::mutex ws_threads_mutex_;
  std::vector<std::thread> ws_threads_;
};

class ScopedEnv {
 public:
  void set(const std::string& name, const std::string& value) {
    remember(name);
    ::setenv(name.c_str(), value.c_str(), 1);
  }
  void unset(const std::string& name) {
    remember(name);
    ::unsetenv(name.c_str());
  }
  ~ScopedEnv() {
    for (auto it = saved_.rbegin(); it != saved_.rend(); ++it) {
      if (it->second.has_value()) {
        ::setenv(it->first.c_str(), it->second->c_str(), 1);
      } else {
        ::unsetenv(it->first.c_str());
      }
    }
  }

 private:
  void remember(const std::string& name) {
    if (std::any_of(saved_.begin(), saved_.end(), [&](const auto& item) {
          return item.first == name;
        })) {
      return;
    }
    const char* raw = std::getenv(name.c_str());
    if (raw == nullptr) {
      saved_.push_back({name, std::nullopt});
    } else {
      saved_.push_back({name, std::string(raw)});
    }
  }

  std::vector<std::pair<std::string, std::optional<std::string>>> saved_;
};

void clear_selftest_env(ScopedEnv* env) {
  for (const char* name : {
           "NEMOTRON_STATS_ENABLED",
           "NEMOTRON_STATS_WINDOW",
           "NEMOTRON_DENSITY_BATCH_STEADY",
           "NEMOTRON_DENSITY_ADMISSION_ACTIVE_CAP",
           "NEMOTRON_DENSITY_ADMISSION_BACKLOG_CAP",
           "NEMOTRON_DENSITY_BATCH_MAX",
           "NEMOTRON_DENSITY_BATCH_WINDOW_MS",
           "NEMOTRON_DENSITY_BATCH_LONE_TIMEOUT_MS",
           "NEMOTRON_DENSITY_BATCH_QUEUE_CAPACITY",
       }) {
    env->unset(name);
  }
}

struct HttpClientResponse {
  int status = 0;
  std::string body;
  std::string raw;
};

UniqueFd connect_localhost(int port) {
  UniqueFd fd(::socket(AF_INET, SOCK_STREAM, 0));
  if (fd.get() < 0) throw std::runtime_error(std::string("client socket failed: ") + std::strerror(errno));
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = htons(static_cast<uint16_t>(port));
  if (::connect(fd.get(), reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    throw std::runtime_error(std::string("client connect failed: ") + std::strerror(errno));
  }
  return fd;
}

std::string recv_until_close(int fd) {
  std::string out;
  char buf[4096];
  for (;;) {
    ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
    if (n > 0) {
      out.append(buf, static_cast<size_t>(n));
      continue;
    }
    if (n < 0 && errno == EINTR) continue;
    break;
  }
  return out;
}

HttpClientResponse http_request(int port, const std::string& request) {
  UniqueFd fd = connect_localhost(port);
  if (!send_all(fd.get(), request)) throw std::runtime_error("client send failed");
  std::string raw = recv_until_close(fd.get());
  HttpClientResponse out;
  out.raw = raw;
  size_t line_end = raw.find("\r\n");
  if (line_end == std::string::npos) throw std::runtime_error("HTTP response missing status line");
  std::istringstream status_line(raw.substr(0, line_end));
  std::string http_version;
  status_line >> http_version >> out.status;
  size_t header_end = raw.find("\r\n\r\n");
  if (header_end != std::string::npos) out.body = raw.substr(header_end + 4);
  return out;
}

struct ClientFrame {
  uint8_t opcode = 0;
  std::vector<uint8_t> payload;
};

bool recv_exact(int fd, void* data, size_t size) {
  char* p = static_cast<char*>(data);
  size_t got = 0;
  while (got < size) {
    ssize_t n = ::recv(fd, p + got, size - got, 0);
    if (n > 0) {
      got += static_cast<size_t>(n);
      continue;
    }
    if (n < 0 && errno == EINTR) continue;
    return false;
  }
  return true;
}

ClientFrame read_server_frame(int fd) {
  uint8_t hdr[2]{};
  if (!recv_exact(fd, hdr, sizeof(hdr))) throw std::runtime_error("missing websocket frame header");
  bool masked = (hdr[1] & 0x80) != 0;
  uint64_t len = hdr[1] & 0x7f;
  if (len == 126) {
    uint8_t ext[2]{};
    if (!recv_exact(fd, ext, sizeof(ext))) throw std::runtime_error("missing websocket frame ext16");
    len = (static_cast<uint64_t>(ext[0]) << 8) | ext[1];
  } else if (len == 127) {
    uint8_t ext[8]{};
    if (!recv_exact(fd, ext, sizeof(ext))) throw std::runtime_error("missing websocket frame ext64");
    len = 0;
    for (uint8_t b : ext) len = (len << 8) | b;
  }
  uint8_t mask[4]{};
  if (masked && !recv_exact(fd, mask, sizeof(mask))) throw std::runtime_error("missing websocket mask");
  if (len > 1024 * 1024) throw std::runtime_error("selftest websocket frame too large");
  ClientFrame frame;
  frame.opcode = hdr[0] & 0x0f;
  frame.payload.resize(static_cast<size_t>(len));
  if (len > 0 && !recv_exact(fd, frame.payload.data(), frame.payload.size())) {
    throw std::runtime_error("missing websocket payload");
  }
  if (masked) {
    for (size_t i = 0; i < frame.payload.size(); ++i) frame.payload[i] ^= mask[i % 4];
  }
  return frame;
}

UniqueFd websocket_connect(int port, int* http_status) {
  UniqueFd fd = connect_localhost(port);
  std::string request =
      "GET / HTTP/1.1\r\n"
      "Host: 127.0.0.1\r\n"
      "Upgrade: websocket\r\n"
      "Connection: Upgrade\r\n"
      "Sec-WebSocket-Version: 13\r\n"
      "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
      "\r\n";
  if (!send_all(fd.get(), request)) throw std::runtime_error("websocket handshake send failed");

  std::string header;
  char ch = '\0';
  while (header.find("\r\n\r\n") == std::string::npos) {
    ssize_t n = ::recv(fd.get(), &ch, 1, 0);
    if (n == 1) {
      header.push_back(ch);
      if (header.size() > 8192) throw std::runtime_error("websocket handshake response too large");
      continue;
    }
    if (n < 0 && errno == EINTR) continue;
    throw std::runtime_error("websocket handshake response closed early");
  }
  size_t line_end = header.find("\r\n");
  std::istringstream status_line(header.substr(0, line_end));
  std::string http_version;
  status_line >> http_version >> *http_status;
  return fd;
}

uint16_t close_code(const ClientFrame& frame) {
  if (frame.payload.size() < 2) return 0;
  return static_cast<uint16_t>((static_cast<uint16_t>(frame.payload[0]) << 8) |
                               static_cast<uint16_t>(frame.payload[1]));
}

std::string close_reason(const ClientFrame& frame) {
  if (frame.payload.size() <= 2) return {};
  return std::string(frame.payload.begin() + 2, frame.payload.end());
}

struct SelftestResult {
  int id = 0;
  std::string name;
  bool pass = false;
  std::string diagnostic;
};

ServerConfig selftest_config(const ServerConfig& base, bool lightweight_runtime = true) {
  ServerConfig cfg = base;
  cfg.port = 0;
  cfg.port_set = true;
  cfg.admission_active_cap = 4;
  cfg.admission_active_cap_set = true;
  cfg.admission_backlog_cap = kDefaultAdmissionBacklogCap;
  cfg.admission_backlog_cap_set = true;
  cfg.process_label.clear();
  cfg.selftest_ws_close_delay_ms = 0;
  cfg.selftest_lightweight_runtime = lightweight_runtime;
  validate_config(&cfg, true);
  return cfg;
}

SelftestResult run_case(int id, const std::string& name, const std::function<void(SelftestResult*)>& fn) {
  SelftestResult result;
  result.id = id;
  result.name = name;
  try {
    fn(&result);
    if (result.diagnostic.empty()) result.diagnostic = "ok";
  } catch (const std::exception& e) {
    result.pass = false;
    result.diagnostic = e.what();
  }
  std::cout << "SELFTEST " << id << " " << (result.pass ? "PASS" : "FAIL")
            << " - " << name << " - " << result.diagnostic << "\n";
  std::cout.flush();
  return result;
}

bool json_has_bool(const std::string& text, const std::string& key, bool expected) {
  nlohmann::json parsed = nlohmann::json::parse(text);
  return parsed.contains(key) && parsed[key].is_boolean() && parsed[key].get<bool>() == expected;
}

int run_selftest(const ServerConfig& parsed) {
  ServerConfig base = parsed;
  std::vector<SelftestResult> results;

  results.push_back(run_case(1, "Default env, valid artifacts", [&](SelftestResult* r) {
    ScopedEnv env;
    clear_selftest_env(&env);
    ServerConfig cfg = selftest_config(base, false);
    WsServer server(cfg);
    server.start();
    HttpClientResponse health = http_request(server.port(), "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n");
    server.stop();
    r->pass = health.status == 200 && json_has_bool(health.body, "model_loaded", true);
    r->diagnostic = "health_status=" + std::to_string(health.status);
  }));

  results.push_back(run_case(2, "NEMOTRON_STATS_ENABLED=0", [&](SelftestResult* r) {
    ScopedEnv env;
    clear_selftest_env(&env);
    env.set("NEMOTRON_STATS_ENABLED", "0");
    ServerConfig cfg = selftest_config(base);
    WsServer server(cfg);
    server.start();
    HttpClientResponse stats = http_request(server.port(), "GET /stats HTTP/1.1\r\nHost: localhost\r\n\r\n");
    server.stop();
    r->pass = stats.status == 200 && json_has_bool(stats.body, "enabled", false);
    r->diagnostic = "stats_status=" + std::to_string(stats.status) + " body=" + stats.body.substr(0, 64);
  }));

  results.push_back(run_case(3, "NEMOTRON_STATS_WINDOW=abc startup failure", [&](SelftestResult* r) {
    ScopedEnv env;
    clear_selftest_env(&env);
    env.set("NEMOTRON_STATS_WINDOW", "abc");
    ServerConfig cfg = base;
    cfg.port = 0;
    cfg.port_set = true;
    cfg.admission_active_cap = 4;
    cfg.admission_active_cap_set = true;
    try {
      validate_config(&cfg, true);
      WsServer server(cfg);
      server.start();
      server.stop();
      r->pass = false;
      r->diagnostic = "startup unexpectedly succeeded";
    } catch (const std::exception& e) {
      r->pass = std::string(e.what()).find("NEMOTRON_STATS_WINDOW") != std::string::npos;
      r->diagnostic = e.what();
    }
  }));

  results.push_back(run_case(4, "Scheduler ON missing MANIFEST startup failure", [&](SelftestResult* r) {
    ScopedEnv env;
    clear_selftest_env(&env);
    env.set("NEMOTRON_DENSITY_BATCH_STEADY", "1");
    fs::path tmp = fs::temp_directory_path() /
                   ("ws-server-missing-manifest-" + std::to_string(::getpid()));
    fs::create_directories(tmp);
    ServerConfig cfg = base;
    cfg.port = 0;
    cfg.port_set = true;
    cfg.admission_active_cap = 4;
    cfg.admission_active_cap_set = true;
    cfg.steady_batch_dir = tmp.string();
    cfg.steady_batch_dir_explicit = true;
    try {
      validate_config(&cfg, true);
      WsServer server(cfg);
      server.start();
      server.stop();
      r->pass = false;
      r->diagnostic = "startup unexpectedly succeeded";
    } catch (const std::exception& e) {
      r->pass = std::string(e.what()).find("MANIFEST") != std::string::npos;
      r->diagnostic = e.what();
    }
    fs::remove_all(tmp);
  }));

  results.push_back(run_case(5, "Scheduler ON valid artifacts", [&](SelftestResult* r) {
    ScopedEnv env;
    clear_selftest_env(&env);
    env.set("NEMOTRON_DENSITY_BATCH_STEADY", "1");
    ServerConfig cfg = selftest_config(base, false);
    WsServer server(cfg);
    server.start();
    HttpClientResponse telemetry =
        http_request(server.port(), "GET /scheduler_telemetry HTTP/1.1\r\nHost: localhost\r\n\r\n");
    server.stop();
    r->pass = telemetry.status == 200 && telemetry.body.find("\"counts\"") != std::string::npos;
    r->diagnostic = "scheduler_status=" + std::to_string(telemetry.status);
  }));

  results.push_back(run_case(6, "--port 0 auto-bind", [&](SelftestResult* r) {
    ScopedEnv env;
    clear_selftest_env(&env);
    ServerConfig cfg = selftest_config(base);
    WsServer server(cfg);
    server.start();
    int bound = server.port();
    HttpClientResponse health = http_request(bound, "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n");
    server.stop();
    r->pass = bound > 0 && health.status == 200;
    r->diagnostic = "bound_port=" + std::to_string(bound) + " health_status=" + std::to_string(health.status);
  }));

  results.push_back(run_case(7, "--admission-active-cap 0 startup failure", [&](SelftestResult* r) {
    ScopedEnv env;
    clear_selftest_env(&env);
    ServerConfig cfg = base;
    cfg.port = 0;
    cfg.port_set = true;
    cfg.admission_active_cap = 0;
    cfg.admission_active_cap_set = true;
    try {
      validate_config(&cfg, true);
      r->pass = false;
      r->diagnostic = "validation unexpectedly succeeded";
    } catch (const std::exception& e) {
      r->pass = std::string(e.what()).find("positive") != std::string::npos;
      r->diagnostic = e.what();
    }
  }));

  results.push_back(run_case(8, "Bound port health + stats + WS handshake-only", [&](SelftestResult* r) {
    ScopedEnv env;
    clear_selftest_env(&env);
    ServerConfig cfg = selftest_config(base);
    WsServer server(cfg);
    server.start();
    HttpClientResponse health = http_request(server.port(), "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n");
    HttpClientResponse stats = http_request(server.port(), "GET /stats?last=1 HTTP/1.1\r\nHost: localhost\r\n\r\n");
    int status = 0;
    UniqueFd ws = websocket_connect(server.port(), &status);
    ClientFrame ready = read_server_frame(ws.get());
    ClientFrame close = read_server_frame(ws.get());
    server.stop();
    std::string ready_payload(ready.payload.begin(), ready.payload.end());
    r->pass = health.status == 200 &&
              stats.status == 200 &&
              json_has_bool(stats.body, "enabled", true) &&
              status == 101 &&
              ready.opcode == static_cast<uint8_t>(ws_framing::Opcode::TEXT) &&
              ready_payload == "{\"type\":\"ready\"}" &&
              close.opcode == static_cast<uint8_t>(ws_framing::Opcode::CLOSE) &&
              close_code(close) == 1011 &&
              close_reason(close) == "server-not-ready";
    r->diagnostic = "health=" + std::to_string(health.status) +
                    " stats=" + std::to_string(stats.status) +
                    " ws_status=" + std::to_string(status) +
                    " close=" + std::to_string(close_code(close));
  }));

  results.push_back(run_case(9, "Cap=1, two WS connections, second post-handshake WS-1013", [&](SelftestResult* r) {
    ScopedEnv env;
    clear_selftest_env(&env);
    ServerConfig cfg = selftest_config(base);
    cfg.admission_active_cap = 1;
    cfg.admission_backlog_cap = 0;
    cfg.selftest_ws_close_delay_ms = 300;
    WsServer server(cfg);
    server.start();
    int first_status = 0;
    UniqueFd first = websocket_connect(server.port(), &first_status);
    ClientFrame first_ready = read_server_frame(first.get());
    int second_status = 0;
    UniqueFd second = websocket_connect(server.port(), &second_status);
    ClientFrame second_close = read_server_frame(second.get());
    ClientFrame first_close = read_server_frame(first.get());
    server.stop();
    r->pass = first_status == 101 &&
              first_ready.opcode == static_cast<uint8_t>(ws_framing::Opcode::TEXT) &&
              second_status == 101 &&
              second_close.opcode == static_cast<uint8_t>(ws_framing::Opcode::CLOSE) &&
              close_code(second_close) == 1013 &&
              close_reason(second_close) == "admission_backpressure" &&
              close_code(first_close) == 1011;
    r->diagnostic = "first_status=" + std::to_string(first_status) +
                    " second_status=" + std::to_string(second_status) +
                    " second_close=" + std::to_string(close_code(second_close)) +
                    " reason=" + close_reason(second_close);
  }));

  results.push_back(run_case(10, "Malformed first HTTP request line", [&](SelftestResult* r) {
    ScopedEnv env;
    clear_selftest_env(&env);
    ServerConfig cfg = selftest_config(base);
    WsServer server(cfg);
    server.start();
    HttpClientResponse bad = http_request(server.port(), "not http\r\n\r\n");
    HttpClientResponse health = http_request(server.port(), "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n");
    server.stop();
    r->pass = bad.status == 400 && health.status == 200;
    r->diagnostic = "bad_status=" + std::to_string(bad.status) +
                    " followup_health=" + std::to_string(health.status);
  }));

  results.push_back(run_case(11, "Oversize headers", [&](SelftestResult* r) {
    ScopedEnv env;
    clear_selftest_env(&env);
    ServerConfig cfg = selftest_config(base);
    WsServer server(cfg);
    server.start();
    std::string request = "GET /health HTTP/1.1\r\nHost: localhost\r\nX-Fill: " +
                          std::string(ws_handshake::kMaxHttpHeaderBytes + 100, 'a') + "\r\n\r\n";
    HttpClientResponse large = http_request(server.port(), request);
    HttpClientResponse health = http_request(server.port(), "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n");
    server.stop();
    r->pass = large.status == 431 && health.status == 200;
    r->diagnostic = "large_status=" + std::to_string(large.status) +
                    " followup_health=" + std::to_string(health.status);
  }));

  results.push_back(run_case(12, "Two ws_server instances on different ports both healthy", [&](SelftestResult* r) {
    ScopedEnv env;
    clear_selftest_env(&env);
    ServerConfig cfg_a = selftest_config(base);
    ServerConfig cfg_b = selftest_config(base);
    cfg_a.process_label = "selftest-a";
    cfg_b.process_label = "selftest-b";
    WsServer server_a(cfg_a);
    WsServer server_b(cfg_b);
    server_a.start();
    server_b.start();
    HttpClientResponse health_a =
        http_request(server_a.port(), "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n");
    HttpClientResponse health_b =
        http_request(server_b.port(), "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n");
    int port_a = server_a.port();
    int port_b = server_b.port();
    server_b.stop();
    server_a.stop();
    r->pass = port_a > 0 && port_b > 0 && port_a != port_b &&
              health_a.status == 200 && health_b.status == 200 &&
              health_a.body.find("\"process_label\":\"selftest-a\"") != std::string::npos &&
              health_b.body.find("\"process_label\":\"selftest-b\"") != std::string::npos;
    r->diagnostic = "port_a=" + std::to_string(port_a) +
                    " port_b=" + std::to_string(port_b) +
                    " health_a=" + std::to_string(health_a.status) +
                    " health_b=" + std::to_string(health_b.status);
  }));

  bool all_pass = std::all_of(results.begin(), results.end(), [](const SelftestResult& result) {
    return result.pass;
  });
  std::cout << "SELFTEST_SUMMARY pass=" << json_bool(all_pass)
            << " passed=" << std::count_if(results.begin(), results.end(), [](const SelftestResult& r) {
                 return r.pass;
               })
            << " total=" << results.size() << "\n";
  return all_pass ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    ServerConfig cfg = parse_args(argc, argv);
    if (cfg.selftest_and_exit) {
      return run_selftest(cfg);
    }

    validate_config(&cfg, true);
    if (cfg.print_config) {
      std::cout << config_table(cfg);
    }

    WsServer server(cfg);
    server.start();
    std::cout << "ws_server listening on 127.0.0.1:" << server.port() << "\n";
    std::cout.flush();

    for (;;) {
      std::this_thread::sleep_for(std::chrono::hours(24));
    }
  } catch (const std::exception& e) {
    std::cerr << "ws_server startup error: " << e.what() << "\n";
    return 1;
  }
}
