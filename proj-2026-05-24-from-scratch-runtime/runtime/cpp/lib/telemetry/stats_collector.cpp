#include "stats_collector.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

size_t read_env_size_t(const char* name, size_t fallback) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') return fallback;
  errno = 0;
  char* end = nullptr;
  unsigned long long value = std::strtoull(raw, &end, 10);
  if (errno != 0 || end == raw || *end != '\0' || value == 0 ||
      value > static_cast<unsigned long long>(std::numeric_limits<size_t>::max())) {
    throw std::runtime_error(std::string("invalid positive integer env var ") + name + "=" + raw);
  }
  return static_cast<size_t>(value);
}

bool read_env_enabled(const char* name, bool fallback) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') return fallback;
  return std::string(raw) != "0";
}

double unix_now_seconds() {
  using namespace std::chrono;
  return duration<double>(system_clock::now().time_since_epoch()).count();
}

void append_optional_number(std::ostringstream& oss, std::optional<double> value) {
  if (value.has_value()) {
    oss << *value;
  } else {
    oss << "null";
  }
}

std::string quantile_summary_json(std::vector<double> values) {
  std::ostringstream oss;
  oss << std::setprecision(17);
  if (values.empty()) {
    return "{\"p50\":null,\"p90\":null,\"p95\":null,\"p99\":null,\"max\":null,\"count\":0}";
  }
  std::sort(values.begin(), values.end());
  const size_t n = values.size();
  auto percentile = [&](double p) {
    size_t idx = static_cast<size_t>(std::llround(p * static_cast<double>(n - 1)));
    if (idx >= n) idx = n - 1;
    return values[idx];
  };
  oss << "{\"p50\":" << percentile(0.50)
      << ",\"p90\":" << percentile(0.90)
      << ",\"p95\":" << percentile(0.95)
      << ",\"p99\":" << percentile(0.99)
      << ",\"max\":" << values.back()
      << ",\"count\":" << n
      << "}";
  return oss.str();
}

}  // namespace

StatsCollector::StatsCollector(size_t window_size, bool enabled)
    : enabled_(read_env_enabled("NEMOTRON_STATS_ENABLED", enabled)),
      window_size_(read_env_size_t("NEMOTRON_STATS_WINDOW", window_size)) {
  if (window_size_ == 0) {
    throw std::runtime_error("StatsCollector window_size must be > 0");
  }
  if (!enabled) enabled_ = false;
}

void StatsCollector::record(SessionTiming timing) {
  if (!enabled_) return;
  std::lock_guard<std::mutex> lock(mutex_);
  samples_.push_back({unix_now_seconds(), std::move(timing)});
  while (samples_.size() > window_size_) samples_.pop_front();
  ++lifetime_records_;
}

std::string StatsCollector::snapshot_json(std::optional<size_t> last_n) const {
  std::vector<Sample> samples;
  uint64_t lifetime_records = 0;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    lifetime_records = lifetime_records_;
    size_t start = 0;
    if (last_n.has_value() && *last_n > 0 && *last_n < samples_.size()) {
      start = samples_.size() - *last_n;
    }
    samples.assign(samples_.begin() + static_cast<std::ptrdiff_t>(start), samples_.end());
  }

  std::vector<double> vad_stop_to_sent;
  std::vector<double> fork_flush_wall;
  std::vector<double> vad_stop_recv_to_process;
  std::vector<double> lock_wait;
  std::vector<double> vad_stop_to_finalize_start;
  std::vector<double> active_sessions;
  vad_stop_to_sent.reserve(samples.size());
  fork_flush_wall.reserve(samples.size());
  vad_stop_recv_to_process.reserve(samples.size());
  lock_wait.reserve(samples.size());
  vad_stop_to_finalize_start.reserve(samples.size());
  active_sessions.reserve(samples.size());

  for (const auto& sample : samples) {
    const auto& timing = sample.timing;
    if (timing.vad_stop_to_sent_ms) vad_stop_to_sent.push_back(*timing.vad_stop_to_sent_ms);
    if (timing.fork_flush_wall_ms) fork_flush_wall.push_back(*timing.fork_flush_wall_ms);
    if (timing.vad_stop_recv_to_process_ms) {
      vad_stop_recv_to_process.push_back(*timing.vad_stop_recv_to_process_ms);
    }
    if (timing.lock_wait_ms) lock_wait.push_back(*timing.lock_wait_ms);
    if (timing.vad_stop_to_finalize_start_ms) {
      vad_stop_to_finalize_start.push_back(*timing.vad_stop_to_finalize_start_ms);
    }
    active_sessions.push_back(static_cast<double>(timing.active_sessions_at_emit));
  }

  std::optional<double> since;
  std::optional<double> until;
  if (!samples.empty()) {
    since = samples.front().ts_unix;
    until = samples.back().ts_unix;
  }

  std::ostringstream oss;
  oss << std::boolalpha << std::setprecision(17);
  oss << "{\"enabled\":" << (enabled_ ? "true" : "false")
      << ",\"window_size\":" << window_size_
      << ",\"samples\":" << samples.size()
      << ",\"since_unix\":";
  append_optional_number(oss, since);
  oss << ",\"until_unix\":";
  append_optional_number(oss, until);
  oss << ",\"emitted_in_window\":" << samples.size()
      << ",\"suppressed_in_window\":0"
      << ",\"lifetime_emitted\":" << lifetime_records
      << ",\"lifetime_suppressed\":0"
      << ",\"metrics\":{"
      << "\"vad_stop_to_sent_ms\":" << quantile_summary_json(std::move(vad_stop_to_sent))
      << ",\"fork_flush_wall_ms\":" << quantile_summary_json(std::move(fork_flush_wall))
      << ",\"vad_stop_recv_to_process_ms\":" << quantile_summary_json(std::move(vad_stop_recv_to_process))
      << ",\"lock_wait_ms\":" << quantile_summary_json(std::move(lock_wait))
      << ",\"vad_stop_to_finalize_start_ms\":" << quantile_summary_json(std::move(vad_stop_to_finalize_start))
      << "},\"active_sessions_at_emit\":" << quantile_summary_json(std::move(active_sessions))
      << "}";
  return oss.str();
}
