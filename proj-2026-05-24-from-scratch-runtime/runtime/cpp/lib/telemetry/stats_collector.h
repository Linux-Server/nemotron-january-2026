#pragma once

#include "session_timing.h"

#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>
#include <string>

class StatsCollector {
 public:
  explicit StatsCollector(size_t window_size = 2048, bool enabled = true);

  void record(SessionTiming timing);
  std::string snapshot_json(std::optional<size_t> last_n = std::nullopt) const;

  bool enabled() const { return enabled_; }
  size_t window_size() const { return window_size_; }

 private:
  struct Sample {
    double ts_unix = 0.0;
    SessionTiming timing;
  };

  bool enabled_ = true;
  size_t window_size_ = 2048;
  mutable std::mutex mutex_;
  std::deque<Sample> samples_;
  uint64_t lifetime_records_ = 0;
};
