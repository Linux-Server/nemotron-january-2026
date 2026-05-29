#pragma once

#include "lib/runtime_io/json.hpp"

#include <cstdint>
#include <optional>
#include <string>

struct SessionTiming {
  std::optional<std::string> reason;
  std::optional<double> vad_stop_ts;
  std::optional<double> vad_stop_recv_ts;
  std::optional<double> debounce_expiry_ts;
  std::optional<double> fork_flush_start_ts;
  std::optional<double> fork_flush_done_ts;
  std::optional<double> final_sent_ts;
  std::optional<double> inference_lock_acquire_wait_ms;
  bool gil_attrib_enabled = false;

  uint64_t finalize_seq = 0;
  int active_sessions_at_emit = 0;
  bool was_suppressed = false;
  std::optional<std::string> close_reason;

  nlohmann::json to_wire_json() const {
    nlohmann::json out;
    out["reason"] = reason.has_value() ? nlohmann::json(*reason) : nlohmann::json(nullptr);
    out["vad_stop"] = vad_stop_ts.has_value() ? nlohmann::json(*vad_stop_ts) : nlohmann::json(nullptr);
    out["vad_stop_recv"] = vad_stop_recv_ts.has_value() ? nlohmann::json(*vad_stop_recv_ts) : nlohmann::json(nullptr);
    out["debounce_expiry"] = debounce_expiry_ts.has_value() ? nlohmann::json(*debounce_expiry_ts) : nlohmann::json(nullptr);
    out["fork_flush_start"] = fork_flush_start_ts.has_value() ? nlohmann::json(*fork_flush_start_ts) : nlohmann::json(nullptr);
    out["fork_flush_done"] = fork_flush_done_ts.has_value() ? nlohmann::json(*fork_flush_done_ts) : nlohmann::json(nullptr);
    out["final_sent"] = final_sent_ts.has_value() ? nlohmann::json(*final_sent_ts) : nlohmann::json(nullptr);
    out["inference_lock_acquire_wait_ms"] = inference_lock_acquire_wait_ms.has_value()
                                                ? nlohmann::json(*inference_lock_acquire_wait_ms)
                                                : nlohmann::json(nullptr);
    out["gil_attrib_enabled"] = gil_attrib_enabled;
    return out;
  }
};
