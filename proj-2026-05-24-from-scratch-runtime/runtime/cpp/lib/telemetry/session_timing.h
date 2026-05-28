#pragma once

#include <cstdint>
#include <optional>

struct SessionTiming {
  std::optional<double> vad_stop_to_sent_ms;
  std::optional<double> fork_flush_wall_ms;
  std::optional<double> vad_stop_recv_to_process_ms;
  std::optional<double> lock_wait_ms;
  std::optional<double> vad_stop_to_finalize_start_ms;

  uint64_t finalize_seq = 0;
  int active_sessions_at_emit = 0;
};
