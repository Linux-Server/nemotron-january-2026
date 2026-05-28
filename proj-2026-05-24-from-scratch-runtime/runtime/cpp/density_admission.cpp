#include "density_admission.h"

DensityAdmission::DensityAdmission(uint64_t active_cap, uint64_t backlog_cap)
    : active_cap_(active_cap), backlog_cap_(backlog_cap) {}

bool DensityAdmission::try_increment_below(std::atomic<uint64_t>& counter,
                                           uint64_t cap,
                                           uint64_t* value_after) {
  uint64_t current = counter.load(std::memory_order_acquire);
  while (current < cap) {
    if (counter.compare_exchange_weak(current,
                                      current + 1,
                                      std::memory_order_acq_rel,
                                      std::memory_order_acquire)) {
      if (value_after != nullptr) *value_after = current + 1;
      return true;
    }
  }
  return false;
}

void DensityAdmission::decrement_if_positive(std::atomic<uint64_t>& counter) {
  uint64_t current = counter.load(std::memory_order_acquire);
  while (current > 0) {
    if (counter.compare_exchange_weak(current,
                                      current - 1,
                                      std::memory_order_acq_rel,
                                      std::memory_order_acquire)) {
      return;
    }
  }
}

void DensityAdmission::update_peak(std::atomic<uint64_t>& peak, uint64_t value) {
  uint64_t current = peak.load(std::memory_order_acquire);
  while (value > current &&
         !peak.compare_exchange_weak(current,
                                     value,
                                     std::memory_order_acq_rel,
                                     std::memory_order_acquire)) {}
}

void DensityAdmission::remember_stream(const std::string& stream_id, StreamSlot slot) {
  if (stream_id.empty()) return;
  std::lock_guard<std::mutex> lock(streams_mutex_);
  streams_[stream_id] = slot;
}

AdmitResult DensityAdmission::try_admit(const std::string& stream_id) {
  offered_.fetch_add(1, std::memory_order_relaxed);

  uint64_t active_after = 0;
  if (try_increment_below(active_count_, active_cap_, &active_after)) {
    admitted_.fetch_add(1, std::memory_order_relaxed);
    update_peak(active_peak_, active_after);
    remember_stream(stream_id, StreamSlot::ACTIVE);
    return {AdmissionDecision::ADMITTED};
  }

  uint64_t backlog_after = 0;
  if (try_increment_below(backlog_count_, backlog_cap_, &backlog_after)) {
    admitted_.fetch_add(1, std::memory_order_relaxed);
    update_peak(backlog_peak_, backlog_after);
    remember_stream(stream_id, StreamSlot::BACKLOG);
    return {AdmissionDecision::QUEUED};
  }

  if (backlog_cap_ == 0) {
    active_cap_hits_.fetch_add(1, std::memory_order_relaxed);
    shed_close_count_.fetch_add(1, std::memory_order_relaxed);
    return {AdmissionDecision::SHED_ACTIVE_CAP};
  }

  backlog_cap_hits_.fetch_add(1, std::memory_order_relaxed);
  shed_close_count_.fetch_add(1, std::memory_order_relaxed);
  return {AdmissionDecision::SHED_BACKLOG_CAP};
}

bool DensityAdmission::try_admit_complete(const std::string& stream_id) {
  if (stream_id.empty()) return false;

  {
    std::lock_guard<std::mutex> lock(streams_mutex_);
    auto it = streams_.find(stream_id);
    if (it == streams_.end() || it->second != StreamSlot::BACKLOG) return false;
  }

  uint64_t active_after = 0;
  if (!try_increment_below(active_count_, active_cap_, &active_after)) return false;

  bool promoted = false;
  {
    std::lock_guard<std::mutex> lock(streams_mutex_);
    auto it = streams_.find(stream_id);
    if (it != streams_.end() && it->second == StreamSlot::BACKLOG) {
      it->second = StreamSlot::ACTIVE;
      promoted = true;
    }
  }
  if (!promoted) {
    decrement_if_positive(active_count_);
    return false;
  }

  decrement_if_positive(backlog_count_);
  update_peak(active_peak_, active_after);
  return true;
}

void DensityAdmission::on_admit_complete(const std::string& stream_id) {
  (void)try_admit_complete(stream_id);
}

void DensityAdmission::on_close(const std::string& stream_id) {
  if (!stream_id.empty()) {
    StreamSlot slot = StreamSlot::ACTIVE;
    bool found = false;
    {
      std::lock_guard<std::mutex> lock(streams_mutex_);
      auto it = streams_.find(stream_id);
      if (it != streams_.end()) {
        slot = it->second;
        streams_.erase(it);
        found = true;
      }
    }
    if (!found) return;
    if (slot == StreamSlot::ACTIVE) {
      decrement_if_positive(active_count_);
    } else {
      decrement_if_positive(backlog_count_);
    }
    return;
  }

  if (active_count_.load(std::memory_order_acquire) > 0) {
    decrement_if_positive(active_count_);
  } else {
    decrement_if_positive(backlog_count_);
  }
}

AdmissionTelemetry DensityAdmission::telemetry_snapshot() const {
  AdmissionTelemetry out;
  out.active_cap = active_cap_;
  out.backlog_cap = backlog_cap_;
  out.offered = offered_.load(std::memory_order_acquire);
  out.admitted = admitted_.load(std::memory_order_acquire);
  out.active_count = active_count_.load(std::memory_order_acquire);
  out.backlog_count = backlog_count_.load(std::memory_order_acquire);
  out.active_peak = active_peak_.load(std::memory_order_acquire);
  out.backlog_peak = backlog_peak_.load(std::memory_order_acquire);
  out.active_cap_hits = active_cap_hits_.load(std::memory_order_acquire);
  out.backlog_cap_hits = backlog_cap_hits_.load(std::memory_order_acquire);
  out.shed_close_count = shed_close_count_.load(std::memory_order_acquire);
  out.shed_close_rate = out.offered == 0
                            ? 0.0
                            : static_cast<double>(out.shed_close_count) /
                                  static_cast<double>(out.offered);
  return out;
}
