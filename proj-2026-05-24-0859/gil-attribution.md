# GIL Attribution Probe

Date: 2026-05-24

Scope: probe-only logging add in `src/nemotron_speech/server.py`, gated by `NEMOTRON_GIL_ATTRIB=1` and default off.

## Logging Add

- Flag: `NEMOTRON_GIL_ATTRIB=1`; unset is a no-op.
- Emission: one `gil_attribution_record` JSON log at shutdown, using the existing `_continuous_finalize_timing` / scheduler batch/finalize telemetry surface.
- Buckets: `decode`, `dispatch`, and `glue`; glue is split into `scatter_gather`, `host_sync`, `inference_lock_wait`, and residual `scheduling_socket_io`.
- Sanity: bucket sum delta is emitted and was `0.0ms` at p50/p95/p99 for chunk and finalize.
- GIL wait: `py-spy --gil` attach failed with ptrace permission denied, so this run used the built-in asyncio event-loop tick-lag proxy.

## Run

Server:

```bash
NEMOTRON_CONTINUOUS=1
NEMOTRON_FINALIZE_SILENCE_MS=0
NEMOTRON_WARMUP_MS=200
NEMOTRON_SCHEDULER_B1=1
NEMOTRON_BATCH_SCHED=1
NEMOTRON_BATCH_BARRIER_DRAIN=1
NEMOTRON_BATCH_FINALIZE=1
NEMOTRON_BATCH_FINALIZE_PREPROC=1
NEMOTRON_MODEL_LANES=2
NEMOTRON_ENCODER_CUDAGRAPH=1
NEMOTRON_ENCODER_CUDAGRAPH_MAX_B=8
NEMOTRON_ENCODER_CUDAGRAPH_FINALIZE=1
NEMOTRON_ENCODER_CUDAGRAPH_FINALIZE_PADDED=1
NEMOTRON_SYNC_COMPRESS=1
NEMOTRON_FINALIZE_PRIORITY=1
NEMOTRON_GIL_ATTRIB=1
```

Loadgen: single process, 16 concurrent streams, 3 rounds, `proj-2026-05-20-modal-cost/loadgen_audio`.

Result: `ok=48`, `errors=0`, `TTFS p50/p95=12/21ms`, `proc-lag p50/p95=32/41ms`, keep-up `YES`.

## JSON Record

```json
{
  "schema": "nemotron_gil_attribution_v1",
  "reason": "shutdown",
  "config": {
    "model_lanes": 2,
    "scheduler_enabled": true,
    "batch_enabled": true,
    "batch_finalize": true,
    "batch_finalize_preproc": true,
    "encoder_cudagraph": true,
    "encoder_cudagraph_finalize": true,
    "sync_compress": true,
    "finalize_priority": true
  },
  "gil_wait_proxy": {
    "source": "asyncio_event_loop_tick_lag",
    "interval_ms": 5.0,
    "samples": 11655,
    "lag_ms": {"p50": 0.528, "p95": 1.041, "p99": 1.321}
  },
  "operations": {
    "chunk": {
      "samples": 3139,
      "batch_size_hist": {"1": 3069, "2": 69, "3": 1},
      "thread_busy_ms": {"p50": 10.426, "p95": 13.575, "p99": 15.026},
      "bucket_sum_delta_ms": {"p50": 0.0, "p95": 0.0, "p99": 0.0},
      "gpu_idle_pct_while_thread_busy": {"p50": 17.243, "p95": 22.114, "p99": 25.977},
      "buckets": {
        "decode": {
          "ms": {"p50": 8.175, "p95": 10.492, "p99": 12.015},
          "pct_thread_busy": {"p50": 78.603, "p95": 82.209, "p99": 83.707}
        },
        "dispatch": {
          "ms": {"p50": 0.038, "p95": 0.090, "p99": 0.199},
          "pct_thread_busy": {"p50": 0.372, "p95": 0.795, "p99": 1.743}
        },
        "glue_total": {
          "ms": {"p50": 2.192, "p95": 3.370, "p99": 4.276},
          "pct_thread_busy": {"p50": 20.959, "p95": 27.281, "p99": 31.691}
        },
        "glue_scatter_gather": {
          "ms": {"p50": 0.140, "p95": 0.299, "p99": 1.035},
          "pct_thread_busy": {"p50": 1.388, "p95": 2.572, "p99": 8.079}
        },
        "glue_host_sync": {
          "ms": {"p50": 0.011, "p95": 0.092, "p99": 0.192},
          "pct_thread_busy": {"p50": 0.102, "p95": 0.868, "p99": 1.729}
        },
        "glue_inference_lock_wait": {
          "ms": {"p50": 0.013, "p95": 0.029, "p99": 0.274},
          "pct_thread_busy": {"p50": 0.120, "p95": 0.265, "p99": 2.464}
        },
        "glue_scheduling_socket_io": {
          "ms": {"p50": 1.993, "p95": 2.972, "p99": 3.685},
          "pct_thread_busy": {"p50": 19.025, "p95": 24.587, "p99": 28.612}
        }
      }
    },
    "finalize": {
      "samples": 103,
      "batch_size_hist": {"1": 103},
      "thread_busy_ms": {"p50": 11.189, "p95": 21.251, "p99": 22.866},
      "bucket_sum_delta_ms": {"p50": 0.0, "p95": 0.0, "p99": 0.0},
      "gpu_idle_pct_while_thread_busy": {"p50": 5.608, "p95": 51.138, "p99": 55.669},
      "buckets": {
        "decode": {
          "ms": {"p50": 9.327, "p95": 10.547, "p99": 11.254},
          "pct_thread_busy": {"p50": 85.026, "p95": 87.939, "p99": 88.225}
        },
        "dispatch": {
          "ms": {"p50": 0.063, "p95": 0.082, "p99": 0.112},
          "pct_thread_busy": {"p50": 0.495, "p95": 0.657, "p99": 1.003}
        },
        "glue_total": {
          "ms": {"p50": 1.595, "p95": 11.501, "p99": 13.570},
          "pct_thread_busy": {"p50": 14.409, "p95": 55.149, "p99": 59.229}
        },
        "glue_scatter_gather": {
          "ms": {"p50": 0.0, "p95": 0.183, "p99": 0.204},
          "pct_thread_busy": {"p50": 0.0, "p95": 1.617, "p99": 1.745}
        },
        "glue_host_sync": {
          "ms": {"p50": 0.003, "p95": 0.007, "p99": 0.015},
          "pct_thread_busy": {"p50": 0.025, "p95": 0.055, "p99": 0.135}
        },
        "glue_inference_lock_wait": {
          "ms": {"p50": 0.030, "p95": 10.296, "p99": 12.323},
          "pct_thread_busy": {"p50": 0.267, "p95": 49.116, "p99": 53.974}
        },
        "glue_scheduling_socket_io": {
          "ms": {"p50": 1.319, "p95": 2.300, "p99": 2.817},
          "pct_thread_busy": {"p50": 12.136, "p95": 16.488, "p99": 21.328}
        }
      }
    }
  }
}
```

## Verdict

CONJUNCT 2 SURVIVES, with medium confidence because `py-spy --gil` was blocked and the fallback event-loop lag proxy did not show large coroutine starvation (`p99=1.32ms`).

The loaded steady chunk path is decode-bound, not dispatch/glue-bound: decode is `8.18/10.49/12.02ms p50/p95/p99`, or `78.6/82.2/83.7%` of thread-busy time. Dispatch is effectively gone (`0.04/0.09/0.20ms`) and host sync is small (`0.01/0.09/0.19ms`). Finalize p50 is also decode-bound (`85.0%` decode), while finalize p95/p99 are dominated by pinned-lane/inference-lock waiting, not encoder dispatch.

Decision for `proj-2026-05-24-from-scratch-runtime/0.1`: keep the native dispatch + on-GPU decode direction. This record does not support a STOP verdict based on MPS/context-launch/bandwidth dominance. Re-check GIL wait with ptrace-enabled `py-spy --gil` on the cloud/WAN box before treating the GIL-starvation part as fully proven.

## Blockers

- `py-spy record --gil --threads -p <pid>` failed locally: `Permission Denied`.
- The GPU-idle number is CUDA-event elapsed around the stream work versus thread-busy wall; it is a cheap proxy, not SM active utilization.
