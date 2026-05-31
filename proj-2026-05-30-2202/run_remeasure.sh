#!/usr/bin/env bash
# Step-6 cold-boot remeasure after Steps 2-5 unify.
# Runs deploy-mode bg-warmup ON for both first-encoder modes:
#   {NEMOTRON_WS_ENC_FIRST_TS=1,0} x {cold-artifacts,warm}
#
# Method intentionally matches run_cold_baseline.sh:
# - COLD = per-file posix_fadvise(POSIX_FADV_DONTNEED) eviction of artifacts/ + steady_b_artifacts/
# - WARM = cat artifacts/ + steady_b_artifacts/ into page cache
# - Same ws_server launch env, same per-pid GPU/RSS peak sampler
# - Graceful shutdown, then reclaim owned /tmp AOTI extraction dirs
set -uo pipefail

RT="/home/khkramer/src/nemotron-january-2026/proj-2026-05-24-from-scratch-runtime/runtime"
NOTES="/home/khkramer/src/nemotron-january-2026/proj-2026-05-30-2202"
OUT="$NOTES/remeasure_runs"
BIN="$RT/cpp/build_step10/ws_server"
TORCH_LIB=$(ls -d "$RT"/.venv/lib/python*/site-packages/torch/lib 2>/dev/null | head -1)
REPO="/home/khkramer/src/nemotron-january-2026"
PORT="${PORT:-8091}"
CAP="${CAP:-64}"
mkdir -p "$OUT"

SUMMARY="$OUT/summary.tsv"
PHASES="$OUT/phases.txt"

evict_artifacts() {  # non-root page-cache eviction of the artifact tree
  python3 - "$RT/artifacts" "$RT/steady_b_artifacts" <<'PY'
import os, sys, glob
total=0
for d in sys.argv[1:]:
    for f in glob.glob(os.path.join(d,'*')):
        if not os.path.isfile(f): continue
        try:
            fd=os.open(f, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd); total+=os.path.getsize(f)
        except Exception as e:
            print(f"  fadvise-skip {f}: {e}")
print(f"  evicted ~{total/1e9:.2f} GB from page cache (artifacts+steady_b)")
PY
}

warm_artifacts() {  # pull the artifact tree INTO page cache (for the WARM cells)
  cat "$RT"/artifacts/* "$RT"/steady_b_artifacts/* >/dev/null 2>&1 || true
}

phase_elapsed() {
  local log="$1" phase="$2"
  grep -E "COLD_START_PHASE phase=${phase} " "$log" 2>/dev/null | tail -1 \
    | sed -E 's/.*elapsed_ms=([0-9.]+).*/\1/'
}

phase_cumulative() {
  local log="$1" phase="$2"
  grep -E "COLD_START_PHASE phase=${phase} " "$log" 2>/dev/null | tail -1 \
    | sed -E 's/.*cumulative_ms=([0-9.]+).*/\1/'
}

run_one() {  # $1=label  $2=enc_first_ts(0/1)  $3=cache(cold/warm)
  local label="$1" enc_first_ts="$2" cache="$3" bg=1
  local log="$OUT/${label}.srvlog"
  echo "============================================================"
  echo "[cell] $label  enc_first_ts=$enc_first_ts  bg_warmup=$bg  cache=$cache"
  if [ "$cache" = cold ]; then evict_artifacts; else echo "  warming artifact cache..."; warm_artifacts; fi
  sync
  local tmp_before; tmp_before=$(du -sm /tmp 2>/dev/null | awk '{print $1}')

  ( cd "$RT" && exec env \
      HF_HUB_OFFLINE=1 "PYTHONPATH=$REPO/src" "LD_LIBRARY_PATH=$TORCH_LIB:${LD_LIBRARY_PATH:-}" \
      NEMOTRON_CONTINUOUS=1 NEMOTRON_FINALIZE_SILENCE_MS=0 "NEMOTRON_ARTIFACT_DIR=$RT/artifacts" \
      NEMOTRON_WS_SCHEDULER=1 NEMOTRON_DENSITY_BATCH_STEADY=1 NEMOTRON_DENSITY_BATCH_MAX=4 \
      NEMOTRON_DENSITY_BATCH_WINDOW_MS=10 NEMOTRON_DENSITY_BATCH_LONE_TIMEOUT_MS=0 \
      "NEMOTRON_DENSITY_ADMISSION_ACTIVE_CAP=$CAP" "NEMOTRON_WS_LANES=$CAP" \
      NEMOTRON_WS_FINALIZE_RUNNERS=2 NEMOTRON_DENSITY_FINALIZE_RUNNERS=2 \
      "NEMOTRON_WS_BACKGROUND_WARMUP=$bg" "NEMOTRON_WS_ENC_FIRST_TS=$enc_first_ts" \
      "$BIN" --port "$PORT" --admission-active-cap "$CAP" --steady-batch-dir "$RT/steady_b_artifacts" \
  ) > "$log" 2>&1 &
  local pid=$!
  echo "  pid=$pid log=$log"

  local peak_gpu=0 peak_rss=0
  local deadline=$(( SECONDS + 400 ))
  while kill -0 "$pid" 2>/dev/null; do
    local rss
    rss=$(ps -o rss= -p "$pid" 2>/dev/null | tr -d ' ')
    [ -n "$rss" ] && [ "$rss" -gt "$peak_rss" ] && peak_rss=$rss
    local g
    g=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
      | awk -F', ' -v p="$pid" '$1==p{print $2}')
    [ -n "$g" ] && [ "$g" -gt "$peak_gpu" ] && peak_gpu=$g
    if grep -q "listening on" "$log" 2>/dev/null && grep -q "background_warm_complete" "$log"; then
      sleep 2
      break
    fi
    [ "$SECONDS" -gt "$deadline" ] && { echo "  TIMEOUT 400s"; break; }
    sleep 1
  done

  local tmp_after; tmp_after=$(du -sm /tmp 2>/dev/null | awk '{print $1}')
  local peak_rss_mib=$(( peak_rss/1024 ))
  local tmp_delta_mb=$(( tmp_after - tmp_before ))
  local listen_ms bg_complete_ms bg_elapsed_ms enc_first_ms shared_ms enc_steady_ms
  listen_ms=$(phase_cumulative "$log" sync_warm_done)
  [ -z "$listen_ms" ] && listen_ms=$(phase_cumulative "$log" scheduler_warmup_start)
  bg_complete_ms=$(phase_cumulative "$log" background_warm_complete)
  bg_elapsed_ms=$(phase_elapsed "$log" background_warm_complete)
  enc_first_ms=$(phase_elapsed "$log" enc_first_load)
  shared_ms=$(phase_elapsed "$log" shared_encoder_constants_load)
  enc_steady_ms=$(phase_elapsed "$log" enc_steady_load)
  [ -z "$enc_steady_ms" ] && enc_steady_ms="ABSENT"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%d\t%d\t%d\n" \
    "$label" "$enc_first_ts" "$cache" "$bg" "${listen_ms:-NA}" "${bg_complete_ms:-NA}" \
    "${bg_elapsed_ms:-NA}" "${enc_first_ms:-NA}" "${shared_ms:-NA}" "$enc_steady_ms" \
    "$peak_gpu" "$peak_rss_mib" "$tmp_delta_mb" >> "$SUMMARY"

  {
    echo "### $label (enc_first_ts=$enc_first_ts bg=$bg cache=$cache)"
    grep -E "COLD_START_PHASE|shared encoder constants loaded|shared runtime first encoder ready|inference lane pool built|listening on|sync_warm_done|background_warm_complete|warmup_complete" "$log"
    echo "  >> peak_gpu_mib=$peak_gpu peak_host_rss_mib=$peak_rss_mib tmp_delta_mb=$tmp_delta_mb"
    echo
  } >> "$PHASES"

  kill -INT "$pid" 2>/dev/null
  for _ in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill -KILL "$pid" 2>/dev/null
  sleep 3
  find /tmp -maxdepth 1 -type d -regextype posix-extended -regex '/tmp/[A-Za-z0-9]{6}' -user "$(id -un)" -exec rm -rf {} + 2>/dev/null
  echo "  peak_gpu_mib=$peak_gpu peak_host_rss_mib=$peak_rss_mib tmp_delta_mb=$tmp_delta_mb"
  echo "  [cell done] $label"
}

echo "=== artifact sizes ===" | tee "$OUT/artifacts.txt"
ls -la "$RT"/artifacts/enc_first.ts "$RT"/artifacts/enc_first_aoti.pt2 \
  "$RT"/artifacts/enc_steady_aoti.pt2 "$RT"/artifacts/finalize_shared_weights.* \
  "$RT"/steady_b_artifacts/*.pt2 2>/dev/null | awk '{print $5, $NF}' | tee -a "$OUT/artifacts.txt"
printf "label\tenc_first_ts\tcache\tbg\tlisten_ms\tbackground_complete_ms\tbackground_elapsed_ms\tenc_first_load_ms\tshared_encoder_constants_load_ms\tenc_steady_load_ms\tpeak_gpu_mib\tpeak_host_rss_mib\ttmp_delta_mb\n" > "$SUMMARY"
: > "$PHASES"

run_one encfirst_ts_cold_on 1 cold
run_one encfirst_ts_warm_on 1 warm
run_one encfirst_aoti_cold_on 0 cold
run_one encfirst_aoti_warm_on 0 warm

echo "=== DONE — see $OUT/{summary.tsv,phases.txt,artifacts.txt,*.srvlog} ==="
