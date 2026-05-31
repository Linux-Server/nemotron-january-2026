#!/usr/bin/env bash
# Step-0 COLD baseline for the cold-start Phase-2 plan (proj-2026-05-30-2202).
# Runs a 4-cell matrix on the LOCAL RTX 5090: {cold-artifacts, warm} x {bg-warmup OFF, ON}.
# COLD method: per-file posix_fadvise(POSIX_FADV_DONTNEED) eviction of the artifact tree
#   (non-root; targets the dominant ~7.5GB of 2.48GB blobs). NOT a global drop_caches
#   (that needs sudo + also evicts torch libs — noted as a methodology divergence in baseline.md).
# Captures: COLD_START_PHASE breakdown, sync_warm_done/background_warm_complete, time-to-listening,
#   GPU + host(RSS) peak memory, /tmp growth. No code change; emits raw logs + a summary.
set -uo pipefail
RT="/home/khkramer/src/nemotron-january-2026/proj-2026-05-24-from-scratch-runtime/runtime"
OUT="/home/khkramer/src/nemotron-january-2026/proj-2026-05-30-2202/baseline_runs"
BIN="$RT/cpp/build_step10/ws_server"
TORCH_LIB=$(ls -d "$RT"/.venv/lib/python*/site-packages/torch/lib 2>/dev/null | head -1)
REPO="/home/khkramer/src/nemotron-january-2026"
PORT=8091
CAP=64
mkdir -p "$OUT"

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

run_one() {  # $1=label  $2=bgwarmup(0/1)  $3=cache(cold/warm)
  local label="$1" bg="$2" cache="$3"
  local log="$OUT/${label}.srvlog"
  echo "============================================================"
  echo "[cell] $label  bg_warmup=$bg  cache=$cache"
  # ---- cache state
  if [ "$cache" = cold ]; then evict_artifacts; else echo "  warming artifact cache..."; warm_artifacts; fi
  sync
  local tmp_before; tmp_before=$(du -sm /tmp 2>/dev/null | awk '{print $1}')
  # ---- launch under env, tee stdout
  ( cd "$RT" && exec env \
      HF_HUB_OFFLINE=1 "PYTHONPATH=$REPO/src" "LD_LIBRARY_PATH=$TORCH_LIB:${LD_LIBRARY_PATH:-}" \
      NEMOTRON_CONTINUOUS=1 NEMOTRON_FINALIZE_SILENCE_MS=0 "NEMOTRON_ARTIFACT_DIR=$RT/artifacts" \
      NEMOTRON_WS_SCHEDULER=1 NEMOTRON_DENSITY_BATCH_STEADY=1 NEMOTRON_DENSITY_BATCH_MAX=4 \
      NEMOTRON_DENSITY_BATCH_WINDOW_MS=10 NEMOTRON_DENSITY_BATCH_LONE_TIMEOUT_MS=0 \
      "NEMOTRON_DENSITY_ADMISSION_ACTIVE_CAP=$CAP" "NEMOTRON_WS_LANES=$CAP" \
      NEMOTRON_WS_FINALIZE_RUNNERS=2 NEMOTRON_DENSITY_FINALIZE_RUNNERS=2 \
      "NEMOTRON_WS_BACKGROUND_WARMUP=$bg" \
      "$BIN" --port "$PORT" --admission-active-cap "$CAP" --steady-batch-dir "$RT/steady_b_artifacts" \
  ) > "$log" 2>&1 &
  local pid=$!
  echo "  pid=$pid log=$log"
  # ---- peak sampler (GPU mem of this pid + host RSS)
  local peak_gpu=0 peak_rss=0
  local deadline=$(( SECONDS + 400 ))
  while kill -0 "$pid" 2>/dev/null; do
    local rss; rss=$(ps -o rss= -p "$pid" 2>/dev/null | tr -d ' '); [ -n "$rss" ] && [ "$rss" -gt "$peak_rss" ] && peak_rss=$rss
    local g; g=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | awk -F', ' -v p="$pid" '$1==p{print $2}'); [ -n "$g" ] && [ "$g" -gt "$peak_gpu" ] && peak_gpu=$g
    # ready when serving + (sync: sync_warm_done | bg: background_warm_complete) present
    if grep -q "listening on" "$log" 2>/dev/null; then
      if [ "$bg" = 0 ] && grep -q "sync_warm_done" "$log"; then sleep 2; break; fi
      if [ "$bg" = 1 ] && grep -q "background_warm_complete" "$log"; then sleep 2; break; fi
    fi
    [ "$SECONDS" -gt "$deadline" ] && { echo "  TIMEOUT 400s"; break; }
    sleep 1
  done
  local tmp_after; tmp_after=$(du -sm /tmp 2>/dev/null | awk '{print $1}')
  echo "  peak_gpu_mib=$peak_gpu peak_host_rss_mib=$(( peak_rss/1024 )) tmp_delta_mb=$(( tmp_after - tmp_before ))" | tee -a "$OUT/summary.txt"
  # ---- record key phase lines
  { echo "### $label (bg=$bg cache=$cache)";
    grep -E "COLD_START_PHASE|listening on|sync_warm_done|background_warm_complete|warmup_complete" "$log" | tail -40;
    echo "  >> peak_gpu_mib=$peak_gpu peak_host_rss_mib=$(( peak_rss/1024 )) tmp_delta_mb=$(( tmp_after - tmp_before ))"; echo; } >> "$OUT/phases.txt"
  # ---- clean shutdown (graceful so AOTI /tmp dirs clean) + wait GPU free
  kill -INT "$pid" 2>/dev/null; for i in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill -KILL "$pid" 2>/dev/null; sleep 3
  # reclaim any leaked AOTI extraction dirs we own
  find /tmp -maxdepth 1 -type d -regextype posix-extended -regex '/tmp/[A-Za-z0-9]{6}' -user "$(id -un)" -exec rm -rf {} + 2>/dev/null
  echo "  [cell done] $label"
}

echo "=== artifact sizes ===" | tee "$OUT/summary.txt"
ls -la "$RT"/artifacts/enc_first.ts "$RT"/artifacts/enc_steady_aoti.pt2 "$RT"/artifacts/finalize_shared_weights.* "$RT"/steady_b_artifacts/*.pt2 2>/dev/null | awk '{print $5, $NF}' | tee -a "$OUT/summary.txt"
: > "$OUT/phases.txt"
run_one cold_off 0 cold
run_one warm_off 0 warm
run_one cold_on  1 cold
run_one warm_on  1 warm
echo "=== DONE — see $OUT/{summary.txt,phases.txt,*.srvlog} ==="
