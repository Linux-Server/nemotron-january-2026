#!/usr/bin/env bash
# Step-0 clean-knee: measure ONE concurrency level N.
# Runs rt_loadgen for N over a steady window while concurrently capturing:
#   /scheduler_telemetry (before+after snapshots), /stats?last= (after), nvidia-smi dmon -s u @1Hz.
# Emits a JSON row + the loadgen table line. Telemetry source = HTTP curl to the ws_server port.
#
# usage: measure_one_n.sh <N> <PROCS> <WINDOW_S> <TAG>
set -uo pipefail
N="${1:?N}"; PROCS="${2:-32}"; WINDOW="${3:-45}"; TAG="${4:-r0}"
PORT="${PORT:-8080}"
RT="$HOME/density"
AUDIO="$RT/loadgen_audio_smoke"
OUTDIR="$RT/step0_out"
mkdir -p "$OUTDIR"
STAMP=$(date +%H%M%S)
BASE="$OUTDIR/N${N}_${TAG}_${STAMP}"

curl_telem() { curl -s --max-time 8 "http://127.0.0.1:$PORT/scheduler_telemetry"; }
curl_stats() { curl -s --max-time 8 "http://127.0.0.1:$PORT/stats?last=4096"; }

echo "===== N=$N procs=$PROCS window=${WINDOW}s tag=$TAG ====="
# settle
sleep 2
curl_telem > "$BASE.telem_before.json"

# dmon @1Hz in background for window+slack
DMON_SECS=$(python3 -c "print(int($WINDOW)+8)")
( nvidia-smi dmon -s u -d 1 -c "$DMON_SECS" > "$BASE.dmon.txt" 2>&1 ) &
DMON_PID=$!

# realtime loadgen for this single N
"$HOME/torch280-sm89-venv/bin/python" "$RT/rt_loadgen.py" \
  --url "ws://127.0.0.1:$PORT" --audio-dir "$AUDIO" \
  --sweep "$N" --procs "$PROCS" --window-s "$WINDOW" \
  --output "$BASE.loadgen.json" 2>&1 | tee "$BASE.loadgen.txt"

curl_telem > "$BASE.telem_after.json"
curl_stats > "$BASE.stats_after.json"
wait $DMON_PID 2>/dev/null || true

# Reduce dmon sm% (col 2) mean+max over the window (skip the 2 header lines).
python3 - "$BASE.dmon.txt" <<'PY'
import sys
sm=[]
with open(sys.argv[1]) as f:
    for line in f:
        line=line.strip()
        if not line or line.startswith('#'): continue
        parts=line.split()
        if len(parts)>=3 and parts[0].isdigit():
            try: sm.append(int(parts[1]))
            except: pass
if sm:
    print(f"NVML_sm_pct mean={sum(sm)/len(sm):.1f} max={max(sm)} n={len(sm)}")
else:
    print("NVML_sm_pct mean=nan max=nan n=0")
PY

# Pull the telemetry/stats fields of interest into a compact line.
python3 - "$BASE" <<'PY'
import json,sys
base=sys.argv[1]
def L(p):
    try:
        return json.load(open(p))
    except Exception as e:
        return {"_err":str(e)}
tb=L(base+".telem_before.json"); ta=L(base+".telem_after.json")
st=L(base+".stats_after.json"); lg=L(base+".loadgen.json")
def g(d,*ks,default=None):
    for k in ks:
        if not isinstance(d,dict): return default
        d=d.get(k,default)
    return d
# counts deltas
cb=g(tb,"counts",default={}) or {}; ca=g(ta,"counts",default={}) or {}
def dlt(k): return (ca.get(k,0) or 0)-(cb.get(k,0) or 0)
disp_cpu = g(ta,"dispatcher_cpu_pct")            # LIFETIME-cumulative ratio (diluted by idle history)
disp_util = g(ta,"dispatcher_stream_util_pct")   # LIFETIME-cumulative ratio
# Windowed dispatcher utilization from us-deltas (faithful per-window signal):
def df(k): return (g(ta,k) or 0)-(g(tb,k) or 0)
d_cpu_us=df("dispatcher_cpu_us"); d_wall_us=df("dispatcher_wall_us"); d_run_us=df("dispatcher_stream_run_us")
disp_cpu_win = (100.0*d_cpu_us/d_wall_us) if d_wall_us>0 else None
disp_util_win = (100.0*d_run_us/d_wall_us) if d_wall_us>0 else None
qd = g(ta,"queue_depth",default={}) or {}
backlog = dlt("backlog_gt_bmax")
exc = dlt("dispatcher_exceptions")
# stats finalize timing
m=g(st,"metrics",default={}) or {}
def q(name,pct):
    v=m.get(name,{})
    return v.get(pct) if isinstance(v,dict) else None
fw50=q("scheduler_future_wait_ms","p50"); fw95=q("scheduler_future_wait_ms","p95"); fw99=q("scheduler_future_wait_ms","p99")
ew50=q("scheduler_enqueue_wait_ms","p50"); ew95=q("scheduler_enqueue_wait_ms","p95"); ew99=q("scheduler_enqueue_wait_ms","p99")
# loadgen summary
N=lg.get("levels",[None])[0]
s=g(lg,"summaries",str(N),default={}) or {}
row={
 "N":N,"completed":s.get("completed"),"errors":s.get("errors"),
 "ttfs_p50":s.get("ttfs_p50"),"ttfs_p95":s.get("ttfs_p95"),"ttfs_p99":s.get("ttfs_p99"),"ttfs_max":s.get("ttfs_max"),
 "lag_p50":s.get("lag_p50"),"lag_p95":s.get("lag_p95"),"lag_p99":s.get("lag_p99"),"maxjit_ms":s.get("maxjit_ms"),
 "dispatcher_cpu_pct":disp_cpu,"dispatcher_stream_util_pct":disp_util,
 "dispatcher_cpu_pct_win":disp_cpu_win,"dispatcher_stream_util_pct_win":disp_util_win,
 "queue_depth_p50":qd.get("p50"),"queue_depth_p95":qd.get("p95"),"queue_depth_p99":qd.get("p99"),
 "backlog_gt_bmax_delta":backlog,"dispatcher_exceptions_delta":exc,
 "dispatch_cycles_delta":dlt("dispatch_cycles"),"B1_delta":dlt("B1"),"B4_delta":dlt("B4"),
 "future_wait_p50":fw50,"future_wait_p95":fw95,"future_wait_p99":fw99,
 "enqueue_wait_p50":ew50,"enqueue_wait_p95":ew95,"enqueue_wait_p99":ew99,
 "stats_samples":st.get("samples"),
}
open(base+".row.json","w").write(json.dumps(row,indent=2))
print("ROW "+json.dumps(row))
PY
echo "===== done N=$N tag=$TAG (artifacts: $BASE.*) ====="
