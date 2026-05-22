#!/usr/bin/env bash
# Step 4 hard gate: byte-exact CUDA-graph encoder at scale, local.
# 4 configs for clean attribution (cudagraph isolated from lanes):
#   cg_off    : graph OFF, lanes=1   (the baseline arm)
#   cg_on_l1  : graph ON,  lanes=1   (cudagraph effect, single lane)
#   cg_off_l2 : graph OFF, lanes=2   (lanes control)
#   cg_on_l2  : graph ON,  lanes=2   (cudagraph effect under per-lane streams)
# Gates: cg_on_l1==cg_off AND cg_on_l2==cg_off_l2 (HARD: cudagraph byte-exact);
#        cg_off_l2==cg_off (lanes byte-exact control); cg_off==silence0_warm200_c12
#        (default-off vs historical baseline, informational re venv). Plus replay
#        engagement (incl. per-lane for l2) and FORK_ASSERT clean. Scratch tags are
#        LEFT in the db for review; delete manually after.
set -uo pipefail
cd /home/khkramer/src/nemotron-january-2026
SRV=/home/khkramer/src/nemotron-nano-omni/.venv-asr/bin/python
HV=stt-benchmark/.venv/bin/python
MODEL=$(cat /tmp/en-nemo-path)
DB=stt-benchmark/stt_benchmark_data/results.db
LIMIT="${LIMIT:-100}"; CONC="${CONC:-12}"
COMMON=(NEMOTRON_CONTINUOUS=1 NEMOTRON_FINALIZE_SILENCE_MS=0 NEMOTRON_WARMUP_MS=200
        NEMOTRON_FORK_ASSERT=1 NEMOTRON_SCHEDULER_B1=1 NEMOTRON_BATCH_SCHED=1 PYTHONPATH=src)

run_cfg(){
  local tag=$1 log=$2; shift 2; local extra=("$@")
  pkill -f "server.py --model" 2>/dev/null; sleep 4
  env "${COMMON[@]}" "${extra[@]}" "$SRV" src/nemotron_speech/server.py --model "$MODEL" \
      --host 127.0.0.1 --port 8080 --right-context 1 > "$log" 2>&1 &
  local ok=0
  for _ in $(seq 1 150); do grep -q "ASR server listening" "$log" 2>/dev/null && { ok=1; break; }; sleep 2; done
  if [ $ok != 1 ]; then echo "[$tag] server FAILED to start"; tail -30 "$log"; pkill -f "server.py --model"; return 1; fi
  sleep 2
  echo "[$tag] ready; harness limit=$LIMIT conc=$CONC extra='${extra[*]}'"
  "$HV" proj-2026-05-19-eou-endpointing/run_full1000_conc12.py --model-tag "$tag" --concurrency "$CONC" --limit "$LIMIT" 2>&1 | tail -3
  pkill -f "server.py --model"; sleep 3
}

echo "########## cg_off (OFF, lanes1) ##########";    run_cfg cg_off    /tmp/srv_cg_off.log
echo "########## cg_on_l1 (ON, lanes1) ##########";   run_cfg cg_on_l1  /tmp/srv_cg_on_l1.log  NEMOTRON_ENCODER_CUDAGRAPH=1
echo "########## cg_off_l2 (OFF, lanes2) ##########"; run_cfg cg_off_l2 /tmp/srv_cg_off_l2.log NEMOTRON_MODEL_LANES=2
echo "########## cg_on_l2 (ON, lanes2) ##########";   run_cfg cg_on_l2  /tmp/srv_cg_on_l2.log  NEMOTRON_ENCODER_CUDAGRAPH=1 NEMOTRON_MODEL_LANES=2

echo "########## BYTE-EXACT DIFFS ##########"
"$HV" - <<PY
import sqlite3
con=sqlite3.connect("$DB")
def rows(t): return {r[0]:(r[1] or "") for r in con.execute("SELECT sample_id, transcription FROM results WHERE model_name=?", (t,))}
off=rows("cg_off"); on1=rows("cg_on_l1"); off2=rows("cg_off_l2"); on2=rows("cg_on_l2"); base=rows("silence0_warm200_c12")
print("counts:", {k:len(v) for k,v in [("cg_off",off),("cg_on_l1",on1),("cg_off_l2",off2),("cg_on_l2",on2),("baseline",base)]})
def cmp(a,an,b,bn):
    ids=sorted(set(a)&set(b)); mm=[i for i in ids if a[i]!=b[i]]
    print(f"{an} vs {bn}: {len(ids)-len(mm)}/{len(ids)} identical; mismatches={len(mm)}")
    for i in mm[:8]: print("   MM",i,"\n     A:",repr(a[i]),"\n     B:",repr(b[i]))
    return len(mm), len(ids)
m_g1,n_g1=cmp(off,"cg_off",on1,"cg_on_l1")
m_ctl,n_ctl=cmp(off,"cg_off",off2,"cg_off_l2")
m_g2,n_g2=cmp(off2,"cg_off_l2",on2,"cg_on_l2")
m_b,n_b=cmp(off,"cg_off",base,"baseline")
print()
print("HARD GATE cudagraph byte-exact (on==off, lanes1 AND lanes2):",
      "PASS" if (m_g1==0 and n_g1>0 and m_g2==0 and n_g2>0) else "FAIL")
print("lanes byte-exact control (off_l2==off):", "PASS" if (m_ctl==0 and n_ctl>0) else f"DIFFERS {m_ctl}/{n_ctl}")
print("default-off==historical-baseline:", "PASS" if (m_b==0 and n_b>0) else f"DIFFERS {m_b}/{n_b} (likely venv vs the baseline run; cudagraph gate above is the proof)")
con.close()
PY

echo "########## ENGAGEMENT + FORK_ASSERT ##########"
echo "-- on_l1 --"; grep -E "encoder_cuda_graph_enabled=|encoder_cuda_graph_manager_captured" /tmp/srv_cg_on_l1.log; grep "encoder_cuda_graph_status" /tmp/srv_cg_on_l1.log | tail -2
echo "-- on_l2 (expect self.model + 2 lane managers, replays on lanes) --"; grep -E "encoder_cuda_graph_enabled=|encoder_cuda_graph_manager_captured|lane_stream_managers" /tmp/srv_cg_on_l2.log; grep "encoder_cuda_graph_status" /tmp/srv_cg_on_l2.log | tail -3
echo "-- FORK_ASSERT / tracebacks (all 4) --"; grep -iE "FORK_ASSERT|Traceback|AssertionError" /tmp/srv_cg_off.log /tmp/srv_cg_on_l1.log /tmp/srv_cg_off_l2.log /tmp/srv_cg_on_l2.log | head -10 || true
echo "(if no FORK_ASSERT/Traceback lines above, fork path is clean)"
pkill -f "server.py --model" 2>/dev/null; sleep 1
echo "########## STEP4 CANARY DONE (scratch tags cg_* left in db for review) ##########"
