import sqlite3, json, statistics as st
from collections import Counter

DB = "/home/khkramer/src/nemotron-january-2026/stt-benchmark/stt_benchmark_data/results.db"
con = sqlite3.connect(DB)
ML, EN = "ml_silence0_warm200_c12", "silence0_warm200_c12"

def load(model):
    rows = con.execute(
        "SELECT w.sample_id, w.wer, w.substitutions, w.deletions, w.insertions, w.reference_words, "
        "w.errors, w.normalized_reference, w.normalized_hypothesis, r.audio_duration_seconds "
        "FROM wer_metrics w JOIN results r ON r.sample_id=w.sample_id AND r.model_name=w.model_name "
        "WHERE w.model_name=?", (model,)).fetchall()
    d = {}
    for sid, wer, s, dl, i, ref, errs, nref, nhyp, dur in rows:
        d[sid] = dict(wer=wer, sub=s, dl=dl, ins=i, ref=ref, errs=errs, nref=nref or "", nhyp=nhyp or "", dur=dur)
    return d

ml, en = load(ML), load(EN)
common = sorted(set(ml) & set(en))
print(f"common samples: {len(common)}  (ml={len(ml)} en={len(en)})\n")

def leading_del_run(errs):
    """consecutive deletions starting at reference position 1 (front of utterance dropped)."""
    try:
        E = json.loads(errs) if errs else []
    except Exception:
        return 0
    delpos = sorted(e["position"] for e in E if e.get("error_type") == "deletion" and isinstance(e.get("position"), int))
    run = 0; expect = 1
    for p in delpos:
        if p == expect:
            run += 1; expect += 1
        elif p < expect:
            continue
        else:
            break
    return run

def del_frac_front(errs, ref_len):
    try:
        E = json.loads(errs) if errs else []
    except Exception:
        return 0.0, 0
    dpos = [e["position"] for e in E if e.get("error_type") == "deletion" and isinstance(e.get("position"), int)]
    if not dpos or not ref_len:
        return 0.0, len(dpos)
    front = sum(1 for p in dpos if p <= ref_len/3.0)
    return front/len(dpos), len(dpos)

# ---- 1. ml-specific failures: en perfect, ml not ----
ml_only = [s for s in common if en[s]["wer"] == 0 and ml[s]["wer"] > 0]
both_bad = [s for s in common if en[s]["wer"] > 0 and ml[s]["wer"] > 0]
print(f"=== ml-SPECIFIC failures (en perfect, ml imperfect): {len(ml_only)} ===")
ts=sum(ml[s]['sub'] for s in ml_only); td=sum(ml[s]['dl'] for s in ml_only); ti=sum(ml[s]['ins'] for s in ml_only)
print(f"   their error split: sub={ts} del={td} ins={ti}  (deletions dominate? {td}>{ts}={td>ts})")

# ---- 2. leading-deletion-run (front-of-utterance dropped) ml vs en ----
def front_drop_stats(d, label):
    runs = {s: leading_del_run(d[s]["errs"]) for s in d}
    big = [s for s,r in runs.items() if r >= 3]
    print(f"   {label}: samples with leading-deletion-run>=3 (front dropped): {len(big)}; "
          f">=5: {sum(1 for r in runs.values() if r>=5)}; max run: {max(runs.values())}")
    return big
print("\n=== FRONT-OF-UTTERANCE DROPPED (leading deletion run) ===")
ml_front = front_drop_stats(ml, "ml")
en_front = front_drop_stats(en, "en")

# ---- 3. on ml front-drop samples, what does EN do? (does en capture the front?) ----
print(f"\n=== HEAD-TO-HEAD on ml front-drop samples (n={len(ml_front)}) ===")
en_perfect_here = sum(1 for s in ml_front if en[s]["wer"] == 0)
en_lowdel_here = sum(1 for s in ml_front if en[s]["dl"] <= 1)
print(f"   of the {len(ml_front)} samples where ML dropped the front: "
      f"EN perfect on {en_perfect_here}, EN deletions<=1 on {en_lowdel_here}")
print(f"   => if EN captures the front on most, the audio HAS the front and ML's code/model dropped it.")

# length ratio (truncation signal)
def wc(s): return len(s.split())
ratios_ml = [wc(ml[s]["nhyp"])/max(wc(ml[s]["nref"]),1) for s in ml_front]
ratios_en = [wc(en[s]["nhyp"])/max(wc(en[s]["nref"]),1) for s in ml_front]
print(f"   hyp/ref word-count ratio on these: ML median={st.median(ratios_ml):.2f}  EN median={st.median(ratios_en):.2f}")
durs = [ml[s]["dur"] for s in ml_front if ml[s]["dur"]]
alld = [ml[s]["dur"] for s in common if ml[s]["dur"]]
print(f"   duration: front-drop median={st.median(durs):.1f}s vs all-common median={st.median(alld):.1f}s  (longer => multi-segment)")

# ---- 4. show concrete examples ----
print("\n=== EXAMPLES (ml dropped front, en's take) ===")
for s in sorted(ml_front, key=lambda x: -ml[x]["dl"])[:6]:
    print(f"  [{s[:8]}] dur={ml[s]['dur']:.1f}s  ref_words={ml[s]['ref']} ml_del={ml[s]['dl']} en_wer={en[s]['wer']:.2f}")
    print(f"     REF : {ml[s]['nref'][:105]}")
    print(f"     ML  : {ml[s]['nhyp'][:105]}")
    print(f"     EN  : {en[s]['nhyp'][:105]}")

# ---- 5. looping / repetition check in ml hyps ----
def has_loop(t):
    w = t.split()
    return any(w[i:i+3] == w[i+3:i+6] for i in range(len(w)-5)) if len(w) >= 6 else False
ml_loops = [s for s in common if has_loop(ml[s]["nhyp"])]
en_loops = [s for s in common if has_loop(en[s]["nhyp"])]
print(f"\n=== looping (repeated trigram) ===  ml={len(ml_loops)}  en={len(en_loops)}")

# ---- 6. distributed substitution-only errors (model-quality signature) ----
sub_only = [s for s in common if ml[s]["wer"]>0 and ml[s]["dl"]==0 and ml[s]["ins"]==0]
print(f"=== ml imperfect that are SUBSTITUTION-ONLY (model-quality signature): {len(sub_only)} ===")
