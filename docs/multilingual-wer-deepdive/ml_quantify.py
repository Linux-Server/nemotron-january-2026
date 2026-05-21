import sqlite3, json, statistics as st

DB = "/home/khkramer/src/nemotron-january-2026/stt-benchmark/stt_benchmark_data/results.db"
con = sqlite3.connect(DB)
ML, EN = "ml_silence0_warm200_c12", "silence0_warm200_c12"

def load(model):
    rows = con.execute(
        "SELECT w.sample_id, w.wer, w.substitutions, w.deletions, w.insertions, w.reference_words, "
        "w.errors, r.audio_duration_seconds "
        "FROM wer_metrics w JOIN results r ON r.sample_id=w.sample_id AND r.model_name=w.model_name "
        "WHERE w.model_name=?", (model,)).fetchall()
    return {r[0]: dict(wer=r[1], sub=r[2], dl=r[3], ins=r[4], ref=r[5], errs=r[6], dur=r[7]) for r in rows}

ml, en = load(ML), load(EN)
common = sorted(set(ml) & set(en))

def lead_run(errs):
    try: E = json.loads(errs) if errs else []
    except: return 0
    dp = sorted(e["position"] for e in E if e.get("error_type")=="deletion" and isinstance(e.get("position"),int))
    run=0; exp=1
    for p in dp:
        if p==exp: run+=1; exp+=1
        elif p<exp: continue
        else: break
    return run

front = [s for s in common if lead_run(ml[s]["errs"]) >= 3]
def pooled(samples, d):
    e = sum(d[s]["sub"]+d[s]["dl"]+d[s]["ins"] for s in samples)
    r = sum(d[s]["ref"] for s in samples)
    return e/r*100, e, r

allp, alle, allr = pooled(common, ml)
nofront, ne, nr = pooled([s for s in common if s not in set(front)], ml)
enp, _, _ = pooled(common, en)
print(f"ml pooled WER (all {len(common)}):           {allp:.2f}%  ({alle} err / {allr} ref)")
print(f"ml pooled WER (excl {len(front)} front-drops): {nofront:.2f}%  ({ne} err / {nr} ref)")
print(f"  -> the {len(front)} front-drop clips contribute {alle-ne} errors = {(alle-ne)/alle*100:.1f}% of ALL ml errors")
print(f"en pooled WER (same {len(common)}):           {enp:.2f}%")
print(f"  gap to en: total={allp-enp:.2f}pp ; after removing front-drops={nofront-enp:.2f}pp "
      f"(=> front-drop bug explains {(allp-nofront)/(allp-enp)*100:.0f}% of the en->ml gap)")

# deletions accounting
ml_del_total = sum(ml[s]["dl"] for s in common)
ml_del_front = sum(ml[s]["dl"] for s in front)
print(f"\ndeletions: ml total={ml_del_total}, in front-drop clips={ml_del_front} "
      f"({ml_del_front/ml_del_total*100:.0f}% of all ml deletions from {len(front)} clips)")

# duration systematic? how many long clips, how many dropped
longs = [s for s in common if (ml[s]["dur"] or 0) >= 12]
print(f"\nlong clips (>=12s): {len(longs)};  front-dropped: {sum(1 for s in longs if s in set(front))}  "
      f"=> {'NOT systematic (only some long clips)' if sum(1 for s in longs if s in set(front))<len(longs)*0.5 else 'SYSTEMATIC'}")
durdist = sorted((ml[s]['dur'] or 0) for s in front)
print(f"front-drop durations: {[f'{d:.1f}' for d in durdist]}")

# substitution-driven (model-quality) share
sub_only = [s for s in common if ml[s]["wer"]>0 and ml[s]["dl"]==0 and ml[s]["ins"]==0]
sop,soe,sor = pooled(sub_only, ml)
print(f"\nsubstitution-only imperfect samples: {len(sub_only)}, contributing {soe} errors "
      f"({soe/alle*100:.0f}% of all ml errors) = the model-quality (distributed) component")
