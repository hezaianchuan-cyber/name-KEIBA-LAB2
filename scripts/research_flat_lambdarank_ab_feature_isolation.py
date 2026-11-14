#!/usr/bin/env python3
"""Research-only four-way feature isolation test for LambdaRank."""
from __future__ import annotations
import datetime as dt, json, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import research_flat_lambdarank_feature_ablation as base
import nar_ml_challenger_backtest as suite
OUT=ROOT/"data"/"flat-lambdarank-ab-feature-isolation.json"
F1="race_rel_recent_finish_pct"
F2="race_rel_recent_top3_rate"
METRICS=("top1","winner_in_top3","top7_podium_coverage","top7_full_podium")
FOLDS=[
("fold_01",dt.date(2024,7,1),dt.date(2024,10,31)),
("fold_02",dt.date(2024,11,1),dt.date(2025,2,28)),
("fold_03",dt.date(2025,3,1),dt.date(2025,6,30)),
("fold_04",dt.date(2025,7,1),dt.date(2025,10,31)),
("fold_05",dt.date(2025,11,1),dt.date(2026,2,28)),
("fold_06",dt.date(2026,3,1),dt.date(2026,6,30)),
("fold_07",dt.date(2026,7,1),dt.date(2026,10,31))]
VARIANTS={"A_45_features":set(),"B_without_recent_finish":{F1},"C_without_recent_top3":{F2},"D_without_both":{F1,F2}}
BOOT=5000
def score(scored):
    out={m:[] for m in METRICS}; keys=[]
    for group,scores in scored:
        winner=next((i for i,r in enumerate(group) if int(r["finish"])==1),None)
        if winner is None: continue
        order=sorted(range(len(group)),key=lambda i:(-float(scores[i]),int(group[i]["horse_no"])))
        podium={i for i,r in enumerate(group) if int(r["finish"])<=3}
        top7=set(order[:7])
        out["top1"].append(float(order[0]==winner))
        out["winner_in_top3"].append(float(winner in order[:3]))
        out["top7_podium_coverage"].append(len(podium&top7)/max(1,len(podium)))
        out["top7_full_podium"].append(float(podium.issubset(top7)))
        keys.append(group[0]["race_key"])
    return keys,out
def summary(v): return {m:round(float(np.mean(v[m])*100),3) for m in METRICS}
def paired(a,b,seed):
    x=np.asarray(b)-np.asarray(a)
    if len(x)==0 or len(x)!=len(a): raise RuntimeError("paired metric mismatch")
    rng=np.random.default_rng(seed); n=len(x); samples=np.empty(BOOT)
    for start in range(0,BOOT,100):
        k=min(100,BOOT-start); samples[start:start+k]=x[rng.integers(0,n,(k,n))].mean(axis=1)
    lo,hi=np.quantile(samples,[.025,.975])
    return {"races":int(n),"delta_pp_B_minus_A":round(float(x.mean()*100),3),"ci95_pp":[round(float(lo*100),3),round(float(hi*100),3)],"bootstrap_iterations":BOOT,"method":"paired race-level bootstrap"}
def main():
    suite.RAW_NUMERIC=[f for f in suite.RAW_NUMERIC if f not in base.EXCLUDED_BODY_WEIGHT]
    suite.RAW_FEATURES=suite.RAW_NUMERIC+suite.CATEGORICAL
    features=list(suite.RAW_FEATURES)
    if len(features)!=45 or not {F1,F2}.issubset(features): raise RuntimeError("Frozen 45-feature schema mismatch")
    selected={k:[f for f in features if f not in removed] for k,removed in VARIANTS.items()}
    if [len(selected[k]) for k in VARIANTS] != [45,44,44,43]: raise RuntimeError("Variant feature count mismatch")
    races,horses,scope=base.load_flat()
    rows,dates=suite.build_rows(races,horses)
    if not dates: raise RuntimeError("No race dates")
    by_date={}
    for r in rows: by_date.setdefault(dt.date.fromisoformat(r["date"]),[]).append(r)
    pooled={k:{m:[] for m in METRICS} for k in VARIANTS}
    fold_reports=[]
    for fi,(fold,start,end) in enumerate(FOLDS):
        train_dates=sorted(d for d in dates if d<start)
        valid_dates=sorted(d for d in dates if start<=d<=end)
        if len(train_dates)<180 or not valid_dates: raise RuntimeError(f"Insufficient dates in {fold}")
        cut=int(len(train_dates)*.90); fit_dates=set(train_dates[:cut]); early_dates=set(train_dates[cut:])
        def collect(ds): return [r for day in sorted(ds) for r in by_date.get(day,[])]
        fit_rows=collect(fit_dates)
        fit_groups=suite.group_rows(fit_rows); early_groups=suite.group_rows(collect(early_dates))
        valid_groups=suite.group_rows(collect(set(valid_dates)))
        if not fit_groups or not early_groups or len(valid_groups)<100: raise RuntimeError(f"Empty/short groups in {fold}")
        maps={key:{value:i for i,value in enumerate(sorted({r[key] for r in fit_rows}))} for key in suite.CATEGORICAL}
        fold_vectors={}; fold_keys={}; models={}
        for vi,(label,variant_features) in enumerate(selected.items()):
            model=base.fit_ranker(fit_groups,early_groups,variant_features,maps)
            keys,vectors=score(base.predict_groups(model,valid_groups,variant_features,maps))
            fold_keys[label]=keys; fold_vectors[label]=vectors
            models[label]={"feature_count":len(variant_features),"removed_features":sorted(VARIANTS[label]),"best_iteration":int(getattr(model,"best_iteration_",0) or 0),"metrics":summary(vectors)}
        if len(set(tuple(v) for v in fold_keys.values()))!=1: raise RuntimeError(f"Race alignment mismatch {fold}")
        for label in VARIANTS:
            for metric in METRICS: pooled[label][metric].extend(fold_vectors[label][metric])
        comparisons={}
        for vi,label in enumerate(list(VARIANTS)[1:]):
            comparisons[label]={m:paired(fold_vectors["A_45_features"][m],fold_vectors[label][m],20261114+fi*100+vi*13+mi) for mi,m in enumerate(METRICS)}
        report={"fold":fold,"training":{"first_date":min(train_dates).isoformat(),"last_date":max(train_dates).isoformat(),"fit_races":len(fit_groups),"early_stopping_races":len(early_groups),"fit_dates":len(fit_dates),"early_stopping_dates":len(early_dates)},"validation":{"start":min(valid_dates).isoformat(),"end":max(valid_dates).isoformat(),"dates":len(valid_dates),"races":len(valid_groups)},"models":models,"paired_vs_A":comparisons}
        fold_reports.append(report); print("FEATURE_ISOLATION_FOLD "+json.dumps(report,ensure_ascii=False),flush=True)
    pooled_summary={k:summary(v) for k,v in pooled.items()}
    pooled_comparisons={}
    for vi,label in enumerate(list(VARIANTS)[1:]):
        pooled_comparisons[label]={m:paired(pooled["A_45_features"][m],pooled[label][m],20261114+1000+vi*17+mi) for mi,m in enumerate(METRICS)}
    report={"generated_at":dt.datetime.now(dt.timezone.utc).isoformat(),"scope":scope,"data":{"runner_rows":len(rows),"total_dates":len(dates),"source_races":scope.get("flat_races"),"pooled_validation_races":len(pooled["A_45_features"]["top1"]),"pooled_validation_dates":sum(f["validation"]["dates"] for f in fold_reports)},"variants":{k:{"feature_count":len(selected[k]),"removed_features":sorted(VARIANTS[k])} for k in VARIANTS},"fixed_parameters":base.PARAMS,"folds":fold_reports,"pooled_oos_metrics":pooled_summary,"pooled_paired_vs_A":pooled_comparisons,"method":{"design":"Four matched models retrained on identical expanding-window chronological folds; same fit/early-stop/validation dates and category maps.","bootstrap":"5,000 paired race-level bootstrap resamples per metric and comparison.","feature_variants":"A=45; B removes only race_rel_recent_finish_pct; C removes only race_rel_recent_top3_rate; D removes both."},"limitations":["Historical validation windows have been inspected in earlier project work; not a pristine holdout.","All four models are trained independently with same frozen hyperparameters; model best iterations can differ naturally.","Ranking metrics only; no ROI or executable edge is established.","Exploratory four-model comparison; multiple comparisons warrant caution.","Research-only: no production model, live weights, database, or ticket generation changed."]}
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print("FEATURE_ISOLATION_COMPLETE "+json.dumps({"data":report["data"],"pooled_oos_metrics":pooled_summary,"paired_vs_A":pooled_comparisons},ensure_ascii=False),flush=True)
if __name__=="__main__": main()
