#!/usr/bin/env python3
"""Research-only four-way one-feature-at-a-time expanding-window LambdaRank comparison."""
from __future__ import annotations
import datetime as dt, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"scripts"))
import numpy as np
import research_flat_lambdarank_feature_ablation as base
import nar_ml_challenger_backtest as suite
OUT=ROOT/"data"/"flat-lambdarank-ab-four-way.json"
F1="race_rel_recent_finish_pct"; F2="race_rel_recent_top3_rate"
FOLDS=[
("fold_01",dt.date(2024,7,1),dt.date(2024,10,31)),
("fold_02",dt.date(2024,11,1),dt.date(2025,2,28)),
("fold_03",dt.date(2025,3,1),dt.date(2025,6,30)),
("fold_04",dt.date(2025,7,1),dt.date(2025,10,31)),
("fold_05",dt.date(2025,11,1),dt.date(2026,2,28)),
("fold_06",dt.date(2026,3,1),dt.date(2026,6,30)),
("fold_07",dt.date(2026,7,1),dt.date(2026,10,31))]
METRICS=("top1","winner_in_top3","top7_podium_coverage","top7_full_podium")
def score(scored):
 out={k:[] for k in METRICS}; keys=[]
 for g,s in scored:
  win=next((i for i,r in enumerate(g) if int(r["finish"])==1),None)
  if win is None: continue
  order=sorted(range(len(g)),key=lambda i:(-float(s[i]),int(g[i]["horse_no"])))
  podium={i for i,r in enumerate(g) if int(r["finish"])<=3}; top7=set(order[:7])
  out["top1"].append(float(order[0]==win)); out["winner_in_top3"].append(float(win in order[:3]))
  out["top7_podium_coverage"].append(len(podium&top7)/max(1,len(podium)))
  out["top7_full_podium"].append(float(podium.issubset(top7))); keys.append(g[0]["race_key"])
 return keys,out
def boot(a,b,seed):
 d=np.asarray(b,dtype=float)-np.asarray(a,dtype=float); rng=np.random.default_rng(seed); n=len(d)
 if n==0 or len(a)!=len(b): raise RuntimeError("paired metric vectors invalid")
 samples=np.empty(5000)
 for start in range(0,5000,100):
  m=min(100,5000-start); samples[start:start+m]=d[rng.integers(0,n,size=(m,n))].mean(axis=1)
 lo,hi=np.quantile(samples,[.025,.975])
 return {"delta_pp_vs_A":round(float(d.mean()*100),3),"ci95_pp":[round(float(lo*100),3),round(float(hi*100),3)],"races":n}
def main():
 suite.RAW_NUMERIC=[f for f in suite.RAW_NUMERIC if f not in base.EXCLUDED_BODY_WEIGHT]
 suite.RAW_FEATURES=suite.RAW_NUMERIC+suite.CATEGORICAL
 allf=list(suite.RAW_FEATURES)
 if len(allf)!=45 or F1 not in allf or F2 not in allf: raise RuntimeError("45-feature schema mismatch")
 variants={"A_45_all":allf,"B_44_without_relative_finish":[f for f in allf if f!=F1],
 "C_44_without_relative_top3":[f for f in allf if f!=F2],
 "D_43_without_both":[f for f in allf if f not in {F1,F2}]}
 races,horses,scope=base.load_flat(); rows,dates=suite.build_rows(races,horses)
 bydate={}
 for r in rows: bydate.setdefault(dt.date.fromisoformat(r["date"]),[]).append(r)
 pooled={v:{m:[] for m in METRICS} for v in variants}; folds=[]
 for fi,(name,start,end) in enumerate(FOLDS):
  td=sorted(d for d in dates if d<start); vd=sorted(d for d in dates if start<=d<=end)
  if len(td)<180 or not vd: raise RuntimeError(f"invalid date coverage {name}")
  cut=int(len(td)*.9); fitd=set(td[:cut]); earlyd=set(td[cut:])
  def collect(ds): return [r for d in sorted(ds) for r in bydate.get(d,[])]
  fitrows=collect(fitd); fitgroups=suite.group_rows(fitrows); earlygroups=suite.group_rows(collect(earlyd)); valid=suite.group_rows(collect(set(vd)))
  maps={k:{v:i for i,v in enumerate(sorted({r[k] for r in fitrows}))} for k in suite.CATEGORICAL}
  foldmodels={}; foldvectors={}; foldkeys={}
  for vi,(label,features) in enumerate(variants.items()):
   model=base.fit_ranker(fitgroups,earlygroups,features,maps)
   keys,vec=score(base.predict_groups(model,valid,features,maps))
   foldkeys[label]=keys; foldvectors[label]=vec
   foldmodels[label]={"feature_count":len(features),"best_iteration":int(getattr(model,"best_iteration_",0) or 0),
    "metrics":{m:round(float(np.mean(v)*100),3) for m,v in vec.items()}}
  if any(foldkeys[x]!=foldkeys["A_45_all"] for x in variants): raise RuntimeError("race alignment mismatch")
  for label in variants:
   for m in METRICS: pooled[label][m].extend(foldvectors[label][m])
  comparisons={}
  for label in list(variants)[1:]:
   comparisons[label]={m:boot(foldvectors["A_45_all"][m],foldvectors[label][m],20261114+fi*97+list(METRICS).index(m)*13+list(variants).index(label)) for m in METRICS}
  fold={"fold":name,"validation_start":min(vd).isoformat(),"validation_end":max(vd).isoformat(),"validation_dates":len(vd),"validation_races":len(valid),"models":foldmodels,"paired_vs_A":comparisons}
  folds.append(fold); print("FOUR_WAY_FOLD "+json.dumps(fold,ensure_ascii=False),flush=True)
 pooled_results={}
 for label in variants:
  pooled_results[label]={m:round(float(np.mean(pooled[label][m])*100),3) for m in METRICS}
 pooled_comparisons={label:{m:boot(pooled["A_45_all"][m],pooled[label][m],20261200+list(variants).index(label)*19+list(METRICS).index(m)) for m in METRICS} for label in list(variants)[1:]}
 report={"generated_at":dt.datetime.now(dt.timezone.utc).isoformat(),"scope":scope,
 "data":{"runner_rows":len(rows),"dates":len(dates),"pooled_validation_races":len(pooled["A_45_all"]["top1"])},
 "variants":{k:{"feature_count":len(v),"removed":sorted(set(allf)-set(v))} for k,v in variants.items()},
 "fixed_parameters":base.PARAMS,"folds":folds,"pooled_metrics":pooled_results,"pooled_paired_vs_A":pooled_comparisons,
 "method":"Same expanding-window temporal folds, training and early stopping dates, and fixed LightGBM parameters for all four variants. Each model refit independently. Paired race-level bootstrap 5000 resamples.",
 "guardrails":["Historical periods have been inspected in earlier project work; not a pristine holdout.","No tuning from validation results; this run is a diagnostic comparison.","Ranking metrics do not establish ROI or profitability.","Research-only; no production model, weights, DB or ticket logic modified."]}
 OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
 print("FOUR_WAY_COMPLETE "+json.dumps({"data":report["data"],"pooled_metrics":pooled_results,"paired_vs_A":pooled_comparisons},ensure_ascii=False),flush=True)
if __name__=="__main__": main()
