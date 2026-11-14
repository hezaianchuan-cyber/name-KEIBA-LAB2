#!/usr/bin/env python3
"""Research-only horse-level diagnostics for the two removed LambdaRank features."""
from __future__ import annotations
import datetime as dt, json, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import research_flat_lambdarank_feature_ablation as base
import nar_ml_challenger_backtest as suite
OUT=ROOT/"data"/"flat-lambdarank-ab-feature-strata.json"
FEATURES=("race_rel_recent_finish_pct","race_rel_recent_top3_rate")
REMOVED=set(FEATURES)
FOLDS=[
("fold_01",dt.date(2024,7,1),dt.date(2024,10,31)),
("fold_02",dt.date(2024,11,1),dt.date(2025,2,28)),
("fold_03",dt.date(2025,3,1),dt.date(2025,6,30)),
("fold_04",dt.date(2025,7,1),dt.date(2025,10,31)),
("fold_05",dt.date(2025,11,1),dt.date(2026,2,28)),
("fold_06",dt.date(2026,3,1),dt.date(2026,6,30)),
("fold_07",dt.date(2026,7,1),dt.date(2026,10,31))]
BINS=("low","mid_low","mid_high","high")
def rank_pct(scores,group):
    order=sorted(range(len(group)),key=lambda i:(-float(scores[i]),int(group[i]["horse_no"])))
    ranks=[0.0]*len(group)
    for pos,idx in enumerate(order): ranks[idx]=(pos+1)/len(group)
    return ranks,order
def main():
    suite.RAW_NUMERIC=[f for f in suite.RAW_NUMERIC if f not in base.EXCLUDED_BODY_WEIGHT]
    suite.RAW_FEATURES=suite.RAW_NUMERIC+suite.CATEGORICAL
    all_features=list(suite.RAW_FEATURES)
    if len(all_features)!=45 or not REMOVED.issubset(all_features): raise RuntimeError("Frozen feature schema mismatch")
    features_b=[f for f in all_features if f not in REMOVED]
    races,horses,scope=base.load_flat()
    rows,dates=suite.build_rows(races,horses)
    by_date=defaultdict(list)
    for r in rows: by_date[dt.date.fromisoformat(r["date"])].append(r)
    pooled={f:{b:{"horses":0,"races":0,"wins":0,"top3":0,"A_top3":0,"B_top3":0,"rank_delta_sum":0.0,"abs_rank_delta_sum":0.0,"rank_improved":0} for b in BINS} for f in FEATURES}
    fold_reports=[]
    for fi,(name,start,end) in enumerate(FOLDS):
        train_dates=sorted(d for d in dates if d<start)
        valid_dates=sorted(d for d in dates if start<=d<=end)
        if len(train_dates)<180 or not valid_dates: raise RuntimeError(f"bad date coverage {name}")
        cut=int(len(train_dates)*.90); fit_dates=set(train_dates[:cut]); early_dates=set(train_dates[cut:])
        def collect(ds): return [r for d in sorted(ds) for r in by_date.get(d,[])]
        fit_rows=collect(fit_dates)
        fit_groups=suite.group_rows(fit_rows)
        early_groups=suite.group_rows(collect(early_dates))
        valid_groups=suite.group_rows(collect(set(valid_dates)))
        maps={k:{v:i for i,v in enumerate(sorted({r[k] for r in fit_rows}))} for k in suite.CATEGORICAL}
        models={}
        for label,features in (("A",all_features),("B",features_b)):
            model=base.fit_ranker(fit_groups,early_groups,features,maps)
            models[label]=base.predict_groups(model,valid_groups,features,maps)
        if len(models["A"])!=len(models["B"]): raise RuntimeError("A/B race count mismatch")
        fold={}
        for feature in FEATURES:
            # Quantile boundaries come only from this fold's fitting rows.
            vals=np.asarray([float(r[feature]) for r in fit_rows if r.get(feature) is not None],dtype=float)
            cuts=np.quantile(vals,[.25,.5,.75]).tolist()
            acc={b:{"horses":0,"races":0,"wins":0,"top3":0,"A_top3":0,"B_top3":0,"rank_delta_sum":0.0,"abs_rank_delta_sum":0.0,"rank_improved":0} for b in BINS}
            for (ga,sa),(gb,sb) in zip(models["A"],models["B"]):
                if [r["race_key"] for r in ga]!=[r["race_key"] for r in gb]: raise RuntimeError("A/B alignment mismatch")
                ar,ao=rank_pct(sa,ga); br,bo=rank_pct(sb,gb)
                atop=set(ao[:3]); btop=set(bo[:3])
                for i,row in enumerate(ga):
                    v=float(row[feature])
                    q=0 if v<=cuts[0] else 1 if v<=cuts[1] else 2 if v<=cuts[2] else 3
                    bucket=BINS[q]; x=acc[bucket]
                    x["horses"]+=1; x["races"]+=1
                    x["wins"]+=int(int(row["finish"])==1); x["top3"]+=int(int(row["finish"])<=3)
                    x["A_top3"]+=int(i in atop); x["B_top3"]+=int(i in btop)
                    delta=ar[i]-br[i] # positive means B ranked horse higher (smaller rank percentile)
                    x["rank_delta_sum"]+=delta; x["abs_rank_delta_sum"]+=abs(delta)
                    x["rank_improved"]+=int(delta>0)
            fold[feature]={"fit_quantile_cuts":[round(float(x),6) for x in cuts],"bins":{}}
            for bucket,x in acc.items():
                n=x["horses"]
                fold[feature]["bins"][bucket]={**x,
                    "win_rate_pct":round(100*x["wins"]/n,3) if n else None,
                    "top3_rate_pct":round(100*x["top3"]/n,3) if n else None,
                    "A_top3_share_pct":round(100*x["A_top3"]/n,3) if n else None,
                    "B_top3_share_pct":round(100*x["B_top3"]/n,3) if n else None,
                    "mean_rank_percentile_B_minus_A":round(100*x["rank_delta_sum"]/n,3) if n else None,
                    "mean_abs_rank_shift_pp":round(100*x["abs_rank_delta_sum"]/n,3) if n else None,
                    "B_ranks_higher_share_pct":round(100*x["rank_improved"]/n,3) if n else None}
                for k in ("rank_delta_sum","abs_rank_delta_sum","rank_improved"): x.pop(k)
                for k in ("horses","races","wins","top3","A_top3","B_top3"): pooled[feature][bucket][k]+=x[k]
        fold_reports.append({"fold":name,"validation_start":min(valid_dates).isoformat(),"validation_end":max(valid_dates).isoformat(),"validation_races":len(valid_groups),"features":fold})
        print("FEATURE_STRATA_FOLD "+json.dumps(fold_reports[-1],ensure_ascii=False),flush=True)
    pooled_out={}
    for feature,bins in pooled.items():
        pooled_out[feature]={}
        for bucket,x in bins.items():
            n=x["horses"]
            pooled_out[feature][bucket]={**x,
                "win_rate_pct":round(100*x["wins"]/n,3) if n else None,
                "top3_rate_pct":round(100*x["top3"]/n,3) if n else None,
                "A_top3_share_pct":round(100*x["A_top3"]/n,3) if n else None,
                "B_top3_share_pct":round(100*x["B_top3"]/n,3) if n else None}
    report={"generated_at":dt.datetime.now(dt.timezone.utc).isoformat(),"scope":scope,
      "data":{"runner_rows":len(rows),"dates":len(dates),"validation_races":sum(x["validation_races"] for x in fold_reports)},
      "models":{"A":{"features":45},"B":{"features":43,"removed":sorted(REMOVED)}},
      "method":{"bins":"Four feature-value quartiles; thresholds computed from each fold's fit rows only.",
       "rank_delta":"Per-race normalized rank percentile; positive B_minus_A means B ranks the horse higher.",
       "metrics":"Horse win/top3 rates and A/B top3 selection share; descriptive, no ROI or production inference."},
      "folds":fold_reports,"pooled_counts":pooled_out,
      "limitations":["Exploratory subgroup analysis; historical periods previously inspected.","Quartile membership uses training-only cut points, but validation outcomes are descriptive.","No multiple-testing-adjusted causal claim.","Ranking diagnostics only, not profitability.","No production model, live weights, database, or ticket logic changed."]}
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print("FEATURE_STRATA_COMPLETE "+json.dumps({"data":report["data"],"pooled_counts":pooled_out},ensure_ascii=False),flush=True)
if __name__=="__main__": main()
