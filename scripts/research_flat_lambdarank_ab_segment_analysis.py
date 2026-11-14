#!/usr/bin/env python3
"""Research-only segment diagnostics for the frozen 45f vs 43f LambdaRank A/B."""
from __future__ import annotations
import datetime as dt
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import research_flat_lambdarank_feature_ablation as base
import nar_ml_challenger_backtest as suite

OUT = ROOT / "data" / "flat-lambdarank-ab-segment-analysis.json"
REMOVED_B = {"race_rel_recent_finish_pct", "race_rel_recent_top3_rate"}
METRICS = ("top1", "winner_in_top3", "top7_podium_coverage", "top7_full_podium")
BOOTSTRAP_ITERATIONS = 5000
MIN_SEGMENT_RACES = 150
FOLDS = [
    ("fold_01", dt.date(2024, 7, 1), dt.date(2024, 10, 31)),
    ("fold_02", dt.date(2024, 11, 1), dt.date(2025, 2, 28)),
    ("fold_03", dt.date(2025, 3, 1), dt.date(2025, 6, 30)),
    ("fold_04", dt.date(2025, 7, 1), dt.date(2025, 10, 31)),
    ("fold_05", dt.date(2025, 11, 1), dt.date(2026, 2, 28)),
    ("fold_06", dt.date(2026, 3, 1), dt.date(2026, 6, 30)),
    ("fold_07", dt.date(2026, 7, 1), dt.date(2026, 10, 31)),
]

def distance_band(value):
    d = float(value or 0)
    if d < 1200: return "<1200m"
    if d < 1400: return "1200-1399m"
    if d < 1600: return "1400-1599m"
    if d < 1800: return "1600-1799m"
    if d < 2000: return "1800-1999m"
    return "2000m+"

def field_band(value):
    n = int(float(value or 0))
    if n <= 8: return "8頭以下"
    if n <= 12: return "9-12頭"
    return "13頭以上"

def race_vectors(scored):
    keys = []
    vectors = {m: [] for m in METRICS}
    segments = {"venue": [], "distance_band": [], "field_size_band": []}
    for group, scores in scored:
        winner = next((i for i, r in enumerate(group) if int(r["finish"]) == 1), None)
        if winner is None:
            continue
        order = sorted(range(len(group)), key=lambda i: (-float(scores[i]), str(group[i]["horse_no"])))
        podium = {i for i, r in enumerate(group) if int(r["finish"]) <= 3}
        top7 = set(order[:7])
        vectors["top1"].append(float(order[0] == winner))
        vectors["winner_in_top3"].append(float(winner in order[:3]))
        vectors["top7_podium_coverage"].append(len(podium & top7) / max(1, len(podium)))
        vectors["top7_full_podium"].append(float(len(podium) == 3 and podium.issubset(top7)))
        keys.append(group[0]["race_key"])
        segments["venue"].append(str(group[0].get("venue") or "unknown"))
        segments["distance_band"].append(distance_band(group[0].get("distance_m")))
        segments["field_size_band"].append(field_band(group[0].get("field_size", len(group))))
    return keys, vectors, segments

def paired_ci(a, b, seed):
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(d)
    samples = np.empty(BOOTSTRAP_ITERATIONS)
    for start in range(0, BOOTSTRAP_ITERATIONS, 100):
        size = min(100, BOOTSTRAP_ITERATIONS-start)
        samples[start:start+size] = d[rng.integers(0, n, size=(size,n))].mean(axis=1)
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return {"delta_pp_B_minus_A": round(float(d.mean()*100),3),
            "ci95_pp": [round(float(lo*100),3),round(float(hi*100),3)]}

def summarize_segment(a, b, seed):
    result = {"races": len(a["top1"]), "A": {}, "B": {}, "paired_B_minus_A": {}}
    for i,m in enumerate(METRICS):
        result["A"][m] = round(float(np.mean(a[m])*100),3)
        result["B"][m] = round(float(np.mean(b[m])*100),3)
        result["paired_B_minus_A"][m] = paired_ci(a[m],b[m],seed+i)
    return result

def main():
    suite.RAW_NUMERIC = [f for f in suite.RAW_NUMERIC if f not in base.EXCLUDED_BODY_WEIGHT]
    suite.RAW_FEATURES = suite.RAW_NUMERIC + suite.CATEGORICAL
    features_a = list(suite.RAW_FEATURES)
    if len(features_a) != 45 or not REMOVED_B.issubset(features_a):
        raise RuntimeError(f"Unexpected feature schema: {len(features_a)} features")
    features_b = [f for f in features_a if f not in REMOVED_B]
    if len(features_b) != 43: raise RuntimeError("B feature count is not 43")

    races, horses, scope = base.load_flat()
    rows, dates = suite.build_rows(races, horses)
    rows_by_date = defaultdict(list)
    for row in rows: rows_by_date[dt.date.fromisoformat(row["date"])].append(row)

    pooled = {
        dim: defaultdict(lambda: {"A": {m: [] for m in METRICS}, "B": {m: [] for m in METRICS}})
        for dim in ("venue","distance_band","field_size_band")
    }
    fold_reports=[]
    for fold_i,(name,start,end) in enumerate(FOLDS):
        valid_dates=[d for d in dates if start <= d <= end]
        train_dates=[d for d in dates if d < start]
        if len(train_dates)<180 or not valid_dates: raise RuntimeError(f"{name}: invalid date coverage")
        cut=int(len(train_dates)*0.90)
        fit_dates=set(train_dates[:cut]); early_dates=set(train_dates[cut:])
        def collect(ds):
            return [r for day in sorted(ds) for r in rows_by_date.get(day,[])]
        fit_rows=collect(fit_dates)
        fit_groups=suite.group_rows(fit_rows)
        early_groups=suite.group_rows(collect(early_dates))
        valid_groups=suite.group_rows(collect(set(valid_dates)))
        if len(valid_groups)<100: raise RuntimeError(f"{name}: insufficient validation races")
        maps={k:{v:i for i,v in enumerate(sorted({r[k] for r in fit_rows}))} for k in suite.CATEGORICAL}
        scored_by_model={}
        for label,features in (("A",features_a),("B",features_b)):
            model=base.fit_ranker(fit_groups,early_groups,features,maps)
            scored=base.predict_groups(model,valid_groups,features,maps)
            keys,vectors,segments=race_vectors(scored)
            scored_by_model[label]=(keys,vectors,segments)
        if scored_by_model["A"][0] != scored_by_model["B"][0] or scored_by_model["A"][2] != scored_by_model["B"][2]:
            raise RuntimeError(f"{name}: A/B alignment mismatch")
        seg_reports={}
        for dim in ("venue","distance_band","field_size_band"):
            values=scored_by_model["A"][2][dim]
            seg_reports[dim]={}
            for segment in sorted(set(values)):
                indices=[i for i,v in enumerate(values) if v==segment]
                if not indices: continue
                a={m:[scored_by_model["A"][1][m][i] for i in indices] for m in METRICS}
                b={m:[scored_by_model["B"][1][m][i] for i in indices] for m in METRICS}
                # Collect individual race-level paired observations across folds.
                for label,v in (("A",a),("B",b)):
                    for m in METRICS: pooled[dim][segment][label][m].extend(v[m])
                seg_reports[segment]=summarize_segment(a,b,20261114+fold_i*1009+len(segment))
        fold_reports.append({"fold":name,"validation_start":min(valid_dates).isoformat(),
            "validation_end":max(valid_dates).isoformat(),"races":len(valid_groups),
            "segments":seg_reports})
        print("SEGMENT_FOLD "+json.dumps(fold_reports[-1],ensure_ascii=False),flush=True)

    pooled_report={}
    for dim,segments in pooled.items():
        pooled_report[dim]={}
        for segment,data in sorted(segments.items()):
            n=len(data["A"]["top1"])
            if n<MIN_SEGMENT_RACES: continue
            pooled_report[dim][segment]=summarize_segment(data["A"],data["B"],20261114+len(segment)+len(dim))
    report={
        "generated_at":dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope":scope,
        "data":{"runner_rows":len(rows),"dates":len(dates),
                "validation_races":sum(f["races"] for f in fold_reports)},
        "models":{"A":{"features":45},"B":{"features":43,"removed":sorted(REMOVED_B)}},
        "segment_definitions":{
            "venue":"Official venue string from race metadata",
            "distance_band":["<1200m","1200-1399m","1400-1599m","1600-1799m","1800-1999m","2000m+"],
            "field_size_band":["8頭以下","9-12頭","13頭以上"],
            "minimum_pooled_races":MIN_SEGMENT_RACES
        },
        "folds":fold_reports,
        "pooled_segments":pooled_report,
        "limitations":[
            "Descriptive subgroup diagnostics, not a new model-selection set.",
            "The same historical periods have been inspected in prior project work; this is not a pristine holdout.",
            "Segment comparisons are exploratory multiple comparisons; small subgroup effects require confirmation.",
            "Metrics describe ranking only, not profitability.",
            "No production model, live weights, database, or ticket logic is modified."
        ]
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print("SEGMENT_ANALYSIS_COMPLETE "+json.dumps({"data":report["data"],
        "segment_counts":{d:{k:v["races"] for k,v in s.items()} for d,s in pooled_report.items()}},
        ensure_ascii=False),flush=True)

if __name__=="__main__": main()
