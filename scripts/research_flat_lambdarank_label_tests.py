#!/usr/bin/env python3
"""Research-only comparison of LambdaRank relevance-label schemes.

Uses the corrected venue-only Banei exclusion, same 45 leakage-audited features,
same chronological 70/10/20 split, and identical LightGBM parameters.
No production files are changed.
"""
from __future__ import annotations
import datetime as dt
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import nar_backtest as nb
import nar_ml_challenger_backtest as suite

OUT = ROOT / "data" / "flat-lambdarank-label-tests.json"
EXCLUDED_BODY_WEIGHT = {
    "body_weight", "body_weight_change", "body_weight_delta_pct",
    "race_rel_body_weight_delta_pct",
}
BANEI_VENUE = re.compile(r"帯広|ばんえい", re.IGNORECASE)
LABELS = {
    "baseline_3_2_1": {"description": "1着=3、2着=2、3着=1、4着以下=0", "values": (3, 2, 1, 0)},
    "podium_binary": {"description": "3着以内=1、4着以下=0（着順差を無視）", "values": (1, 1, 1, 0)},
    "podium_weighted_3_2_2": {"description": "1着=3、2・3着=2、4着以下=0", "values": (3, 2, 2, 0)},
    "winner_emphasis_4_2_2": {"description": "1着=4、2・3着=2、4着以下=0", "values": (4, 2, 2, 0)},
}
PARAMS = {
    "objective": "lambdarank", "metric": "ndcg", "n_estimators": 900,
    "learning_rate": 0.035, "num_leaves": 15, "min_child_samples": 80,
    "reg_lambda": 2.0, "random_state": 42, "n_jobs": 4, "verbosity": -1,
    "label_gain": [0, 1, 2, 3, 4],
}

def load_flat():
    races, horses, payouts, downloads = nb.load_archives()
    banei = {k:r for k,r in races.items() if BANEI_VENUE.search(str(r.get("venue") or ""))}
    flat = {k:r for k,r in races.items() if k not in banei}
    flat_horses = {k:v for k,v in horses.items() if k in flat}
    flat_payouts = {k:v for k,v in payouts.items() if k in flat}
    name_only = {k:r for k,r in flat.items() if BANEI_VENUE.search(str(r.get("name") or ""))}
    if not flat or any(BANEI_VENUE.search(str(r.get("venue") or "")) for r in flat.values()):
        raise RuntimeError("Venue-only flat filter failed")
    audit = {
        "source_races": len(races), "excluded_banei_venue_races": len(banei),
        "flat_races": len(flat), "flat_name_only_banei_word_races_retained": len(name_only),
        "excluded_venues": sorted({str(r.get("venue") or "") for r in banei.values()}),
        "filter": "Venue only; title/sponsorship mentions do not remove a flat race.",
        "download_months": len(downloads),
    }
    print("FLAT_SCOPE " + json.dumps(audit, ensure_ascii=False), flush=True)
    return flat, flat_horses, audit

def fit_with_labels(train_groups, early_groups, features, category_maps, label_values):
    import numpy as np
    from lightgbm import LGBMRanker, early_stopping, log_evaluation
    def flatten(groups):
        data, sizes, labels = [], [], []
        for group in groups:
            sizes.append(len(group))
            data.extend(group)
            for r in group:
                finish = int(r["finish"])
                labels.append(label_values[0] if finish == 1 else
                              label_values[1] if finish == 2 else
                              label_values[2] if finish == 3 else label_values[3])
        x = np.asarray([[r[k] if k not in suite.CATEGORICAL else category_maps[k].get(r[k], -1)
                         for k in features] for r in data], dtype=float)
        return x, np.asarray(labels, dtype=int), sizes
    xtr, ytr, gtr = flatten(train_groups)
    xdv, ydv, gdv = flatten(early_groups)
    if max(int(ytr.max()), int(ydv.max())) >= len(PARAMS["label_gain"]):
        raise RuntimeError("label_gain does not cover the chosen relevance labels")
    model = LGBMRanker(**PARAMS)
    model.fit(xtr, ytr, group=gtr, eval_set=[(xdv, ydv)], eval_group=[gdv],
              eval_at=[1, 3, 7],
              callbacks=[early_stopping(60, verbose=False), log_evaluation(0)])
    return model

def bootstrap_delta(base, challenger, seed=20261114, iterations=3000):
    n = min(len(base), len(challenger))
    if n < 30:
        return {"races": n, "ci95": None, "reason": "fewer_than_30_races"}
    diffs = [challenger[i] - base[i] for i in range(n)]
    rng = random.Random(seed)
    samples = []
    for _ in range(iterations):
        samples.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    samples.sort()
    return {"races": n, "mean_delta_pp": round(100 * sum(diffs) / n, 3),
            "ci95_pp": [round(100 * samples[int(.025*(iterations-1))], 3),
                        round(100 * samples[int(.975*(iterations-1))], 3)],
            "method": "paired race bootstrap", "iterations": iterations}

def main():
    suite.RAW_NUMERIC = [f for f in suite.RAW_NUMERIC if f not in EXCLUDED_BODY_WEIGHT]
    suite.RAW_FEATURES = suite.RAW_NUMERIC + suite.CATEGORICAL
    races, horses, scope = load_flat()
    rows, dates = suite.build_rows(races, horses)
    if len(dates) < 100:
        raise RuntimeError("Insufficient chronological dates")
    cut1, cut2 = int(len(dates)*.70), int(len(dates)*.80)
    train_dates, dev_dates, hold_dates = set(dates[:cut1]), set(dates[cut1:cut2]), set(dates[cut2:])
    def selected(ds):
        return [r for r in rows if dt.date.fromisoformat(r["date"]) in ds]
    train_rows = selected(train_dates)
    train = suite.group_rows(train_rows)
    dev = suite.group_rows(selected(dev_dates))
    holdout = suite.group_rows(selected(hold_dates))
    early_stop = dev[:len(dev)//2]
    if not train or not early_stop or not holdout:
        raise RuntimeError("Empty chronological split")
    category_maps = {k:{v:i for i,v in enumerate(sorted({r[k] for r in train_rows}))}
                     for k in suite.CATEGORICAL}
    features = list(suite.RAW_FEATURES)
    scored_by_name, metrics = {}, {}
    for name, spec in LABELS.items():
        model = fit_with_labels(train, early_stop, features, category_maps, spec["values"])
        scored = []
        for group in holdout:
            x = [[r[k] if k not in suite.CATEGORICAL else category_maps[k].get(r[k], -1)
                  for k in features] for r in group]
            scored.append((group, model.predict(x)))
        scored_by_name[name] = scored
        metrics[name] = suite.evaluate_scores(name, scored)
        metrics[name]["label_description"] = spec["description"]
        metrics[name]["label_values_1st_2nd_3rd_other"] = list(spec["values"])
        metrics[name]["best_iteration"] = int(getattr(model, "best_iteration_", 0) or 0)
        print("LABEL_RESULT " + json.dumps({"name":name, "metrics":metrics[name]}, ensure_ascii=False), flush=True)
    base = scored_by_name["baseline_3_2_1"]
    deltas = {}
    for name, scored in scored_by_name.items():
        if name == "baseline_3_2_1":
            continue
        base_top1, alt_top1, base_top3, alt_top3 = [], [], [], []
        for (g0,s0),(g1,s1) in zip(base,scored):
            winner = next((i for i,r in enumerate(g0) if r["winner"]), None)
            if winner is None: continue
            o0=sorted(range(len(g0)),key=lambda i:(-float(s0[i]),g0[i]["horse_no"]))
            o1=sorted(range(len(g1)),key=lambda i:(-float(s1[i]),g1[i]["horse_no"]))
            base_top1.append(float(o0[0]==winner)); alt_top1.append(float(o1[0]==winner))
            base_top3.append(float(winner in o0[:3])); alt_top3.append(float(winner in o1[:3]))
        deltas[name] = {
            "top1": bootstrap_delta(base_top1,alt_top1,seed=101),
            "winner_in_top3": bootstrap_delta(base_top3,alt_top3,seed=103),
        }
    report = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": scope,
        "data": {"runner_rows":len(rows),"dates":len(dates),"train_races":len(train),
                 "development_races":len(dev),"early_stopping_races":len(early_stop),
                 "holdout_races":len(holdout)},
        "split": {"train_end":dates[cut1-1].isoformat(),"development_start":dates[cut1].isoformat(),
                  "development_end":dates[cut2-1].isoformat(),"holdout_start":dates[cut2].isoformat(),
                  "holdout_end":dates[-1].isoformat()},
        "features": {"count":len(features),"fields":features,
                     "excluded_current_race_body_weight":sorted(EXCLUDED_BODY_WEIGHT),
                     "prior_body_weight_avg_retained":"prior_body_weight_avg" in features},
        "training": {"parameters":PARAMS,"gain_policy":"Explicit linear label_gain [0,1,2,3,4] for all variants; all other parameters identical.",
                     "holdout_used_for_tuning":False},
        "labels":metrics,"paired_deltas_vs_baseline":deltas,
        "interpretation":"Research-only ranking label comparison. All variants were predeclared; same data, features, split and model parameters. Holdout results are not proof of future profitability. No production changes.",
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print("LABEL_TEST_COMPLETE " + json.dumps({"data":report["data"],"labels":metrics,
          "paired_deltas_vs_baseline":deltas},ensure_ascii=False),flush=True)

if __name__ == "__main__":
    main()
