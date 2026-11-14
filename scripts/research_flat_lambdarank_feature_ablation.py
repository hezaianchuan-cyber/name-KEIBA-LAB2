#!/usr/bin/env python3
"""Research-only LambdaRank feature-group ablation on flat NAR racing.

Compare the current 45-feature baseline with one-group-at-a-time removals.
Chronological split and all training parameters are held fixed. Holdout is
evaluated only after the feature groups have been defined; no tuning occurs.
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

OUT = ROOT / "data" / "flat-lambdarank-feature-ablation.json"
EXCLUDED_BODY_WEIGHT = {
    "body_weight", "body_weight_change", "body_weight_delta_pct",
    "race_rel_body_weight_delta_pct",
}
BANEI_VENUE = re.compile(r"帯広|ばんえい", re.IGNORECASE)
PARAMS = {
    "objective": "lambdarank", "metric": "ndcg", "n_estimators": 900,
    "learning_rate": 0.035, "num_leaves": 15, "min_child_samples": 80,
    "reg_lambda": 2.0, "random_state": 42, "n_jobs": 4, "verbosity": -1,
    "label_gain": [0, 1, 2, 3, 4],
}
GROUPS = {
    "recent_form": [
        "prior_runs", "career_win_rate", "career_top3_rate",
        "recent_finish_pct", "recent_top3_rate", "recent_margin_quality",
        "recent_last3f_rank", "recent_time_per_km", "recent_time_trend",
        "recent_first_corner_pct", "recent_last_corner_pct", "recent_position_gain",
        "race_rel_recent_finish_pct", "race_rel_recent_top3_rate",
        "race_rel_recent_last_corner_pct", "race_rel_recent_last3f_rank",
        "race_rel_recent_time_per_km",
    ],
    "opponent_strength": [
        "recent_opponent_strength", "race_rel_recent_opponent_strength",
    ],
    "venue_distance_suitability": [
        "venue_runs", "venue_top3_rate", "distance_runs", "distance_top3_rate",
        "race_rel_venue_top3_rate", "race_rel_distance_top3_rate",
    ],
    "jockey": ["jockey_runs", "jockey_top3_rate", "race_rel_jockey_top3_rate", "jockey"],
}

def load_flat():
    races, horses, payouts, downloads = nb.load_archives()
    banei = {k: r for k, r in races.items()
             if BANEI_VENUE.search(str(r.get("venue") or ""))}
    flat = {k: r for k, r in races.items() if k not in banei}
    flat_horses = {k: v for k, v in horses.items() if k in flat}
    if not flat or any(BANEI_VENUE.search(str(r.get("venue") or "")) for r in flat.values()):
        raise RuntimeError("Venue-only flat filter failed")
    scope = {
        "source_races": len(races), "excluded_banei_venue_races": len(banei),
        "flat_races": len(flat),
        "flat_name_only_banei_word_races_retained": sum(
            1 for r in flat.values() if BANEI_VENUE.search(str(r.get("name") or ""))),
        "excluded_venues": sorted({str(r.get("venue") or "") for r in banei.values()}),
        "filter": "Venue only; title/sponsorship mentions do not remove a flat race.",
        "download_months": len(downloads),
    }
    print("FLAT_SCOPE " + json.dumps(scope, ensure_ascii=False), flush=True)
    return flat, flat_horses, scope

def flatten(groups, features, category_maps):
    import numpy as np
    data, sizes, labels = [], [], []
    for group in groups:
        sizes.append(len(group))
        data.extend(group)
        labels.extend([max(0, 4 - int(r["finish"])) for r in group])
    x = np.asarray([
        [r[k] if k not in suite.CATEGORICAL else category_maps[k].get(r[k], -1)
         for k in features] for r in data
    ], dtype=float)
    return x, np.asarray(labels, dtype=int), sizes

def fit_ranker(train_groups, early_groups, features, category_maps):
    from lightgbm import LGBMRanker, early_stopping, log_evaluation
    xtr, ytr, gtr = flatten(train_groups, features, category_maps)
    xdv, ydv, gdv = flatten(early_groups, features, category_maps)
    model = LGBMRanker(**PARAMS)
    model.fit(xtr, ytr, group=gtr, eval_set=[(xdv, ydv)], eval_group=[gdv],
              eval_at=[1, 3, 7],
              callbacks=[early_stopping(60, verbose=False), log_evaluation(0)])
    return model

def predict_groups(model, groups, features, category_maps):
    output = []
    for group in groups:
        x = [[r[k] if k not in suite.CATEGORICAL else category_maps[k].get(r[k], -1)
              for k in features] for r in group]
        output.append((group, model.predict(x)))
    return output

def paired_bootstrap(base, challenger, seed, iterations=3000):
    if len(base) != len(challenger):
        raise RuntimeError("Paired holdout race count mismatch")
    n = len(base)
    diffs = [challenger[i] - base[i] for i in range(n)]
    rng = random.Random(seed)
    samples = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n
                     for _ in range(iterations))
    return {
        "races": n, "mean_delta_pp": round(100 * sum(diffs) / n, 3),
        "ci95_pp": [round(100 * samples[int(.025 * (iterations - 1))], 3),
                    round(100 * samples[int(.975 * (iterations - 1))], 3)],
        "method": "paired race bootstrap", "iterations": iterations,
    }

def paired_metrics(base_scored, alt_scored):
    base_top1, alt_top1, base_top3, alt_top3 = [], [], [], []
    for (g0, s0), (g1, s1) in zip(base_scored, alt_scored):
        if [r["race_key"] for r in g0] != [r["race_key"] for r in g1]:
            raise RuntimeError("Holdout race alignment mismatch")
        winner = next((i for i, r in enumerate(g0) if r["winner"]), None)
        if winner is None:
            continue
        o0 = sorted(range(len(g0)), key=lambda i: (-float(s0[i]), int(g0[i]["horse_no"])))
        o1 = sorted(range(len(g1)), key=lambda i: (-float(s1[i]), int(g1[i]["horse_no"])))
        base_top1.append(float(o0[0] == winner)); alt_top1.append(float(o1[0] == winner))
        base_top3.append(float(winner in o0[:3])); alt_top3.append(float(winner in o1[:3]))
    return {
        "top1": paired_bootstrap(base_top1, alt_top1, 101),
        "winner_in_top3": paired_bootstrap(base_top3, alt_top3, 103),
    }

def main():
    suite.RAW_NUMERIC = [f for f in suite.RAW_NUMERIC if f not in EXCLUDED_BODY_WEIGHT]
    suite.RAW_FEATURES = suite.RAW_NUMERIC + suite.CATEGORICAL
    races, horses, scope = load_flat()
    rows, dates = suite.build_rows(races, horses)
    if len(dates) < 100:
        raise RuntimeError("Insufficient chronological dates")
    cut1, cut2 = int(len(dates) * .70), int(len(dates) * .80)
    train_dates, dev_dates, hold_dates = set(dates[:cut1]), set(dates[cut1:cut2]), set(dates[cut2:])
    def rows_for(ds):
        return [r for r in rows if dt.date.fromisoformat(r["date"]) in ds]
    train_rows = rows_for(train_dates)
    train_groups = suite.group_rows(train_rows)
    dev_groups = suite.group_rows(rows_for(dev_dates))
    holdout_groups = suite.group_rows(rows_for(hold_dates))
    early_groups = dev_groups[:len(dev_groups)//2]
    selection_groups = dev_groups[len(dev_groups)//2:]
    if not train_groups or not early_groups or not selection_groups or not holdout_groups:
        raise RuntimeError("Empty chronological split")
    category_maps = {
        k: {v: i for i, v in enumerate(sorted({r[k] for r in train_rows}))}
        for k in suite.CATEGORICAL
    }
    baseline_features = list(suite.RAW_FEATURES)
    for name, members in GROUPS.items():
        unknown = set(members) - set(baseline_features)
        if unknown:
            raise RuntimeError(f"Feature group {name} contains unknown features: {sorted(unknown)}")
    variants = {"baseline_all_features": baseline_features}
    for name, members in GROUPS.items():
        variants[f"without_{name}"] = [f for f in baseline_features if f not in set(members)]
    baseline_scored = None
    results = {}
    for index, (name, features) in enumerate(variants.items()):
        model = fit_ranker(train_groups, early_groups, features, category_maps)
        scored = predict_groups(model, holdout_groups, features, category_maps)
        metrics = suite.evaluate_scores(name, scored)
        if name == "baseline_all_features":
            baseline_scored = scored
        results[name] = {
            "feature_count": len(features), "removed_features": sorted(set(baseline_features)-set(features)),
            "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
            "holdout_metrics": metrics,
            "paired_vs_baseline": None if name == "baseline_all_features"
                else paired_metrics(baseline_scored, scored),
        }
        print("ABLATION_RESULT " + json.dumps({"variant": name, **results[name]}, ensure_ascii=False), flush=True)
    report = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": scope,
        "data": {
            "runner_rows": len(rows), "dates": len(dates),
            "train_races": len(train_groups), "development_races": len(dev_groups),
            "early_stopping_races": len(early_groups),
            "development_selection_races": len(selection_groups),
            "holdout_races": len(holdout_groups),
        },
        "split": {
            "train_end": dates[cut1-1].isoformat(),
            "development_start": dates[cut1].isoformat(),
            "development_end": dates[cut2-1].isoformat(),
            "holdout_start": dates[cut2].isoformat(),
            "holdout_end": dates[-1].isoformat(),
            "policy": "Chronological 70/10/20; identical split for every ablation.",
        },
        "features": {
            "baseline_count": len(baseline_features), "baseline_fields": baseline_features,
            "groups_removed_one_at_a_time": GROUPS,
            "excluded_current_race_body_weight": sorted(EXCLUDED_BODY_WEIGHT),
            "prior_body_weight_avg_retained": "prior_body_weight_avg" in baseline_features,
        },
        "training": {
            "fixed_parameters": PARAMS,
            "only_varied_dimension": "feature group removal",
            "relevance": "1着=3、2着=2、3着=1、4着以下=0",
            "holdout_used_for_tuning": False,
            "note": "Pure ablation diagnostic; no feature group or parameter selected using holdout.",
        },
        "results": results,
        "limitations": [
            "Research only; no production model, ticket generator, or canonical report changed.",
            "Ablation measures dependence on existing feature groups, not the value of newly engineered features.",
            "Ranking metrics do not establish betting profitability.",
            "No final odds or payout values are used as features.",
            "Historical holdout cannot guarantee future performance.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("FEATURE_ABLATION_COMPLETE " + json.dumps({
        "data": report["data"], "results": results,
    }, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
