#!/usr/bin/env python3
"""Locked forward validation: A vs B on November 2026, trained only through October."""
from __future__ import annotations
import datetime as dt
import json
import random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import research_flat_lambdarank_feature_ablation as base
import nar_ml_challenger_backtest as suite

OUT = ROOT / "data" / "flat-lambdarank-november-forward-validation.json"
TEST_START = dt.date(2026, 11, 1)
TEST_END = dt.date(2026, 11, 30)
MIN_TEST_RACES = 300
REMOVED = {"race_rel_recent_finish_pct", "race_rel_recent_top3_rate"}

def paired_bootstrap(base_values, candidate_values, seed, iterations=10000):
    if len(base_values) != len(candidate_values) or not base_values:
        raise RuntimeError("Paired race arrays are empty or misaligned")
    diffs = [candidate_values[i] - base_values[i] for i in range(len(base_values))]
    n = len(diffs)
    rng = random.Random(seed)
    samples = sorted(
        sum(diffs[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(iterations)
    )
    return {
        "races": n,
        "delta_pp": round(100 * sum(diffs) / n, 3),
        "ci95_pp": [
            round(100 * samples[int(.025 * (iterations - 1))], 3),
            round(100 * samples[int(.975 * (iterations - 1))], 3),
        ],
        "method": "paired race bootstrap",
        "iterations": iterations,
    }

def race_vectors(scored):
    vectors = {"top1": [], "winner_in_top3": [],
               "top7_podium_coverage": [], "top7_full_podium": []}
    for group, scores in scored:
        winner = next((i for i, r in enumerate(group) if r.get("winner")), None)
        if winner is None:
            continue
        order = sorted(range(len(group)),
                       key=lambda i: (-float(scores[i]), int(group[i]["horse_no"])))
        actual = {i for i, r in enumerate(group)
                  if r.get("finish") is not None and int(r["finish"]) <= 3}
        selected = set(order[:7])
        vectors["top1"].append(float(order[0] == winner))
        vectors["winner_in_top3"].append(float(winner in order[:3]))
        vectors["top7_podium_coverage"].append(
            len(actual & selected) / max(1, len(actual)))
        vectors["top7_full_podium"].append(float(len(actual) == 3 and len(actual & selected) == 3))
    return vectors

def main():
    suite.RAW_NUMERIC = [f for f in suite.RAW_NUMERIC if f not in base.EXCLUDED_BODY_WEIGHT]
    suite.RAW_FEATURES = suite.RAW_NUMERIC + suite.CATEGORICAL
    races, horses, scope = base.load_flat()
    rows, dates = suite.build_rows(races, horses)
    dates = sorted(dates)
    test_dates = [d for d in dates if TEST_START <= d <= TEST_END]
    train_dates = [d for d in dates if d < TEST_START]
    if not test_dates or max(test_dates) < TEST_END:
        raise RuntimeError(
            f"November archive incomplete: observed {min(test_dates) if test_dates else 'none'}"
            f" through {max(test_dates) if test_dates else 'none'}; require {TEST_END}."
        )
    def rows_for(ds):
        selected = set(ds)
        return [r for r in rows if dt.date.fromisoformat(r["date"]) in selected]
    train_date_list = sorted(train_dates)
    if len(train_date_list) < 100:
        raise RuntimeError("Insufficient pre-November training dates")
    fit_cut = int(len(train_date_list) * .90)
    fit_dates, early_dates = set(train_date_list[:fit_cut]), set(train_date_list[fit_cut:])
    fit_rows = rows_for(fit_dates)
    fit_groups = suite.group_rows(fit_rows)
    early_groups = suite.group_rows(rows_for(early_dates))
    test_groups = suite.group_rows(rows_for(test_dates))
    test_races = sum(1 for group in test_groups if any(r.get("winner") for r in group))
    if test_races < MIN_TEST_RACES:
        raise RuntimeError(f"Only {test_races} valid November races; require {MIN_TEST_RACES}.")
    category_maps = {
        k: {v: i for i, v in enumerate(sorted({r[k] for r in fit_rows}))}
        for k in suite.CATEGORICAL
    }
    all_features = list(suite.RAW_FEATURES)
    if not REMOVED.issubset(set(all_features)):
        raise RuntimeError("Expected relative-finish features are missing")
    variants = {
        "A_baseline_45": all_features,
        "B_without_relative_finish_43": [f for f in all_features if f not in REMOVED],
    }
    results, vectors = {}, {}
    for name, features in variants.items():
        model = base.fit_ranker(fit_groups, early_groups, features, category_maps)
        scored = base.predict_groups(model, test_groups, features, category_maps)
        results[name] = {
            "feature_count": len(features),
            "removed_features": sorted(set(all_features) - set(features)),
            "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
            "november_metrics": suite.evaluate_scores(name, scored),
        }
        vectors[name] = race_vectors(scored)
        print("FORWARD_RESULT " + json.dumps({"model": name, **results[name]},
                                               ensure_ascii=False), flush=True)
    paired = {
        metric: paired_bootstrap(vectors["A_baseline_45"][metric],
                                 vectors["B_without_relative_finish_43"][metric],
                                 seed=700 + i)
        for i, metric in enumerate(vectors["A_baseline_45"])
    }
    report = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "completed",
        "scope": scope,
        "period": {"train_end": max(train_dates).isoformat(),
                   "evaluation_start": TEST_START.isoformat(),
                   "evaluation_end": TEST_END.isoformat()},
        "data": {"source_races": len(races), "runner_rows": len(rows),
                 "source_dates": len(dates), "fit_races": len(fit_groups),
                 "early_stopping_races": len(early_groups),
                 "evaluation_races": test_races},
        "protocol": {
            "A": "Original 45-feature model.",
            "B": "Same model and parameters, excluding only race_rel_recent_finish_pct and race_rel_recent_top3_rate.",
            "training_cutoff": "2026-10-31",
            "November_results_used_for_training": False,
            "same_races_and_same_parameters": True,
            "category_maps_fit_on_fit_period_only": True,
            "same_day_result_withholding": "Inherited from chronological feature builder.",
            "current_race_body_weight_excluded": True,
            "prior_body_weight_avg_retained": "prior_body_weight_avg" in all_features,
            "no_holdout_parameter_tuning": True,
        },
        "results": results,
        "paired_comparisons_B_minus_A": paired,
        "limitations": [
            "One calendar month is an independent temporal confirmation, not proof of long-term profitability.",
            "Ranking metrics do not establish positive betting returns.",
            "No production model, ticket generator, or live setting was changed.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("FORWARD_VALIDATION_COMPLETE " + json.dumps({
        "period": report["period"], "data": report["data"],
        "results": results, "paired": paired,
    }, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
