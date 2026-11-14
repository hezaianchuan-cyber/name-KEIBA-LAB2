#!/usr/bin/env python3
"""Pre-registered research-only A/B validation on November 2026 NAR results.

A is the frozen 45-feature LambdaRank baseline. B removes only the two
race-relative recent-finish fields. Train/early-stop data ends before
2026-11-01; November is evaluated once and is never used for tuning.
"""
from __future__ import annotations
import datetime as dt
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import research_flat_lambdarank_feature_ablation as base
import nar_ml_challenger_backtest as suite

OUT = ROOT / "data" / "flat-lambdarank-november-fresh-validation.json"
CUTOFF = dt.date(2026, 11, 1)
RELATIVE_FINISH = {"race_rel_recent_finish_pct", "race_rel_recent_top3_rate"}
MIN_VALIDATION_RACES = 100
MIN_VALIDATION_DATES = 10

def race_vectors(scored):
    metrics = {k: [] for k in ("top1", "winner_in_top3", "top7_podium_coverage", "top7_full_podium")}
    for group, scores in scored:
        order = sorted(range(len(group)), key=lambda i: (-float(scores[i]), int(group[i]["horse_no"])))
        winner = next((i for i, row in enumerate(group) if row["winner"]), None)
        if winner is None:
            continue
        podium = {i for i, row in enumerate(group) if int(row["finish"]) <= 3}
        top7 = set(order[:7])
        metrics["top1"].append(float(order[0] == winner))
        metrics["winner_in_top3"].append(float(winner in order[:3]))
        metrics["top7_podium_coverage"].append(len(podium & top7) / max(1, len(podium)))
        metrics["top7_full_podium"].append(float(podium.issubset(top7)))
    return metrics

def paired_delta(a, b, seed, iterations=5000):
    if len(a) != len(b) or not a:
        raise RuntimeError("Paired evaluation vectors are empty or misaligned")
    diffs = [b[i] - a[i] for i in range(len(a))]
    rng = random.Random(seed)
    samples = sorted(sum(diffs[rng.randrange(len(diffs))] for _ in diffs) / len(diffs)
                     for _ in range(iterations))
    return {
        "races": len(diffs),
        "delta_pp": round(100 * sum(diffs) / len(diffs), 3),
        "ci95_pp": [
            round(100 * samples[int(.025 * (iterations - 1))], 3),
            round(100 * samples[int(.975 * (iterations - 1))], 3),
        ],
        "method": "paired race bootstrap",
        "iterations": iterations,
    }

def main():
    suite.RAW_NUMERIC = [f for f in suite.RAW_NUMERIC if f not in base.EXCLUDED_BODY_WEIGHT]
    suite.RAW_FEATURES = suite.RAW_NUMERIC + suite.CATEGORICAL
    races, horses, scope = base.load_flat()
    rows, all_dates = suite.build_rows(races, horses)
    train_dates = [d for d in all_dates if d < CUTOFF]
    validation_dates = [d for d in all_dates if d >= CUTOFF]
    if not train_dates:
        raise RuntimeError("No pre-November training dates found")
    if len(validation_dates) < MIN_VALIDATION_DATES:
        raise RuntimeError(
            f"Fresh November archive not ready: validation_dates={len(validation_dates)}; "
            f"need at least {MIN_VALIDATION_DATES}. No model was evaluated."
        )
    def rows_for(ds):
        selected = set(ds)
        return [r for r in rows if dt.date.fromisoformat(r["date"]) in selected]

    train_date_list = sorted(train_dates)
    fit_cut = int(len(train_date_list) * .90)
    fit_dates, early_dates = set(train_date_list[:fit_cut]), set(train_date_list[fit_cut:])
    fit_groups = suite.group_rows(rows_for(fit_dates))
    early_groups = suite.group_rows(rows_for(early_dates))
    validation_groups = suite.group_rows(rows_for(validation_dates))
    if not fit_groups or not early_groups or not validation_groups:
        raise RuntimeError("Empty fit, early-stopping, or fresh validation split")
    if len(validation_groups) < MIN_VALIDATION_RACES:
        raise RuntimeError(
            f"Insufficient fresh validation races={len(validation_groups)}; "
            f"need at least {MIN_VALIDATION_RACES}. No model was evaluated."
        )

    fit_rows = rows_for(fit_dates)
    category_maps = {
        k: {v: i for i, v in enumerate(sorted({r[k] for r in fit_rows}))}
        for k in suite.CATEGORICAL
    }
    all_features = list(suite.RAW_FEATURES)
    if not RELATIVE_FINISH.issubset(set(all_features)):
        raise RuntimeError("Pre-registered B features are not present in the source feature set")
    variants = {
        "A_baseline_45": all_features,
        "B_without_relative_finish_43": [f for f in all_features if f not in RELATIVE_FINISH],
    }
    results, vectors = {}, {}
    for name, features in variants.items():
        model = base.fit_ranker(fit_groups, early_groups, features, category_maps)
        scored = base.predict_groups(model, validation_groups, features, category_maps)
        vectors[name] = race_vectors(scored)
        results[name] = {
            "feature_count": len(features),
            "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
            "validation_metrics": suite.evaluate_scores(name, scored),
        }
        print("FRESH_VALIDATION_RESULT " + json.dumps(
            {"variant": name, **results[name]}, ensure_ascii=False), flush=True)

    paired = {
        metric: paired_delta(vectors["A_baseline_45"][metric],
                             vectors["B_without_relative_finish_43"][metric],
                             seed=700 + i)
        for i, metric in enumerate(vectors["A_baseline_45"])
    }
    report = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": scope,
        "pre_registration": {
            "cutoff": CUTOFF.isoformat(),
            "candidate": "B_without_relative_finish_43",
            "removed_features": sorted(RELATIVE_FINISH),
            "no_parameter_or_feature_tuning_on_validation": True,
            "minimum_validation_dates": MIN_VALIDATION_DATES,
            "minimum_validation_races": MIN_VALIDATION_RACES,
        },
        "data": {
            "runner_rows": len(rows),
            "fit_races": len(fit_groups),
            "early_stopping_races": len(early_groups),
            "fresh_validation_races": len(validation_groups),
            "fresh_validation_dates": len(validation_dates),
            "validation_start": min(validation_dates).isoformat(),
            "validation_end": max(validation_dates).isoformat(),
            "validation_race_count_is_group_count": True,
        },
        "training": {
            "fixed_parameters": base.PARAMS,
            "train_period_end": max(train_dates).isoformat(),
            "current_race_body_weight_excluded": True,
            "prior_body_weight_avg_retained": "prior_body_weight_avg" in all_features,
            "final_odds_and_payout_not_used_as_features": True,
        },
        "results": results,
        "paired_B_minus_A": paired,
        "limitations": [
            "Research-only; no production model or live ticket settings changed.",
            "One fresh temporal validation period does not prove long-term profitability.",
            "No betting ROI claim is made by ranking metrics.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("FRESH_NOVEMBER_VALIDATION_COMPLETE " + json.dumps({
        "data": report["data"], "results": results, "paired_B_minus_A": paired,
    }, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
