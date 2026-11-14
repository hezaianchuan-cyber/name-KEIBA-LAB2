#!/usr/bin/env python3
"""Development-only comparison of three predeclared LambdaRank feature sets."""
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

OUT = ROOT / "data" / "flat-lambdarank-recent-form-candidate-compare.json"
RELATIVE_FINISH = ["race_rel_recent_finish_pct", "race_rel_recent_top3_rate"]
CLOSING_KICK = ["recent_last3f_rank", "race_rel_recent_last3f_rank"]

def bootstrap_delta(base_values, candidate_values, seed, iterations=5000):
    if len(base_values) != len(candidate_values):
        raise RuntimeError("Paired race arrays have different lengths")
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

def race_level_scores(scored):
    top1, winner_top3, podium_coverage, full_podium = [], [], [], []
    for group, scores in scored:
        order = sorted(
            range(len(group)),
            key=lambda i: (-float(scores[i]), int(group[i]["horse_no"]))
        )
        winner = next((i for i, row in enumerate(group) if row["winner"]), None)
        if winner is None:
            continue
        actual_podium = {i for i, row in enumerate(group) if int(row["finish"]) <= 3}
        selected7 = set(order[:7])
        top1.append(float(order[0] == winner))
        winner_top3.append(float(winner in order[:3]))
        podium_coverage.append(len(actual_podium & selected7) / max(1, len(actual_podium)))
        full_podium.append(float(actual_podium.issubset(selected7)))
    return {
        "top1": top1,
        "winner_in_top3": winner_top3,
        "top7_podium_coverage": podium_coverage,
        "top7_full_podium": full_podium,
    }

def main():
    suite.RAW_NUMERIC = [f for f in suite.RAW_NUMERIC if f not in base.EXCLUDED_BODY_WEIGHT]
    suite.RAW_FEATURES = suite.RAW_NUMERIC + suite.CATEGORICAL
    races, horses, scope = base.load_flat()
    rows, dates = suite.build_rows(races, horses)
    if len(dates) < 100:
        raise RuntimeError("Insufficient chronological dates")
    cut1, cut2 = int(len(dates) * .70), int(len(dates) * .80)
    train_dates = set(dates[:cut1])
    dev_dates = set(dates[cut1:cut2])
    holdout_dates = set(dates[cut2:])

    def rows_for(date_set):
        return [r for r in rows if dt.date.fromisoformat(r["date"]) in date_set]

    train_date_list = sorted(train_dates)
    fit_cut = int(len(train_date_list) * .90)
    fit_dates = set(train_date_list[:fit_cut])
    early_dates = set(train_date_list[fit_cut:])
    fit_groups = suite.group_rows(rows_for(fit_dates))
    early_groups = suite.group_rows(rows_for(early_dates))
    dev_groups = suite.group_rows(rows_for(dev_dates))
    if not fit_groups or not early_groups or not dev_groups:
        raise RuntimeError("Empty fit, early-stopping, or development split")

    fit_rows = rows_for(fit_dates)
    category_maps = {
        k: {v: i for i, v in enumerate(sorted({r[k] for r in fit_rows}))}
        for k in suite.CATEGORICAL
    }
    all_features = list(suite.RAW_FEATURES)
    for group in (RELATIVE_FINISH, CLOSING_KICK):
        unknown = set(group) - set(all_features)
        if unknown:
            raise RuntimeError(f"Unknown features: {sorted(unknown)}")

    variants = {
        "A_baseline_45": all_features,
        "B_without_relative_finish": [
            f for f in all_features if f not in set(RELATIVE_FINISH)
        ],
        "C_without_relative_finish_and_closing_kick": [
            f for f in all_features
            if f not in set(RELATIVE_FINISH + CLOSING_KICK)
        ],
    }
    scored_by_name, result_by_name, vectors = {}, {}, {}
    for name, features in variants.items():
        model = base.fit_ranker(fit_groups, early_groups, features, category_maps)
        scored = base.predict_groups(model, dev_groups, features, category_maps)
        metrics = suite.evaluate_scores(name, scored)
        vectors[name] = race_level_scores(scored)
        scored_by_name[name] = scored
        result_by_name[name] = {
            "feature_count": len(features),
            "removed_features": sorted(set(all_features) - set(features)),
            "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
            "development_metrics": metrics,
        }
        print("CANDIDATE_RESULT " + json.dumps(
            {"variant": name, **result_by_name[name]}, ensure_ascii=False), flush=True)

    pairwise = {}
    names = list(variants)
    for candidate in names[1:]:
        pairwise[candidate + "_vs_A_baseline"] = {
            metric: bootstrap_delta(vectors[names[0]][metric], vectors[candidate][metric],
                                    seed=100 + i)
            for i, metric in enumerate(vectors[names[0]])
        }
    pairwise["C_vs_B"] = {
        metric: bootstrap_delta(vectors["B_without_relative_finish"][metric],
                                vectors["C_without_relative_finish_and_closing_kick"][metric],
                                seed=200 + i)
        for i, metric in enumerate(vectors["B_without_relative_finish"])
    }

    report = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": scope,
        "data": {
            "runner_rows": len(rows), "dates": len(dates),
            "fit_races": len(fit_groups), "early_stopping_races": len(early_groups),
            "development_races": len(dev_groups),
            "holdout_races_reserved": sum(1 for r in rows
                if dt.date.fromisoformat(r["date"]) in holdout_dates),
            "holdout_evaluated": False,
        },
        "split": {
            "fit_start": train_date_list[0].isoformat(),
            "fit_end": train_date_list[fit_cut-1].isoformat(),
            "early_stopping_start": train_date_list[fit_cut].isoformat(),
            "early_stopping_end": train_date_list[-1].isoformat(),
            "development_start": dates[cut1].isoformat(),
            "development_end": dates[cut2-1].isoformat(),
            "holdout_start_reserved": dates[cut2].isoformat(),
            "holdout_end_reserved": dates[-1].isoformat(),
            "policy": "Chronological 70/10/20; train portion split 90/10; all candidates compared on identical development races.",
        },
        "candidates": {
            "A": "Current 45-feature baseline.",
            "B": "Remove race-relative recent finish and top-three-rate features.",
            "C": "Remove the two relative-finish features plus recent and race-relative last-3f rank.",
        },
        "training": {
            "fixed_parameters": base.PARAMS,
            "only_varied_dimension": "predeclared feature set",
            "holdout_used_for_tuning_or_evaluation": False,
            "current_race_body_weight_excluded": True,
            "prior_body_weight_avg_retained": "prior_body_weight_avg" in all_features,
        },
        "results": result_by_name,
        "paired_comparisons": pairwise,
        "limitations": [
            "Development-only screening; no holdout result was computed.",
            "Feature ablation is not causal evidence and does not prove improved future performance.",
            "No production model, ticket generator, or live setting changed.",
            "Ranking metrics do not establish profitability.",
            "A fresh future period or untouched holdout is required before adoption.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("CANDIDATE_COMPARE_COMPLETE " + json.dumps({
        "data": report["data"], "results": result_by_name,
        "paired_comparisons": pairwise,
    }, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
