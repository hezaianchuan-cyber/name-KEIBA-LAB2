#!/usr/bin/env python3
"""Pre-registered research-only expanding-window A/B validation for flat LambdaRank.

A = frozen 45-feature baseline.
B = A without race_rel_recent_finish_pct and race_rel_recent_top3_rate.
Each validation window is strictly later than its fit/early-stop data. Windows do
not overlap. No feature/parameter selection is performed from validation results.
"""
from __future__ import annotations
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import research_flat_lambdarank_feature_ablation as base
import nar_ml_challenger_backtest as suite

OUT = ROOT / "data" / "flat-lambdarank-walk-forward-ab.json"
REMOVED_B = {"race_rel_recent_finish_pct", "race_rel_recent_top3_rate"}
BOOTSTRAP_ITERATIONS = 5000
MIN_TRAIN_DATES = 180
MIN_VALIDATION_RACES = 100

# Fixed before this run. Expanding training window; non-overlapping 4-month tests.
FOLDS = [
    ("fold_01", dt.date(2024, 7, 1), dt.date(2024, 10, 31)),
    ("fold_02", dt.date(2024, 11, 1), dt.date(2025, 2, 28)),
    ("fold_03", dt.date(2025, 3, 1), dt.date(2025, 6, 30)),
    ("fold_04", dt.date(2025, 7, 1), dt.date(2025, 10, 31)),
    ("fold_05", dt.date(2025, 11, 1), dt.date(2026, 2, 28)),
    ("fold_06", dt.date(2026, 3, 1), dt.date(2026, 6, 30)),
    ("fold_07", dt.date(2026, 7, 1), dt.date(2026, 10, 31)),
]

METRICS = ("top1", "winner_in_top3", "top7_podium_coverage", "top7_full_podium")

def score_vectors(scored):
    vectors = {name: [] for name in METRICS}
    race_keys = []
    for group, scores in scored:
        if [r["race_key"] for r in group] != sorted(
            [r["race_key"] for r in group], key=lambda x: x
        ):
            # Runner order is source order; it must be identical between A and B,
            # but does not need lexical ordering.
            pass
        winner = next((i for i, row in enumerate(group) if int(row["finish"]) == 1), None)
        if winner is None:
            continue
        order = sorted(
            range(len(group)),
            key=lambda i: (-float(scores[i]), int(group[i]["horse_no"]))
        )
        podium = {i for i, row in enumerate(group) if int(row["finish"]) <= 3}
        top7 = set(order[:7])
        vectors["top1"].append(float(order[0] == winner))
        vectors["winner_in_top3"].append(float(winner in order[:3]))
        vectors["top7_podium_coverage"].append(
            len(podium & top7) / max(1, len(podium))
        )
        vectors["top7_full_podium"].append(float(podium.issubset(top7)))
        race_keys.append(group[0]["race_key"])
    return race_keys, vectors

def paired_bootstrap(a, b, seed):
    if len(a) != len(b) or not a:
        raise RuntimeError("Paired vectors are empty or misaligned")
    diffs = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(diffs)
    samples = np.empty(BOOTSTRAP_ITERATIONS, dtype=float)
    # Batch resampling to keep memory bounded on large validation windows.
    cursor = 0
    while cursor < BOOTSTRAP_ITERATIONS:
        batch = min(100, BOOTSTRAP_ITERATIONS - cursor)
        indices = rng.integers(0, n, size=(batch, n))
        samples[cursor:cursor + batch] = diffs[indices].mean(axis=1)
        cursor += batch
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "races": int(n),
        "delta_pp_B_minus_A": round(float(diffs.mean() * 100), 3),
        "ci95_pp": [round(float(low * 100), 3), round(float(high * 100), 3)],
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "method": "paired race-level bootstrap",
    }

def summarize(vectors):
    return {
        metric: round(float(np.mean(values) * 100), 3) if values else None
        for metric, values in vectors.items()
    }

def main():
    # Exclude current-race bodyweight and all derived current-race bodyweight
    # fields. Historical prior_body_weight_avg remains eligible.
    suite.RAW_NUMERIC = [
        f for f in suite.RAW_NUMERIC if f not in base.EXCLUDED_BODY_WEIGHT
    ]
    suite.RAW_FEATURES = suite.RAW_NUMERIC + suite.CATEGORICAL

    all_features = list(suite.RAW_FEATURES)
    if len(all_features) != 45:
        raise RuntimeError(
            f"Expected frozen 45-feature baseline, found {len(all_features)}: {all_features}"
        )
    if not REMOVED_B.issubset(set(all_features)):
        raise RuntimeError("Pre-registered B fields are missing from the feature schema")
    features_a = all_features
    features_b = [f for f in all_features if f not in REMOVED_B]
    if len(features_b) != 43:
        raise RuntimeError(f"Expected 43 B features, found {len(features_b)}")

    races, horses, scope = base.load_flat()
    rows, dates = suite.build_rows(races, horses)
    if not dates:
        raise RuntimeError("No chronological race dates were constructed")
    rows_by_date = {}
    for row in rows:
        rows_by_date.setdefault(dt.date.fromisoformat(row["date"]), []).append(row)

    pooled = {
        "A": {m: [] for m in METRICS},
        "B": {m: [] for m in METRICS},
    }
    fold_results = []

    for fold_index, (fold_name, start, end) in enumerate(FOLDS):
        validation_dates = [d for d in dates if start <= d <= end]
        train_dates = [d for d in dates if d < start]
        if len(train_dates) < MIN_TRAIN_DATES:
            raise RuntimeError(
                f"{fold_name}: only {len(train_dates)} prior dates; need {MIN_TRAIN_DATES}"
            )
        if not validation_dates:
            raise RuntimeError(f"{fold_name}: no validation dates in requested interval")

        train_date_list = sorted(train_dates)
        fit_cut = int(len(train_date_list) * 0.90)
        fit_dates = set(train_date_list[:fit_cut])
        early_dates = set(train_date_list[fit_cut:])
        validation_date_set = set(validation_dates)

        def collect(date_set):
            selected = []
            for day in sorted(date_set):
                selected.extend(rows_by_date.get(day, []))
            return selected

        fit_groups = suite.group_rows(collect(fit_dates))
        early_groups = suite.group_rows(collect(early_dates))
        validation_groups = suite.group_rows(collect(validation_date_set))
        if not fit_groups or not early_groups:
            raise RuntimeError(f"{fold_name}: empty fit or early-stopping groups")
        if len(validation_groups) < MIN_VALIDATION_RACES:
            raise RuntimeError(
                f"{fold_name}: only {len(validation_groups)} validation races; "
                f"need {MIN_VALIDATION_RACES}"
            )

        fit_rows = collect(fit_dates)
        category_maps = {
            key: {
                value: index
                for index, value in enumerate(sorted({row[key] for row in fit_rows}))
            }
            for key in suite.CATEGORICAL
        }

        fold_models = {}
        fold_vectors = {}
        fold_keys = {}
        for label, selected_features in (("A", features_a), ("B", features_b)):
            model = base.fit_ranker(
                fit_groups, early_groups, selected_features, category_maps
            )
            scored = base.predict_groups(
                model, validation_groups, selected_features, category_maps
            )
            keys, vectors = score_vectors(scored)
            fold_keys[label] = keys
            fold_vectors[label] = vectors
            fold_models[label] = {
                "feature_count": len(selected_features),
                "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
                "metrics": summarize(vectors),
            }

        if fold_keys["A"] != fold_keys["B"]:
            raise RuntimeError(f"{fold_name}: A/B race alignment mismatch")
        for label in ("A", "B"):
            for metric in METRICS:
                pooled[label][metric].extend(fold_vectors[label][metric])

        paired = {
            metric: paired_bootstrap(
                fold_vectors["A"][metric],
                fold_vectors["B"][metric],
                seed=20261114 + fold_index * 31 + metric_index,
            )
            for metric_index, metric in enumerate(METRICS)
        }
        fold_report = {
            "fold": fold_name,
            "training": {
                "first_date": min(train_dates).isoformat(),
                "last_date": max(train_dates).isoformat(),
                "fit_races": len(fit_groups),
                "early_stopping_races": len(early_groups),
                "fit_dates": len(fit_dates),
                "early_stopping_dates": len(early_dates),
            },
            "validation": {
                "start": min(validation_dates).isoformat(),
                "end": max(validation_dates).isoformat(),
                "dates": len(validation_dates),
                "races": len(validation_groups),
            },
            "models": fold_models,
            "paired_B_minus_A": paired,
        }
        fold_results.append(fold_report)
        print("WALK_FORWARD_FOLD " + json.dumps(fold_report, ensure_ascii=False), flush=True)

    pooled_paired = {
        metric: paired_bootstrap(
            pooled["A"][metric], pooled["B"][metric],
            seed=20261114 + 1000 + metric_index,
        )
        for metric_index, metric in enumerate(METRICS)
    }
    report = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": scope,
        "pre_registration": {
            "models": {
                "A": {"feature_count": 45, "features": features_a},
                "B": {"feature_count": 43, "removed_features": sorted(REMOVED_B)},
            },
            "fixed_parameters": base.PARAMS,
            "folds": [
                {"name": name, "start": start.isoformat(), "end": end.isoformat()}
                for name, start, end in FOLDS
            ],
            "policy": (
                "Expanding-window chronological validation. For each fold, only dates "
                "strictly before validation start are used for fit/early stopping. "
                "The last 10% of prior dates are early-stopping only. A/B use identical "
                "fit, early-stop and validation races. No validation-based tuning."
            ),
            "current_race_body_weight_excluded": True,
            "prior_body_weight_average_retained": "prior_body_weight_avg" in features_a,
            "final_odds_and_payout_not_used_as_features": True,
        },
        "data": {
            "runner_rows": len(rows),
            "total_dates": len(dates),
            "source_races": scope.get("flat_races"),
            "pooled_validation_races": len(pooled["A"]["top1"]),
            "pooled_validation_dates": sum(f["validation"]["dates"] for f in fold_results),
        },
        "folds": fold_results,
        "pooled_oos_metrics": {
            "A": summarize(pooled["A"]),
            "B": summarize(pooled["B"]),
            "paired_B_minus_A": pooled_paired,
        },
        "interpretation_guardrails": [
            "This is a historical temporal robustness test, not a pristine untouched holdout: earlier project work has inspected some of these historical periods.",
            "The feature set and parameters are frozen for this run; do not change them in response to fold results and still call these same folds independent.",
            "A/B differences are predictive ranking metrics only; they do not establish profitability or a deployable betting edge.",
            "Research-only: no production model, live weights, database, or ticket-generation behavior is modified.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("WALK_FORWARD_COMPLETE " + json.dumps({
        "data": report["data"],
        "pooled_oos_metrics": report["pooled_oos_metrics"],
    }, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
