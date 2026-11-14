#!/usr/bin/env python3
"""Research-only diagnostic: which actual finishing bands are lifted by the
45-feature LambdaRank model's race-relative recent-finish feature.

A = frozen 45-feature research baseline.
B = same model without race_rel_recent_finish_pct (44 features).
Seven expanding-window temporal folds; no production data or ticket logic writes.
"""
from __future__ import annotations
import datetime as dt
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import research_flat_lambdarank_feature_ablation as base
import nar_ml_challenger_backtest as suite

OUT = ROOT / "data" / "flat-lambdarank-relative-finish-rank-bands.json"
FEATURE = "race_rel_recent_finish_pct"
FOLDS = [
    ("fold_01", dt.date(2024, 7, 1), dt.date(2024, 10, 31)),
    ("fold_02", dt.date(2024, 11, 1), dt.date(2025, 2, 28)),
    ("fold_03", dt.date(2025, 3, 1), dt.date(2025, 6, 30)),
    ("fold_04", dt.date(2025, 7, 1), dt.date(2025, 10, 31)),
    ("fold_05", dt.date(2025, 11, 1), dt.date(2026, 2, 28)),
    ("fold_06", dt.date(2026, 3, 1), dt.date(2026, 6, 30)),
    ("fold_07", dt.date(2026, 7, 1), dt.date(2026, 10, 31)),
]
BANDS = ("1st", "2nd", "3rd", "4-5", "6-8", "9+")
BOOTSTRAPS = 5000

def finish_band(finish):
    f = int(finish)
    if f <= 3:
        return ("1st", "2nd", "3rd")[f - 1]
    if f <= 5:
        return "4-5"
    if f <= 8:
        return "6-8"
    return "9+"

def rank_percentiles(group, scores):
    order = sorted(range(len(group)),
                   key=lambda i: (-float(scores[i]), int(group[i]["horse_no"])))
    result = [0.0] * len(group)
    for pos, idx in enumerate(order):
        result[idx] = (pos + 1) / len(group)
    top3 = set(order[:3])
    return result, top3

def bootstrap_ci(values, seed):
    if not values:
        return None
    rng = random.Random(seed)
    n = len(values)
    samples = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(BOOTSTRAPS)
    )
    return [
        round(100 * samples[int(0.025 * (BOOTSTRAPS - 1))], 3),
        round(100 * samples[int(0.975 * (BOOTSTRAPS - 1))], 3),
    ]

def main():
    # Freeze exactly the previously evaluated 45-feature research schema.
    suite.RAW_NUMERIC = [
        f for f in suite.RAW_NUMERIC if f not in base.EXCLUDED_BODY_WEIGHT
    ]
    suite.RAW_FEATURES = suite.RAW_NUMERIC + suite.CATEGORICAL
    features_a = list(suite.RAW_FEATURES)
    if len(features_a) != 45 or FEATURE not in features_a:
        raise RuntimeError(f"Frozen A45 schema mismatch: {len(features_a)} features")
    features_b = [f for f in features_a if f != FEATURE]
    if len(features_b) != 44:
        raise RuntimeError("B44 schema mismatch")

    races, horses, scope = base.load_flat()
    rows, dates = suite.build_rows(races, horses)
    by_date = defaultdict(list)
    for row in rows:
        by_date[dt.date.fromisoformat(row["date"])].append(row)

    pooled = {
        band: {
            "horses": 0, "races": 0, "actual_top3": 0,
            "A_rank_sum": 0.0, "B_rank_sum": 0.0,
            "A_top3": 0, "B_top3": 0,
            "race_rank_delta": [], "race_top3_delta": [],
        } for band in BANDS
    }
    fold_reports = []

    for fold_index, (name, start, end) in enumerate(FOLDS):
        train_dates = sorted(d for d in dates if d < start)
        valid_dates = sorted(d for d in dates if start <= d <= end)
        if len(train_dates) < 180 or not valid_dates:
            raise RuntimeError(f"Invalid temporal coverage: {name}")
        cut = int(len(train_dates) * 0.90)
        fit_dates, early_dates = set(train_dates[:cut]), set(train_dates[cut:])

        def collect(date_set):
            return [r for day in sorted(date_set) for r in by_date.get(day, [])]

        fit_rows = collect(fit_dates)
        fit_groups = suite.group_rows(fit_rows)
        early_groups = suite.group_rows(collect(early_dates))
        valid_groups = suite.group_rows(collect(set(valid_dates)))
        if not fit_groups or not early_groups or not valid_groups:
            raise RuntimeError(f"Empty model split in {name}")

        category_maps = {
            key: {value: i for i, value in enumerate(sorted({r[key] for r in fit_rows}))}
            for key in suite.CATEGORICAL
        }
        models = {}
        for label, features in (("A45", features_a), ("B44", features_b)):
            model = base.fit_ranker(fit_groups, early_groups, features, category_maps)
            models[label] = base.predict_groups(model, valid_groups, features, category_maps)

        if len(models["A45"]) != len(models["B44"]):
            raise RuntimeError("A/B race count mismatch")
        fold_acc = {b: {"horses": 0, "races": 0, "actual_top3": 0,
                        "A_rank_sum": 0.0, "B_rank_sum": 0.0,
                        "A_top3": 0, "B_top3": 0,
                        "race_rank_delta": [], "race_top3_delta": []}
                    for b in BANDS}

        for (ga, sa), (gb, sb) in zip(models["A45"], models["B44"]):
            if [r["race_key"] for r in ga] != [r["race_key"] for r in gb]:
                raise RuntimeError("A/B runner alignment mismatch")
            ar, atop = rank_percentiles(ga, sa)
            br, btop = rank_percentiles(gb, sb)
            for band in BANDS:
                indices = [i for i, row in enumerate(ga)
                           if finish_band(row["finish"]) == band]
                if not indices:
                    continue
                x = fold_acc[band]
                x["horses"] += len(indices)
                x["races"] += 1
                x["actual_top3"] += sum(int(int(ga[i]["finish"]) <= 3) for i in indices)
                x["A_rank_sum"] += sum(ar[i] for i in indices)
                x["B_rank_sum"] += sum(br[i] for i in indices)
                x["A_top3"] += sum(int(i in atop) for i in indices)
                x["B_top3"] += sum(int(i in btop) for i in indices)
                # Positive B-A means A ranked these eventual finishers higher.
                x["race_rank_delta"].append(
                    sum(br[i] - ar[i] for i in indices) / len(indices)
                )
                x["race_top3_delta"].append(
                    sum(int(i in atop) - int(i in btop) for i in indices) / len(indices)
                )

        fold_metrics = {}
        for band, x in fold_acc.items():
            n = x["horses"]
            fold_metrics[band] = {
                "horses": n, "races": x["races"],
                "actual_top3_rate_pct": round(100*x["actual_top3"]/n, 3) if n else None,
                "A_mean_rank_percentile_pct": round(100*x["A_rank_sum"]/n, 3) if n else None,
                "B_mean_rank_percentile_pct": round(100*x["B_rank_sum"]/n, 3) if n else None,
                "A_top3_selection_share_pct": round(100*x["A_top3"]/n, 3) if n else None,
                "B_top3_selection_share_pct": round(100*x["B_top3"]/n, 3) if n else None,
                "A_lift_vs_B_rank_percentile_pp": round(
                    100*sum(x["race_rank_delta"])/len(x["race_rank_delta"]), 3
                ) if x["race_rank_delta"] else None,
                "A_top3_selection_advantage_pp": round(
                    100*sum(x["race_top3_delta"])/len(x["race_top3_delta"]), 3
                ) if x["race_top3_delta"] else None,
            }
            for key in ("horses", "races", "actual_top3", "A_rank_sum", "B_rank_sum",
                        "A_top3", "B_top3"):
                pooled[band][key] += x[key]
            pooled[band]["race_rank_delta"].extend(x["race_rank_delta"])
            pooled[band]["race_top3_delta"].extend(x["race_top3_delta"])

        fold_report = {
            "fold": name,
            "fit_start": min(fit_dates).isoformat(),
            "fit_end": max(fit_dates).isoformat(),
            "validation_start": min(valid_dates).isoformat(),
            "validation_end": max(valid_dates).isoformat(),
            "validation_races": len(valid_groups),
            "bands": fold_metrics,
        }
        fold_reports.append(fold_report)
        print("RANK_BAND_FOLD " + json.dumps(fold_report, ensure_ascii=False), flush=True)

    pooled_report = {}
    for index, band in enumerate(BANDS):
        x = pooled[band]
        n = x["horses"]
        pooled_report[band] = {
            "horses": n, "races": x["races"],
            "actual_top3_rate_pct": round(100*x["actual_top3"]/n, 3) if n else None,
            "A_mean_rank_percentile_pct": round(100*x["A_rank_sum"]/n, 3) if n else None,
            "B_mean_rank_percentile_pct": round(100*x["B_rank_sum"]/n, 3) if n else None,
            "A_top3_selection_share_pct": round(100*x["A_top3"]/n, 3) if n else None,
            "B_top3_selection_share_pct": round(100*x["B_top3"]/n, 3) if n else None,
            "A_lift_vs_B_rank_percentile_pp": round(
                100*sum(x["race_rank_delta"])/len(x["race_rank_delta"]), 3
            ) if x["race_rank_delta"] else None,
            "A_lift_rank_delta_ci95_pp": bootstrap_ci(
                x["race_rank_delta"], 20261114 + index*11
            ),
            "A_top3_selection_advantage_pp": round(
                100*sum(x["race_top3_delta"])/len(x["race_top3_delta"]), 3
            ) if x["race_top3_delta"] else None,
            "A_top3_advantage_ci95_pp": bootstrap_ci(
                x["race_top3_delta"], 20261114 + index*17 + 3
            ),
        }

    report = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": scope,
        "data": {
            "runner_rows": len(rows), "dates": len(dates),
            "validation_races": sum(f["validation_races"] for f in fold_reports),
            "validation_folds": len(fold_reports),
        },
        "frozen_baseline": {
            "name": "A45",
            "feature_count": len(features_a),
            "features": features_a,
            "fixed_parameters": base.PARAMS,
            "excluded_current_race_body_weight": sorted(base.EXCLUDED_BODY_WEIGHT),
            "prior_body_weight_avg_retained": "prior_body_weight_avg" in features_a,
        },
        "diagnostic_comparison": {
            "A": "Frozen 45-feature baseline.",
            "B": "Same training protocol and parameters, with only race_rel_recent_finish_pct removed.",
            "rank_percentile": "Lower is a better predicted rank; positive A_lift_vs_B means A ranks eventual finishers higher than B.",
            "top3_selection": "Share of horses in each actual finish band selected in the model's predicted top three.",
            "bootstrap": "5,000 paired race-cluster bootstrap resamples; descriptive subgroup intervals.",
        },
        "folds": fold_reports,
        "pooled_finish_bands": pooled_report,
        "guardrails": [
            "The feature set and model parameters are frozen; only one feature is removed in B.",
            "Actual finish bands are used only for post-prediction diagnostics, never as training inputs.",
            "Temporal windows were previously inspected in project work; not a pristine untouched holdout.",
            "Subgroup intervals are descriptive and do not correct for multiple comparisons.",
            "Ranking diagnostics do not establish profitability or ROI.",
            "Research only; no production model, live weights, database, or ticket logic changed.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("RANK_BAND_COMPLETE " + json.dumps({
        "data": report["data"], "pooled_finish_bands": pooled_report
    }, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
