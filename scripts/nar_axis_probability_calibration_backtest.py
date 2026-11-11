#!/usr/bin/env python3
"""Chronological, research-only calibration test for fixed LAB axis top-three probabilities.

The axis and probabilities are replayed race-by-race using only prior history.
A calibration mapping is learned on earlier dates and evaluated on later,
untouched dates. Nothing is written to production model or ticket files.
"""
from __future__ import annotations

import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import nar_backtest as nb
from nar_axis_pair_walkforward_backtest import collect_fixed_axes
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

OUT = ROOT / "data" / "axis-probability-calibration-backtest.json"
EPS = 1e-6
BOOTSTRAP_REPS = 2000


def _clip(p):
    return min(1.0 - EPS, max(EPS, float(p)))


def _logit(p):
    p = _clip(p)
    return math.log(p / (1.0 - p))


def _sigmoid(value):
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def chronological_split(records, train_fraction=0.70):
    dates = sorted({row["date"] for row in records})
    if len(dates) < 20:
        raise ValueError(f"Need at least 20 distinct race dates; found {len(dates)}")
    cut = max(1, min(len(dates) - 1, int(len(dates) * train_fraction)))
    train_dates, test_dates = set(dates[:cut]), set(dates[cut:])
    train = [row for row in records if row["date"] in train_dates]
    test = [row for row in records if row["date"] in test_dates]
    if len(train) < 100 or len(test) < 30:
        raise ValueError(
            f"Insufficient chronological sample: train={len(train)}, test={len(test)}"
        )
    return train, test, {
        "train_start": dates[0],
        "train_end": dates[cut - 1],
        "test_start": dates[cut],
        "test_end": dates[-1],
        "train_dates": len(train_dates),
        "test_dates": len(test_dates),
    }


def fit_calibrators(train):
    probabilities = [_clip(row["predicted"]) for row in train]
    outcomes = [int(row["actual"]) for row in train]
    if len(set(outcomes)) < 2:
        raise ValueError("Training data must contain both top-three hits and misses")

    # Method 1: one global additive correction, estimated on training dates only.
    additive_offset = sum(y - p for p, y in zip(probabilities, outcomes)) / len(train)

    # Method 2: Platt-style logistic recalibration on the log-odds.
    platt = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    platt.fit([[_logit(p)] for p in probabilities], outcomes)

    # Method 3: isotonic mapping, only when the training sample supports it.
    isotonic = None
    if len(train) >= 200 and len({round(p, 6) for p in probabilities}) >= 10:
        isotonic = IsotonicRegression(out_of_bounds="clip")
        isotonic.fit(probabilities, outcomes)

    return {
        "additive_offset": additive_offset,
        "platt": platt,
        "isotonic": isotonic,
    }


def predict_methods(models, probabilities):
    raw = [_clip(p) for p in probabilities]
    additive = [_clip(p + models["additive_offset"]) for p in raw]
    platt = [
        _clip(float(models["platt"].predict_proba([[_logit(p)]])[0][1]))
        for p in raw
    ]
    result = {
        "uncalibrated": raw,
        "global_offset": additive,
        "platt": platt,
    }
    if models["isotonic"] is not None:
        result["isotonic"] = [
            _clip(float(p)) for p in models["isotonic"].predict(raw)
        ]
    return result


def metrics(probabilities, outcomes, bins=10):
    n = len(outcomes)
    if not n:
        return {"races": 0}
    brier = sum((p - y) ** 2 for p, y in zip(probabilities, outcomes)) / n
    logloss = -sum(
        y * math.log(_clip(p)) + (1 - y) * math.log(1 - _clip(p))
        for p, y in zip(probabilities, outcomes)
    ) / n
    ece = 0.0
    calibration_bins = []
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        members = [
            (p, y) for p, y in zip(probabilities, outcomes)
            if lower <= p < upper or (index == bins - 1 and p == 1.0)
        ]
        if not members:
            continue
        mean_p = sum(p for p, _ in members) / len(members)
        observed = sum(y for _, y in members) / len(members)
        ece += len(members) / n * abs(mean_p - observed)
        calibration_bins.append({
            "range_pct": f"{int(lower * 100)}-{int(upper * 100)}%",
            "races": len(members),
            "mean_predicted_pct": round(100 * mean_p, 2),
            "observed_top3_pct": round(100 * observed, 2),
        })
    return {
        "races": n,
        "mean_predicted_pct": round(100 * sum(probabilities) / n, 2),
        "observed_top3_pct": round(100 * sum(outcomes) / n, 2),
        "brier_score": round(brier, 6),
        "log_loss": round(logloss, 6),
        "expected_calibration_error": round(ece, 6),
        "bins": calibration_bins,
    }


def paired_brier_difference_ci(raw, adjusted, outcomes, seed=20261112):
    n = len(outcomes)
    if n < 30:
        return None
    rng = random.Random(seed)
    differences = []
    for _ in range(BOOTSTRAP_REPS):
        indices = [rng.randrange(n) for _ in range(n)]
        raw_score = sum((raw[i] - outcomes[i]) ** 2 for i in indices) / n
        adjusted_score = sum((adjusted[i] - outcomes[i]) ** 2 for i in indices) / n
        differences.append(adjusted_score - raw_score)
    differences.sort()
    return [
        round(differences[int(0.025 * (len(differences) - 1))], 6),
        round(differences[int(0.975 * (len(differences) - 1))], 6),
    ]


def build_records(races, horses, axis_by_key):
    records = []
    exclusions = {
        "missing_race_or_field": 0,
        "missing_axis_prediction": 0,
        "not_three_classified_podium": 0,
        "invalid_probability": 0,
    }
    for key, axis_info in axis_by_key.items():
        parts = str(key).split("|")
        if len(parts) != 3:
            continue
        race_key = nb.race_key(parts[0], parts[1], parts[2])
        race = races.get(race_key)
        field = horses.get(race_key) or []
        if not race or not field:
            exclusions["missing_race_or_field"] += 1
            continue
        axis = str(axis_info.get("axis") or "")
        prediction = (axis_info.get("predictions") or {}).get(axis) or {}
        probability = prediction.get("top3_probability")
        try:
            probability = float(probability)
            date_value = str(parts[1])
            if len(date_value) != 8:
                raise ValueError("invalid date")
            datetime.strptime(date_value, "%Y%m%d")
        except (TypeError, ValueError):
            exclusions["invalid_probability"] += 1
            continue
        if not 0.0 <= probability <= 1.0:
            exclusions["invalid_probability"] += 1
            continue
        finish_by_no = {
            str(int(row["no"])): int(row["finish"])
            for row in field
            if row.get("no") is not None and row.get("finish") is not None
        }
        podium = {
            no for no, finish in finish_by_no.items()
            if 1 <= finish <= 3
        }
        if len(podium) != 3:
            exclusions["not_three_classified_podium"] += 1
            continue
        if axis not in finish_by_no:
            exclusions["missing_axis_prediction"] += 1
            continue
        records.append({
            "date": datetime.strptime(date_value, "%Y%m%d").date().isoformat(),
            "race_key": "|".join(parts),
            "axis_horse_no": axis,
            "venue": str(race.get("venue") or "unknown"),
            "distance": race.get("distance"),
            "predicted": probability,
            "actual": int(axis in podium),
        })
    records.sort(key=lambda row: (row["date"], row["race_key"]))
    return records, exclusions


def condition_breakdown(test, predictions, min_group_races=100):
    """Compare globally fitted calibrators inside untouched venue/distance groups."""
    outcomes = [row["actual"] for row in test]
    raw = predictions["uncalibrated"]
    report = {"minimum_group_races": min_group_races, "venue": {}, "distance_band": {}}

    def add_group(target, label, indices):
        group_outcomes = [outcomes[i] for i in indices]
        row = {"races": len(indices), "methods": {}}
        if len(indices) < min_group_races:
            row["status"] = "insufficient_sample_descriptive_only"
            target[label] = row
            return
        row["status"] = "evaluated"
        raw_group = [raw[i] for i in indices]
        raw_score = metrics(raw_group, group_outcomes)["brier_score"]
        for method, values in predictions.items():
            selected = [values[i] for i in indices]
            result = metrics(selected, group_outcomes)
            if method != "uncalibrated":
                result["brier_delta_vs_raw"] = round(result["brier_score"] - raw_score, 6)
                result["brier_difference_vs_raw_95ci"] = paired_brier_difference_ci(
                    raw_group, selected, group_outcomes
                )
            row["methods"][method] = result
        target[label] = row

    venues = sorted({str(row.get("venue") or "unknown") for row in test})
    for venue in venues:
        add_group(
            report["venue"], venue,
            [i for i, row in enumerate(test) if str(row.get("venue") or "unknown") == venue]
        )

    def distance_band(value):
        try:
            distance = int(value)
        except (TypeError, ValueError):
            return "unknown"
        if distance <= 1200:
            return "1200m以下"
        if distance <= 1600:
            return "1300-1600m"
        if distance <= 2000:
            return "1700-2000m"
        return "2100m以上"

    bands = sorted({distance_band(row.get("distance")) for row in test})
    for band in bands:
        add_group(
            report["distance_band"], band,
            [i for i, row in enumerate(test) if distance_band(row.get("distance")) == band]
        )
    report["interpretation"] = (
        "All mappings are fitted once on the earlier global training period. "
        "Venue/distance rows are untouched holdout diagnostics only; no subgroup "
        "mapping is fitted. Groups below the minimum are descriptive only."
    )
    return report


def self_test():
    # Synthetic monotonic example: calibration methods must return valid probabilities
    # and the split must preserve date order.
    sample = []
    for day in range(1, 41):
        for race in range(5):
            p = 0.35 + 0.01 * (day % 20)
            y = int((race + day) % 10 < 4)
            sample.append({
                "date": f"2026-{1 + (day - 1) // 28:02d}-{1 + (day - 1) % 28:02d}",
                "race_key": f"venue|{day:08d}|{race}",
                "predicted": p,
                "actual": y,
            })
    train, test, split = chronological_split(sample, 0.70)
    models = fit_calibrators(train)
    predictions = predict_methods(models, [row["predicted"] for row in test])
    assert len(test) > 0 and split["train_end"] < split["test_start"]
    for values in predictions.values():
        assert len(values) == len(test)
        assert all(0.0 < value < 1.0 for value in values)
        assert metrics(values, [row["actual"] for row in test])["races"] == len(test)
    print("CALIBRATION_SELF_TEST=passed", flush=True)


def main():
    if "--self-test" in sys.argv:
        self_test()
        return

    print(
        f"AXIS_CALIBRATION_ARCHIVE_WINDOW start={nb.START.isoformat()} end={nb.END.isoformat()}",
        flush=True,
    )
    races, horses, payouts, months = nb.load_archives()
    print(
        f"AXIS_CALIBRATION_ARCHIVES races={len(races)} horse_races={len(horses)} months={len(months)}",
        flush=True,
    )
    # Fixed axis replay is chronological and uses only each race's prior history.
    axis_by_key = collect_fixed_axes(races, horses, payouts)
    records, exclusions = build_records(races, horses, axis_by_key)
    train, test, split = chronological_split(records)
    models = fit_calibrators(train)
    test_predictions = predict_methods(
        models, [row["predicted"] for row in test]
    )
    outcomes = [row["actual"] for row in test]
    comparisons = {}
    raw = test_predictions["uncalibrated"]
    for name, probabilities in test_predictions.items():
        comparisons[name] = metrics(probabilities, outcomes)
        if name != "uncalibrated":
            comparisons[name]["brier_difference_vs_raw_95ci"] = paired_brier_difference_ci(
                raw, probabilities, outcomes
            )
            comparisons[name]["brier_delta_vs_raw"] = round(
                comparisons[name]["brier_score"] - comparisons["uncalibrated"]["brier_score"],
                6,
            )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "research_only_completed",
        "data_source": "official NAR monthly archives and chronological fixed-axis replay",
        "sample": {
            "races_with_valid_axis_probability": len(records),
            "train_races": len(train),
            "holdout_races": len(test),
            "first_date": records[0]["date"] if records else None,
            "last_date": records[-1]["date"] if records else None,
            "exclusions": exclusions,
        },
        "chronological_split": split,
        "calibration_fit": {
            "global_offset": round(models["additive_offset"], 6),
            "platt_coefficients": [
                round(float(value), 6)
                for value in models["platt"].coef_[0]
            ],
            "platt_intercept": round(float(models["platt"].intercept_[0]), 6),
            "isotonic_fitted": models["isotonic"] is not None,
            "fit_policy": "Only the earlier 70% of distinct dates; holdout dates are never used to fit or choose a calibration mapping.",
        },
        "holdout_comparison": comparisons,
        "condition_breakdown": condition_breakdown(test, test_predictions),
        "decision_policy": [
            "A calibration method is only a research candidate if holdout Brier and log-loss improve without a material ECE deterioration.",
            "The paired bootstrap interval for Brier difference should be inspected; if it crosses zero, improvement is uncertain.",
            "Do not select a method solely because it has the best score on this one holdout.",
            "No production probability, ranking, axis, or ticket logic is modified.",
        ],
        "limitations": [
            "Calibration is evaluated on a chronological holdout; underlying axis predictions are generated by the canonical prior-history replay.",
            "Only races with three classified official podium finishers and a valid fixed-axis top-three probability are included.",
            "This tests probability calibration, not ticket profitability or live betting performance.",
            "Offline virtual research only; no deployment or production changes.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "sample": report["sample"],
        "chronological_split": split,
        "holdout_comparison": {
            name: {
                "races": result["races"],
                "brier_score": result["brier_score"],
                "log_loss": result["log_loss"],
                "expected_calibration_error": result["expected_calibration_error"],
                "brier_delta_vs_raw": result.get("brier_delta_vs_raw"),
            }
            for name, result in comparisons.items()
        },
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
