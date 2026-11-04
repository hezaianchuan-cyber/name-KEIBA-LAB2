#!/usr/bin/env python3
"""Leakage-safe retrospective G1 probability-calibration evaluation.

The base chronological C-model produces race scores using only earlier results.
G1 fits one softmax temperature on the first 70% of eligible dates from
2026-03-01 onward and evaluates it once on the untouched final 30%.
This is a probability-quality experiment, not a live-betting ROI claim.
"""
from __future__ import annotations

import datetime as dt
import json
import random
from pathlib import Path

import nar_backtest as nb

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "model-g1-calibration-backtest.json"
START_DATE = dt.date(2026, 3, 1)
BASELINE_TEMPERATURE = 0.18
MIN_HOLDOUT_RACES = 100
BOOTSTRAP_REPS = 2000


def valid_records(records):
    output = []
    for row in records:
        try:
            day = dt.date.fromisoformat(str(row.get("date") or ""))
        except ValueError:
            continue
        if day < START_DATE or not row.get("scores") or not row.get("winner_indices"):
            continue
        if any(i < 0 or i >= len(row["scores"]) for i in row["winner_indices"]):
            continue
        output.append(row)
    return sorted(output, key=lambda r: (r["date"], r.get("venue", ""), int(r.get("race") or 0)))


def metric_delta_ci(records, baseline_temp, calibrated_temp):
    if len(records) < 30:
        return {"method": "paired race bootstrap", "replicates": 0,
                "log_loss_delta_95ci": None, "brier_delta_95ci": None,
                "ece_delta_95ci": None}
    rng = random.Random(20261104 + len(records))
    deltas = {"log_loss": [], "brier": [], "ece": []}
    n = len(records)
    for _ in range(BOOTSTRAP_REPS):
        sample = [records[rng.randrange(n)] for _ in range(n)]
        before = nb.calibration_metrics(sample, baseline_temp)
        after = nb.calibration_metrics(sample, calibrated_temp)
        for key in deltas:
            if before.get(key) is not None and after.get(key) is not None:
                deltas[key].append(after[key] - before[key])

    def interval(values):
        values.sort()
        if not values:
            return None
        lo = values[int(0.025 * (len(values) - 1))]
        hi = values[int(0.975 * (len(values) - 1))]
        return [round(lo, 6), round(hi, 6)]

    return {
        "method": "paired race-level bootstrap; calibrated minus baseline; lower is better",
        "replicates": BOOTSTRAP_REPS,
        "log_loss_delta_95ci": interval(deltas["log_loss"]),
        "brier_delta_95ci": interval(deltas["brier"]),
        "ece_delta_95ci": interval(deltas["ece"]),
    }


def main():
    races, horses, payouts, downloaded = nb.load_archives()
    _, _, _, _, _, all_calibration_records = nb.run_backtest(races, horses, payouts)
    records = valid_records(all_calibration_records)
    dates = sorted({row["date"] for row in records})

    report = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "feature": "G1_probability_calibration",
        "status": "insufficient_data",
        "model_scope": "Existing chronological C-model scores; E model and live model code are unchanged.",
        "evaluation_period_start": START_DATE.isoformat(),
        "evaluation_period_end": dates[-1] if dates else None,
        "eligible_races": len(records),
        "eligible_dates": len(dates),
        "official_archive_months": downloaded,
        "base_training": "Chronological NAR history before each target race; same-day outcomes are committed only after all races on that date are scored.",
        "split": "Dates sorted chronologically; first 70% for G1 temperature selection, final 30% untouched holdout.",
        "baseline_temperature": BASELINE_TEMPERATURE,
        "holdout_minimum_races": MIN_HOLDOUT_RACES,
        "metrics": "Race-level winner-mass log loss, multiclass Brier score, top-ranked confidence ECE (10 bins), and top-1 accuracy. Paired race bootstrap 95% CI for calibrated-minus-baseline deltas.",
        "final_odds_note": "G1 probability calibration is scored against official race outcomes. Final odds are not calibration labels and are reserved for G2 odds-divergence/value evaluation; no pre-off odds are fabricated.",
        "e_model_changed": False,
        "profitable_claim": False,
    }

    if len(dates) < 10:
        report["status"] = "insufficient_dates_for_chronological_split"
    else:
        split = max(1, min(len(dates) - 1, int(len(dates) * 0.70)))
        dev_dates = set(dates[:split])
        development = [r for r in records if r["date"] in dev_dates]
        holdout = [r for r in records if r["date"] not in dev_dates]
        temperature = nb.fit_temperature(development)
        dev_before = nb.calibration_metrics(development, BASELINE_TEMPERATURE)
        dev_after = nb.calibration_metrics(development, temperature)
        holdout_before = nb.calibration_metrics(holdout, BASELINE_TEMPERATURE)
        holdout_after = nb.calibration_metrics(holdout, temperature)
        ci = metric_delta_ci(holdout, BASELINE_TEMPERATURE, temperature)

        log_ci = ci["log_loss_delta_95ci"]
        brier_ci = ci["brier_delta_95ci"]
        ece_ci = ci["ece_delta_95ci"]
        adequate = len(holdout) >= MIN_HOLDOUT_RACES
        statistically_better = bool(
            adequate and log_ci and brier_ci
            and log_ci[1] < 0 and brier_ci[1] < 0
            and (ece_ci is None or ece_ci[0] <= 0)
        )
        report.update({
            "status": "G1_improved_on_untouched_holdout" if statistically_better
                      else "G1_not_confirmed_improved" if adequate
                      else "insufficient_holdout_sample",
            "development_period": [dates[0], dates[split - 1]],
            "holdout_period": [dates[split], dates[-1]],
            "development_races": len(development),
            "holdout_races": len(holdout),
            "selected_temperature": round(float(temperature), 5),
            "temperature_search": "0.02 to 0.50 inclusive, step 0.005; minimum development winner-mass log loss.",
            "development_metrics": {
                "baseline": dev_before,
                "G1_calibrated": dev_after,
            },
            "holdout_metrics": {
                "baseline": holdout_before,
                "G1_calibrated": holdout_after,
                "paired_delta_95ci": ci,
            },
            "holdout_changes": {
                "log_loss_delta": round(holdout_after["log_loss"] - holdout_before["log_loss"], 6),
                "brier_delta": round(holdout_after["brier"] - holdout_before["brier"], 6),
                "ece_delta": round(holdout_after["ece"] - holdout_before["ece"], 6),
                "top1_accuracy_delta_percentage_points": round(
                    holdout_after["top1_accuracy"] - holdout_before["top1_accuracy"], 4),
            },
            "interpretation": (
                "G1 passes this experiment only when the untouched holdout has at least "
                "100 races, both log-loss and Brier 95% delta upper bounds are below zero, "
                "and ECE does not show a statistically clear deterioration. This is not "
                "evidence of profitable betting or live pre-off performance."
            ),
            "g1_pass": statistically_better,
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
