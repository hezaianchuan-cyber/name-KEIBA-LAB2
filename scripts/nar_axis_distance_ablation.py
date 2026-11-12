#!/usr/bin/env python3
"""Research-only one-factor axis ablation: remove distance matching only.

A: current suitability history (same venue OR within 200m OR same going).
B: remove only within-200m distance matching (same venue OR same going).
All other model axes, weights, probabilities, dates and holdout races stay fixed.
"""
from __future__ import annotations

import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path

import nar_backtest as nb
import nar_ml_challenger_backtest as ml
import horse_evaluation_model as hem
from nar_axis_pair_walkforward_backtest import collect_fixed_axes, _axis_variant_rows, _axis_variant_metrics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "axis-distance-ablation.json"
BOOTSTRAP_REPS = 2000


def _without_distance_evaluator(field, history_by_name, race, *args, **kwargs):
    """Re-score the same field after removing only distance-based history matches."""
    result = _ORIGINAL_EVALUATOR(field, history_by_name, race, *args, **kwargs)
    venue = str(race.get("venue") or "")
    going = str(race.get("going") or "")
    for prediction in result.get("predictions", []):
        name = str(prediction.get("name") or "")
        history = history_by_name.get(name, [])
        recent = history if len(history) <= 5 else history[-5:]
        comparable = []
        for row in recent:
            same_venue = bool(venue) and str(row.get("venue") or "") == venue
            same_going = bool(going) and str(row.get("going") or "") == going
            if same_venue or same_going:
                quality = hem._rank_quality(row)
                if quality is not None:
                    comparable.append(quality)
        suitability = hem._weighted_recent(comparable) if comparable else 0.5
        prediction.setdefault("axes", {})["suitability"] = hem.clamp01(suitability)
        weights = result.get("weights") or hem.AXIS_WEIGHTS
        prediction["score"] = sum(
            hem.clamp01(prediction["axes"].get(key)) * float(weight)
            for key, weight in weights.items()
        )

    predictions = result.get("predictions", [])
    win_temperature = kwargs.get("win_temperature", 0.12)
    top3_temperature = kwargs.get("top3_temperature", 0.10)
    win_probs = hem._softmax([row["score"] for row in predictions], win_temperature)
    top3_probs = hem._top3_probabilities([row["score"] for row in predictions], top3_temperature)
    for row, win_p, top3_p in zip(predictions, win_probs, top3_probs):
        row["win_probability"] = win_p
        row["top3_probability"] = top3_p
    result["predictions"] = sorted(predictions, key=lambda row: row["score"], reverse=True)
    result["win_ranking"] = [
        row["horse_no"] for row in sorted(result["predictions"],
        key=lambda row: row["win_probability"], reverse=True)
    ]
    result["top3_ranking"] = [
        row["horse_no"] for row in sorted(result["predictions"],
        key=lambda row: row["top3_probability"], reverse=True)
    ]
    result["distance_match_removed_research_only"] = True
    return result


def _paired_compare(groups, baseline_axes, without_distance_axes):
    a = {row["race_key"]: row for row in _axis_variant_rows(groups, baseline_axes)}
    b = {row["race_key"]: row for row in _axis_variant_rows(groups, without_distance_axes)}
    common = sorted(set(a) & set(b))
    aa, bb = [a[key] for key in common], [b[key] for key in common]
    n = len(common)
    if not n:
        raise RuntimeError("No common classified holdout races")
    top3_delta = [y["axis_top3"] - x["axis_top3"] for x, y in zip(aa, bb)]
    top7_delta = [(y["top7_podium_count"] - x["top7_podium_count"]) / 3 for x, y in zip(aa, bb)]
    rng = random.Random(20261113)
    boot_top3, boot_top7 = [], []
    for _ in range(BOOTSTRAP_REPS):
        indexes = [rng.randrange(n) for _ in range(n)]
        boot_top3.append(100 * sum(top3_delta[i] for i in indexes) / n)
        boot_top7.append(100 * sum(top7_delta[i] for i in indexes) / n)
    def ci(values):
        values.sort()
        return [round(values[int(.025 * (len(values)-1))], 2),
                round(values[int(.975 * (len(values)-1))], 2)]
    improved = sum(y["axis_top3"] > x["axis_top3"] for x, y in zip(aa, bb))
    worsened = sum(y["axis_top3"] < x["axis_top3"] for x, y in zip(aa, bb))
    changed_axis = sum(
        str(baseline_axes[key].get("axis")) != str(without_distance_axes[key].get("axis"))
        for key in common
    )
    by_venue = {}
    for venue in sorted({row["venue"] for row in aa}):
        indexes = [i for i, row in enumerate(aa) if row["venue"] == venue]
        av, bv = [aa[i] for i in indexes], [bb[i] for i in indexes]
        by_venue[venue] = {
            "races": len(indexes),
            "baseline_axis_top3_rate_pct": round(100 * sum(x["axis_top3"] for x in av) / len(av), 2),
            "without_distance_axis_top3_rate_pct": round(100 * sum(x["axis_top3"] for x in bv) / len(bv), 2),
            "delta_pp": round(100 * sum(bv[j]["axis_top3"] - av[j]["axis_top3"] for j in range(len(av))) / len(av), 2),
            "note": "探索的な競馬場別集計。単独で重み変更の根拠にしない。",
        }
    return {
        "races": n,
        "baseline_current_suitability": _axis_variant_metrics(aa),
        "without_distance_match": _axis_variant_metrics(bb),
        "paired_difference_without_distance_minus_baseline": {
            "axis_top3_rate_delta_pp": round(100 * sum(top3_delta) / n, 2),
            "axis_top3_rate_delta_95ci_paired_bootstrap_pp": ci(boot_top3),
            "top7_podium_coverage_delta_pp": round(100 * sum(top7_delta) / n, 2),
            "top7_podium_coverage_delta_95ci_paired_bootstrap_pp": ci(boot_top7),
            "axis_changed_races": changed_axis,
            "axis_top3_improved_races": improved,
            "axis_top3_worsened_races": worsened,
            "axis_top3_unchanged_outcome_races": n - improved - worsened,
        },
        "by_venue": by_venue,
    }


def main():
    global _ORIGINAL_EVALUATOR
    print("DISTANCE_ABLATION loading archives", flush=True)
    races, horses, payouts, months = nb.load_archives()
    print(f"DISTANCE_ABLATION archives races={len(races)} months={len(months)}", flush=True)

    # A: exact current production-research evaluator, without changing it.
    _ORIGINAL_EVALUATOR = nb.evaluate_field
    baseline_axes = collect_fixed_axes(races, horses, payouts, suitability_mode="combined")

    # B: wrap the evaluator only for the second chronological replay.
    nb.evaluate_field = _without_distance_evaluator
    try:
        without_distance_axes = collect_fixed_axes(races, horses, payouts, suitability_mode="combined")
    finally:
        nb.evaluate_field = _ORIGINAL_EVALUATOR

    rows, all_dates = ml.build_rows(races, horses, include_unplaced=True)
    groups = ml.group_rows(rows)
    if len(all_dates) < 20:
        raise RuntimeError(f"Insufficient chronological dates: {len(all_dates)}")
    holdout_start = max(1, int(len(all_dates) * .85))
    holdout_dates = {day.isoformat() for day in all_dates[holdout_start:]}
    holdout_groups = [group for group in groups if group[0]["date"] in holdout_dates]
    comparison = _paired_compare(holdout_groups, baseline_axes, without_distance_axes)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "research_only_evaluated",
        "experiment": "Aは同競馬場 OR 距離±200m OR 同馬場。Bは距離±200mだけを除き、同競馬場 OR 同馬場を維持。",
        "partitions": {
            "holdout_start": all_dates[holdout_start].isoformat(),
            "holdout_end": all_dates[-1].isoformat(),
            "holdout_dates": len(holdout_dates),
            "archive_months": months,
        },
        "holdout": comparison,
        "policy": "同じ時系列ホールドアウトで距離一致条件だけを除外。他の評価軸・重み・予測温度は固定。結果を見てホールドアウト上で調整しない。ばんえい200mは競馬場適性の条件と混同せず、競馬場別に独立して確認する。",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "holdout": comparison}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
