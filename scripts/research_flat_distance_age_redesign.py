#!/usr/bin/env python3
"""Research-only flat-racing redesign of distance and age features.

The original six-feature model and its weights are retained. Only the shape of
its distance/age inputs is changed in isolated challenger variants. No production
predictions, tickets, weights, or database records are modified.
"""
from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path

import nar_backtest as nb

OUT = Path(__file__).resolve().parents[1] / "data" / "research-flat-distance-age-redesign.json"
FEATURES = ("form", "venue", "distance", "sire", "jockey", "age")
MODELS = (
    "C_performance_pace",
    "shape_distance_exact_100",
    "shape_distance_kernel_350",
    "shape_age_smooth",
    "shape_age_neutral",
    "shape_distance_kernel_350_age_smooth",
)


def is_banei(race: dict) -> bool:
    venue = str(race.get("venue") or "")
    return any(token in venue for token in ("帯広", "ばんえい", "帯広ば"))


def make_shape_runner(original):
    """Wrap the existing scorer; alter only the selected pre-race feature shape."""
    def score_runner(h, history, sire_stats, jockey_stats, race, model_weights,
                     include_recent_history=True, parts_cache=None):
        base_score, original_parts = original(
            h, history, sire_stats, jockey_stats, race, model_weights,
            include_recent_history=include_recent_history, parts_cache=parts_cache,
        )
        variant = model_weights.get("__shape_variant")
        if not variant:
            return base_score, original_parts

        parts = dict(original_parts)
        past_rows = history.get(h.get("name", ""), []) if include_recent_history else []
        past = past_rows[-5:]
        target_distance = nb.number(race.get("distance"))

        if variant in ("shape_distance_exact_100",
                       "shape_distance_kernel_350",
                       "shape_distance_kernel_350_age_smooth"):
            samples = []
            for index, row in enumerate(past, start=1):
                prior_distance = nb.number(row.get("distance"))
                quality = nb.rank_quality(row.get("finish"), row.get("field_size"))
                if target_distance is None or prior_distance is None or quality is None:
                    continue
                gap = abs(prior_distance - target_distance)
                if variant == "shape_distance_exact_100":
                    if gap <= 100:
                        samples.append((float(index), quality))
                else:
                    # Continuous distance similarity with recency weighting.
                    weight = float(index) * math.exp(-gap / 350.0)
                    if weight > 0.01:
                        samples.append((weight, quality))
            prior_strength = 3.0 if variant == "shape_distance_exact_100" else 2.0
            weighted_sum = sum(weight * value for weight, value in samples)
            total_weight = sum(weight for weight, _ in samples)
            parts["distance"] = (
                (weighted_sum + 0.48 * prior_strength) /
                (total_weight + prior_strength)
            )

        age = nb.number(h.get("age"))
        if variant in ("shape_age_smooth", "shape_distance_kernel_350_age_smooth"):
            if age is None:
                parts["age"] = 0.5
            else:
                # Smooth peak around age five; avoids hard age-band jumps.
                parts["age"] = 0.45 + 0.27 * math.exp(-0.5 * ((age - 5.0) / 2.0) ** 2)
        elif variant == "shape_age_neutral":
            parts["age"] = 0.5

        score = sum(float(model_weights.get(key, 0.0)) * parts[key]
                    for key in FEATURES)
        return score, parts
    return score_runner


def main() -> None:
    print("DISTANCE_AGE_REDESIGN loading official NAR archives", flush=True)
    races, horses, payouts, months = nb.load_archives()
    source_count = len(races)
    flat_races = {key: race for key, race in races.items() if not is_banei(race)}
    excluded = source_count - len(flat_races)
    flat_keys = set(flat_races)
    flat_horses = {key: value for key, value in horses.items() if key in flat_keys}
    flat_payouts = {key: value for key, value in payouts.items() if key in flat_keys}
    dates = sorted({race["date"] for race in flat_races.values()})
    cutoff = nb.END - dt.timedelta(days=365)
    train_dates = [date for date in dates if date <= cutoff]
    test_dates = [date for date in dates if date > cutoff]
    if not train_dates or not test_dates:
        raise RuntimeError("時系列の学習期間または評価期間が空です")

    train_set, test_set = set(train_dates), set(test_dates)
    train_count = sum(r["date"] in train_set for r in flat_races.values())
    test_count = sum(r["date"] in test_set for r in flat_races.values())
    venues = sorted({str(r.get("venue") or "") for r in flat_races.values()})
    print("DISTANCE_AGE_REDESIGN_SCOPE " + json.dumps({
        "source_races": source_count, "excluded_banei": excluded,
        "flat_races": len(flat_races), "venues": venues,
        "train_start": train_dates[0].isoformat(),
        "train_end": train_dates[-1].isoformat(), "train_races": train_count,
        "test_start": test_dates[0].isoformat(),
        "test_end": test_dates[-1].isoformat(), "test_races": test_count,
    }, ensure_ascii=False), flush=True)

    # Use the same chronological fold and the same fixed ticket construction as
    # the prior six-feature audit. Only candidate feature shapes differ.
    original_score_runner = nb.score_runner
    nb.score_runner = make_shape_runner(original_score_runner)
    try:
        fold = nb.evaluate_holdout_fold(
            flat_races, flat_horses, flat_payouts,
            train_dates, test_dates, model_names=list(MODELS),
        )
    finally:
        nb.score_runner = original_score_runner

    model_rows = {}
    for name in MODELS:
        metrics = (fold.get("models") or {}).get(name) or {}
        trio = (metrics.get("bet_types") or {}).get("trio") or {}
        model_rows[name] = {
            "races": metrics.get("races"),
            "winner_top1_rate": metrics.get("winner_top1_rate"),
            "winner_top3_rate": metrics.get("winner_top3_rate"),
            "winner_topk_rate": metrics.get("winner_topk_rate"),
            "trio_race_count": trio.get("race_count"),
            "trio_tickets": trio.get("tickets"),
            "trio_hit_rate": trio.get("hit_rate"),
            "trio_roi": trio.get("roi"),
            "trio_profit_yen": trio.get("profit"),
            "trio_stake_yen": trio.get("stake"),
            "trio_return_yen": trio.get("return"),
        }
    report = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "research_only",
        "scope": {
            "source_races": source_count, "excluded_banei_races": excluded,
            "flat_races": len(flat_races), "included_venues": venues,
            "banei_present_after_filter": any(is_banei(r) for r in flat_races.values()),
            "train_period": [train_dates[0].isoformat(), train_dates[-1].isoformat()],
            "train_races": fold.get("train_races", train_count),
            "test_period": [test_dates[0].isoformat(), test_dates[-1].isoformat()],
            "test_races": fold.get("test_races", test_count),
        },
        "models": model_rows,
        "methods": {
            "distance_exact_100": "Prior starts within 100m; recency-weighted quality shrunk toward neutral 0.48 with prior strength 3.",
            "distance_kernel_350": "Continuous exp(-distance_gap/350m) similarity multiplied by recency weight; shrunk toward neutral 0.48 with prior strength 2.",
            "age_smooth": "Continuous Gaussian age curve peaking at age five, with maximum score 0.72 and floor 0.45.",
            "age_neutral": "Age feature held at neutral 0.50.",
            "combined": "Distance kernel plus smooth age curve.",
            "controls": "Banei excluded before training/evaluation; chronological train/test split; identical holdout races, other scoring overlays, and fixed ticket generation.",
            "interpretation": "This is a feature-shape sensitivity experiment, not causal proof or a production deployment. ROI is retrospective.",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("DISTANCE_AGE_REDESIGN_RESULT " + json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
