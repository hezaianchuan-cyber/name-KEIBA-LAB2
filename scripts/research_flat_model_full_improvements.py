#!/usr/bin/env python3
"""Research-only combined audit for distance, age, jockey/ticket separation and pre-off EV.

All horse features are built from prior races only. Banei is removed before fitting.
No production files, predictions, tickets, or database records are modified.
"""
from __future__ import annotations

import datetime as dt
import json
import math
from collections import defaultdict
from pathlib import Path

import nar_backtest as nb

OUT = Path(__file__).resolve().parents[1] / "data" / "research-flat-model-full-improvements.json"
FEATURES = ("form", "venue", "distance", "sire", "jockey", "age")


def is_banei(race):
    venue = str(race.get("venue") or "")
    return any(token in venue for token in ("帯広", "ばんえい", "帯広ば"))


def age_stage(age):
    age = nb.number(age)
    if age is None:
        return "unknown"
    if age <= 3:
        return "developing_2_3"
    if age <= 6:
        return "prime_4_6"
    return "mature_7_plus"


def install_feature_variant(variant):
    """Patch only this process' research scorer; caller restores originals."""
    originals = {
        "score_runner": nb.score_runner,
        "field_performance_features": nb.field_performance_features,
        "update_model": nb.update_model,
    }
    age_stats = defaultdict(lambda: {"runs": 0, "quality_sum": 0.0})
    prior_quality = 0.48
    age_prior_strength = 30.0

    def add_age_to_history(field):
        rows = originals["field_performance_features"](field)
        for horse in field:
            no = str(horse.get("no", ""))
            if no in rows:
                rows[no] = {**rows[no], "age": nb.number(horse.get("age"))}
        return rows

    def tracked_update(field, parts_by_no, model_weights):
        updated = originals["update_model"](field, parts_by_no, model_weights)
        # This state is only used in the age challenger. It is updated after
        # the day's predictions by the existing chronological learner.
        if variant in ("age_stage", "distance_plus_age"):
            for horse in field:
                age = nb.number(horse.get("age"))
                finish = horse.get("finish")
                if age is None or not finish:
                    continue
                quality = nb.rank_quality(finish, horse.get("field_size") or len(field))\n                quality = (float(quality) / 100.0) if quality is not None else None
                if quality is None:
                    continue
                bucket = age_stats[age_stage(age)]
                bucket["runs"] += 1
                bucket["quality_sum"] += float(quality)
        return updated

    def research_score_runner(horse, history, sire_stats, jockey_stats, race,
                              model_weights, include_recent_history=True,
                              parts_cache=None):
        score, original_parts = originals["score_runner"](
            horse, history, sire_stats, jockey_stats, race, model_weights,
            include_recent_history=include_recent_history, parts_cache=parts_cache
        )
        parts = dict(original_parts)
        past = history.get(horse.get("name"), []) if include_recent_history else []
        past = past[-5:]

        if variant in ("distance_kernel", "distance_plus_age"):
            target_distance = nb.number(race.get("distance"))
            weighted_sum = 0.0
            weight_total = 0.0
            if target_distance is not None:
                for index, row in enumerate(past, 1):
                    old_distance = nb.number(row.get("distance"))
                    quality = nb.rank_quality(row.get("finish"), row.get("field_size"))
                    if old_distance is None or quality is None:
                        continue
                    delta = abs(old_distance - target_distance)
                    kernel = math.exp(-delta / 300.0)
                    weight = index * kernel
                    weighted_sum += (float(quality) / 100.0) * weight
                    weight_total += weight
            # Shrink sparse evidence toward neutral instead of treating one
            # nearby start as a definitive distance-specialist signal.
            parts["distance"] = (
                (weighted_sum + prior_quality * 3.0) / (weight_total + 3.0)
                if weight_total else prior_quality
            )

        if variant in ("age_stage", "distance_plus_age"):
            stage = age_stage(horse.get("age"))
            bucket = age_stats.get(stage)
            if bucket and bucket["runs"]:
                parts["age"] = (
                    bucket["quality_sum"] + prior_quality * age_prior_strength
                ) / (bucket["runs"] + age_prior_strength)
            else:
                parts["age"] = prior_quality

        score = sum(float(model_weights.get(key, 0.0)) * float(parts.get(key, 0.5))
                    for key in model_weights)
        return score, parts

    nb.score_runner = research_score_runner
    nb.field_performance_features = add_age_to_history
    nb.update_model = tracked_update
    return originals, age_stats


def restore_feature_variant(originals):
    nb.score_runner = originals["score_runner"]
    nb.field_performance_features = originals["field_performance_features"]
    nb.update_model = originals["update_model"]


def summarize_model(model):
    trio = (model.get("bet_types") or {}).get("trio") or {}
    return {
        "races": model.get("races"),
        "winner_top1_rate": model.get("winner_top1_rate"),
        "winner_top3_rate": model.get("winner_top3_rate"),
        "winner_topk_rate": model.get("winner_topk_rate"),
        "trio": trio,
        "trio_diagnostics": model.get("trio_diagnostics"),
        "role_trifecta_diagnostics": model.get("role_trifecta_diagnostics"),
        "finish_position_diagnostics": model.get("finish_position_diagnostics"),
    }


def run_fold(label, variant, races, horses, payouts, train_dates, test_dates, models):
    print(f"FULL_IMPROVEMENTS start variant={variant}", flush=True)
    originals, age_stats = install_feature_variant(variant)
    try:
        fold = nb.evaluate_holdout_fold(
            races, horses, payouts, train_dates, test_dates,
            model_names=models,
        )
    finally:
        restore_feature_variant(originals)
    selected = {}
    for model_name in models:
        row = (fold.get("models") or {}).get(model_name) or {}
        selected[model_name] = summarize_model(row)
    return {
        "label": label,
        "variant": variant,
        "train_races": fold.get("train_races"),
        "test_races": fold.get("test_races"),
        "trained_weights": fold.get("trained_weights"),
        "models": selected,
        "calibration": fold.get("calibration"),
        "age_stage_update_counts": dict(age_stats),
    }


def main():
    print("FULL_IMPROVEMENTS loading official archives", flush=True)
    races, horses, payouts, months = nb.load_archives()
    source_count = len(races)
    flat_races = {key: race for key, race in races.items() if not is_banei(race)}
    flat_keys = set(flat_races)
    flat_horses = {key: value for key, value in horses.items() if key in flat_keys}
    flat_payouts = {key: value for key, value in payouts.items() if key in flat_keys}
    dates = sorted({race["date"] for race in flat_races.values()})
    cutoff = nb.END - dt.timedelta(days=365)
    train_dates = [date for date in dates if date <= cutoff]
    test_dates = [date for date in dates if date > cutoff]
    train_races = sum(1 for race in flat_races.values() if race["date"] in set(train_dates))
    test_races = sum(1 for race in flat_races.values() if race["date"] in set(test_dates))
    scope = {
        "source_races": source_count,
        "excluded_banei_races": source_count - len(flat_races),
        "flat_races": len(flat_races),
        "venues": sorted({str(r.get("venue") or "") for r in flat_races.values()}),
        "banei_present_after_filter": any(is_banei(r) for r in flat_races.values()),
        "train_period": [train_dates[0].isoformat(), train_dates[-1].isoformat()],
        "test_period": [test_dates[0].isoformat(), test_dates[-1].isoformat()],
        "train_races": train_races,
        "test_races": test_races,
    }
    print("FULL_IMPROVEMENTS scope " + json.dumps(scope, ensure_ascii=False), flush=True)

    # Baseline and jockey ablation share the same fold and identical race/ticket budget.
    baseline = run_fold(
        "Original six-feature model", "original",
        flat_races, flat_horses, flat_payouts, train_dates, test_dates,
        ["C_performance_pace", "ablate_jockey"],
    )
    # Fixed, predeclared challengers: no threshold is selected on the holdout.
    distance = run_fold(
        "Continuous distance suitability", "distance_kernel",
        flat_races, flat_horses, flat_payouts, train_dates, test_dates,
        ["C_performance_pace"],
    )
    age = run_fold(
        "Data-driven age-stage prior", "age_stage",
        flat_races, flat_horses, flat_payouts, train_dates, test_dates,
        ["C_performance_pace"],
    )
    combined = run_fold(
        "Distance plus age-stage", "distance_plus_age",
        flat_races, flat_horses, flat_payouts, train_dates, test_dates,
        ["C_performance_pace"],
    )

    # Existing role-specific trifecta diagnostics are a separate ticket-layer
    # comparison; jockey remains in horse ranking, while ticket roles are
    # generated from prior performance/style evidence.
    ticket_layer = {
        "fixed_trio_and_jockey_ablation": {
            "original": baseline["models"].get("C_performance_pace"),
            "without_jockey": baseline["models"].get("ablate_jockey"),
        },
        "role_specific_trifecta": (
            baseline["models"].get("C_performance_pace") or {}
        ).get("role_trifecta_diagnostics"),
        "note": "Horse ranking and ticket construction are evaluated separately. Role-specific ticket roles use prior performance/style signals; no current-race outcomes are features.",
    }

    # The repo currently stores a short recent window of pre-off snapshots.
    # Keep this separate from the year-long horse-ranking holdout and report
    # whether the sample is large enough to support any EV threshold.
    snapshot_inventory = nb.odds_snapshot_inventory()
    try:
        ev_audit = nb.preoff_value_backtest(flat_payouts)
    except Exception as exc:
        ev_audit = {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
    ev_audit = {
        "snapshot_inventory": snapshot_inventory,
        "existing_preoff_ev_signal_audit": ev_audit,
        "interpretation": (
            "This evaluates already-recorded pre-off predictions/expected_value against official payouts. "
            "It is not a newly calibrated probability model. A new EV model requires a larger chronological "
            "snapshot archive; do not adopt a threshold from this short recent window."
        ),
    }

    report = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "research_only",
        "scope": scope,
        "source_months": months,
        "experiments": {
            "original_baseline_and_jockey_ablation": baseline,
            "distance_kernel": distance,
            "age_stage": age,
            "distance_plus_age": combined,
        },
        "ticket_layer": ticket_layer,
        "expected_value": ev_audit,
        "method": {
            "distance": "Replace the fixed +/-200m average with a recency-weighted exponential distance kernel (300m scale), shrunk toward neutral with three pseudo-runs.",
            "age": "Replace the fixed age score with a chronological age-stage prior learned from past runners only (2-3, 4-6, 7+); prior strength 30; update after each date's predictions.",
            "jockey_ticket_separation": "Compare the same fixed trio ticket builder with and without the jockey feature, and separately retain role-specific ticket diagnostics. Same races and ticket cost per policy.",
            "ev": "Audit pre-off snapshots captured 2-30 minutes before post time. The current archive is a short recent sample; this run does not fit or deploy a new probability model.",
            "leakage": "Banei is excluded before fitting and scoring. Predictions for all races on a date precede updates from that date. Current-race results, payouts, and final odds are not predictor inputs.",
            "deployment": "No production weights, live predictions, tickets, or database records changed.",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("FULL_IMPROVEMENTS_RESULT " + json.dumps({
        "scope": scope,
        "baseline": baseline["models"],
        "distance": distance["models"],
        "age": age["models"],
        "combined": combined["models"],
        "ticket_layer": ticket_layer,
        "expected_value": ev_audit,
        "report": str(OUT),
    }, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()
