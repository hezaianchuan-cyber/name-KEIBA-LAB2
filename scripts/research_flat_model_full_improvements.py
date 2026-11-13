#!/usr/bin/env python3
"""Research-only combined audit for distance, age, jockey/ticket separation and pre-off EV.

All horse features are built from prior races only. Banei is removed before fitting.
No production files, predictions, tickets, or database records are modified.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import random
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
                quality = nb.rank_quality(finish, horse.get("field_size") or len(field))
                quality = (float(quality) / 100.0) if quality is not None else None
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


def merge_recent_daily_payouts(payouts):
    """Merge recent official daily payout records for the pre-off snapshot window."""
    path = Path(__file__).resolve().parents[1] / "data" / "daily-payouts.json"
    if not path.exists():
        return {"file_present": False, "merged_races": 0, "date_count": 0}
    payload = json.loads(path.read_text(encoding="utf-8"))
    merged = 0
    dates = set()
    for day in payload.get("days", []):
        date_text = str(day.get("date") or "")
        for race in day.get("races", []):
            venue = str(race.get("venue") or "")
            if any(token in venue for token in ("帯広", "ばんえい", "帯広ば")):
                continue
            parts = str(race.get("race_id") or "").split("|")
            if len(parts) != 3:
                continue
            key = nb.race_key(parts[0], parts[1], parts[2])
            if not key:
                continue
            current = payouts.setdefault(key, {
                "win": [], "place": [], "quinella": [], "exacta": [],
                "wide": [], "trio": [], "trifecta": [],
            })
            entries = race.get("payouts") or {}
            for kind in ("win", "place", "quinella", "exacta", "wide", "trio", "trifecta"):
                for entry in entries.get(kind, []) or []:
                    numbers = [str(x) for x in entry.get("numbers", [])]
                    pay = int(entry.get("pay_100") or 0)
                    item = (numbers, pay)
                    if numbers and item not in current.setdefault(kind, []):
                        current[kind].append(item)
            merged += 1
            if date_text:
                dates.add(date_text)
    return {
        "file_present": True,
        "merged_races": merged,
        "date_count": len(dates),
        "first_date": min(dates) if dates else None,
        "last_date": max(dates) if dates else None,
        "source_period": payload.get("period"),
    }


def preoff_probability_ev_audit(payouts):
    """Fit race-wise temperature calibration on earlier snapshot dates, then test EV on later dates."""
    records = nb.load_odds_snapshots()
    eligible = []
    for key, record in records.items():
        payout = payouts.get(key)
        if not payout or not payout.get("win"):
            continue
        try:
            captured = dt.datetime.fromisoformat(str(record["captured_at"]))
            scheduled = dt.datetime.fromisoformat(str(record["scheduled_post_time"]))
            minutes = float(record.get("minutes_to_post"))
        except (KeyError, TypeError, ValueError):
            continue
        if captured.tzinfo is None or scheduled.tzinfo is None:
            continue
        if captured >= scheduled or not 2 <= minutes <= 30:
            continue
        predictions = []
        for row in record.get("predictions") or []:
            no = "".join(character for character in str(row.get("horse_no") or "") if character.isdigit())
            no = str(int(no)) if no else ""
            probability = nb.number(row.get("win_probability"))
            odds = nb.number(row.get("win_odds"))
            if not no or probability is None or probability <= 0:
                continue
            predictions.append({
                "no": no,
                "probability": float(probability),
                "odds": float(odds) if odds is not None and odds > 0 else None,
            })
        def normalize_no(value):
            digits = "".join(character for character in str(value) if character.isdigit())
            return str(int(digits)) if digits else ""

        winners = {
            normalize_no(no)
            for numbers, _pay in payout.get("win", [])
            for no in numbers
            if normalize_no(no)
        }
        if not predictions or not winners or not any(row["no"] in winners for row in predictions):
            continue
        date_text = str(record.get("date") or key[1])
        if len(date_text) == 8 and date_text.isdigit():
            date_text = f"{date_text[:4]}-{date_text[4:6]}-{date_text[6:8]}"
        eligible.append({
            "date": date_text,
            "key": key,
            "predictions": predictions,
            "winners": winners,
            "payout": payout,
        })
    eligible.sort(key=lambda row: (row["date"], str(row["key"])))
    dates = sorted({row["date"] for row in eligible})
    if len(dates) < 10:
        return {
            "status": "insufficient_dates_for_chronological_calibration",
            "eligible_races": len(eligible),
            "eligible_dates": len(dates),
            "first_date": dates[0] if dates else None,
            "last_date": dates[-1] if dates else None,
            "required_dates": 10,
        }

    split_index = max(1, min(len(dates) - 1, int(len(dates) * 0.70)))
    dev_dates = set(dates[:split_index])
    holdout_dates = set(dates[split_index:])
    development = [row for row in eligible if row["date"] in dev_dates]
    holdout = [row for row in eligible if row["date"] in holdout_dates]

    def calibrated(row, temperature):
        raw = [max(1e-12, item["probability"]) for item in row["predictions"]]
        logs = [math.log(value) / temperature for value in raw]
        peak = max(logs)
        exps = [math.exp(max(-700.0, value - peak)) for value in logs]
        total = sum(exps) or 1.0
        return [value / total for value in exps]

    def logloss_for(rows, temperature):
        losses = []
        for row in rows:
            probs = calibrated(row, temperature)
            mass = sum(
                probs[index]
                for index, item in enumerate(row["predictions"])
                if item["no"] in row["winners"]
            )
            losses.append(-math.log(max(1e-12, mass)))
        return sum(losses) / len(losses) if losses else None

    temperatures = [round(0.50 + index * 0.025, 3) for index in range(61)]
    temperature = min(
        temperatures,
        key=lambda candidate: logloss_for(development, candidate)
        if development else float("inf"),
    )

    def calibration_metrics(rows, temp):
        brier_sum = 0.0
        logloss_sum = 0.0
        top1_hits = 0
        top1_confidence = []
        ece_bins = [{"count": 0, "confidence": 0.0, "accuracy": 0.0} for _ in range(10)]
        horse_rows = 0
        for row in rows:
            probs = calibrated(row, temp)
            winner_indices = {
                index for index, item in enumerate(row["predictions"])
                if item["no"] in row["winners"]
            }
            if not winner_indices:
                continue
            target_mass = 1.0 / len(winner_indices)
            brier_sum += sum(
                (prob - (target_mass if index in winner_indices else 0.0)) ** 2
                for index, prob in enumerate(probs)
            )
            mass = sum(probs[index] for index in winner_indices)
            logloss_sum += -math.log(max(1e-12, mass))
            best = max(range(len(probs)), key=lambda index: probs[index])
            correct = float(best in winner_indices)
            confidence = probs[best]
            top1_hits += int(correct)
            top1_confidence.append(confidence)
            bucket = min(9, int(confidence * 10))
            ece_bins[bucket]["count"] += 1
            ece_bins[bucket]["confidence"] += confidence
            ece_bins[bucket]["accuracy"] += correct
            horse_rows += len(probs)
        race_count = len(top1_confidence)
        ece = 0.0
        for bucket in ece_bins:
            if bucket["count"]:
                ece += bucket["count"] / max(1, race_count) * abs(
                    bucket["confidence"] / bucket["count"]
                    - bucket["accuracy"] / bucket["count"]
                )
        return {
            "races": race_count,
            "horse_probability_rows": horse_rows,
            "top1_rate_pct": round(100 * top1_hits / race_count, 2) if race_count else None,
            "multiclass_brier": round(brier_sum / race_count, 6) if race_count else None,
            "race_log_loss": round(logloss_sum / race_count, 6) if race_count else None,
            "top1_ece": round(ece, 6) if race_count else None,
        }

    thresholds = (1.00, 1.05, 1.10, 1.15, 1.20, 1.30)

    def ev_summary(rows, temp):
        output = {}
        for threshold in thresholds:
            ticket_count = hit_count = stake = returned = selected_races = 0
            race_results = []
            ev_sum = 0.0
            for row in rows:
                probs = calibrated(row, temp)
                race_stake = race_return = race_tickets = 0
                for index, item in enumerate(row["predictions"]):
                    odds = item["odds"]
                    if odds is None:
                        continue
                    ev = probs[index] * odds
                    if ev + 1e-12 < threshold:
                        continue
                    race_tickets += 1
                    ticket_count += 1
                    stake += 100
                    race_stake += 100
                    ev_sum += ev
                    if item["no"] in row["winners"]:
                        hit_count += 1
                        # Use the official payout for this exact horse.
                        payout_value = 0
                        for numbers, pay in row["payout"].get("win", []):
                            if numbers and str(int("".join(ch for ch in str(numbers[0]) if ch.isdigit()) or 0)) == item["no"]:
                                payout_value = int(pay or 0)
                                break
                        returned += payout_value
                        race_return += payout_value
                if race_tickets:
                    selected_races += 1
                    race_results.append((race_stake, race_return))
            ci_low = ci_high = None
            if len(race_results) >= 30:
                rng = random.Random(20261113 + int(round(threshold * 100)))
                samples = []
                count = len(race_results)
                for _ in range(1000):
                    chosen = [race_results[rng.randrange(count)] for _ in range(count)]
                    sample_stake = sum(item[0] for item in chosen)
                    sample_return = sum(item[1] for item in chosen)
                    if sample_stake:
                        samples.append(100.0 * sample_return / sample_stake)
                samples.sort()
                if samples:
                    ci_low = round(samples[int(0.025 * (len(samples) - 1))], 2)
                    ci_high = round(samples[int(0.975 * (len(samples) - 1))], 2)
            roi = 100.0 * returned / stake if stake else None
            output[f"{threshold:.2f}"] = {
                "selected_races": selected_races,
                "tickets": ticket_count,
                "hits": hit_count,
                "hit_rate_pct": round(100 * hit_count / ticket_count, 2) if ticket_count else None,
                "stake_yen": stake,
                "return_yen": returned,
                "profit_yen": returned - stake,
                "roi_pct": round(roi, 2) if roi is not None else None,
                "average_recalculated_ev": round(ev_sum / ticket_count, 4) if ticket_count else None,
                "roi_bootstrap_95_low": ci_low,
                "roi_bootstrap_95_high": ci_high,
                "sample_gate_100_races_300_tickets": selected_races >= 100 and ticket_count >= 300,
            }
        return output

    dev_metrics = calibration_metrics(development, temperature)
    holdout_metrics = calibration_metrics(holdout, temperature)
    dev_ev = ev_summary(development, temperature)
    holdout_ev = ev_summary(holdout, temperature)
    qualified = [
        (float(key), value)
        for key, value in dev_ev.items()
        if value["sample_gate_100_races_300_tickets"]
        and value["roi_bootstrap_95_low"] is not None
        and value["roi_bootstrap_95_low"] > 100.0
    ]
    selected_threshold = (
        max(qualified, key=lambda item: item[1]["roi_pct"] or 0)[0]
        if qualified else None
    )
    return {
        "status": "chronological_temperature_calibration_research_only",
        "snapshot_races": len(eligible),
        "snapshot_dates": len(dates),
        "first_date": dates[0],
        "last_date": dates[-1],
        "development_period": [dates[0], dates[split_index - 1]],
        "holdout_period": [dates[split_index], dates[-1]],
        "development_races": len(development),
        "holdout_races": len(holdout),
        "fitted_temperature": temperature,
        "development_calibration": dev_metrics,
        "holdout_calibration": holdout_metrics,
        "development_ev_thresholds": dev_ev,
        "holdout_ev_thresholds": holdout_ev,
        "development_selected_threshold": selected_threshold,
        "holdout_selected_threshold_result": (
            holdout_ev.get(f"{selected_threshold:.2f}") if selected_threshold is not None else None
        ),
        "note": "Temperature is fitted only on earlier snapshot dates. EV is recalculated as calibrated win probability times the captured pre-off win odds. Final payouts are labels only. The short snapshot window is exploratory and cannot establish a deployable positive-EV edge.",
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
    calibrated_ev_audit = preoff_probability_ev_audit(flat_payouts)
    try:
        ev_audit = nb.preoff_value_backtest(flat_payouts)
    except Exception as exc:
        ev_audit = {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
    ev_audit = {
        "snapshot_inventory": snapshot_inventory,
        "chronological_probability_calibration": calibrated_ev_audit,
        "existing_preoff_ev_signal_audit": ev_audit,
        "interpretation": (
            "Win probabilities are recalibrated by temperature using earlier snapshot dates only. "
            "Expected value is recalculated as calibrated probability times captured pre-off odds. "
            "This is research-only; the short snapshot window is not sufficient for deployment."
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
        "recent_daily_payout_merge": recent_payout_merge,
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
        "recent_daily_payout_merge": recent_payout_merge,
        "report": str(OUT),
    }, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()
