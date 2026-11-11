#!/usr/bin/env python3
"""Walk-forward research comparison for the axis-conditioned trio pair ranker.

All predictions/candidate membership are generated from strictly prior history
by the existing chronological NAR backtest. Current-race finishes are retained
only as labels after evaluate_field has produced its pre-race predictions.
This script writes aggregate research metrics only; it never changes live files.
"""
from __future__ import annotations

import itertools
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import nar_backtest as nb
from nar_axis_conditional_pair_ranker import (
    build_axis_pair_training_data,
    fit_axis_pair_model,
    rank_axis_partner_pairs,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "axis-conditional-pair-backtest.json"
BOOTSTRAP_REPS = 2000
TICKETS_PER_RACE = 6
STAKE_PER_TICKET = 100


def _mean_history(rows, key, default=0.5):
    values = []
    for row in rows[-5:]:
        try:
            value = float(row.get(key))
            if math.isfinite(value):
                values.append(value)
        except (TypeError, ValueError):
            pass
    return sum(values) / len(values) if values else default


def _time_trend(rows):
    values = []
    for row in rows[-5:]:
        try:
            value = float(row.get("time_per_km"))
            if math.isfinite(value) and value > 0:
                values.append(value)
        except (TypeError, ValueError):
            pass
    if len(values) < 3:
        return 0.5
    old = sum(values[:-2]) / len(values[:-2])
    recent = sum(values[-2:]) / 2
    if old <= 0:
        return 0.5
    improvement = (old - recent) / old
    return max(0.0, min(1.0, 0.5 + improvement * 8.0))


def _capture_record(field, history_by_name, race, result, payouts):
    predictions = result.get("predictions") or []
    score_by_no = {
        str(row["horse_no"]): float(row["score"])
        for row in predictions
        if row.get("horse_no") is not None and row.get("score") is not None
    }
    win_order = [str(no) for no in result.get("win_ranking") or []]
    ability_order = [str(no) for no in result.get("top3_ranking") or []]
    if not win_order or not ability_order:
        return None
    axis_no = win_order[0]
    candidate_nos = [axis_no] + [no for no in ability_order if no != axis_no][:6]
    if len(candidate_nos) < 3 or any(no not in score_by_no for no in candidate_nos):
        return None

    styles = race.get("predicted_styles") or []
    front_count = sum(1 for style in styles if style in ("front", "逃げ"))
    field_size = max(1, len(field))
    group = []
    for horse in field:
        no = str(horse.get("no"))
        prior = history_by_name.get(str(horse.get("name") or ""), [])[-5:]
        current_weight = horse.get("body_weight")
        weight_change = horse.get("body_weight_change")
        try:
            body_delta_pct = (
                float(weight_change) / float(current_weight)
                if current_weight and float(current_weight) > 0 and weight_change is not None
                else 0.0
            )
        except (TypeError, ValueError):
            body_delta_pct = 0.0
        group.append({
            "horse_no": no,
            # Outcome is carried solely for the label builder after predictions.
            "finish": horse.get("finish"),
            "recent_first_corner_pct": _mean_history(prior, "first_corner_pct"),
            "recent_last_corner_pct": _mean_history(prior, "last_corner_pct"),
            "recent_last3f_rank": _mean_history(prior, "last3f_rank_score"),
            "recent_time_trend": _time_trend(prior),
            "recent_opponent_strength": _mean_history(prior, "opponent_strength"),
            "predicted_front_count": float(front_count),
            "front_crowding_ratio": front_count / field_size,
            "carried_weight": horse.get("carried"),
            "body_weight_delta_pct": body_delta_pct,
        })
    key = race.get("key")
    return {
        "date": race["date"].isoformat(),
        "venue": str(race.get("venue") or ""),
        "race": str(race.get("race") or ""),
        "group": group,
        "axis_horse_no": axis_no,
        "candidate_horse_nos": candidate_nos,
        "individual_scores": score_by_no,
        "payout": payouts.get(key) or {},
    }


def collect_walk_forward_records(races, horses, payouts):
    """Reuse the canonical date-batched backtest to expose pre-off candidate sets."""
    records = []
    original_evaluator = nb.evaluate_field

    def capture_evaluator(field, history_by_name, race, *args, **kwargs):
        result = original_evaluator(field, history_by_name, race, *args, **kwargs)
        record = _capture_record(field, history_by_name, race, result, payouts)
        if record is not None:
            records.append(record)
        return result

    nb.evaluate_field = capture_evaluator
    try:
        # training_only preserves chronological history updates but skips the
        # unrelated ticket/report calculations. No live model or file is written.
        nb.run_backtest(races, horses, payouts, training_only=True)
    finally:
        nb.evaluate_field = original_evaluator
    records.sort(key=lambda row: (row["date"], row["venue"], int(row["race"])))
    return records


def _six_pairs_from_top_four(record):
    axis = str(record["axis_horse_no"])
    scores = record["individual_scores"]
    opponents = [str(no) for no in record["candidate_horse_nos"] if str(no) != axis]
    opponents.sort(key=lambda no: (-float(scores[no]), int(no)))
    top_four = opponents[:4]
    return list(itertools.combinations(top_four, 2))


def _tickets_for_pairs(axis, pairs):
    return [
        tuple(sorted((str(axis), str(a), str(b)), key=int))
        for a, b in pairs
    ]


def _roi_interval(race_rows):
    if len(race_rows) < 30:
        return None
    rng = random.Random(20261111)
    values = []
    for _ in range(BOOTSTRAP_REPS):
        sample = [race_rows[rng.randrange(len(race_rows))] for _ in race_rows]
        stake = sum(row[0] for row in sample)
        returned = sum(row[1] for row in sample)
        if stake:
            values.append(100.0 * returned / stake)
    values.sort()
    if not values:
        return None
    return [
        round(values[int(0.025 * (len(values) - 1))], 2),
        round(values[int(0.975 * (len(values) - 1))], 2),
    ]


def evaluate_selector(records, fitted=None):
    races = ticket_hits = hit_races = stake = returned = 0
    axis_podium = axis_races = 0
    candidate_pair_covered = selected_pair_covered = 0
    race_rows = []
    for record in records:
        payout = record.get("payout") or {}
        if not payout.get("trio") or not payout.get("win"):
            continue
        group_by_no = {str(row["horse_no"]): row for row in record["group"]}
        axis = str(record["axis_horse_no"])
        if axis not in group_by_no:
            continue
        opponents = [str(no) for no in record["candidate_horse_nos"] if str(no) != axis]
        if len(opponents) < 4:
            continue
        baseline_pairs = _six_pairs_from_top_four(record)
        if fitted is None:
            selected_pairs = baseline_pairs
        else:
            ranked = rank_axis_partner_pairs(
                record["group"], axis, record["candidate_horse_nos"],
                record["individual_scores"], fitted,
            )
            selected_pairs = [tuple(item["pair"]) for item in ranked[:TICKETS_PER_RACE]]
        tickets = _tickets_for_pairs(axis, selected_pairs)
        if len(tickets) != TICKETS_PER_RACE:
            continue
        race_return = sum(
            nb.payout_for("trio", ticket, payout) or 0
            for ticket in tickets
        )
        race_hits = sum(
            1 for ticket in tickets
            if (nb.payout_for("trio", ticket, payout) or 0) > 0
        )
        race_stake = TICKETS_PER_RACE * STAKE_PER_TICKET
        races += 1
        ticket_hits += race_hits
        hit_races += int(race_hits > 0)
        stake += race_stake
        returned += race_return
        race_rows.append((race_stake, race_return))

        actual_podium = {
            no for no, row in group_by_no.items()
            if row.get("finish") is not None and int(row["finish"]) <= 3
        }
        axis_races += 1
        axis_in = axis in actual_podium
        axis_podium += int(axis_in)
        if axis_in:
            actual_partners = actual_podium - {axis}
            candidate_pair_covered += int(
                any(set(pair).issubset(set(opponents)) for pair in [tuple(actual_partners)])
                if len(actual_partners) == 2 else False
            )
            selected_pair_covered += int(
                any(set(pair) == actual_partners for pair in selected_pairs)
                if len(actual_partners) == 2 else False
            )
    return {
        "races": races,
        "tickets": races * TICKETS_PER_RACE,
        "ticket_hits": ticket_hits,
        "hit_races": hit_races,
        "ticket_hit_rate_pct": round(100 * ticket_hits / (races * TICKETS_PER_RACE), 2) if races else None,
        "race_hit_rate_pct": round(100 * hit_races / races, 2) if races else None,
        "stake_jpy": stake,
        "return_jpy": returned,
        "profit_jpy": returned - stake,
        "roi_pct": round(100 * returned / stake, 2) if stake else None,
        "roi_95ci_bootstrap_pct": _roi_interval(race_rows),
        "axis_podium_races": axis_races,
        "axis_top3_rate_pct": round(100 * axis_podium / axis_races, 2) if axis_races else None,
        "both_partners_in_candidate_pool": candidate_pair_covered,
        "both_partners_selected": selected_pair_covered,
        "partner_pair_capture_pct": (
            round(100 * selected_pair_covered / axis_podium, 2) if axis_podium else None
        ),
        "note": "Offline simulation only; six unordered trio tickets per eligible race, fixed axis, equal 100-yen virtual unit.",
    }


def main():
    print(
        f"PAIR_RESEARCH_ARCHIVE_WINDOW start={nb.START.isoformat()} end={nb.END.isoformat()}",
        flush=True,
    )
    races, horses, payouts, months = nb.load_archives()
    print(
        f"PAIR_RESEARCH_ARCHIVES races={len(races)} horse_races={len(horses)} "
        f"payout_races={len(payouts)} months={len(months)}",
        flush=True,
    )
    records = collect_walk_forward_records(races, horses, payouts)
    dates = sorted({row["date"] for row in records})
    if len(dates) < 10:
        raise RuntimeError(f"Insufficient chronological dates for development/holdout split: {len(dates)}")
    split_index = max(1, min(len(dates) - 1, int(len(dates) * 0.70)))
    development_dates = set(dates[:split_index])
    development = [row for row in records if row["date"] in development_dates]
    holdout = [row for row in records if row["date"] not in development_dates]
    print(
        f"PAIR_RESEARCH_SPLIT records={len(records)} dev={len(development)} "
        f"holdout={len(holdout)} dev_end={dates[split_index-1]} "
        f"holdout_start={dates[split_index]}",
        flush=True,
    )

    x_rows, labels, diagnostics = build_axis_pair_training_data(development)
    fitted = fit_axis_pair_model(development)
    baseline_dev = evaluate_selector(development)
    baseline_holdout = evaluate_selector(holdout)
    pair_holdout = evaluate_selector(holdout, fitted)
    report = {
        "generated_at": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
        "status": "research_only_evaluated",
        "data_source": "NAR official monthly race archives; predictions replayed through canonical chronological prior-history evaluator",
        "period": {"start": dates[0], "end": dates[-1]},
        "development": {
            "date_count": len(development_dates),
            "race_records": len(development),
            "end_date": dates[split_index - 1],
            "axis_pair_training": diagnostics,
            "training_pair_rows": len(labels),
            "baseline_ability_top4": baseline_dev,
            "note": "Development outcomes are used only for fitting; development ROI is descriptive and not an out-of-sample claim.",
        },
        "holdout": {
            "date_count": len(dates) - split_index,
            "race_records": len(holdout),
            "start_date": dates[split_index],
            "end_date": dates[-1],
            "baseline_ability_top4": baseline_holdout,
            "conditional_pair_ranker": pair_holdout,
            "same_races_and_budget": True,
        },
        "model": {
            "type": "StandardScaler + balanced regularized LogisticRegression",
            "candidate_pool": "fixed pre-off axis plus six non-axis ability candidates",
            "pair_universe": 15,
            "selected_pairs": 6,
            "training_rule": "only historical races where the preselected axis finished top three; finishes used only as pair labels",
            "features": fitted["features"],
            "hyperparameters": {"C": 0.5, "max_iter": 600, "class_weight": "balanced", "random_state": 42},
        },
        "policy": "No holdout tuning, no final-odds features, no live integration, no deployment. Holdout is chronological and untouched.",
        "archive_months": months,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "development": report["development"]["axis_pair_training"],
        "holdout_baseline": baseline_holdout,
        "holdout_pair_ranker": pair_holdout,
        "report_path": str(OUT),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
