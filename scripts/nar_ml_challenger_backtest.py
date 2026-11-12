#!/usr/bin/env python3
"""Research-only NAR challenger suite: CatBoost, LightGBM, LambdaRank and Plackett-Luce.

The bet-type evaluator also compares six-ticket and ten-ticket trio constructions.

All features are built from information available before the race date. Results from
all races on a date are added only after every race on that date has been featurized.
The final chronological holdout is never used for tuning or early stopping.
"""
from __future__ import annotations
import datetime as dt
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import nar_backtest as nb
from nar_bet_type_comparison import evaluate_bet_types

OUT = ROOT / "data" / "ml-challenger-backtest.json"
BASE_WEIGHTS = {"form": 0.55, "venue": 0.12, "distance": 0.12,
                "sire": 0.10, "jockey": 0.07, "age": 0.04}
LAB_NUMERIC = ["lab_score", "form", "venue_score", "distance_score", "sire_score",
               "jockey_score", "age_score", "field_size", "distance_m"]
RAW_NUMERIC = [
    "prior_runs", "career_win_rate", "career_top3_rate", "recent_finish_pct",
    "recent_top3_rate", "recent_margin_quality", "recent_last3f_rank",
    "recent_time_per_km", "recent_time_trend", "recent_opponent_strength",
    "recent_first_corner_pct", "recent_last_corner_pct", "recent_position_gain",
    "prior_body_weight_avg", "body_weight_delta_pct", "body_weight",
    "body_weight_change", "frame_no", "carried_weight", "predicted_front_count",
    "front_crowding_ratio", "race_rel_recent_finish_pct",
    "race_rel_recent_top3_rate", "race_rel_recent_last_corner_pct",
    "race_rel_recent_last3f_rank", "race_rel_recent_opponent_strength",
    "race_rel_recent_time_per_km", "race_rel_venue_top3_rate",
    "race_rel_distance_top3_rate", "race_rel_jockey_top3_rate",
    "race_rel_body_weight_delta_pct", "venue_runs", "venue_top3_rate", "distance_runs",
    "distance_top3_rate", "going_runs", "going_top3_rate", "jockey_runs",
    "jockey_top3_rate", "sire_runs", "sire_top3_rate", "age", "field_size",
    "distance_m"
]
CATEGORICAL = ["venue", "surface", "going", "jockey", "sire"]
LAB_FEATURES = LAB_NUMERIC + ["venue", "surface", "going"]
RAW_FEATURES = RAW_NUMERIC + CATEGORICAL


def horse_history_key(h):
    """Build a conservative identity key from official NAR runner attributes.

    The published monthly CSV has no permanent horse ID. Use birth date, sex,
    sire, dam and name when complete; isolate incomplete identities per race
    rather than silently merging uncertain horses by name.
    """
    name = nb.norm(h.get("name"))
    birth = re.sub(r"\D", "", str(h.get("birth_date") or ""))
    sex = nb.norm(h.get("sex"))
    sire = nb.norm(h.get("sire"))
    dam = nb.norm(h.get("dam"))
    if len(birth) == 8 and name and sex and sire and dam:
        return "identified|" + "|".join((birth, sex, sire, dam, name)), "strong"
    if len(birth) == 8 and name:
        return "partial|" + "|".join((birth, sex, sire, dam, name)), "partial"
    return "isolated|" + str(h.get("race_key") or "") + "|" + str(h.get("no") or ""), "isolated"


def mean_or(values, default=0.0):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(values) / len(values) if values else default


def rate(rows, prior=0.5, strength=8.0):
    if not rows:
        return prior
    hits = sum(1 for x in rows if x.get("finish") is not None and x["finish"] <= 3)
    return (hits + prior * strength) / (len(rows) + strength)


def add_race_relative_features(race_rows):
    """Add field-relative z-scores using only pre-race runner features."""
    source_features = {
        "recent_finish_pct": "race_rel_recent_finish_pct",
        "recent_top3_rate": "race_rel_recent_top3_rate",
        "recent_last_corner_pct": "race_rel_recent_last_corner_pct",
        "recent_last3f_rank": "race_rel_recent_last3f_rank",
        "recent_opponent_strength": "race_rel_recent_opponent_strength",
        "recent_time_per_km": "race_rel_recent_time_per_km",
        "venue_top3_rate": "race_rel_venue_top3_rate",
        "distance_top3_rate": "race_rel_distance_top3_rate",
        "jockey_top3_rate": "race_rel_jockey_top3_rate",
        "body_weight_delta_pct": "race_rel_body_weight_delta_pct",
    }
    for source, target in source_features.items():
        values = []
        for row in race_rows:
            value = float(row.get(source) or 0.0)
            if math.isfinite(value) and (source != "recent_time_per_km" or value > 0):
                values.append(value)
        center = mean_or(values, 0.0)
        variance = mean_or([(value - center) ** 2 for value in values], 0.0)
        scale = math.sqrt(variance) if variance > 1e-12 else 0.0
        for row in race_rows:
            value = float(row.get(source) or 0.0)
            # Lower seconds per km means faster recent performance, so reverse
            # the field-relative direction for time while keeping higher-is-better
            # semantics for the other pre-race quality features.
            direction = -1.0 if source == "recent_time_per_km" else 1.0
            z = direction * (value - center) / scale if scale > 1e-6 and math.isfinite(value) else 0.0
            row[target] = max(-5.0, min(5.0, z))
    return race_rows


def build_rows(races, horses, include_unplaced=False):
    history = defaultdict(list)
    sire_stats = defaultdict(lambda: {"runs": 0, "top3": 0, "wins": 0})
    jockey_stats = defaultdict(lambda: {"runs": 0, "top3": 0, "wins": 0})
    rows = []
    ordered = sorted(races.values(), key=lambda r: (r["date"], r["venue"], int(r["race"])))
    dates = sorted({r["date"] for r in ordered})
    by_date = defaultdict(list)
    for race in ordered:
        by_date[race["date"]].append(race)

    for day_index, day in enumerate(dates, 1):
        day_pending = []
        day_races = by_date[day]
        for race in day_races:
            field = horses.get(race["key"], [])
            if len(field) < 3:
                continue
            identity_by_no = {}
            field_histories = {}
            for candidate in field:
                candidate_key, _ = horse_history_key({
                    **candidate, "race_key": "|".join(map(str, race["key"]))
                })
                identity_by_no[str(candidate.get("no", ""))] = candidate_key
                field_histories[str(candidate.get("no", ""))] = history.get(candidate_key, [])[-5:]
            front_count = sum(
                1 for prior in field_histories.values()
                if prior and mean_or([x.get("first_corner_pct") for x in prior], 1.0) <= 0.12
            )
            front_ratio = front_count / max(len(field), 1)
            race_rows = []
            for h in field:
                history_key, identity_quality = horse_history_key({**h, "race_key": "|".join(map(str, race["key"]))})
                _, parts = nb.score_runner({**h, "name": history_key}, history, sire_stats, jockey_stats, race, BASE_WEIGHTS)
                score = sum(BASE_WEIGHTS[k] * parts[k] for k in BASE_WEIGHTS)
                finish = nb.rank_value(h.get("finish"))
                finish_raw = str(h.get("finish_raw") or "").strip()
                started_without_rank = any(
                    marker in finish_raw for marker in ("失格", "中止", "落馬")
                )
                if finish is None and not (include_unplaced and started_without_rank):
                    continue
                past = history.get(history_key, [])
                recent = past[-5:]
                prior_weights = [nb.number(x.get("body_weight")) for x in past]
                prior_weights = [x for x in prior_weights if x is not None and x > 0]
                prior_weight_avg = mean_or(prior_weights[-5:], 0.0)
                current_weight = nb.number(h.get("body_weight"))
                weight_delta_pct = (
                    (current_weight - prior_weight_avg) / prior_weight_avg
                    if current_weight is not None and prior_weight_avg > 0 else 0.0
                )
                prior_times = [nb.number(x.get("time_per_km")) for x in recent]
                prior_times = [x for x in prior_times if x is not None and x > 0]
                time_trend = 0.5
                if len(prior_times) >= 3:
                    baseline_time = mean_or(prior_times[:-2], 0.0)
                    recent_time = mean_or(prior_times[-2:], 0.0)
                    if baseline_time > 0:
                        time_trend = max(0.0, min(1.0,
                            0.5 + (baseline_time - recent_time) / baseline_time * 8.0))
                first_positions = [nb.number(x.get("first_corner_pct")) for x in recent]
                first_positions = [x for x in first_positions if x is not None]
                last_positions = [nb.number(x.get("last_corner_pct")) for x in recent]
                last_positions = [x for x in last_positions if x is not None]
                position_gains = [nb.number(x.get("position_gain")) for x in recent]
                position_gains = [x for x in position_gains if x is not None]
                opponent_strength = mean_or(
                    [x.get("opponent_strength") for x in recent if x.get("opponent_strength") is not None],
                    0.5,
                )
                venue_past = [x for x in past if x.get("venue") == race.get("venue")]
                distance = nb.number(race.get("distance"))
                distance_past = [
                    x for x in past if distance is not None and nb.number(x.get("distance")) is not None
                    and abs(nb.number(x.get("distance")) - distance) <= 200
                ]
                venue_distance_past = [
                    x for x in past if x.get("venue") == race.get("venue")
                    and distance is not None and nb.number(x.get("distance")) is not None
                    and abs(nb.number(x.get("distance")) - distance) <= 200
                ]
                going_past = [
                    x for x in past if race.get("going") and x.get("going") == race.get("going")
                    and (not race.get("surface") or not x.get("surface") or x.get("surface") == race.get("surface"))
                ]
                finish_pct = [
                    max(0.0, min(1.0, (float(x.get("field_size") or 12) - float(x["finish"])) /
                                      max(float(x.get("field_size") or 12) - 1, 1)))
                    for x in recent if x.get("finish") is not None
                ]
                recent_times = [nb.number(x.get("time_per_km")) for x in recent]
                recent_last3f = [nb.number(x.get("last3f_rank_score")) for x in recent]
                recent_margins = [nb.number(x.get("margin_quality")) for x in recent]
                js = jockey_stats[h.get("jockey")] if h.get("jockey") else {"runs": 0, "top3": 0, "wins": 0}
                ss = sire_stats[h.get("sire")] if h.get("sire") else {"runs": 0, "top3": 0, "wins": 0}
                row = {
                    "race_key": "|".join(map(str, race["key"])),
                    "horse_identity_quality": identity_quality,
                    "date": race["date"].isoformat(),
                    "horse_no": str(h.get("no", "")),
                    "finish": finish, "finish_raw": finish_raw,
                    "winner": int(finish == 1),
                    "lab_score": float(score),
                    "form": float(parts["form"]),
                    "venue_score": float(parts["venue"]),
                    "distance_score": float(parts["distance"]),
                    "sire_score": float(parts["sire"]),
                    "jockey_score": float(parts["jockey"]),
                    "age_score": float(parts["age"]),
                    "field_size": float(race.get("field_size") or len(field)),
                    "distance_m": float(distance or 0),
                    "venue": str(race.get("venue") or "unknown"),
                    "surface": str(race.get("surface") or "unknown"),
                    "going": str(race.get("going") or "unknown"),
                    "jockey": str(h.get("jockey") or "unknown"),
                    "sire": str(h.get("sire") or "unknown"),
                    "prior_runs": float(len(past)),
                    "career_win_rate": float((sum(1 for x in past if x.get("finish") == 1) + 0.5 * 8) / (len(past) + 8)),
                    "career_top3_rate": float(rate(past)),
                    "recent_finish_pct": float(mean_or(finish_pct, 0.5)),
                    "recent_top3_rate": float(rate(recent)),
                    "recent_margin_quality": float(mean_or(recent_margins, 0.5)),
                    "recent_last3f_rank": float(mean_or(recent_last3f, 0.5)),
                    "recent_time_per_km": float(mean_or(recent_times, 0.0)),
                    "venue_runs": float(len(venue_past)),
                    "venue_top3_rate": float(rate(venue_past)),
                    "distance_runs": float(len(distance_past)),
                    "distance_top3_rate": float(rate(distance_past)),
                    "venue_distance_runs": float(len(venue_distance_past)),
                    "venue_distance_top3_rate": float(rate(venue_distance_past)),
                    "going_runs": float(len(going_past)),
                    "going_top3_rate": float(rate(going_past)),
                    "jockey_runs": float(js["runs"]),
                    "jockey_top3_rate": float((js["top3"] + 0.5 * 10) / (js["runs"] + 10)),
                    "sire_runs": float(ss["runs"]),
                    "sire_top3_rate": float((ss["top3"] + 0.5 * 10) / (ss["runs"] + 10)),
                    "age": float(h.get("age") or 0),
                    "frame_no": float(h.get("frame_no") or 0),
                    "carried_weight": float(h.get("carried") or 0),
                    "body_weight": float(current_weight or 0),
                    "body_weight_change": float(h.get("body_weight_change") or 0),
                    "prior_body_weight_avg": float(prior_weight_avg),
                    "body_weight_delta_pct": float(weight_delta_pct),
                    "recent_time_trend": float(time_trend),
                    "recent_opponent_strength": float(opponent_strength),
                    "recent_first_corner_pct": float(mean_or(first_positions, 0.5)),
                    "recent_last_corner_pct": float(mean_or(last_positions, 0.5)),
                    "recent_position_gain": float(mean_or(position_gains, 0.0)),
                    "predicted_front_count": float(front_count),
                    "front_crowding_ratio": float(front_ratio),
                }
                race_rows.append(row)
            # Relative-to-field signals let LambdaRank judge whether a horse's
            # form, pace profile and suitability are strong versus today's rivals.
            rows.extend(add_race_relative_features(race_rows))

        # Update each race exactly once, only after all races on this date are scored.
        for race in day_races:
            field = horses.get(race["key"], [])
            if len(field) < 3:
                continue
            identity_by_no = {}
            for runner in field:
                history_key, _ = horse_history_key({
                    **runner, "race_key": "|".join(map(str, race["key"]))
                })
                identity_by_no[str(runner.get("no", ""))] = history_key
            performance = nb.field_performance_features(field)
            opposition_field = []
            for runner in field:
                history_key = identity_by_no.get(str(runner.get("no", "")))
                if history_key is not None:
                    opposition_field.append({**runner, "name": history_key})
            opposition_by_no = nb.historical_opponent_strength(opposition_field, history)
            for runner in field:
                if not runner.get("finish"):
                    continue
                sire = runner.get("sire")
                jockey = runner.get("jockey")
                if sire:
                    sire_stats[sire]["runs"] += 1
                    sire_stats[sire]["top3"] += int(runner["finish"] <= 3)
                    sire_stats[sire]["wins"] += int(runner["finish"] == 1)
                if jockey:
                    jockey_stats[jockey]["runs"] += 1
                    jockey_stats[jockey]["top3"] += int(runner["finish"] <= 3)
                    jockey_stats[jockey]["wins"] += int(runner["finish"] == 1)
                history_key = identity_by_no.get(str(runner.get("no", "")))
                if history_key is None:
                    continue
                corner = nb.corner_history_for_runner(runner, race)
                first_corner = corner.get("first_corner_pct")
                last_corner = corner.get("last_corner_pct")
                history[history_key].append({
                    **performance.get(runner["no"], {}),
                    "date": race["date"], "venue": race["venue"],
                    "distance": race.get("distance"), "going": race.get("going"),
                    "surface": race.get("surface"), "field_size": race.get("field_size"),
                    "finish": runner["finish"],
                    "margin_quality": performance.get(runner["no"], {}).get("margin_quality"),
                    "last3f_rank_score": performance.get(runner["no"], {}).get("last3f_rank_score"),
                    "going_quality": (nb.rank_quality(runner["finish"], race.get("field_size")) or 50) / 100.0,
                    "time_seconds": runner.get("time_seconds"),
                    "time_per_km": (runner.get("time_seconds") * 1000 / race["distance"]
                                    if runner.get("time_seconds") and race.get("distance") else None),
                    "frame_no": runner.get("frame_no"),
                    "carried": runner.get("carried"),
                    "body_weight": runner.get("body_weight"),
                    "body_weight_change": runner.get("body_weight_change"),
                    "first_corner_pct": first_corner,
                    "last_corner_pct": last_corner,
                    "position_gain": (first_corner - last_corner
                                      if first_corner is not None and last_corner is not None else None),
                    "opponent_strength": opposition_by_no.get(str(runner.get("no", "")), 0.5),
                    "last3f": runner.get("last3f"),
                })
        if day_index == 1 or day_index % max(1, len(dates)//10) == 0 or day_index == len(dates):
            print(f"ML_DATA_PROGRESS {day_index}/{len(dates)} date={day}", flush=True)
    return rows, dates


def group_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["race_key"]].append(row)
    return [g for g in groups.values() if len(g) >= 3 and any(r["winner"] for r in g)]


def evaluate_scores(name, grouped_scores):
    races = top1 = top3 = 0
    podium_horses = podium_captured = full_podium_covered = any_podium_covered = 0
    for group, scores in grouped_scores:
        winner = next((i for i, r in enumerate(group) if r["winner"]), None)
        if winner is None:
            continue
        order = sorted(range(len(group)), key=lambda i: (-float(scores[i]), group[i]["horse_no"]))
        actual_podium = {i for i, r in enumerate(group)
                         if r.get("finish") is not None and int(r["finish"]) <= 3}
        top7 = set(order[:7])
        races += 1
        top1 += int(order[0] == winner)
        top3 += int(winner in order[:3])
        podium_horses += len(actual_podium)
        captured = len(actual_podium & top7)
        podium_captured += captured
        any_podium_covered += int(captured > 0)
        full_podium_covered += int(len(actual_podium) == 3 and captured == 3)
    return {"name": name, "races": races, "top1_hits": top1, "top3_hits": top3,
            "top1_rate": round(100 * top1 / races, 3) if races else None,
            "top3_capture_rate": round(100 * top3 / races, 3) if races else None,
            "top7_podium_horses_captured": podium_captured,
            "top7_podium_coverage_pct": round(100 * podium_captured / podium_horses, 3) if podium_horses else None,
            "top7_any_podium_rate": round(100 * any_podium_covered / races, 3) if races else None,
            "top7_full_podium_rate": round(100 * full_podium_covered / races, 3) if races else None,
            "top7_policy": "Top 7 predicted runners; coverage is the share of all actual top-3 finishers included. Full-podium rate requires all three actual top-3 finishers in the selected seven."}


def percentile_rank_scores(scores):
    order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
    result = [0.0] * len(scores)
    denom = max(len(scores) - 1, 1)
    for rank, index in enumerate(order):
        result[index] = 1.0 - rank / denom
    return result


def blend_rank_scores(primary, secondary, primary_weight=0.5):
    p = percentile_rank_scores(primary)
    q = percentile_rank_scores(secondary)
    return [primary_weight * x + (1.0 - primary_weight) * y for x, y in zip(p, q)]


def market_odds_start_date(requested_start):
    """Official NAR monthly odds archive is available from 2026-03 onward."""
    return max(requested_start, dt.date(2026, 3, 1))


def race_segment(group, dimension):
    race = group[0]
    if dimension == "venue":
        return str(race.get("venue") or "unknown")
    if dimension == "going":
        return str(race.get("going") or "unknown")
    if dimension == "distance_band":
        distance = float(race.get("distance_m") or 0)
        if distance < 1200: return "<1200m"
        if distance < 1600: return "1200-1599m"
        if distance < 2000: return "1600-1999m"
        return "2000m+"
    if dimension == "field_size_band":
        size = float(race.get("field_size") or len(group))
        if size <= 8: return "8頭以下"
        if size <= 12: return "9-12頭"
        return "13頭以上"
    raise ValueError(f"Unknown race segment: {dimension}")


def summarize_stratified_comparison(scored_groups, dimensions):
    """Summarize each ranking strategy on identical races within each segment."""
    report = {}
    for dimension in dimensions:
        buckets = defaultdict(list)
        for group, strategy_scores in scored_groups:
            buckets[race_segment(group, dimension)].append((group, strategy_scores))
        report[dimension] = {}
        for segment, entries in sorted(buckets.items()):
            names = sorted({name for _, scores in entries for name in scores})
            report[dimension][segment] = {
                "races": len(entries),
                "models": [
                    evaluate_scores(name, [(group, scores[name]) for group, scores in entries])
                    for name in names
                ],
            }
    return report


def evaluate_betting_returns(scored_groups, payouts):
    """Fixed-stake retrospective win and trio returns on identical odds-matched races."""
    strategies = ("LAB", "independent_raw_CatBoost", "independent_raw_LightGBM",
                  "LambdaRank", "LambdaRank_tuned", "LambdaRank_LightGBM_rank_blend_50_50",
                  "Market_favorite")
    stats = {
        name: {
            "win": {"races": 0, "hits": 0, "stake_jpy": 0, "return_jpy": 0},
            "trio_2_4_5_narrow12": {"races": 0, "hits": 0, "tickets": 0,
                                    "stake_jpy": 0, "return_jpy": 0},
        } for name in strategies
    }
    for group, strategy_scores in scored_groups:
        parts = str(group[0]["race_key"]).split("|")
        if len(parts) != 3:
            continue
        race_key = nb.race_key(parts[0], parts[1], parts[2])
        payout = payouts.get(race_key)
        if not payout or not payout.get("win"):
            continue
        for name in strategies:
            scores = strategy_scores.get(name)
            if scores is None:
                continue
            order = sorted(range(len(group)),
                           key=lambda i: (-float(scores[i]), int(group[i]["horse_no"])))
            ranked = [{"no": group[i]["horse_no"]} for i in order]
            win = stats[name]["win"]
            win["races"] += 1
            win["stake_jpy"] += 100
            win["return_jpy"] += nb.payout_for("win", [ranked[0]["no"]], payout)
            win["hits"] += int(nb.payout_for("win", [ranked[0]["no"]], payout) > 0)

            trio = stats[name]["trio_2_4_5_narrow12"]
            if len(ranked) < 3:
                continue
            tickets = nb.build_trio_formation_2_4_5(ranked)
            tickets = nb.narrow_trio_by_rank_sum(tickets, ranked, 12)
            units = nb.allocate_formation_units(tickets, ranked, "equal", budget_units=19)
            trio["races"] += 1
            trio["tickets"] += len(tickets)
            trio["stake_jpy"] += sum(units) * 100
            returned = sum(unit * nb.payout_for("trio", ticket, payout)
                           for ticket, unit in zip(tickets, units))
            trio["return_jpy"] += returned
            trio["hits"] += int(returned > 0)

    for name in strategies:
        for bet_type, values in stats[name].items():
            values["hit_rate_pct"] = round(100 * values["hits"] / values["races"], 3) if values["races"] else None
            values["roi_pct"] = round(100 * values["return_jpy"] / values["stake_jpy"], 3) if values["stake_jpy"] else None
            values["profit_jpy"] = values["return_jpy"] - values["stake_jpy"]
            values["stake_per_race_jpy"] = round(values["stake_jpy"] / values["races"], 2) if values["races"] else None
    return {
        "odds_matched_races_with_official_payouts": max(
            (stats[name]["win"]["races"] for name in strategies), default=0),
        "fixed_stake_methods": {
            "win": "100 yen on each strategy's top-ranked runner",
            "trio_2_4_5_narrow12": "2-4-5 formation narrowed to 12 combinations; equal allocation totaling 1,900 yen per race",
        },
        "strategies": stats,
        "caveat": "Retrospective ranking and official payout simulation; market favorite uses final odds, not executable pre-off odds. No slippage or operational constraints modeled."
    }


def score_model(name, model, test_groups, feature_names, category_maps=None, categorical=False):
    output = []
    for group in test_groups:
        if categorical:
            x = [[r[k] if k in CATEGORICAL else r[k] for k in feature_names] for r in group]
        else:
            x = [[r[k] if k in feature_names and k not in CATEGORICAL else
                  (category_maps[k].get(r[k], -1) if k in CATEGORICAL else r[k])
                  for k in feature_names] for r in group]
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(x)[:, 1]
        else:
            scores = model.predict(x)
        output.append((group, scores))
    return evaluate_scores(name, output)


def fit_classifier(kind, train, dev, features, category_maps):
    import numpy as np
    y_train = np.asarray([r["winner"] for r in train], dtype=int)
    y_dev = np.asarray([r["winner"] for r in dev], dtype=int)
    cat_features = [i for i, k in enumerate(features) if k in CATEGORICAL]
    def matrix(data, cat=False):
        if cat:
            return [[r[k] if k in CATEGORICAL else r[k] for k in features] for r in data]
        return np.asarray([[r[k] if k not in CATEGORICAL else category_maps[k].get(r[k], -1)
                            for k in features] for r in data], dtype=float)
    if kind == "CatBoost":
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(iterations=700, depth=6, learning_rate=0.04,
            loss_function="Logloss", eval_metric="AUC", verbose=False,
            random_seed=42, allow_writing_files=False, thread_count=4)
        model.fit(matrix(train, True), y_train, cat_features=cat_features,
                  eval_set=(matrix(dev, True), y_dev), early_stopping_rounds=60, verbose=False)
        return model
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation
    model = LGBMClassifier(n_estimators=900, learning_rate=0.035, num_leaves=15,
        max_depth=-1, min_child_samples=100, reg_lambda=2.0,
        subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=4, verbosity=-1)
    model.fit(matrix(train), y_train, eval_set=[(matrix(dev), y_dev)],
              callbacks=[early_stopping(60, verbose=False), log_evaluation(0)])
    return model


def fit_lambdarank(train_groups, dev_groups, features, category_maps, params=None):
    import numpy as np
    from lightgbm import LGBMRanker, early_stopping, log_evaluation
    def flatten(groups):
        data, sizes, labels = [], [], []
        for group in groups:
            sizes.append(len(group))
            data.extend(group)
            # Graded relevance: winner > second > third > other finishers.
            labels.extend([max(0, 4 - int(r["finish"])) if r.get("finish") is not None else 0 for r in group])
        x = np.asarray([[r[k] if k not in CATEGORICAL else category_maps[k].get(r[k], -1)
                         for k in features] for r in data], dtype=float)
        return x, np.asarray(labels, dtype=int), sizes
    xtr, ytr, gtr = flatten(train_groups)
    xdv, ydv, gdv = flatten(dev_groups)
    config = {
        "objective": "lambdarank", "metric": "ndcg", "n_estimators": 900,
        "learning_rate": 0.035, "num_leaves": 15, "min_child_samples": 80,
        "reg_lambda": 2.0, "random_state": 42, "n_jobs": 4, "verbosity": -1,
    }
    config.update(params or {})
    model = LGBMRanker(**config)
    model.fit(xtr, ytr, group=gtr, eval_set=[(xdv, ydv)], eval_group=[gdv],
              eval_at=[1, 3], callbacks=[early_stopping(60, verbose=False), log_evaluation(0)])
    return model


def evaluate_wide_top3(scored_groups, payouts):
    """Evaluate three wide pairs from the same top-three ranking, 100 JPY each."""
    import itertools
    races = hits = stake = returned = 0
    for group, scores in scored_groups:
        if len(group) < 3:
            continue
        parts = str(group[0]["race_key"]).split("|")
        if len(parts) != 3:
            continue
        payout = payouts.get(nb.race_key(parts[0], parts[1], parts[2]))
        if not payout:
            continue
        order = sorted(range(len(group)),
                       key=lambda i: (-float(scores[i]), int(group[i]["horse_no"])))
        top = [str(group[i]["horse_no"]) for i in order[:3]]
        tickets = [list(pair) for pair in itertools.combinations(top, 2)]
        race_return = sum(nb.payout_for("wide", ticket, payout) for ticket in tickets)
        races += 1
        stake += 300
        returned += race_return
        hits += int(race_return > 0)
    roi = 100.0 * returned / stake if stake else 0.0
    hit_rate = 100.0 * hits / races if races else 0.0
    # Geometric mean balances ROI and hit rate; it penalizes improving one by
    # sacrificing the other. Selection is performed on development only.
    balance_score = math.sqrt(max(roi, 0.0) * max(hit_rate, 0.0))
    return {
        "races": races, "hit_races": hits, "hit_rate_pct": round(hit_rate, 3),
        "tickets_per_race": 3, "stake_jpy": stake, "return_jpy": returned,
        "roi_pct": round(roi, 3), "profit_jpy": returned - stake,
        "roi_hit_balance_score": round(balance_score, 3),
        "policy": "Top 3 ranked runners; all 3 wide pairs; 100 JPY per pair (300 JPY per race).",
    }


def optimize_lambdarank_for_wide(train_groups, early_stop_groups, selection_groups,
                                 features, category_maps, payouts):
    """Tune LambdaRank on development only; holdout is never read by this selector."""
    candidates = [
        ("baseline", {"learning_rate": 0.035, "num_leaves": 15,
                      "min_child_samples": 80, "reg_lambda": 2.0}),
        ("compact_regularized", {"learning_rate": 0.030, "num_leaves": 7,
                                 "min_child_samples": 120, "reg_lambda": 4.0}),
        ("wide_regularized", {"learning_rate": 0.025, "num_leaves": 31,
                              "min_child_samples": 120, "reg_lambda": 4.0}),
        ("strong_regularization", {"learning_rate": 0.025, "num_leaves": 15,
                                   "min_child_samples": 160, "reg_lambda": 8.0}),
    ]
    if not early_stop_groups or not selection_groups:
        raise RuntimeError("LambdaRank wide optimization requires two non-empty development blocks")
    reports = []
    fitted = {}
    for name, params in candidates:
        model = fit_lambdarank(train_groups, early_stop_groups, features,
                               category_maps, params=params)
        scored = []
        for group in selection_groups:
            x = [[r[k] if k not in CATEGORICAL else category_maps[k].get(r[k], -1)
                  for k in features] for r in group]
            scored.append((group, model.predict(x)))
        metrics = evaluate_wide_top3(scored, payouts)
        reports.append({"name": name, "params": params, "development_metrics": metrics})
        fitted[name] = model
    eligible = [item for item in reports if item["development_metrics"]["races"] >= 100]
    if not eligible:
        raise RuntimeError("Insufficient development payout coverage for wide optimization")
    selected = max(eligible, key=lambda item: (
        item["development_metrics"]["roi_hit_balance_score"],
        item["development_metrics"]["roi_pct"],
        item["development_metrics"]["hit_rate_pct"],
    ))
    return fitted[selected["name"]], {
        "selection_policy": "Select on the second chronological half of development only, after early stopping on the first half. Primary score is geometric mean of wide ROI% and race hit-rate%; tie-break by ROI then hit rate.",
        "early_stopping_races": len(early_stop_groups),
        "selection_races": len(selection_groups),
        "candidates": reports,
        "selected_candidate": selected["name"],
        "selected_params": selected["params"],
        "selected_development_metrics": selected["development_metrics"],
    }


def fit_plackett_luce(train_groups, features, category_maps, train_scale):
    import numpy as np
    from scipy.optimize import minimize
    # Conditional-logit/Plackett-Luce winner likelihood with L2 regularization.
    x_groups, winner_idx = [], []
    for group in train_groups:
        x = np.asarray([[r[k] if k not in CATEGORICAL else category_maps[k].get(r[k], -1)
                         for k in features] for r in group], dtype=float)
        x_groups.append((x / train_scale, train_scale))
        winner_idx.append(next(i for i, r in enumerate(group) if r["winner"]))
    dimension = len(features)
    def objective(beta):
        loss = 0.0
        grad = np.zeros(dimension)
        for (x, _), winner in zip(x_groups, winner_idx):
            z = x @ beta
            z = z - np.max(z)
            expz = np.exp(np.clip(z, -40, 40))
            probs = expz / expz.sum()
            loss += -math.log(max(float(probs[winner]), 1e-15))
            grad += x.T @ probs - x[winner]
        loss /= len(x_groups)
        grad /= len(x_groups)
        loss += 0.01 * float(beta @ beta)
        grad += 0.02 * beta
        return loss, grad
    result = minimize(objective, np.zeros(dimension), jac=True, method="L-BFGS-B",
                      options={"maxiter": 300, "ftol": 1e-9})
    return {"coef": result.x.tolist(), "converged": bool(result.success),
            "message": str(result.message), "iterations": int(result.nit)}


def pl_scores(model, group, features, category_maps, train_scale):
    import numpy as np
    x = np.asarray([[r[k] if k not in CATEGORICAL else category_maps[k].get(r[k], -1)
                     for k in features] for r in group], dtype=float)
    return (x / train_scale) @ np.asarray(model["coef"])


def build_walk_forward_splits(dates, folds=3, initial_train_fraction=0.50):
    """Expanding-window chronological folds; each test block is later than its dev block."""
    ordered = sorted(dates)
    block = max(1, len(ordered) // 10)
    output = []
    for index in range(folds):
        train_end = int(len(ordered) * initial_train_fraction) + index * block
        dev_end = train_end + block
        test_end = dev_end + block
        if test_end > len(ordered) or train_end < 2:
            break
        train_dates = ordered[:train_end]
        dev_dates = ordered[train_end:dev_end]
        test_dates = ordered[dev_end:test_end]
        if train_dates and dev_dates and test_dates:
            output.append({
                "fold": index + 1,
                "train_dates": train_dates,
                "development_dates": dev_dates,
                "test_dates": test_dates,
            })
    return output


def run_walk_forward_validation(rows, dates):
    """Refit independent LambdaRank models over three expanding chronological folds."""
    fold_reports = []
    for split in build_walk_forward_splits(dates):
        train_set = set(split["train_dates"])
        dev_set = set(split["development_dates"])
        test_set = set(split["test_dates"])
        train_rows = [r for r in rows if dt.date.fromisoformat(r["date"]) in train_set]
        dev_rows = [r for r in rows if dt.date.fromisoformat(r["date"]) in dev_set]
        test_rows = [r for r in rows if dt.date.fromisoformat(r["date"]) in test_set]
        train_groups, dev_groups, test_groups = group_rows(train_rows), group_rows(dev_rows), group_rows(test_rows)
        if not train_groups or not dev_groups or not test_groups:
            fold_reports.append({"fold": split["fold"], "status": "skipped_empty_groups"})
            continue
        category_maps = {
            key: {value: index for index, value in enumerate(sorted({r[key] for r in train_rows}))}
            for key in CATEGORICAL
        }
        ranker = fit_lambdarank(train_groups, dev_groups, RAW_FEATURES, category_maps)
        scored = []
        for group in test_groups:
            x = [[r[k] if k not in CATEGORICAL else category_maps[k].get(r[k], -1)
                  for k in RAW_FEATURES] for r in group]
            scored.append((group, ranker.predict(x)))
        metrics = evaluate_scores("LambdaRank_walk_forward", scored)
        fold_reports.append({
            "fold": split["fold"], "status": "evaluated",
            "train_start": split["train_dates"][0].isoformat(),
            "train_end": split["train_dates"][-1].isoformat(),
            "development_start": split["development_dates"][0].isoformat(),
            "development_end": split["development_dates"][-1].isoformat(),
            "test_start": split["test_dates"][0].isoformat(),
            "test_end": split["test_dates"][-1].isoformat(),
            "train_races": len(train_groups), "development_races": len(dev_groups),
            "test_races": len(test_groups), "metrics": metrics,
        })
        print(f"ML_WALK_FORWARD fold={split['fold']} train={len(train_groups)} "
              f"dev={len(dev_groups)} test={len(test_groups)} "
              f"top3={metrics['top3_capture_rate']}", flush=True)
    evaluated = [x for x in fold_reports if x.get("status") == "evaluated"]
    total_races = sum(x["metrics"]["races"] for x in evaluated)
    total_top1 = sum(x["metrics"]["top1_hits"] for x in evaluated)
    total_top3 = sum(x["metrics"]["top3_hits"] for x in evaluated)
    return {
        "policy": "Three expanding chronological folds; training and development always precede each untouched test block. Test blocks do not overlap.",
        "folds": fold_reports,
        "aggregate": {
            "folds_evaluated": len(evaluated), "races": total_races,
            "top1_hits": total_top1, "top3_hits": total_top3,
            "top1_rate": round(100 * total_top1 / total_races, 3) if total_races else None,
            "top3_capture_rate": round(100 * total_top3 / total_races, 3) if total_races else None,
        },
    }


def main():
    races, horses, payouts, downloads = nb.load_archives()
    rows, dates = build_rows(races, horses)
    walk_forward_validation = run_walk_forward_validation(rows, dates)
    if len(dates) < 100:
        raise RuntimeError(f"Insufficient chronological date coverage: {len(dates)}")
    cut_train = max(1, int(len(dates) * 0.70))
    cut_dev = max(cut_train + 1, int(len(dates) * 0.80))
    train_dates = set(dates[:cut_train])
    dev_dates = set(dates[cut_train:cut_dev])
    test_dates = set(dates[cut_dev:])
    train = [r for r in rows if dt.date.fromisoformat(r["date"]) in train_dates]
    dev = [r for r in rows if dt.date.fromisoformat(r["date"]) in dev_dates]
    test = [r for r in rows if dt.date.fromisoformat(r["date"]) in test_dates]
    train_groups, dev_groups, test_groups = group_rows(train), group_rows(dev), group_rows(test)
    if not train_groups or not dev_groups or not test_groups:
        raise RuntimeError("Chronological train/development/holdout split is empty")
    # Category dictionaries and numeric scaling are fit on training data only.
    category_maps = {key: {value: i for i, value in enumerate(sorted({r[key] for r in train}))}
                     for key in CATEGORICAL}
    models = []
    model_objects = {}
    for feature_set, features in (("LAB_feature_enhanced", LAB_FEATURES),
                                  ("independent_raw", RAW_FEATURES)):
        maps = {k: category_maps[k] for k in CATEGORICAL}
        for kind in ("CatBoost", "LightGBM"):
            model = fit_classifier(kind, train, dev, features, maps)
            model_objects[f"{feature_set}_{kind}"] = model
            models.append(score_model(f"{feature_set}_{kind}", model, test_groups,
                                       features, maps, categorical=(kind == "CatBoost")))
    # LAB frozen baseline, on exactly the same eligible holdout races.
    lab_scores = [(g, [r["lab_score"] for r in g]) for g in test_groups]
    models.insert(0, evaluate_scores("LAB_frozen_baseline", lab_scores))

    # LambdaRank and Plackett-Luce use the genuinely LAB-score-free raw feature set.
    ranker = fit_lambdarank(train_groups, dev_groups, RAW_FEATURES, category_maps)
    dev_mid = max(1, len(dev_groups) // 2)
    tuned_ranker, wide_optimization = optimize_lambdarank_for_wide(
        train_groups, dev_groups[:dev_mid], dev_groups[dev_mid:],
        RAW_FEATURES, category_maps, payouts)
    rank_output = []
    for group in test_groups:
        x = [[r[k] if k not in CATEGORICAL else category_maps[k].get(r[k], -1)
              for k in RAW_FEATURES] for r in group]
        rank_output.append((group, ranker.predict(x)))
    models.append(evaluate_scores("LambdaRank_independent_raw", rank_output))
    tuned_rank_output = []
    for group in test_groups:
        x = [[r[k] if k not in CATEGORICAL else category_maps[k].get(r[k], -1)
              for k in RAW_FEATURES] for r in group]
        tuned_rank_output.append((group, tuned_ranker.predict(x)))
    models.append(evaluate_scores("LambdaRank_wide_optimized_independent_raw", tuned_rank_output))
    # LAB + LambdaRank fusion candidates. Each model is converted to within-race
    # percentile ranks before blending, so their raw score scales are never mixed.
    fusion_weights = (0.25, 0.50, 0.75)  # LAB share; LambdaRank supplies the remainder.
    development_fusion_outputs = {w: [] for w in fusion_weights}
    for group in dev_groups:
        x = [[r[k] if k not in CATEGORICAL else category_maps[k].get(r[k], -1)
              for k in RAW_FEATURES] for r in group]
        rank_scores = ranker.predict(x)
        lab_scores = [r["lab_score"] for r in group]
        for weight in fusion_weights:
            development_fusion_outputs[weight].append(
                (group, blend_rank_scores(lab_scores, rank_scores, weight)))
    development_fusion_metrics = [
        {**evaluate_scores(f"LAB_LambdaRank_rank_blend_LAB_{int(w*100)}", scored),
         "lab_weight": w, "lambdarank_weight": 1.0 - w}
        for w, scored in development_fusion_outputs.items()
    ]
    development_bet_inputs = []
    for group_index, group in enumerate(dev_groups):
        scores = {}
        for weight in fusion_weights:
            scores[f"LAB_LambdaRank_rank_blend_LAB_{int(weight*100)}"] = development_fusion_outputs[weight][group_index][1]
        development_bet_inputs.append((group, scores))
    development_bet_returns = evaluate_bet_types(development_bet_inputs, payouts)
    required_fusion_strategies = {
        f"LAB_LambdaRank_rank_blend_LAB_{int(weight*100)}" for weight in fusion_weights
    }
    missing_fusion_strategies = required_fusion_strategies - set(development_bet_returns["strategies"])
    if missing_fusion_strategies:
        raise RuntimeError(
            "Fusion bet-type strategies missing from evaluator: "
            + ", ".join(sorted(missing_fusion_strategies))
        )
    if not any(
        development_bet_returns["strategies"][name]["trio_axis_plus4"]["races"] > 0
        for name in required_fusion_strategies
    ):
        raise RuntimeError("Fusion trio-axis-plus-four evaluation produced zero eligible races")

    hybrid_output = []
    lightgbm_model = model_objects["independent_raw_LightGBM"]
    for group in test_groups:
        x = [[r[k] if k not in CATEGORICAL else category_maps[k].get(r[k], -1)
              for k in RAW_FEATURES] for r in group]
        hybrid_output.append((group, blend_rank_scores(
            ranker.predict(x), lightgbm_model.predict_proba(x)[:, 1], 0.5)))
    models.append(evaluate_scores("LambdaRank_LightGBM_rank_blend_50_50", hybrid_output))
    holdout_fusion_outputs = {}
    for weight in fusion_weights:
        scored = []
        for group in test_groups:
            x = [[r[k] if k not in CATEGORICAL else category_maps[k].get(r[k], -1)
                  for k in RAW_FEATURES] for r in group]
            scored.append((group, blend_rank_scores(
                [r["lab_score"] for r in group], ranker.predict(x), weight)))
        name = f"LAB_LambdaRank_rank_blend_LAB_{int(weight*100)}"
        holdout_fusion_outputs[weight] = scored
        models.append(evaluate_scores(name, scored))
    wide_optimization["holdout_baseline_metrics"] = evaluate_wide_top3(rank_output, payouts)
    wide_optimization["holdout_selected_metrics"] = evaluate_wide_top3(tuned_rank_output, payouts)
    wide_optimization["holdout_policy"] = "Untouched chronological holdout; used only once for final evaluation, never for candidate selection."

    import numpy as np
    scale = np.asarray([[r[k] if k not in CATEGORICAL else category_maps[k].get(r[k], -1)
                         for k in RAW_FEATURES] for r in train], dtype=float).std(axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    pl = fit_plackett_luce(train_groups, RAW_FEATURES, category_maps, scale)
    pl_output = [(g, pl_scores(pl, g, RAW_FEATURES, category_maps, scale)) for g in test_groups]
    models.append(evaluate_scores("PlackettLuce_independent_raw", pl_output))

    # Retrospective market benchmark: compare all challengers on the exact same
    # holdout races and runner subset with official NAR confirmed final win odds.
    # These odds are NOT pre-off prices and are never used as model inputs.
    holdout_start = min(test_dates)
    holdout_end = max(test_dates)
    official_odds_start = market_odds_start_date(holdout_start)
    final_odds, final_odds_downloads = nb.load_final_odds_archives(official_odds_start, holdout_end)
    market_groups = []
    market_output = []
    for group in test_groups:
        parts = str(group[0]["race_key"]).split("|")
        if len(parts) != 3:
            continue
        key = nb.race_key(parts[0], parts[1], parts[2])
        odds_for_race = final_odds.get(key, {})
        eligible = [r for r in group if r["horse_no"] in odds_for_race]
        if len(eligible) < 3 or not any(r["winner"] for r in eligible):
            continue
        market_groups.append(eligible)
        market_output.append((eligible, [-float(odds_for_race[r["horse_no"]]) for r in eligible]))
    market_models = []
    if market_groups:
        market_models.append(evaluate_scores("final_odds_market_favorite", market_output))
        market_models.append(evaluate_scores(
            "LAB_frozen_baseline_same_odds_races",
            [(g, [r["lab_score"] for r in g]) for g in market_groups]))
        for feature_set, features in (("LAB_feature_enhanced", LAB_FEATURES),
                                      ("independent_raw", RAW_FEATURES)):
            for kind in ("CatBoost", "LightGBM"):
                name = f"{feature_set}_{kind}"
                market_models.append(score_model(
                    name + "_same_odds_races", model_objects[name], market_groups,
                    features, category_maps, categorical=(kind == "CatBoost")))
        rank_output_market = []
        for group in market_groups:
            x = [[r[k] if k not in CATEGORICAL else category_maps[k].get(r[k], -1)
                  for k in RAW_FEATURES] for r in group]
            rank_output_market.append((group, ranker.predict(x)))
        market_models.append(evaluate_scores("LambdaRank_independent_raw_same_odds_races",
                                             rank_output_market))
        tuned_rank_output_market = []
        for group in market_groups:
            x = [[r[k] if k not in CATEGORICAL else category_maps[k].get(r[k], -1)
                  for k in RAW_FEATURES] for r in group]
            tuned_rank_output_market.append((group, tuned_ranker.predict(x)))
        market_models.append(evaluate_scores("LambdaRank_wide_optimized_same_odds_races",
                                             tuned_rank_output_market))
        hybrid_market_output = []
        for group in market_groups:
            x = [[r[k] if k not in CATEGORICAL else category_maps[k].get(r[k], -1)
                  for k in RAW_FEATURES] for r in group]
            hybrid_market_output.append((group, blend_rank_scores(
                ranker.predict(x), model_objects["independent_raw_LightGBM"].predict_proba(x)[:, 1], 0.5)))
        market_models.append(evaluate_scores("LambdaRank_LightGBM_rank_blend_50_50_same_odds_races",
                                             hybrid_market_output))
        for weight in fusion_weights:
            scored = []
            for group in market_groups:
                x = [[r[k] if k not in CATEGORICAL else category_maps[k].get(r[k], -1)
                      for k in RAW_FEATURES] for r in group]
                scored.append((group, blend_rank_scores(
                    [r["lab_score"] for r in group], ranker.predict(x), weight)))
            market_models.append(evaluate_scores(
                f"LAB_LambdaRank_rank_blend_LAB_{int(weight*100)}_same_odds_races", scored))
        pl_output_market = [(g, pl_scores(pl, g, RAW_FEATURES, category_maps, scale))
                            for g in market_groups]
        market_models.append(evaluate_scores("PlackettLuce_independent_raw_same_odds_races",
                                             pl_output_market))
    # Segment-level diagnostics compare market and model rankings on exactly
    # the same official-odds-matched races. They are descriptive, not tuning data.
    stratified_inputs = []
    for group in market_groups:
        parts = str(group[0]["race_key"]).split("|")
        key = nb.race_key(parts[0], parts[1], parts[2])
        odds_for_race = final_odds.get(key, {})
        strategy_scores = {
            "LAB": [r["lab_score"] for r in group],
            "Market_favorite": [-float(odds_for_race[r["horse_no"]]) for r in group],
        }
        for feature_set, features in (("LAB_feature_enhanced", LAB_FEATURES),
                                      ("independent_raw", RAW_FEATURES)):
            for kind in ("CatBoost", "LightGBM"):
                name = f"{feature_set}_{kind}"
                model = model_objects[name]
                if kind == "CatBoost":
                    x = [[r[k] for k in features] for r in group]
                else:
                    x = [[r[k] if k not in CATEGORICAL else category_maps[k].get(r[k], -1)
                          for k in features] for r in group]
                strategy_scores[name] = model.predict_proba(x)[:, 1].tolist()
        x = [[r[k] if k not in CATEGORICAL else category_maps[k].get(r[k], -1)
              for k in RAW_FEATURES] for r in group]
        strategy_scores["LambdaRank"] = ranker.predict(x).tolist()
        strategy_scores["LambdaRank_tuned"] = tuned_ranker.predict(x).tolist()
        strategy_scores["LambdaRank_LightGBM_rank_blend_50_50"] = blend_rank_scores(
            strategy_scores["LambdaRank"], strategy_scores["independent_raw_LightGBM"], 0.5)
        for weight in fusion_weights:
            strategy_scores[f"LAB_LambdaRank_rank_blend_LAB_{int(weight*100)}"] = blend_rank_scores(
                strategy_scores["LAB"], strategy_scores["LambdaRank"], weight)
        stratified_inputs.append((group, strategy_scores))
    stratified_comparison = summarize_stratified_comparison(
        stratified_inputs, ("venue", "distance_band", "field_size_band", "going"))

    betting_returns = evaluate_betting_returns(stratified_inputs, payouts)
    multi_bet_type_returns = evaluate_bet_types(stratified_inputs, payouts)

    market_comparison = {
        "odds_source": "NAR official monthly confirmed final win odds; retrospective only, not pre-off odds",
        "holdout_period": {"start": holdout_start.isoformat(), "end": holdout_end.isoformat()},
        "official_odds_available_from": "2026-03-01",
        "requested_odds_start": holdout_start.isoformat(),
        "actual_odds_start": official_odds_start.isoformat(),
        "pre_official_archive_period_excluded": (
            {"start": holdout_start.isoformat(),
             "end": (official_odds_start - dt.timedelta(days=1)).isoformat()}
            if holdout_start < official_odds_start else None
        ),
        "holdout_races_total": len(test_groups),
        "races_with_at_least_3_odds_matched_runners_and_winner": len(market_groups),
        "coverage_pct": round(100 * len(market_groups) / len(test_groups), 2) if test_groups else None,
        "races_without_eligible_odds": len(test_groups) - len(market_groups),
        "models": market_models,
        "stratified_comparison": stratified_comparison,
        "betting_returns": betting_returns,
        "multi_bet_type_returns": multi_bet_type_returns,
        "odds_downloads": final_odds_downloads,
        "policy": "Every listed model and final-odds favorite is evaluated on exactly the same holdout races and the same matched runner subset. Market odds are used only as a retrospective benchmark, never as model input or a live betting price."
    }

    # Native model importance is computed without consulting holdout outcomes.
    importance = {}
    for name, model in model_objects.items():
        if name.endswith("CatBoost"):
            values = model.get_feature_importance()
        else:
            values = model.booster_.feature_importance(importance_type="gain")
        features = LAB_FEATURES if "LAB_feature_enhanced" in name else RAW_FEATURES
        importance[name] = sorted(
            [{"feature": f, "importance": round(float(v), 6)} for f, v in zip(features, values)],
            key=lambda x: -x["importance"])[:20]
    importance["LambdaRank_independent_raw"] = sorted(
        [{"feature": f, "importance": round(float(v), 6)}
         for f, v in zip(RAW_FEATURES, ranker.booster_.feature_importance(importance_type="gain"))],
        key=lambda x: -x["importance"])[:20]
    report = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "evaluated",
        "period": {"start": dates[0].isoformat(), "end": dates[-1].isoformat()},
        "split": {"train_end": dates[cut_train-1].isoformat(),
                  "development_start": dates[cut_train].isoformat(),
                  "development_end": dates[cut_dev-1].isoformat(),
                  "holdout_start": dates[cut_dev].isoformat(), "holdout_end": dates[-1].isoformat(),
                  "policy": "Chronological 70/10/20 split; early stopping uses development only; holdout untouched."},
        "identity_audit": {
            "strong_identity_rows": sum(1 for r in rows if r.get("horse_identity_quality") == "strong"),
            "partial_identity_rows": sum(1 for r in rows if r.get("horse_identity_quality") == "partial"),
            "isolated_identity_rows": sum(1 for r in rows if r.get("horse_identity_quality") == "isolated"),
            "policy": "No permanent NAR horse ID is present in the published monthly runner CSV; complete composite identities are used where possible and incomplete identities are isolated per race to prevent cross-horse history contamination."
        },
        "data": {"races": len(races), "runner_rows": len(rows), "dates": len(dates),
                 "train_races": len(train_groups), "development_races": len(dev_groups),
                 "holdout_races": len(test_groups), "downloaded_months": downloads},
        "feature_sets": {"LAB_feature_enhanced": LAB_FEATURES, "independent_raw": RAW_FEATURES},
        "models": models,
        "lab_lambdarank_fusion_development": {
            "candidate_weights": [{"lab": w, "lambdarank": 1.0-w} for w in fusion_weights],
            "ranking_metrics": development_fusion_metrics,
            "bet_type_returns": development_bet_returns,
            "selection_policy": "Weights are compared on chronological development data only; final chronological holdout is reserved for one-time evaluation. No automatic production promotion.",
        },
        "market_comparison": market_comparison,
        "wide_top3_optimization": wide_optimization,
        "walk_forward_validation": walk_forward_validation,
        "feature_importance": importance,
        "plackett_luce_fit": {"converged": pl["converged"], "message": pl["message"],
                              "iterations": pl["iterations"]},
        "limitations": [
            "Research-only ranking; no production model or ticket logic changed.",
            "LAB_feature_enhanced includes LAB score components; independent_raw excludes all LAB-derived scores.",
            "LambdaRank and Plackett-Luce use independent_raw features only.",
            "Top1/Top3 are ranking metrics, not profitability or betting-return evidence.",
            "Market benchmark uses official confirmed final odds retrospectively, not odds available before the race.",
            "Official monthly NAR odds archives are restricted to 2026-03 onward; earlier holdout dates are excluded from the market comparison.",
            "Venue, distance, field-size and going segments are descriptive diagnostics on the same odds-matched races, not evidence of profitability."
            "NAR monthly CSV has no permanent horse ID; history uses a conservative composite of birth date, sex, sire, dam and name, isolating incomplete identities per race.",
            "Feature importance is model-native gain/importance, not causal attribution.",
            "All same-day race results are withheld until that date's feature generation is complete.",
            "All category mappings and PL scaling are fit on training data only; early stopping uses development only."
        ]
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "data": report["data"],
                      "split": report["split"], "models": report["models"],
                      "plackett_luce_fit": report["plackett_luce_fit"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
