#!/usr/bin/env python3
"""Research-only five-day NAR trio test: LambdaRank fused with timestamped pre-off market odds.

LambdaRank is trained only on races before the five-day window. The fusion weight is
selected on the first three test dates and evaluated on the final two untouched dates.
No final odds, live model files, or production tickets are used or modified.
"""
from __future__ import annotations
import datetime as dt
import itertools
import json
import math
import re
import unicodedata
from pathlib import Path
import nar_backtest as nb
import nar_ml_challenger_backtest as ml
import nar_trio_compatibility as tc
import nar_trio_learned_compatibility as learned_tc

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "nar-trio-lambdarank-market-fusion-5day.json"
BLEND_WEIGHTS = (0.00, 0.25, 0.50, 0.75, 1.00)
COMPATIBILITY_WEIGHTS = (0.00, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00)
LEARNED_COMPATIBILITY_WEIGHTS = (0.00, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00)
TICKETS_PER_RACE = 12


def race_key_text(value):
    parts = str(value or "").split("|")
    if len(parts) != 3:
        return None
    date = re.sub(r"\D", "", parts[1])
    race = re.sub(r"\D", "", parts[2])
    if len(date) != 8 or not race:
        return None
    return f"{parts[0]}|{date}|{int(race)}"


def snapshot_date(snapshot):
    key = race_key_text(snapshot.get("race_key"))
    if not key:
        return None
    try:
        return dt.datetime.strptime(key.split("|")[1], "%Y%m%d").date()
    except ValueError:
        return None


def valid_snapshot(snapshot):
    try:
        captured = dt.datetime.fromisoformat(str(snapshot["captured_at"]).replace("Z", "+00:00"))
        scheduled = dt.datetime.fromisoformat(str(snapshot["scheduled_post_time"]).replace("Z", "+00:00"))
        minutes = float(snapshot["minutes_to_post"])
    except (KeyError, TypeError, ValueError):
        return False
    return captured.tzinfo is not None and scheduled.tzinfo is not None and captured < scheduled and 2 <= minutes <= 30


def market_win_probabilities(snapshot):
    """Normalize inverse win odds from the same timestamped pre-off snapshot."""
    inverse = {}
    for row in snapshot.get("odds") or []:
        label = unicodedata.normalize("NFKC", str(row.get("type") or "")).replace(" ", "")
        if "単勝" not in label:
            continue
        digits = re.sub(r"\D", "", str(row.get("n1") or ""))
        try:
            odds = float(row.get("odds"))
        except (TypeError, ValueError):
            continue
        if digits and odds > 0 and math.isfinite(odds):
            inverse[str(int(digits))] = 1.0 / odds
    total = sum(inverse.values())
    return {horse: value / total for horse, value in inverse.items()} if total > 0 else {}


def percentile_scores(values):
    """Convert arbitrary per-horse scores into stable [0,1] within-race ranks."""
    if not values:
        return {}
    ordered = sorted(values.items(), key=lambda item: (-float(item[1]), int(item[0])))
    if len(ordered) == 1:
        return {ordered[0][0]: 0.5}
    result = {}
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average_rank = (index + end - 1) / 2
        score = 1.0 - average_rank / (len(ordered) - 1)
        for pos in range(index, end):
            result[ordered[pos][0]] = score
        index = end
    return result


def blend_rank_scores(model_scores, market_scores, market_weight):
    if set(model_scores) != set(market_scores):
        raise ValueError("Model and market scores must cover the same runners")
    return {
        horse: (1.0 - market_weight) * model_scores[horse] + market_weight * market_scores[horse]
        for horse in model_scores
    }


def rank_horses(group, scores):
    return sorted(group, key=lambda row: (-scores[str(row["horse_no"])], int(row["horse_no"])))


def build_trio_tickets(group, scores, limit=TICKETS_PER_RACE, compatibility_weight=0.0):
    """Rank trios with separate individual-merit and pair-compatibility terms."""
    candidates = []
    for combo in itertools.combinations((str(row["horse_no"]) for row in group), 3):
        key = tuple(sorted(combo, key=int))
        score = tc.trio_score(group, key, scores, compatibility_weight)
        candidates.append((score, key))
    candidates.sort(key=lambda item: (-item[0], tuple(map(int, item[1]))))
    return [list(key) for _, key in candidates[:limit]]


def evaluate_races(races, weight, compatibility_weight=0.0):
    totals = {
        "races": 0, "winner_top1_hits": 0, "winner_top3_hits": 0,
        "podium_horses_captured": 0, "podium_horses_total": 0,
        "exact_podium_top3_hits": 0, "trio_hit_races": 0,
        "tickets": 0, "stake_jpy": 0, "return_jpy": 0,
        "compatibility_changed_races": 0, "top12_overlap_tickets": 0,
        "top12_overlap_denominator": 0,
    }
    for item in races:
        scores = blend_rank_scores(item["model_rank_scores"], item["market_rank_scores"], weight)
        ranked = rank_horses(item["group"], scores)
        winner = {str(row["horse_no"]) for row in item["group"] if row["finish"] == 1}
        podium = {str(row["horse_no"]) for row in item["group"] if row["finish"] in (1, 2, 3)}
        if not winner or len(podium) != 3:
            continue
        totals["races"] += 1
        totals["winner_top1_hits"] += int(str(ranked[0]["horse_no"]) in winner)
        totals["winner_top3_hits"] += int(any(str(row["horse_no"]) in winner for row in ranked[:3]))
        predicted_top3 = {str(row["horse_no"]) for row in ranked[:3]}
        totals["podium_horses_captured"] += len(predicted_top3 & podium)
        totals["podium_horses_total"] += 3
        totals["exact_podium_top3_hits"] += int(predicted_top3 == podium)
        tickets = build_trio_tickets(item["group"], scores, compatibility_weight=compatibility_weight)
        baseline_tickets = build_trio_tickets(item["group"], scores, compatibility_weight=0.0)
        baseline_set = {tuple(ticket) for ticket in baseline_tickets}
        current_set = {tuple(ticket) for ticket in tickets}
        overlap = len(baseline_set & current_set)
        totals["compatibility_changed_races"] += int(current_set != baseline_set)
        totals["top12_overlap_tickets"] += overlap
        totals["top12_overlap_denominator"] += len(baseline_set)
        totals["tickets"] += len(tickets)
        totals["stake_jpy"] += len(tickets) * 100
        returned = sum(nb.payout_for("trio", ticket, item["payout"]) or 0 for ticket in tickets)
        totals["return_jpy"] += returned
        totals["trio_hit_races"] += int(returned > 0)
    count = totals["races"]
    stake = totals["stake_jpy"]
    totals.update({
        "winner_top1_rate_pct": round(100 * totals["winner_top1_hits"] / count, 2) if count else None,
        "winner_top3_capture_rate_pct": round(100 * totals["winner_top3_hits"] / count, 2) if count else None,
        "podium_capture_rate_pct": round(100 * totals["podium_horses_captured"] / totals["podium_horses_total"], 2) if totals["podium_horses_total"] else None,
        "exact_podium_top3_rate_pct": round(100 * totals["exact_podium_top3_hits"] / count, 2) if count else None,
        "trio_hit_rate_pct": round(100 * totals["trio_hit_races"] / count, 2) if count else None,
        "roi_pct": round(100 * totals["return_jpy"] / stake, 2) if stake else None,
        "profit_jpy": totals["return_jpy"] - stake,
        "tickets_per_race": round(totals["tickets"] / count, 2) if count else None,
        "compatibility_changed_races": totals["compatibility_changed_races"],
        "top12_overlap_pct": round(100 * totals["top12_overlap_tickets"] / totals["top12_overlap_denominator"], 2)
            if totals["top12_overlap_denominator"] else None,
    })
    return totals



def build_trio_tickets_learned(group, scores, pair_scores, limit=TICKETS_PER_RACE, compatibility_weight=0.0):
    candidates = []
    for combo in itertools.combinations((str(row["horse_no"]) for row in group), 3):
        score = learned_tc.learned_trio_score(combo, scores, pair_scores, compatibility_weight)
        candidates.append((score, tuple(sorted(combo, key=int))))
    candidates.sort(key=lambda item: (-item[0], tuple(map(int, item[1]))))
    return [list(key) for _, key in candidates[:limit]]


def evaluate_learned_compatibility(races, market_weight, compatibility_weight):
    totals = {
        "races": 0, "winner_top1_hits": 0, "winner_top3_hits": 0,
        "podium_horses_captured": 0, "podium_horses_total": 0,
        "trio_hit_races": 0, "tickets": 0, "stake_jpy": 0,
        "return_jpy": 0, "compatibility_changed_races": 0,
        "top12_overlap_tickets": 0, "top12_overlap_denominator": 0,
    }
    for item in races:
        scores = blend_rank_scores(item["model_rank_scores"], item["market_rank_scores"], market_weight)
        ranked = rank_horses(item["group"], scores)
        winner = {str(row["horse_no"]) for row in item["group"] if row["finish"] == 1}
        podium = {str(row["horse_no"]) for row in item["group"] if row["finish"] in (1, 2, 3)}
        if not winner or len(podium) != 3:
            continue
        totals["races"] += 1
        totals["winner_top1_hits"] += int(str(ranked[0]["horse_no"]) in winner)
        totals["winner_top3_hits"] += int(any(str(row["horse_no"]) in winner for row in ranked[:3]))
        totals["podium_horses_captured"] += len({str(row["horse_no"]) for row in ranked[:3]} & podium)
        totals["podium_horses_total"] += 3
        tickets = build_trio_tickets_learned(
            item["group"], scores, item["learned_pair_scores"],
            compatibility_weight=compatibility_weight,
        )
        baseline = build_trio_tickets_learned(
            item["group"], scores, item["learned_pair_scores"], compatibility_weight=0.0,
        )
        baseline_set, current_set = {tuple(x) for x in baseline}, {tuple(x) for x in tickets}
        totals["compatibility_changed_races"] += int(current_set != baseline_set)
        totals["top12_overlap_tickets"] += len(baseline_set & current_set)
        totals["top12_overlap_denominator"] += len(baseline_set)
        totals["tickets"] += len(tickets)
        totals["stake_jpy"] += len(tickets) * 100
        returned = sum(nb.payout_for("trio", ticket, item["payout"]) or 0 for ticket in tickets)
        totals["return_jpy"] += returned
        totals["trio_hit_races"] += int(returned > 0)
    count, stake = totals["races"], totals["stake_jpy"]
    totals.update({
        "winner_top1_rate_pct": round(100 * totals["winner_top1_hits"] / count, 2) if count else None,
        "winner_top3_capture_rate_pct": round(100 * totals["winner_top3_hits"] / count, 2) if count else None,
        "podium_capture_rate_pct": round(100 * totals["podium_horses_captured"] / totals["podium_horses_total"], 2) if totals["podium_horses_total"] else None,
        "trio_hit_rate_pct": round(100 * totals["trio_hit_races"] / count, 2) if count else None,
        "roi_pct": round(100 * totals["return_jpy"] / stake, 2) if stake else None,
        "profit_jpy": totals["return_jpy"] - stake,
        "tickets_per_race": round(totals["tickets"] / count, 2) if count else None,
        "top12_overlap_pct": round(100 * totals["top12_overlap_tickets"] / totals["top12_overlap_denominator"], 2)
            if totals["top12_overlap_denominator"] else None,
    })
    return totals




def build_axis_wide_trio_strategy(group, scores, pair_scores, market_probabilities,
                                  compatibility_weight=0.0, temperature=0.10):
    """Build 2 axis-wide tickets and 6 axis-trio tickets using a pre-off EV proxy.

    The axis is the highest fused individual rank. Wide opponents are ranks 2 and 3.
    For trio, score every axis+pair combination using the learned pair model, convert
    those relative scores to a conditional probability proxy with softmax, then
    multiply by a market-implied payout proxy (inverse square-root product of
    pre-off win probabilities). Select the four opponents maximizing the sum of
    the six ticket EV proxies. This is not a calibrated probability or yen EV:
    exact pre-off wide/trio prices are not available in the snapshot feed.
    """
    if len(group) < 5:
        raise ValueError("Axis strategy requires at least five runners")
    ranked = rank_horses(group, scores)
    numbers = [str(row["horse_no"]) for row in ranked]
    axis = numbers[0]
    wide_opponents = numbers[1:3]
    candidates = numbers[1:]
    combos = list(itertools.combinations(candidates, 2))
    raw_scores = []
    for a, b in combos:
        raw_scores.append(learned_tc.learned_trio_score(
            (axis, a, b), scores, pair_scores, compatibility_weight
        ))
    peak = max(raw_scores)
    exps = [math.exp((value - peak) / max(float(temperature), 1e-6)) for value in raw_scores]
    denominator = sum(exps) or 1.0
    conditional_prob_proxy = [value / denominator for value in exps]

    axis_prob = max(float(market_probabilities.get(axis, 0.0)), 1e-6)
    ev_proxy_by_pair = {}
    for (a, b), probability_proxy in zip(combos, conditional_prob_proxy):
        pa = max(float(market_probabilities.get(a, 0.0)), 1e-6)
        pb = max(float(market_probabilities.get(b, 0.0)), 1e-6)
        market_payout_proxy = 1.0 / math.sqrt(axis_prob * pa * pb)
        ev_proxy_by_pair[tuple(sorted((a, b), key=int))] = probability_proxy * market_payout_proxy

    best_subset = None
    best_value = -1.0
    best_rank_tiebreak = -1.0
    for subset in itertools.combinations(candidates, 4):
        pairs = list(itertools.combinations(subset, 2))
        value = sum(ev_proxy_by_pair[tuple(sorted(pair, key=int))] for pair in pairs)
        rank_tiebreak = sum(float(scores[h]) for h in subset)
        key = tuple(sorted(subset, key=int))
        if (value > best_value + 1e-12 or
            (abs(value - best_value) <= 1e-12 and rank_tiebreak > best_rank_tiebreak + 1e-12) or
            (abs(value - best_value) <= 1e-12 and abs(rank_tiebreak - best_rank_tiebreak) <= 1e-12
             and (best_subset is None or tuple(map(int, key)) < tuple(map(int, best_subset)))):
            best_subset, best_value, best_rank_tiebreak = key, value, rank_tiebreak

    trio_tickets = [
        sorted((axis, a, b), key=int)
        for a, b in itertools.combinations(best_subset, 2)
    ]
    return {
        "axis": axis,
        "wide_opponents": wide_opponents,
        "wide_tickets": [[axis, h] for h in wide_opponents],
        "trio_opponents": list(best_subset),
        "trio_tickets": trio_tickets,
        "trio_ev_proxy_sum": round(best_value, 8),
        "trio_probability_proxy_mass": round(sum(
            conditional_prob_proxy[i] for i, (a, b) in enumerate(combos)
            if a in best_subset and b in best_subset
        ), 8),
    }


def evaluate_axis_wide_trio_strategy(races, market_weight, compatibility_weight):
    """Evaluate fixed 2-wide + 6-trio tickets at 100 JPY each, plus rank-only baseline."""
    totals = {
        "races": 0, "axis_winner_hits": 0, "axis_podium_hits": 0,
        "wide_hit_races": 0, "wide_tickets": 0, "wide_stake_jpy": 0,
        "wide_return_jpy": 0, "trio_hit_races": 0, "trio_tickets": 0,
        "trio_stake_jpy": 0, "trio_return_jpy": 0,
        "combined_hit_races": 0, "combined_stake_jpy": 0, "combined_return_jpy": 0,
        "rank_only_trio_hit_races": 0, "rank_only_trio_return_jpy": 0,
        "ev_trio_hit_races": 0, "ev_trio_return_jpy": 0,
        "trio_opponent_overlap": 0, "trio_opponent_overlap_denominator": 0,
        "axis_podium_total": 0,
    }
    for item in races:
        scores = blend_rank_scores(item["model_rank_scores"], item["market_rank_scores"], market_weight)
        group = item["group"]
        podium = {str(row["horse_no"]) for row in group if row["finish"] in (1, 2, 3)}
        winners = {str(row["horse_no"]) for row in group if row["finish"] == 1}
        if len(podium) != 3 or not winners:
            continue
        strategy = build_axis_wide_trio_strategy(
            group, scores, item["learned_pair_scores"], item["market_win_probabilities"],
            compatibility_weight=compatibility_weight,
        )
        ranked = rank_horses(group, scores)
        rank_only_opponents = [str(row["horse_no"]) for row in ranked[1:5]]
        rank_only_tickets = [
            sorted((strategy["axis"], a, b), key=int)
            for a, b in itertools.combinations(rank_only_opponents, 2)
        ]
        wide_return = sum(nb.payout_for("wide", ticket, item["payout"]) or 0
                          for ticket in strategy["wide_tickets"])
        trio_return = sum(nb.payout_for("trio", ticket, item["payout"]) or 0
                          for ticket in strategy["trio_tickets"])
        rank_only_return = sum(nb.payout_for("trio", ticket, item["payout"]) or 0
                               for ticket in rank_only_tickets)
        totals["races"] += 1
        totals["axis_winner_hits"] += int(strategy["axis"] in winners)
        totals["axis_podium_hits"] += int(strategy["axis"] in podium)
        totals["axis_podium_total"] += 1
        totals["wide_tickets"] += 2
        totals["wide_stake_jpy"] += 200
        totals["wide_return_jpy"] += wide_return
        totals["wide_hit_races"] += int(wide_return > 0)
        totals["trio_tickets"] += 6
        totals["trio_stake_jpy"] += 600
        totals["trio_return_jpy"] += trio_return
        totals["trio_hit_races"] += int(trio_return > 0)
        totals["combined_stake_jpy"] += 800
        totals["combined_return_jpy"] += wide_return + trio_return
        totals["combined_hit_races"] += int(wide_return > 0 or trio_return > 0)
        totals["rank_only_trio_return_jpy"] += rank_only_return
        totals["rank_only_trio_hit_races"] += int(rank_only_return > 0)
        totals["ev_trio_return_jpy"] += trio_return
        totals["ev_trio_hit_races"] += int(trio_return > 0)
        totals["trio_opponent_overlap"] += len(set(strategy["trio_opponents"]) & set(rank_only_opponents))
        totals["trio_opponent_overlap_denominator"] += 4
    n = totals["races"]
    for prefix, stake_key, return_key, hit_key in (
        ("wide", "wide_stake_jpy", "wide_return_jpy", "wide_hit_races"),
        ("trio", "trio_stake_jpy", "trio_return_jpy", "trio_hit_races"),
        ("combined", "combined_stake_jpy", "combined_return_jpy", "combined_hit_races"),
    ):
        stake = totals[stake_key]
        totals[f"{prefix}_roi_pct"] = round(100 * totals[return_key] / stake, 2) if stake else None
        totals[f"{prefix}_profit_jpy"] = totals[return_key] - stake
        totals[f"{prefix}_hit_rate_pct"] = round(100 * totals[hit_key] / n, 2) if n else None
    rank_stake = n * 600
    totals["rank_only_trio_roi_pct"] = round(100 * totals["rank_only_trio_return_jpy"] / rank_stake, 2) if rank_stake else None
    totals["rank_only_trio_profit_jpy"] = totals["rank_only_trio_return_jpy"] - rank_stake
    totals["rank_only_trio_hit_rate_pct"] = round(100 * totals["rank_only_trio_hit_races"] / n, 2) if n else None
    totals["axis_winner_rate_pct"] = round(100 * totals["axis_winner_hits"] / n, 2) if n else None
    totals["axis_podium_rate_pct"] = round(100 * totals["axis_podium_hits"] / n, 2) if n else None
    totals["trio_opponent_overlap_pct"] = round(
        100 * totals["trio_opponent_overlap"] / totals["trio_opponent_overlap_denominator"], 2
    ) if totals["trio_opponent_overlap_denominator"] else None
    totals["stake_per_race_jpy"] = 800
    return totals


def select_market_weight(dev_races):
    """Tune only on the first three dates; never inspect final two-day holdout results."""
    results = {f"{weight:.2f}": evaluate_races(dev_races, weight) for weight in BLEND_WEIGHTS}
    selected = max(BLEND_WEIGHTS, key=lambda weight: (
        results[f"{weight:.2f}"]["trio_hit_rate_pct"] if results[f"{weight:.2f}"]["trio_hit_rate_pct"] is not None else -1,
        results[f"{weight:.2f}"]["podium_capture_rate_pct"] if results[f"{weight:.2f}"]["podium_capture_rate_pct"] is not None else -1,
        results[f"{weight:.2f}"]["winner_top3_capture_rate_pct"] if results[f"{weight:.2f}"]["winner_top3_capture_rate_pct"] is not None else -1,
        -weight,
    ))
    return selected, results


def feature_matrix(group, category_maps):
    return [[row[key] if key not in ml.CATEGORICAL else category_maps[key].get(row[key], -1)
             for key in ml.RAW_FEATURES] for row in group]


def main():
    snapshots = nb.load_odds_snapshots()
    snapshot_by_key = {
        "|".join(key): record for key, record in snapshots.items()
        if valid_snapshot(record) and market_win_probabilities(record)
    }
    if not snapshot_by_key:
        raise RuntimeError("No valid 2-30 minute pre-off market snapshots with win odds were found")
    snapshot_end = max(snapshot_date(record) for record in snapshot_by_key.values())
    nb.END = snapshot_end
    nb.START = snapshot_end - dt.timedelta(days=1095)
    print(f"TRIO_FUSION_ARCHIVE_WINDOW start={nb.START} end={nb.END}", flush=True)
    races, horses, payouts, downloads = nb.load_archives()
    rows, all_dates = ml.build_rows(races, horses)

    eligible_dates = set()
    for key, snapshot in snapshot_by_key.items():
        parts = key.split("|")
        race_key = nb.race_key(parts[0], parts[1], parts[2])
        payout = payouts.get(race_key) or {}
        if race_key in races and payout.get("trio"):
            eligible_dates.add(snapshot_date(snapshot))
    test_dates = sorted(eligible_dates)[-5:]
    if len(test_dates) < 5:
        raise RuntimeError(f"Need five completed snapshot dates with trio payouts; found {len(test_dates)}: {test_dates}")
    test_date_set = set(test_dates)
    test_start = test_dates[0]
    training_dates = [day for day in all_dates if day < test_start]
    if len(training_dates) < 100:
        raise RuntimeError(f"Insufficient prior training dates: {len(training_dates)}")

    training_date_set = set(training_dates)
    training_rows = [row for row in rows if dt.date.fromisoformat(row["date"]) in training_date_set]
    train_cut = max(1, int(len(training_dates) * 0.85))
    fit_dates = set(training_dates[:train_cut])
    dev_dates = set(training_dates[train_cut:])
    fit_groups = ml.group_rows([row for row in training_rows if dt.date.fromisoformat(row["date"]) in fit_dates])
    dev_groups = ml.group_rows([row for row in training_rows if dt.date.fromisoformat(row["date"]) in dev_dates])
    if not fit_groups or not dev_groups:
        raise RuntimeError("Chronological LambdaRank fit/development split is empty")

    fit_rows = [row for row in training_rows if dt.date.fromisoformat(row["date"]) in fit_dates]
    category_maps = {
        key: {value: index for index, value in enumerate(sorted({row[key] for row in fit_rows}))}
        for key in ml.CATEGORICAL
    }
    print(f"TRIO_FUSION_LAMBDARANK fit_races={len(fit_groups)} dev_races={len(dev_groups)}", flush=True)
    ranker = ml.fit_lambdarank(fit_groups, dev_groups, ml.RAW_FEATURES, category_maps)

    training_groups_for_pairs = ml.group_rows(training_rows)
    learned_pair_model = learned_tc.fit_pair_model(training_groups_for_pairs)
    test_rows = [row for row in rows if dt.date.fromisoformat(row["date"]) in test_date_set]
    test_groups = ml.group_rows(test_rows)
    matched = []
    for group in test_groups:
        key = race_key_text(group[0]["race_key"])
        snapshot = snapshot_by_key.get(key)
        parts = key.split("|") if key else []
        race_key = nb.race_key(parts[0], parts[1], parts[2]) if len(parts) == 3 else None
        payout = payouts.get(race_key) if race_key else None
        if not snapshot or not payout or not payout.get("trio"):
            continue
        market_prob = market_win_probabilities(snapshot)
        horse_numbers = {str(row["horse_no"]) for row in group}
        if not horse_numbers.issubset(market_prob):
            continue
        market_ranks = percentile_scores(market_prob)
        model_raw = {
            str(row["horse_no"]): float(score)
            for row, score in zip(group, ranker.predict(feature_matrix(group, category_maps)))
        }
        model_ranks = percentile_scores(model_raw)
        matched.append({
            "date": group[0]["date"], "race_key": key, "group": group,
            "payout": payout, "model_rank_scores": model_ranks,
            "market_rank_scores": {horse: market_ranks[horse] for horse in horse_numbers},
            "market_win_probabilities": {horse: market_prob[horse] for horse in horse_numbers},
            "snapshot_minutes_to_post": snapshot.get("minutes_to_post"),
            "snapshot_model_version": snapshot.get("model_version"),
            "learned_pair_scores": learned_tc.predict_pair_compatibility(group, learned_pair_model),
        })
    matched.sort(key=lambda item: (item["date"], item["race_key"]))
    if len(matched) < 20:
        raise RuntimeError(f"Insufficient same-race pre-off market coverage for a useful five-day pilot: {len(matched)} races")

    dev_dates_for_fusion = {day.isoformat() for day in test_dates[:3]}
    holdout_dates = {day.isoformat() for day in test_dates[3:]}
    dev_races = [item for item in matched if item["date"] in dev_dates_for_fusion]
    holdout_races = [item for item in matched if item["date"] in holdout_dates]
    if len(dev_races) < 10 or len(holdout_races) < 10:
        raise RuntimeError(f"Insufficient fusion split: development={len(dev_races)} holdout={len(holdout_races)}")

    selected_weight, dev_results = select_market_weight(dev_races)
    compatibility_weights = COMPATIBILITY_WEIGHTS
    dev_compatibility_results = {
        f"{cw:.2f}": evaluate_races(dev_races, selected_weight, cw)
        for cw in compatibility_weights
    }
    selected_compatibility_weight = max(compatibility_weights, key=lambda cw: (
        dev_compatibility_results[f"{cw:.2f}"]["trio_hit_rate_pct"]
        if dev_compatibility_results[f"{cw:.2f}"]["trio_hit_rate_pct"] is not None else -1,
        dev_compatibility_results[f"{cw:.2f}"]["roi_pct"]
        if dev_compatibility_results[f"{cw:.2f}"]["roi_pct"] is not None else -1,
        -cw,
    ))
    holdout_results = {f"{weight:.2f}": evaluate_races(holdout_races, weight) for weight in BLEND_WEIGHTS}
    holdout_results["selected_from_development"] = evaluate_races(holdout_races, selected_weight)
    development_learned_compatibility = {
        f"{cw:.2f}": evaluate_learned_compatibility(dev_races, selected_weight, cw)
        for cw in LEARNED_COMPATIBILITY_WEIGHTS
    }
    selected_learned_compatibility_weight = max(LEARNED_COMPATIBILITY_WEIGHTS, key=lambda cw: (
        development_learned_compatibility[f"{cw:.2f}"]["trio_hit_rate_pct"]
        if development_learned_compatibility[f"{cw:.2f}"]["trio_hit_rate_pct"] is not None else -1,
        development_learned_compatibility[f"{cw:.2f}"]["podium_capture_rate_pct"]
        if development_learned_compatibility[f"{cw:.2f}"]["podium_capture_rate_pct"] is not None else -1,
        -cw,
    ))
    holdout_learned_compatibility = {
        f"{cw:.2f}": evaluate_learned_compatibility(holdout_races, selected_weight, cw)
        for cw in LEARNED_COMPATIBILITY_WEIGHTS
    }
    holdout_learned_compatibility["selected_from_development"] = evaluate_learned_compatibility(
        holdout_races, selected_weight, selected_learned_compatibility_weight
    )
    holdout_compatibility_results = {
        f"{cw:.2f}": evaluate_races(holdout_races, selected_weight, cw)
        for cw in compatibility_weights
    }
    holdout_compatibility_results["selected_from_development"] = evaluate_races(
        holdout_races, selected_weight, selected_compatibility_weight
    )
    holdout_front_count_segments = {}
    for label, predicate in (
        ("0-1", lambda n: n <= 1),
        ("2", lambda n: n == 2),
        ("3+", lambda n: n >= 3),
    ):
        segment = [
            item for item in holdout_races
            if predicate(max(float(row.get("predicted_front_count", 0.0)) for row in item["group"]))
        ]
        holdout_front_count_segments[label] = {
            "races": len(segment),
            "baseline": evaluate_races(segment, selected_weight, 0.0) if segment else None,
            "compatibility_50pct": evaluate_races(segment, selected_weight, 0.50) if segment else None,
            "compatibility_100pct": evaluate_races(segment, selected_weight, 1.00) if segment else None,
        }
    axis_strategy_dev = evaluate_axis_wide_trio_strategy(
        dev_races, selected_weight, selected_learned_compatibility_weight
    )
    axis_strategy_holdout = evaluate_axis_wide_trio_strategy(
        holdout_races, selected_weight, selected_learned_compatibility_weight
    )
    axis_strategy_all_days = evaluate_axis_wide_trio_strategy(
        matched, selected_weight, selected_learned_compatibility_weight
    )
    report = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "evaluated", "research_only": True,
        "test_window": {
            "dates": [day.isoformat() for day in test_dates],
            "development_dates_for_fusion": [day.isoformat() for day in test_dates[:3]],
            "untouched_holdout_dates": [day.isoformat() for day in test_dates[3:]],
            "matched_preoff_races": len(matched), "development_races": len(dev_races),
            "holdout_races": len(holdout_races),
            "snapshot_policy": "closest valid 2-30 minute pre-off snapshot; same races and runners for every blend",
        },
        "training": {
            "start": min(training_dates).isoformat(), "end": max(training_dates).isoformat(),
            "dates": len(training_dates), "races": len(ml.group_rows(training_rows)),
            "fit_races": len(fit_groups), "development_races": len(dev_groups),
            "model": "LightGBM LambdaRank", "features": list(ml.RAW_FEATURES),
            "relevance": "winner=3, second=2, third=1, other=0",
            "market_features_used_inside_ranker": False,
            "fusion_learning": "Select market percentile weight from {0,.25,.50,.75,1.00} using only the first three test dates; maximize trio 12-ticket race-hit rate, then podium capture; ties prefer lower market weight.",
        },
        "development_weight_selection": {"selected_market_weight": round(selected_weight, 2), "selection_results": dev_results},
        "learned_pair_compatibility_training": {
            "training_races": len(training_groups_for_pairs),
            "training_pairs": learned_pair_model["training_pairs"],
            "positive_pairs": learned_pair_model["positive_pairs"],
            "negative_pairs": learned_pair_model["negative_pairs"],
            "features": learned_pair_model["features"],
            "method": "balanced logistic regression on pre-race pair features; negative pairs deterministically capped per race",
        },
        "development_learned_compatibility_selection": {
            "selected_compatibility_weight": round(selected_learned_compatibility_weight, 2),
            "candidate_weights": list(LEARNED_COMPATIBILITY_WEIGHTS),
            "selection_results": development_learned_compatibility,
        },
        "untouched_holdout_learned_compatibility": holdout_learned_compatibility,
        "development_compatibility_selection": {
            "selected_compatibility_weight": round(selected_compatibility_weight, 2),
            "candidate_weights": list(compatibility_weights),
            "selection_results": dev_compatibility_results,
        },
        "untouched_holdout_compatibility": holdout_compatibility_results,
        "untouched_holdout_front_count_segments": holdout_front_count_segments,
        "axis_wide_trio_strategy": {
            "ticket_policy": {
                "axis": "Highest fused LambdaRank + pre-off market individual rank",
                "wide": "Axis to individual ranks 2 and 3; 2 unordered wide tickets at 100 JPY each",
                "trio": "Axis plus four opponents selected by sum of six combination EV proxies; 6 unordered trio tickets at 100 JPY each",
                "total_stake_per_race_jpy": 800,
                "rank_only_comparator": "Same axis plus next four individual ranks; six trio tickets at 100 JPY each",
            },
            "selection_method": "For each axis+opponent-pair, softmax the learned trio compatibility score into a relative conditional probability proxy, multiply by inverse square-root product of timestamped pre-off win probabilities as a market-implied payout proxy, then choose the four-opponent subset maximizing the sum across its six pairs.",
            "not_calibrated_warning": "This is a relative EV proxy, not calibrated probability or yen expected value. Exact pre-off wide/trio odds are not available in the snapshot feed; no final odds or realized payout is used in selection. Official payouts are used only after the race for retrospective evaluation.",
            "development": axis_strategy_dev,
            "untouched_holdout": axis_strategy_holdout,
            "all_five_days_descriptive": axis_strategy_all_days,
        },
        "untouched_holdout": holdout_results,
        "all_five_days_descriptive": {f"{weight:.2f}": evaluate_races(matched, weight) for weight in BLEND_WEIGHTS},
        "interpretation": {
            "market_weight": "0.00 = LambdaRank only; 1.00 = pre-off market only; 0.50 = equal blend.",
            "trio_method": "Compare additive individual-score baseline against a separate pair-compatibility term; top 12 unordered combinations at 100 JPY per ticket.",
            "axis_strategy": "2 wide tickets and 6 trio tickets from one top-ranked axis; trio opponents selected by pre-off relative EV proxy. This research-only strategy does not alter production.",
            "individual_score": "LambdaRank and pre-off market rank fusion; compatibility does not alter the individual horse ranking.",
            "pair_compatibility": "Transparent exploratory adjustment using historical first-corner running-style proxy and predicted front-runner crowding. Weight selected on first three test dates only.",
            "compatibility_weight_grid": [0.00, 0.05, 0.10, 0.15, 0.20],
            "limitations": [
                "Only the final two of five dates are untouched holdout; this is a pilot, not proof of profitability.",
                "The fusion weight is tuned on three dates and needs validation on a longer later holdout.",
                "Only timestamped pre-off win odds are used; confirmed final odds are excluded.",
                "No live-model.json, live-history.json, or production ticket logic is changed.",
                "Compatibility weights are swept broadly and top-12 ticket overlap is reported to detect inert adjustments.",
                "Front-count segment results are exploratory; predicted_front_count is a historical proxy, not observed in-race pace.",
                "A separate supervised pair model is trained only on races before the five-day evaluation window; its selected weight is chosen on development dates and measured on untouched holdout dates.",
            ],
        },
        "data_source": {"archive_months": downloads, "snapshot_records_considered": len(snapshot_by_key)},
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "dates": report["test_window"]["dates"],
        "matched_races": len(matched), "development_races": len(dev_races),
        "holdout_races": len(holdout_races), "selected_market_weight": selected_weight,
        "holdout_selected": holdout_results["selected_from_development"],
        "report": str(OUT),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
