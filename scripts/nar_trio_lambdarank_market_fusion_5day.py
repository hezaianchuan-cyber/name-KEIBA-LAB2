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
import os
import re
import unicodedata
from pathlib import Path
import nar_backtest as nb
import nar_ml_challenger_backtest as ml
import nar_trio_compatibility as tc
import nar_trio_learned_compatibility as learned_tc
import nar_axis_longshot_research as axis_longshot

ROOT = Path(__file__).resolve().parents[1]
TEST_DAYS = int(os.environ.get("NAR_RESEARCH_TEST_DAYS", "5"))
DEV_FRACTION = float(os.environ.get("NAR_RESEARCH_DEV_FRACTION", "0.60"))
OUT = Path(os.environ.get(
    "NAR_RESEARCH_OUTPUT",
    str(ROOT / "data" / "nar-trio-lambdarank-market-fusion-5day.json"),
))
if TEST_DAYS < 5:
    raise ValueError("NAR_RESEARCH_TEST_DAYS must be at least 5")
if not 0.50 <= DEV_FRACTION <= 0.80:
    raise ValueError("NAR_RESEARCH_DEV_FRACTION must be between 0.50 and 0.80")
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
                                  compatibility_weight=0.0, temperature=0.10,
                                  axis_scores=None):
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
    # Axis selection can use a specialist model independently from opponent ranking.
    # Default behavior is unchanged when axis_scores is omitted.
    axis_ranked = rank_horses(group, axis_scores if axis_scores is not None else scores)
    axis = str(axis_ranked[0]["horse_no"])
    candidates = [
        str(row["horse_no"]) for row in sorted(
            (row for row in group if str(row["horse_no"]) != axis),
            key=lambda row: (-float(scores[str(row["horse_no"])]), int(row["horse_no"]))
        )
    ]
    wide_opponents = candidates[:2]
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
             and (best_subset is None or tuple(map(int, key)) < tuple(map(int, best_subset))))):
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


def evaluate_axis_wide_trio_strategy(races, market_weight, compatibility_weight,
                                      axis_score_key=None):
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
        axis_scores = item[axis_score_key] if axis_score_key else scores
        group = item["group"]
        podium = {str(row["horse_no"]) for row in group if row["finish"] in (1, 2, 3)}
        winners = {str(row["horse_no"]) for row in group if row["finish"] == 1}
        if len(podium) != 3 or not winners:
            continue
        strategy = build_axis_wide_trio_strategy(
            group, scores, item["learned_pair_scores"], item["market_win_probabilities"],
            compatibility_weight=compatibility_weight, axis_scores=axis_scores,
        )
        rank_only_opponents = [
            str(row["horse_no"]) for row in sorted(
                (row for row in group if str(row["horse_no"]) != strategy["axis"]),
                key=lambda row: (-float(scores[str(row["horse_no"])]), int(row["horse_no"]))
            )[:4]
        ]
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


def fit_binary_model(train_rows, dev_rows, target, category_maps):
    """Fit a research-only winner or podium classifier using chronological rows."""
    import numpy as np
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation
    def matrix(rows):
        return np.asarray(feature_matrix(rows, category_maps), dtype=float)
    def labels(rows):
        return np.asarray([
            int(row.get("finish") == 1) if target == "winner"
            else int(row.get("finish") in (1, 2, 3))
            for row in rows
        ], dtype=int)
    y_train, y_dev = labels(train_rows), labels(dev_rows)
    if len(set(y_train.tolist())) < 2 or len(set(y_dev.tolist())) < 2:
        raise RuntimeError(f"{target} classifier requires both classes in fit and early-stop data")
    model = LGBMClassifier(
        objective="binary", n_estimators=900, learning_rate=0.035,
        num_leaves=15, min_child_samples=80, reg_lambda=2.0,
        subsample=0.9, colsample_bytree=0.9, random_state=42,
        n_jobs=4, verbosity=-1,
    )
    model.fit(matrix(train_rows), y_train, eval_set=[(matrix(dev_rows), y_dev)],
              callbacks=[early_stopping(60, verbose=False), log_evaluation(0)])
    return model


def classifier_probabilities(model, group, category_maps):
    return model.predict_proba(feature_matrix(group, category_maps))[:, 1].tolist()


def evaluate_axis_model(races, score_key, probability_key=None, target="winner"):
    races_n = axis_wins = axis_podiums = top7_podiums = podium_total = 0
    probs, labels = [], []
    for item in races:
        group = item["group"]
        scores = item[score_key]
        if len(group) < 3 or len(scores) != len(group):
            continue
        podium = {str(r["horse_no"]) for r in group if r.get("finish") in (1, 2, 3)}
        winners = {str(r["horse_no"]) for r in group if r.get("finish") == 1}
        if len(podium) != 3 or len(winners) != 1:
            continue
        ordered = sorted(range(len(group)), key=lambda i: (-float(scores[str(group[i]["horse_no"])]),
                                                            int(group[i]["horse_no"])))
        axis = str(group[ordered[0]]["horse_no"])
        top7 = {str(group[i]["horse_no"]) for i in ordered[:7]}
        races_n += 1
        axis_wins += int(axis in winners)
        axis_podiums += int(axis in podium)
        top7_podiums += len(top7 & podium)
        podium_total += 3
        if probability_key:
            pmap = item[probability_key]
            for row in group:
                horse = str(row["horse_no"])
                probs.append(min(max(float(pmap[horse]), 1e-7), 1 - 1e-7))
                labels.append(int(row.get("finish") == 1) if target == "winner"
                              else int(row.get("finish") in (1, 2, 3)))
    result = {
        "races": races_n,
        "axis_win_rate_pct": round(100 * axis_wins / races_n, 2) if races_n else None,
        "axis_podium_rate_pct": round(100 * axis_podiums / races_n, 2) if races_n else None,
        "top7_podium_coverage_pct": round(100 * top7_podiums / podium_total, 2) if podium_total else None,
        "top7_podium_horses_captured": top7_podiums,
        "top7_podium_horses_total": podium_total,
    }
    if probs:
        import numpy as np
        y = np.asarray(labels, dtype=float)
        p = np.asarray(probs, dtype=float)
        bins = []
        ece = 0.0
        for lo in np.linspace(0.0, 0.9, 10):
            hi = lo + 0.1
            mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
            if mask.any():
                gap = abs(float(y[mask].mean()) - float(p[mask].mean()))
                ece += float(mask.mean()) * gap
                bins.append({"lower": round(float(lo), 2), "upper": round(float(hi), 2),
                             "count": int(mask.sum()), "mean_probability": round(float(p[mask].mean()), 5),
                             "observed_rate": round(float(y[mask].mean()), 5)})
        result["calibration"] = {
            "target": target,
            "brier_score": round(float(np.mean((p - y) ** 2)), 6),
            "log_loss": round(float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), 6),
            "ece_10_bins": round(ece, 6),
            "bins": bins,
            "warning": "Per-runner binary probabilities are not normalized to sum to one within each race.",
        }
    return result


def blend_win_podium_percentiles(group, winner_scores, podium_scores, podium_weight):
    win_rank = percentile_scores(winner_scores)
    podium_rank = percentile_scores(podium_scores)
    return {
        horse: (1.0 - podium_weight) * win_rank[horse] + podium_weight * podium_rank[horse]
        for horse in win_rank
    }


def select_win_podium_blend(dev_races):
    weights = [round(i / 10, 1) for i in range(11)]
    results = {}
    for weight in weights:
        scored = []
        for item in dev_races:
            scores = blend_win_podium_percentiles(
                item["group"], item["winner_model_scores"], item["podium_model_scores"], weight
            )
            scored.append({**item, "candidate_scores": scores})
        results[f"{weight:.1f}"] = evaluate_axis_model(scored, "candidate_scores")
    selected = max(weights, key=lambda w: (
        results[f"{w:.1f}"]["axis_podium_rate_pct"] if results[f"{w:.1f}"]["axis_podium_rate_pct"] is not None else -1,
        results[f"{w:.1f}"]["axis_win_rate_pct"] if results[f"{w:.1f}"]["axis_win_rate_pct"] is not None else -1,
        results[f"{w:.1f}"]["top7_podium_coverage_pct"] if results[f"{w:.1f}"]["top7_podium_coverage_pct"] is not None else -1,
        -w,
    ))
    return selected, results


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
    test_dates = sorted(eligible_dates)[-TEST_DAYS:]
    if len(test_dates) < TEST_DAYS:
        raise RuntimeError(
            f"Need {TEST_DAYS} completed snapshot dates with trio payouts; "
            f"found {len(test_dates)}: {test_dates}"
        )
    dev_date_count = max(1, min(len(test_dates) - 1, int(len(test_dates) * DEV_FRACTION)))
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
    dev_rows = [row for row in training_rows if dt.date.fromisoformat(row["date"]) in dev_dates]
    winner_model = fit_binary_model(fit_rows, dev_rows, "winner", category_maps)
    podium_model = fit_binary_model(fit_rows, dev_rows, "podium", category_maps)
    print(f"TRIO_AXIS_CLASSIFIERS winner_fit_rows={len(fit_rows)} podium_fit_rows={len(fit_rows)} "
          f"early_stop_rows={len(dev_rows)}", flush=True)

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
        winner_probs_list = classifier_probabilities(winner_model, group, category_maps)
        podium_probs_list = classifier_probabilities(podium_model, group, category_maps)
        winner_probs = {str(row["horse_no"]): float(prob)
                        for row, prob in zip(group, winner_probs_list)}
        podium_probs = {str(row["horse_no"]): float(prob)
                        for row, prob in zip(group, podium_probs_list)}
        matched.append({
            "date": group[0]["date"], "race_key": key, "group": group,
            "payout": payout, "model_rank_scores": model_ranks,
            "winner_model_scores": percentile_scores(winner_probs),
            "podium_model_scores": percentile_scores(podium_probs),
            "winner_model_probabilities": winner_probs,
            "podium_model_probabilities": podium_probs,
            "market_rank_scores": {horse: market_ranks[horse] for horse in horse_numbers},
            "market_win_probabilities": {horse: market_prob[horse] for horse in horse_numbers},
            "snapshot_minutes_to_post": snapshot.get("minutes_to_post"),
            "snapshot_model_version": snapshot.get("model_version"),
            "learned_pair_scores": learned_tc.predict_pair_compatibility(group, learned_pair_model),
        })
    matched.sort(key=lambda item: (item["date"], item["race_key"]))
    if len(matched) < 20:
        raise RuntimeError(f"Insufficient same-race pre-off market coverage for a useful five-day pilot: {len(matched)} races")

    dev_dates_for_fusion = {day.isoformat() for day in test_dates[:dev_date_count]}
    holdout_dates = {day.isoformat() for day in test_dates[dev_date_count:]}
    dev_races = [item for item in matched if item["date"] in dev_dates_for_fusion]
    holdout_races = [item for item in matched if item["date"] in holdout_dates]
    if len(dev_races) < 10 or len(holdout_races) < 10:
        raise RuntimeError(f"Insufficient fusion split: development={len(dev_races)} holdout={len(holdout_races)}")

    selected_weight, dev_results = select_market_weight(dev_races)
    axis_blend_weight, axis_blend_dev_results = select_win_podium_blend(dev_races)
    axis_model_development = {
        "LambdaRank_baseline": evaluate_axis_model(dev_races, "model_rank_scores"),
        "winner_classifier": evaluate_axis_model(dev_races, "winner_model_scores",
                                                  "winner_model_probabilities", "winner"),
        "podium_classifier": evaluate_axis_model(dev_races, "podium_model_scores",
                                                  "podium_model_probabilities", "podium"),
        "winner_podium_blend": axis_blend_dev_results[f"{axis_blend_weight:.1f}"],
        "preoff_market_favorite": evaluate_axis_model(dev_races, "market_rank_scores"),
    }
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
    holdout_axis_scores = []
    for item in holdout_races:
        holdout_axis_scores.append({
            **item,
            "winner_podium_blend_scores": blend_win_podium_percentiles(
                item["group"], item["winner_model_scores"], item["podium_model_scores"],
                axis_blend_weight
            ),
        })
    axis_model_holdout = {
        "LambdaRank_baseline": evaluate_axis_model(holdout_races, "model_rank_scores"),
        "winner_classifier": evaluate_axis_model(holdout_races, "winner_model_scores",
                                                  "winner_model_probabilities", "winner"),
        "podium_classifier": evaluate_axis_model(holdout_races, "podium_model_scores",
                                                  "podium_model_probabilities", "podium"),
        "winner_podium_blend": evaluate_axis_model(holdout_axis_scores, "winner_podium_blend_scores"),
        "preoff_market_favorite": evaluate_axis_model(holdout_races, "market_rank_scores"),
    }
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
    # Research candidate: use the winner specialist only for axis selection,
    # while keeping the exact same fused opponent scores, pair model and ticket count.
    axis_classifier_strategy_dev = evaluate_axis_wide_trio_strategy(
        dev_races, selected_weight, selected_learned_compatibility_weight,
        axis_score_key="winner_model_scores",
    )
    axis_classifier_strategy_holdout = evaluate_axis_wide_trio_strategy(
        holdout_races, selected_weight, selected_learned_compatibility_weight,
        axis_score_key="winner_model_scores",
    )
    axis_classifier_strategy_all_days = evaluate_axis_wide_trio_strategy(
        matched, selected_weight, selected_learned_compatibility_weight,
        axis_score_key="winner_model_scores",
    )
    # Axis uses the individual fused ranking. Opponents are then re-ranked
    # conditionally on that fixed axis using the independently trained pair model.
    longshot_races = []
    winner_axis_longshot_races = []
    for item in matched:
        individual_scores = blend_rank_scores(
            item["model_rank_scores"], item["market_rank_scores"], selected_weight
        )
        axis = str(rank_horses(item["group"], individual_scores)[0]["horse_no"])
        opponent_scores = axis_longshot.build_axis_conditioned_opponent_scores(
            item["group"], axis, individual_scores, item["learned_pair_scores"],
            selected_learned_compatibility_weight
        )
        common = {
            "group": item["group"],
            "market_win_probabilities": item["market_win_probabilities"],
            "payout": item["payout"],
        }
        longshot_races.append({
            **common, "axis_scores": individual_scores, "opponent_scores": opponent_scores,
        })
        # Separate axis specialist candidate; no holdout tuning and no live promotion.
        winner_axis_scores = item["winner_model_scores"]
        winner_axis = str(rank_horses(item["group"], winner_axis_scores)[0]["horse_no"])
        winner_opponent_scores = axis_longshot.build_axis_conditioned_opponent_scores(
            item["group"], winner_axis, winner_axis_scores, item["learned_pair_scores"],
            selected_learned_compatibility_weight
        )
        winner_axis_longshot_races.append({
            **common, "axis_scores": winner_axis_scores,
            "opponent_scores": winner_opponent_scores,
        })
    longshot_dev = axis_longshot.evaluate_axis_longshot([
        race for race, item in zip(longshot_races, matched)
        if item["date"] in dev_dates_for_fusion
    ])
    longshot_holdout = axis_longshot.evaluate_axis_longshot([
        race for race, item in zip(longshot_races, matched)
        if item["date"] in holdout_dates
    ])
    longshot_all_days = axis_longshot.evaluate_axis_longshot(longshot_races)
    winner_axis_longshot_dev = axis_longshot.evaluate_axis_longshot([
        race for race, item in zip(winner_axis_longshot_races, matched)
        if item["date"] in dev_dates_for_fusion
    ])
    winner_axis_longshot_holdout = axis_longshot.evaluate_axis_longshot([
        race for race, item in zip(winner_axis_longshot_races, matched)
        if item["date"] in holdout_dates
    ])
    winner_axis_longshot_all_days = axis_longshot.evaluate_axis_longshot(winner_axis_longshot_races)
    report = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "evaluated", "research_only": True,
        "test_window": {
            "dates": [day.isoformat() for day in test_dates],
            "requested_test_days": TEST_DAYS,
            "development_date_count": dev_date_count,
            "holdout_date_count": len(test_dates) - dev_date_count,
            "development_fraction": DEV_FRACTION,
            "development_dates_for_fusion": [day.isoformat() for day in test_dates[:dev_date_count]],
            "untouched_holdout_dates": [day.isoformat() for day in test_dates[dev_date_count:]],
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
            "fusion_learning": "Select market percentile weight from {0,.25,.50,.75,1.00} using only the chronological development dates; maximize trio 12-ticket race-hit rate, then podium capture; ties prefer lower market weight.",
        },
        "development_weight_selection": {"selected_market_weight": round(selected_weight, 2), "selection_results": dev_results},
        "axis_model_comparison": {
            "purpose": "Research-only comparison of winner, podium and combined ability models against current LambdaRank and same-race pre-off market favorite.",
            "training": {
                "fit_dates": {"start": min(fit_dates).isoformat(), "end": max(fit_dates).isoformat()},
                "early_stopping_dates": {"start": min(dev_dates).isoformat(), "end": max(dev_dates).isoformat()},
                "fit_rows": len(fit_rows), "early_stopping_rows": len(dev_rows),
                "winner_label": "finish == 1", "podium_label": "finish in 1,2,3",
                "features": list(ml.RAW_FEATURES),
                "policy": "Only historical races before the chronological evaluation window are used for fitting; feature dictionaries fit on fit rows only; dev rows are used only for early stopping.",
            },
            "development": axis_model_development,
            "blend_selection": {
                "selected_podium_model_weight": axis_blend_weight,
                "selection_metric_priority": "Axis podium rate, then axis win rate, then top-7 podium coverage; selected using chronological development dates only.",
                "candidate_results": axis_blend_dev_results,
            },
            "untouched_holdout": axis_model_holdout,
            "holdout_policy": "All later chronological holdout dates remain untouched until model and blend selection are complete.",
        },
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
        "axis_longshot_strategy": {
            "development": longshot_dev,
            "untouched_holdout": longshot_holdout,
            "all_days_descriptive": longshot_all_days,
            "winner_classifier_axis_candidate": {
                "development": winner_axis_longshot_dev,
                "untouched_holdout": winner_axis_longshot_holdout,
                "all_days_descriptive": winner_axis_longshot_all_days,
            },
        },
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
            "all_days_descriptive": axis_strategy_all_days,
            "winner_classifier_axis_candidate": {
                "development": axis_classifier_strategy_dev,
                "untouched_holdout": axis_classifier_strategy_holdout,
                "all_days_descriptive": axis_classifier_strategy_all_days,
            },
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
                "The later chronological segment is untouched holdout; sample size and uncertainty must be considered before any promotion.",
                "The fusion weight is tuned only on chronological development dates and evaluated on a later untouched holdout.",
                "Only timestamped pre-off win odds are used; confirmed final odds are excluded.",
                "No live-model.json, live-history.json, or production ticket logic is changed.",
                "Compatibility weights are swept broadly and top-12 ticket overlap is reported to detect inert adjustments.",
                "Front-count segment results are exploratory; predicted_front_count is a historical proxy, not observed in-race pace.",
                "A separate supervised pair model is trained only on races before the five-day evaluation window; its selected weight is chosen on development dates and measured on untouched holdout dates.",
            ],
        },
        "data_source": {"archive_months": downloads, "snapshot_records_considered": len(snapshot_by_key)},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
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