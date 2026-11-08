"""Research-only fixed-budget comparison across NAR bet types.

All tickets are generated from the same pre-race ranking and evaluated against
official payout records. This is retrospective simulation, not a live strategy.
"""
from __future__ import annotations

import itertools
import random
from collections import defaultdict

import nar_backtest as nb

STRATEGIES = (
    "LAB", "independent_raw_CatBoost", "independent_raw_LightGBM",
    "LambdaRank", "LambdaRank_tuned", "Market_favorite",
)
BET_TYPES = ("win", "wide_top3", "quinella_top2", "exacta_top2",
             "trio_top3", "trifecta_top3_box")


def tickets_for_ranked(kind, ranked):
    """Return unordered/ordered tickets with exactly specified ranked runners."""
    nos = [str(row["no"]) for row in ranked]
    if kind == "win":
        return [[nos[0]]]
    if kind == "wide_top3":
        return [list(x) for x in itertools.combinations(nos[:3], 2)]
    if kind == "quinella_top2":
        return [nos[:2]]
    if kind == "exacta_top2":
        return [nos[:2]]
    if kind == "trio_top3":
        return [nos[:3]]
    if kind == "trifecta_top3_box":
        return [list(x) for x in itertools.permutations(nos[:3], 3)]
    raise ValueError(f"Unknown bet type: {kind}")


def _percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def _cluster_bootstrap_roi(race_returns, race_stakes, seed=42, iterations=2000):
    """Race-cluster bootstrap CI; races, not individual tickets, are resampled."""
    if len(race_returns) < 20:
        return {"low": None, "high": None, "method": "race_cluster_bootstrap",
                "iterations": iterations, "reason": "fewer_than_20_races"}
    rng = random.Random(seed)
    n = len(race_returns)
    estimates = []
    for _ in range(iterations):
        indices = [rng.randrange(n) for _ in range(n)]
        stake = sum(race_stakes[i] for i in indices)
        returned = sum(race_returns[i] for i in indices)
        if stake:
            estimates.append(100.0 * returned / stake)
    return {
        "low": round(_percentile(estimates, 0.025), 3),
        "high": round(_percentile(estimates, 0.975), 3),
        "method": "race_cluster_bootstrap",
        "iterations": iterations,
    }


def evaluate_bet_types(scored_groups, payouts):
    """Compare win/wide/quinella/exacta/trio/trifecta on identical races."""
    stats = {
        strategy: {
            kind: {"races": 0, "hits": 0, "tickets": 0, "stake_jpy": 0,
                   "return_jpy": 0, "race_returns": [], "race_stakes": []}
            for kind in BET_TYPES
        } for strategy in STRATEGIES
    }
    for group, strategy_scores in scored_groups:
        parts = str(group[0]["race_key"]).split("|")
        if len(parts) != 3:
            continue
        payout = payouts.get(nb.race_key(parts[0], parts[1], parts[2]))
        if not payout:
            continue
        for strategy in STRATEGIES:
            scores = strategy_scores.get(strategy)
            if scores is None or len(scores) < 3:
                continue
            order = sorted(range(len(group)),
                           key=lambda i: (-float(scores[i]), int(group[i]["horse_no"])))
            ranked = [{"no": str(group[i]["horse_no"])} for i in order]
            for kind in BET_TYPES:
                tickets = tickets_for_ranked(kind, ranked)
                stake = 100 * len(tickets)
                returned = sum(nb.payout_for(kind.split("_")[0], ticket, payout)
                               for ticket in tickets)
                item = stats[strategy][kind]
                item["races"] += 1
                item["tickets"] += len(tickets)
                item["stake_jpy"] += stake
                item["return_jpy"] += returned
                item["hits"] += int(returned > 0)
                item["race_returns"].append(returned)
                item["race_stakes"].append(stake)

    result = {}
    for strategy, by_type in stats.items():
        result[strategy] = {}
        for kind, values in by_type.items():
            races = values["races"]
            stake = values["stake_jpy"]
            result[strategy][kind] = {
                "races": races,
                "hits": values["hits"],
                "hit_rate_pct": round(100 * values["hits"] / races, 3) if races else None,
                "tickets": values["tickets"],
                "stake_jpy": stake,
                "return_jpy": values["return_jpy"],
                "roi_pct": round(100 * values["return_jpy"] / stake, 3) if stake else None,
                "profit_jpy": values["return_jpy"] - stake,
                "stake_per_race_jpy": round(stake / races, 2) if races else None,
                "roi_ci95": _cluster_bootstrap_roi(values["race_returns"], values["race_stakes"]),
            }
    return {
        "strategies": result,
        "ticket_policy": {
            "win": "Top-ranked runner, 100 JPY",
            "wide_top3": "All three unordered pairs among top 3, 100 JPY each",
            "quinella_top2": "Top two as unordered pair, 100 JPY",
            "exacta_top2": "Top two in predicted order, 100 JPY",
            "trio_top3": "Top three as unordered trio, 100 JPY",
            "trifecta_top3_box": "All six permutations of top three, 100 JPY each",
        },
        "caveat": "Fixed ticket simulation from retrospective rankings and official payouts; does not imply executable pre-off prices or profitability.",
    }
