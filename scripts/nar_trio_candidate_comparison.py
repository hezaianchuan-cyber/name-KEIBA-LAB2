"""Comparable four-way NAR trio candidate evaluation (research only).

All candidates use the same winner-classifier axis and six trio tickets.
Weights are selected on chronological development races only; holdout is evaluation-only.
"""
from __future__ import annotations

import itertools


def percentile(values):
    def stable_key(h):
        return tuple(int(part) for part in str(h).split("|"))
    ordered = sorted(values, key=lambda h: (-float(values[h]), stable_key(h)))
    if len(ordered) <= 1:
        return {h: 0.5 for h in ordered}
    result = {}
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and float(values[ordered[j]]) == float(values[ordered[i]]):
            j += 1
        rank = (i + j - 1) / 2
        score = 1.0 - rank / (len(ordered) - 1)
        for k in range(i, j):
            result[ordered[k]] = score
        i = j
    return result


def _subset(group, axis, ability, market, combo, mode, combo_weight=0.5,
            market_weight=0.25):
    horses = [str(row["horse_no"]) for row in group]
    opponents = [h for h in horses if h != axis]
    if len(opponents) < 4:
        raise ValueError("At least five runners are required")
    ability_rank = percentile({h: float(ability[h]) for h in horses})
    market_rank = percentile({h: float(market[h]) for h in horses})
    combo_values = {}
    for members, score in combo.items():
        members = tuple(map(str, members))
        pair = tuple(sorted((h for h in members if h != axis), key=int))
        if len(pair) != 2:
            raise ValueError("Each combination score must identify the two opponents and fixed axis")
        combo_values["|".join(pair)] = float(score)
    combo_rank = percentile(combo_values)
    if mode == "ability":
        return tuple(sorted(sorted(opponents,
                                   key=lambda h: (-ability_rank[h], int(h)))[:4], key=int))
    best, best_score = None, float("-inf")
    for subset in itertools.combinations(opponents, 4):
        score = 0.0
        for a, b in itertools.combinations(subset, 2):
            key = "|".join(sorted((a, b), key=int))
            c = combo_rank[key]
            ab = (ability_rank[a] + ability_rank[b]) / 2
            mk = (market_rank[a] + market_rank[b]) / 2
            if mode == "combo":
                pair_score = c
            elif mode == "ability_combo":
                pair_score = (1.0 - combo_weight) * ab + combo_weight * c
            elif mode == "ability_market_combo":
                ability_weight = 1.0 - market_weight - combo_weight
                pair_score = ability_weight * ab + market_weight * mk + combo_weight * c
            else:
                raise ValueError(f"Unknown mode: {mode}")
            score += pair_score
        key = tuple(sorted(subset, key=int))
        if score > best_score + 1e-12 or (
            abs(score - best_score) <= 1e-12 and
            (best is None or tuple(map(int, key)) < tuple(map(int, best)))
        ):
            best, best_score = key, score
    return best


def _tickets(axis, opponents):
    return [sorted((axis, a, b), key=int)
            for a, b in itertools.combinations(opponents, 2)]


def evaluate(races, mode, combo_weight=0.5, market_weight=0.25):
    import nar_backtest as nb
    totals = {"races": 0, "axis_wins": 0, "axis_podiums": 0,
              "hits": 0, "tickets": 0, "stake_jpy": 0, "return_jpy": 0}
    for item in races:
        group = item["group"]
        axis_scores = item["winner_model_scores"]
        axis = min(axis_scores, key=lambda h: (-float(axis_scores[h]), int(h)))
        opponents = _subset(
            group, axis, item["model_rank_scores"], item["market_rank_scores"],
            item["winner_axis_trio_scores"], mode, combo_weight, market_weight,
        )
        tickets = _tickets(axis, opponents)
        payout = sum(nb.payout_for("trio", ticket, item["payout"]) or 0
                     for ticket in tickets)
        winners = {str(r["horse_no"]) for r in group if r.get("finish") == 1}
        podium = {str(r["horse_no"]) for r in group if r.get("finish") in (1, 2, 3)}
        if len(winners) != 1 or len(podium) != 3:
            continue
        totals["races"] += 1
        totals["axis_wins"] += int(axis in winners)
        totals["axis_podiums"] += int(axis in podium)
        totals["hits"] += int(payout > 0)
        totals["tickets"] += len(tickets)
        totals["stake_jpy"] += len(tickets) * 100
        totals["return_jpy"] += payout
    n, stake = totals["races"], totals["stake_jpy"]
    totals.update({
        "axis_win_rate_pct": round(100 * totals["axis_wins"] / n, 2) if n else None,
        "axis_podium_rate_pct": round(100 * totals["axis_podiums"] / n, 2) if n else None,
        "trio_hit_rate_pct": round(100 * totals["hits"] / n, 2) if n else None,
        "roi_pct": round(100 * totals["return_jpy"] / stake, 2) if stake else None,
        "profit_jpy": totals["return_jpy"] - stake,
        "tickets_per_race": round(totals["tickets"] / n, 2) if n else None,
        "stake_per_race_jpy": 600,
    })
    return totals


def compare_and_select(dev_races, holdout_races):
    """Select C/D weights on dev only, then score all four unchanged on holdout."""
    combo_grid = (0.25, 0.50, 0.75)
    simplex = [(a, m, c) for a in (0.25, 0.50, 0.75)
               for m in (0.0, 0.25, 0.50)
               for c in (0.25, 0.50, 0.75)
               if abs(a + m + c - 1.0) < 1e-9 and a > 0 and m > 0 and c > 0]
    dev = {
        "A_ability_only": evaluate(dev_races, "ability"),
        "B_combo_only": evaluate(dev_races, "combo"),
        "C_ability_plus_combo": {
            f"combo_{w:.2f}": evaluate(dev_races, "ability_combo", combo_weight=w)
            for w in combo_grid
        },
        "D_ability_market_combo": {
            f"ability_{a:.2f}_market_{m:.2f}_combo_{c:.2f}":
                evaluate(dev_races, "ability_market_combo", combo_weight=c, market_weight=m)
            for a, m, c in simplex
        },
    }
    def rank(result):
        return (
            result["trio_hit_rate_pct"] if result["trio_hit_rate_pct"] is not None else -1,
            result["roi_pct"] if result["roi_pct"] is not None else -1,
            result["profit_jpy"],
        )
    c_key = max(dev["C_ability_plus_combo"], key=lambda k: rank(dev["C_ability_plus_combo"][k]))
    d_key = max(dev["D_ability_market_combo"], key=lambda k: rank(dev["D_ability_market_combo"][k]))
    c_weight = float(c_key.split("_")[-1])
    parts = d_key.split("_")
    d_market, d_combo = float(parts[3]), float(parts[5])
    holdout = {
        "A_ability_only": evaluate(holdout_races, "ability"),
        "B_combo_only": evaluate(holdout_races, "combo"),
        "C_ability_plus_combo_selected_on_development": evaluate(
            holdout_races, "ability_combo", combo_weight=c_weight),
        "D_ability_market_combo_selected_on_development": evaluate(
            holdout_races, "ability_market_combo", combo_weight=d_combo,
            market_weight=d_market),
    }
    return {
        "policy": "Same winner-classifier axis and six 100-JPY trio tickets per race. A uses LambdaRank ability only; B uses supervised trio-combination scores only; C blends ability and combination score; D blends ability, same-race pre-off market rank and combination score. C/D weights selected only on chronological development races by trio hit rate, then ROI, then profit.",
        "development": dev,
        "selected_on_development": {
            "C": {"key": c_key, "combo_weight": c_weight},
            "D": {"key": d_key, "market_weight": d_market, "combo_weight": d_combo,
                  "ability_weight": round(1.0 - d_market - d_combo, 2)},
        },
        "untouched_holdout": holdout,
        "research_only": True,
    }
