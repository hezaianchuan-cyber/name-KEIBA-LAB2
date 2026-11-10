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
        pair = tuple(sorted(members if len(members) == 2
                            else (h for h in members if h != axis), key=int))
        if len(pair) != 2:
            raise ValueError("Each combination score must identify exactly two opponents")
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


def evaluate(races, mode, combo_weight=0.5, market_weight=0.25,
             include_diagnostics=False):
    import nar_backtest as nb
    totals = {"races": 0, "axis_wins": 0, "axis_podiums": 0,
              "hits": 0, "tickets": 0, "stake_jpy": 0, "return_jpy": 0}
    diagnostics = []
    for item in races:
        group = item["group"]
        axis_scores = item["winner_model_scores"]
        axis = min(axis_scores, key=lambda h: (-float(axis_scores[h]), int(h)))
        opponents = _subset(
            group, axis, item["model_rank_scores"], item["market_rank_scores"],
            item["winner_axis_trio_scores"], mode, combo_weight, market_weight,
        )
        tickets = _tickets(axis, opponents)
        ticket_returns = {
            tuple(ticket): (nb.payout_for("trio", ticket, item["payout"]) or 0)
            for ticket in tickets
        }
        payout = sum(ticket_returns.values())
        winners = {str(r["horse_no"]) for r in group if r.get("finish") == 1}
        podium = {str(r["horse_no"]) for r in group if r.get("finish") in (1, 2, 3)}
        if len(winners) != 1 or len(podium) != 3:
            continue
        if include_diagnostics:
            ability_rank = percentile({str(r["horse_no"]): float(item["model_rank_scores"][str(r["horse_no"])]) for r in group})
            market_rank = percentile({str(r["horse_no"]): float(item["market_rank_scores"][str(r["horse_no"])]) for r in group})
            diagnostics.append({
                "date": item.get("date"), "race_key": item.get("race_key"),
                "axis": axis, "axis_finish": next((r.get("finish") for r in group if str(r["horse_no"]) == axis), None),
                "axis_in_podium": axis in podium,
                "selected_opponents": list(opponents),
                "actual_podium": sorted(podium, key=int),
                "selected_tickets": [{"horses": list(t), "return_jpy": float(v)} for t, v in ticket_returns.items()],
                "hit": payout > 0, "total_return_jpy": float(payout),
                "horses": [{
                    "horse_no": str(r["horse_no"]), "finish": r.get("finish"),
                    "ability_rank_score": float(ability_rank[str(r["horse_no"])]),
                    "market_rank_score": float(market_rank[str(r["horse_no"])]),
                    "combo_pair_scores": {
                        "|".join(sorted((str(r["horse_no"]), other), key=int)):
                            float(item["winner_axis_trio_scores"].get(tuple(sorted((axis, str(r["horse_no"]), other), key=int)), 0.0))
                        for other in opponents if other != str(r["horse_no"])
                    }
                } for r in group]
            })
        totals["races"] += 1
        totals["axis_wins"] += int(axis in winners)
        totals["axis_podiums"] += int(axis in podium)
        totals["hits"] += int(payout > 0)
        totals["tickets"] += len(tickets)
        totals["stake_jpy"] += len(tickets) * 100
        totals["return_jpy"] += payout
    if include_diagnostics:
        totals["race_diagnostics"] = diagnostics
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
        "A_ability_only": evaluate(holdout_races, "ability", include_diagnostics=True),
        "B_combo_only": evaluate(holdout_races, "combo", include_diagnostics=True),
        "C_ability_plus_combo_selected_on_development": evaluate(
            holdout_races, "ability_combo", combo_weight=c_weight, include_diagnostics=True),
        "D_ability_market_combo_selected_on_development": evaluate(
            holdout_races, "ability_market_combo", combo_weight=d_combo,
            market_weight=d_market, include_diagnostics=True),
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


def evaluate_candidate_pool(races, pool_size, include_diagnostics=False):
    """Keep the winner-classifier axis fixed; select four opponents from the top-N
    LambdaRank ability pool by maximizing the sum of six axis-conditioned trio scores.
    Exactly six unordered trio tickets (600 JPY) are emitted per race.
    """
    import nar_backtest as nb
    import itertools

    if pool_size not in (4, 6, 8):
        raise ValueError("pool_size must be one of 4, 6, or 8")
    totals = {"races": 0, "axis_wins": 0, "axis_podiums": 0,
              "hits": 0, "tickets": 0, "stake_jpy": 0, "return_jpy": 0}
    diagnostics = []
    for item in races:
        group = item["group"]
        axis_scores = item["winner_model_scores"]
        axis = min(axis_scores, key=lambda h: (-float(axis_scores[h]), int(h)))
        ability = item["model_rank_scores"]
        candidates = sorted(
            (h for h in ability if h != axis),
            key=lambda h: (-float(ability[h]), int(h)),
        )
        if len(candidates) < pool_size:
            continue
        pool = candidates[:pool_size]
        trio_scores = item["winner_axis_trio_scores"]
        best_subset, best_score, best_ability = None, float("-inf"), float("-inf")
        for subset in itertools.combinations(pool, 4):
            score = sum(
                float(trio_scores[tuple(sorted((axis, a, b), key=int))])
                for a, b in itertools.combinations(subset, 2)
            )
            ability_tiebreak = sum(float(ability[h]) for h in subset)
            key = tuple(sorted(subset, key=int))
            if (score > best_score + 1e-12 or
                (abs(score - best_score) <= 1e-12 and ability_tiebreak > best_ability + 1e-12) or
                (abs(score - best_score) <= 1e-12 and
                 abs(ability_tiebreak - best_ability) <= 1e-12 and
                 (best_subset is None or tuple(map(int, key)) < tuple(map(int, best_subset))))):
                best_subset, best_score, best_ability = key, score, ability_tiebreak
        tickets = _tickets(axis, best_subset)
        returned = sum(nb.payout_for("trio", ticket, item["payout"]) or 0
                       for ticket in tickets)
        winners = {str(r["horse_no"]) for r in group if r.get("finish") == 1}
        podium = {str(r["horse_no"]) for r in group if r.get("finish") in (1, 2, 3)}
        if len(winners) != 1 or len(podium) != 3:
            continue
        totals["races"] += 1
        totals["axis_wins"] += int(axis in winners)
        totals["axis_podiums"] += int(axis in podium)
        totals["hits"] += int(returned > 0)
        totals["tickets"] += len(tickets)
        totals["stake_jpy"] += len(tickets) * 100
        totals["return_jpy"] += returned
        if include_diagnostics:
            diagnostics.append({
                "date": item.get("date"), "race_key": item.get("race_key"),
                "venue": str(group[0].get("venue") or "unknown"),
                "axis": axis, "axis_finish": next(
                    (r.get("finish") for r in group if str(r["horse_no"]) == axis), None),
                "axis_in_podium": axis in podium,
                "candidate_pool_size": pool_size, "candidate_pool": pool,
                "selected_opponents": list(best_subset),
                "actual_podium": sorted(podium, key=int),
                "selected_tickets": tickets, "hit": returned > 0,
                "return_jpy": int(returned),
            })
    n, stake = totals["races"], totals["stake_jpy"]
    totals.update({
        "candidate_pool_size": pool_size,
        "axis_win_rate_pct": round(100 * totals["axis_wins"] / n, 2) if n else None,
        "axis_podium_rate_pct": round(100 * totals["axis_podiums"] / n, 2) if n else None,
        "trio_hit_rate_pct": round(100 * totals["hits"] / n, 2) if n else None,
        "roi_pct": round(100 * totals["return_jpy"] / stake, 2) if stake else None,
        "profit_jpy": totals["return_jpy"] - stake,
        "tickets_per_race": round(totals["tickets"] / n, 2) if n else None,
        "stake_per_race_jpy": 600,
    })
    if include_diagnostics:
        totals["race_diagnostics"] = diagnostics
    return totals


def compare_candidate_pools(dev_races, holdout_races):
    """Select pool size on development dates only; report 4/6/8 on untouched holdout."""
    development = {
        str(size): evaluate_candidate_pool(dev_races, size)
        for size in (4, 6, 8)
    }
    def rank(result):
        return (
            result["trio_hit_rate_pct"] if result["trio_hit_rate_pct"] is not None else -1,
            result["roi_pct"] if result["roi_pct"] is not None else -1,
            result["profit_jpy"],
        )
    selected_size = max((6, 8), key=lambda size: rank(development[str(size)]))
    holdout = {
        str(size): evaluate_candidate_pool(holdout_races, size, include_diagnostics=True)
        for size in (4, 6, 8)
    }
    venue_results = {}
    venues = sorted({
        str(item["group"][0].get("venue") or "unknown")
        for item in holdout_races if item.get("group")
    })
    for venue in venues:
        segment = [
            item for item in holdout_races
            if str(item["group"][0].get("venue") or "unknown") == venue
        ]
        venue_results[venue] = {
            str(size): evaluate_candidate_pool(segment, size)
            for size in (4, 6, 8)
        }
    return {
        "policy": "Winner-classifier axis is fixed. Opponent pool is top 4/6/8 by the same LambdaRank ability scores; select exactly four opponents by maximizing the sum of six axis-conditioned supervised trio scores; exactly six unordered trio tickets at 100 JPY each. Pool size is selected on chronological development data only.",
        "development": development,
        "selected_pool_size_on_development": selected_size,
        "untouched_holdout": holdout,
        "holdout_by_venue": venue_results,
        "banei_holdout": venue_results.get("帯広") or venue_results.get("ばんえい") or None,
        "research_only": True,
    }
