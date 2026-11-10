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


# Override the initial comparison helper with a strict common-cohort version:
# all three pool sizes are evaluated on exactly the same races (>= 9 runners).
def compare_candidate_pools(dev_races, holdout_races):
    """Select pool size on development dates only, with identical 4/6/8 race cohorts."""
    dev_eligible = [item for item in dev_races if len(item["group"]) >= 9]
    holdout_eligible = [item for item in holdout_races if len(item["group"]) >= 9]
    if len(dev_eligible) < 10 or len(holdout_eligible) < 10:
        raise ValueError(
            "Candidate-pool comparison requires at least 10 common-cohort races "
            f"with >=9 runners; development={len(dev_eligible)}, holdout={len(holdout_eligible)}"
        )
    development = {
        str(size): evaluate_candidate_pool(dev_eligible, size)
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
        str(size): evaluate_candidate_pool(holdout_eligible, size, include_diagnostics=True)
        for size in (4, 6, 8)
    }
    venue_results = {}
    venues = sorted({
        str(item["group"][0].get("venue") or "unknown")
        for item in holdout_eligible
    })
    for venue in venues:
        segment = [
            item for item in holdout_eligible
            if str(item["group"][0].get("venue") or "unknown") == venue
        ]
        venue_results[venue] = {
            str(size): evaluate_candidate_pool(segment, size)
            for size in (4, 6, 8)
        }
    banei_names = ("帯広", "ばんえい", "ばんえい帯広")
    banei = {
        name: venue_results[name] for name in banei_names if name in venue_results
    }
    return {
        "policy": "Winner-classifier axis is fixed. Opponent pool is top 4/6/8 by the same LambdaRank ability scores; select exactly four opponents by maximizing the sum of six axis-conditioned supervised trio scores; exactly six unordered trio tickets at 100 JPY each. All three pool sizes use the same races with at least nine runners. Pool size is selected on chronological development data only.",
        "common_cohort": {
            "development_races": len(dev_eligible),
            "holdout_races": len(holdout_eligible),
            "development_excluded_for_field_size": len(dev_races) - len(dev_eligible),
            "holdout_excluded_for_field_size": len(holdout_races) - len(holdout_eligible),
            "minimum_field_size": 9,
        },
        "development": development,
        "selected_pool_size_on_development": selected_size,
        "untouched_holdout": holdout,
        "holdout_by_venue": venue_results,
        "banei_holdout": banei or None,
        "research_only": True,
    }


def _pool_selector_subset(item, axis, pool, mode, trio_weight=0.5):
    """Select four opponents from one fixed ability-ranked candidate pool.

    Trio and pair inputs are converted to within-race percentile ranks before
    blending, so the selector does not assume their raw score scales match.
    """
    import itertools

    ability = item["model_rank_scores"]
    if mode == "ability":
        return tuple(sorted(sorted(pool, key=lambda h: (-float(ability[h]), int(h)))[:4], key=int))
    if mode not in ("trio", "pair", "blend"):
        raise ValueError(f"Unknown pool selector mode: {mode}")
    if not 0.0 <= float(trio_weight) <= 1.0:
        raise ValueError("trio_weight must be between 0 and 1")

    trio_rank, pair_rank = {}, {}
    if mode in ("trio", "blend"):
        trio_raw = {}
        for a, b in itertools.combinations(pool, 2):
            key = "|".join(sorted((a, b), key=int))
            trio_key = tuple(sorted((axis, a, b), key=int))
            trio_raw[key] = float(item["winner_axis_trio_scores"][trio_key])
        trio_rank = percentile(trio_raw)
    if mode in ("pair", "blend"):
        pair_raw = {}
        for a, b in itertools.combinations(pool, 2):
            key = "|".join(sorted((a, b), key=int))
            pair_key = tuple(sorted((a, b), key=int))
            pair_raw[key] = float(item["learned_pair_scores"][pair_key])
        pair_rank = percentile(pair_raw)
    best_subset, best_score, best_ability = None, float("-inf"), float("-inf")
    for subset in itertools.combinations(pool, 4):
        score = 0.0
        for a, b in itertools.combinations(subset, 2):
            key = "|".join(sorted((a, b), key=int))
            if mode == "trio":
                pair_score = trio_rank[key]
            elif mode == "pair":
                pair_score = pair_rank[key]
            else:
                pair_score = trio_weight * trio_rank[key] + (1.0 - trio_weight) * pair_rank[key]
            score += pair_score
        ability_tiebreak = sum(float(ability[h]) for h in subset)
        key = tuple(sorted(subset, key=int))
        if (score > best_score + 1e-12 or
            (abs(score - best_score) <= 1e-12 and ability_tiebreak > best_ability + 1e-12) or
            (abs(score - best_score) <= 1e-12 and
             abs(ability_tiebreak - best_ability) <= 1e-12 and
             (best_subset is None or tuple(map(int, key)) < tuple(map(int, best_subset))))):
            best_subset, best_score, best_ability = key, score, ability_tiebreak
    return best_subset


def evaluate_pool_selector(races, mode, trio_weight=0.5, include_diagnostics=False):
    """Evaluate selectors with a fixed winner axis, top-six pool and six trio tickets."""
    import itertools
    import nar_backtest as nb

    totals = {
        "races": 0, "axis_wins": 0, "axis_podiums": 0,
        "both_partners_in_pool": 0, "both_partners_selected": 0,
        "hits": 0, "tickets": 0, "stake_jpy": 0, "return_jpy": 0,
    }
    diagnostics = []
    for item in races:
        group = item["group"]
        axis_scores = item["winner_model_scores"]
        axis = min(axis_scores, key=lambda h: (-float(axis_scores[h]), int(h)))
        ability = item["model_rank_scores"]
        pool = sorted(
            (h for h in ability if h != axis),
            key=lambda h: (-float(ability[h]), int(h)),
        )[:6]
        if len(pool) < 6:
            continue
        selected = _pool_selector_subset(item, axis, pool, mode, trio_weight)
        tickets = _tickets(axis, selected)
        returned = sum(nb.payout_for("trio", ticket, item["payout"]) or 0 for ticket in tickets)
        winners = {str(r["horse_no"]) for r in group if r.get("finish") == 1}
        podium = {str(r["horse_no"]) for r in group if r.get("finish") in (1, 2, 3)}
        if len(winners) != 1 or len(podium) != 3:
            continue
        axis_in_podium = axis in podium
        actual_partners = podium - {axis} if axis_in_podium else set()
        both_in_pool = axis_in_podium and actual_partners.issubset(set(pool))
        both_selected = axis_in_podium and actual_partners.issubset(set(selected))
        totals["races"] += 1
        totals["axis_wins"] += int(axis in winners)
        totals["axis_podiums"] += int(axis_in_podium)
        totals["both_partners_in_pool"] += int(both_in_pool)
        totals["both_partners_selected"] += int(both_selected)
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
                "axis_in_podium": axis_in_podium,
                "candidate_pool": list(pool), "selected_opponents": list(selected),
                "actual_podium": sorted(podium, key=int),
                "both_partners_in_pool": bool(both_in_pool),
                "both_partners_selected": bool(both_selected),
                "selected_tickets": tickets, "hit": returned > 0,
                "return_jpy": int(returned),
            })
    n, stake = totals["races"], totals["stake_jpy"]
    totals.update({
        "selector": mode,
        "trio_weight": round(float(trio_weight), 2) if mode == "blend" else None,
        "candidate_pool_size": 6,
        "axis_win_rate_pct": round(100 * totals["axis_wins"] / n, 2) if n else None,
        "axis_podium_rate_pct": round(100 * totals["axis_podiums"] / n, 2) if n else None,
        "both_partners_in_pool_pct_of_axis_podiums": round(
            100 * totals["both_partners_in_pool"] / totals["axis_podiums"], 2
        ) if totals["axis_podiums"] else None,
        "both_partners_selected_pct_of_axis_podiums": round(
            100 * totals["both_partners_selected"] / totals["axis_podiums"], 2
        ) if totals["axis_podiums"] else None,
        "trio_hit_rate_pct": round(100 * totals["hits"] / n, 2) if n else None,
        "roi_pct": round(100 * totals["return_jpy"] / stake, 2) if stake else None,
        "profit_jpy": totals["return_jpy"] - stake,
        "tickets_per_race": round(totals["tickets"] / n, 2) if n else None,
        "stake_per_race_jpy": 600,
    })
    if include_diagnostics:
        totals["race_diagnostics"] = diagnostics
    return totals


def compare_candidate_pool_selectors(dev_races, holdout_races):
    """Compare ability, axis-conditioned trio, pair compatibility and rank blends.

    Candidate pool size is fixed at six; all selectors use the same >=9-runner
    races, fixed winner-classifier axis and six 100-JPY trio tickets.
    Blend weight is tuned only on development dates.
    """
    dev = [item for item in dev_races if len(item["group"]) >= 9]
    holdout = [item for item in holdout_races if len(item["group"]) >= 9]
    if len(dev) < 10 or len(holdout) < 10:
        raise ValueError(
            "Pool-selector comparison requires at least 10 common-cohort races "
            f"with >=9 runners; development={len(dev)}, holdout={len(holdout)}"
        )
    development = {
        "ability_top4": evaluate_pool_selector(dev, "ability"),
        "axis_trio_score": evaluate_pool_selector(dev, "trio"),
        "pair_compatibility": evaluate_pool_selector(dev, "pair"),
        "blends": {
            f"{weight:.2f}": evaluate_pool_selector(dev, "blend", trio_weight=weight)
            for weight in (0.0, 0.25, 0.50, 0.75, 1.0)
        },
    }
    def rank(result):
        return (
            result["trio_hit_rate_pct"] if result["trio_hit_rate_pct"] is not None else -1,
            result["roi_pct"] if result["roi_pct"] is not None else -1,
            result["profit_jpy"],
        )
    selected_weight = max(
        (0.0, 0.25, 0.50, 0.75, 1.0),
        key=lambda weight: rank(development["blends"][f"{weight:.2f}"]),
    )
    holdout_results = {
        "ability_top4": evaluate_pool_selector(holdout, "ability", include_diagnostics=True),
        "axis_trio_score": evaluate_pool_selector(holdout, "trio", include_diagnostics=True),
        "pair_compatibility": evaluate_pool_selector(holdout, "pair", include_diagnostics=True),
        "blends": {
            f"{weight:.2f}": evaluate_pool_selector(
                holdout, "blend", trio_weight=weight, include_diagnostics=True
            )
            for weight in (0.0, 0.25, 0.50, 0.75, 1.0)
        },
        "blend_selected_on_development": evaluate_pool_selector(
            holdout, "blend", trio_weight=selected_weight, include_diagnostics=True
        ),
    }
    venues = sorted({str(item["group"][0].get("venue") or "unknown") for item in holdout})
    venue_results = {}
    for venue in venues:
        segment = [item for item in holdout
                   if str(item["group"][0].get("venue") or "unknown") == venue]
        venue_results[venue] = {
            "ability_top4": evaluate_pool_selector(segment, "ability"),
            "axis_trio_score": evaluate_pool_selector(segment, "trio"),
            "pair_compatibility": evaluate_pool_selector(segment, "pair"),
            "blend_selected_on_development": evaluate_pool_selector(
                segment, "blend", trio_weight=selected_weight
            ),
        }
    return {
        "policy": (
            "Fixed winner-classifier axis; top-six LambdaRank ability candidate pool; "
            "four opponents selected by ability top4, axis-conditioned trio-score sum, "
            "pair-compatibility sum, or a blend of within-race percentile ranks. "
            "Exactly six unordered trio tickets at 100 JPY each. All selectors share "
            "the same >=9-runner cohort. Blend weight selected on development only."
        ),
        "common_cohort": {"development_races": len(dev), "holdout_races": len(holdout),
                          "minimum_field_size": 9},
        "development": development,
        "selected_blend_trio_weight_on_development": selected_weight,
        "untouched_holdout": holdout_results,
        "holdout_by_venue": venue_results,
        "direct_pair_ticket_comparison": compare_direct_pair_tickets(dev_races, holdout_races),
        "research_only": True,
    }



def _direct_pair_tickets(item, axis, pool, mode, trio_weight=0.5):
    """Select six opponent pairs independently; unlike a four-horse box, tickets may
    cover more than four distinct opponents. All ranks are computed within the race.
    """
    import itertools

    ability = percentile({h: float(item["model_rank_scores"][h]) for h in pool})
    ability_pair = {
        "|".join((a, b)): (ability[a] + ability[b]) / 2.0
        for a, b in itertools.combinations(sorted(pool, key=int), 2)
    }
    trio_raw = {
        "|".join((a, b)): float(item["winner_axis_trio_scores"][
            tuple(sorted((axis, a, b), key=int))
        ])
        for a, b in itertools.combinations(sorted(pool, key=int), 2)
    }
    pair_raw = {
        "|".join((a, b)): float(item["learned_pair_scores"][
            tuple(sorted((a, b), key=int))
        ])
        for a, b in itertools.combinations(sorted(pool, key=int), 2)
    }
    ability_rank = percentile(ability_pair)
    trio_rank = percentile(trio_raw)
    pair_rank = percentile(pair_raw)
    if mode == "ability_pair":
        scores = ability_rank
    elif mode == "trio":
        scores = trio_rank
    elif mode == "pair":
        scores = pair_rank
    elif mode == "blend":
        weight = float(trio_weight)
        if not 0.0 <= weight <= 1.0:
            raise ValueError("trio_weight must be between 0 and 1")
        scores = {
            key: weight * trio_rank[key] + (1.0 - weight) * ability_rank[key]
            for key in ability_rank
        }
    else:
        raise ValueError(f"Unknown direct-pair mode: {mode}")
    ranked = sorted(
        scores,
        key=lambda key: (-float(scores[key]), tuple(map(int, key.split("|")))),
    )
    return [tuple(key.split("|")) for key in ranked[:6]]


def evaluate_direct_pair_tickets(races, mode, trio_weight=0.5,
                                 include_diagnostics=False):
    """Evaluate six independently selected axis+opponent-pair tickets on a fixed top-six pool."""
    import nar_backtest as nb

    totals = {
        "races": 0, "axis_wins": 0, "axis_podiums": 0,
        "both_partners_in_pool": 0, "both_partners_selected": 0,
        "hits": 0, "tickets": 0, "stake_jpy": 0, "return_jpy": 0,
        "distinct_opponents_selected": 0,
    }
    diagnostics = []
    for item in races:
        group = item["group"]
        axis_scores = item["winner_model_scores"]
        axis = min(axis_scores, key=lambda h: (-float(axis_scores[h]), int(h)))
        ability = item["model_rank_scores"]
        pool = sorted(
            (h for h in ability if h != axis),
            key=lambda h: (-float(ability[h]), int(h)),
        )[:6]
        if len(pool) != 6:
            continue
        pairs = _direct_pair_tickets(item, axis, pool, mode, trio_weight)
        tickets = [tuple(sorted((axis, a, b), key=int)) for a, b in pairs]
        returned = sum(
            nb.payout_for("trio", ticket, item["payout"]) or 0
            for ticket in tickets
        )
        winners = {str(r["horse_no"]) for r in group if r.get("finish") == 1}
        podium = {str(r["horse_no"]) for r in group if r.get("finish") in (1, 2, 3)}
        if len(winners) != 1 or len(podium) != 3:
            continue
        axis_in_podium = axis in podium
        actual_partners = podium - {axis} if axis_in_podium else set()
        both_in_pool = axis_in_podium and actual_partners.issubset(set(pool))
        selected_pairs = {frozenset(pair) for pair in pairs}
        both_selected = axis_in_podium and frozenset(actual_partners) in selected_pairs
        selected_opponents = sorted({h for pair in pairs for h in pair}, key=int)
        totals["races"] += 1
        totals["axis_wins"] += int(axis in winners)
        totals["axis_podiums"] += int(axis_in_podium)
        totals["both_partners_in_pool"] += int(both_in_pool)
        totals["both_partners_selected"] += int(both_selected)
        totals["hits"] += int(returned > 0)
        totals["tickets"] += len(tickets)
        totals["stake_jpy"] += len(tickets) * 100
        totals["return_jpy"] += returned
        totals["distinct_opponents_selected"] += len(selected_opponents)
        if include_diagnostics:
            diagnostics.append({
                "date": item.get("date"), "race_key": item.get("race_key"),
                "venue": str(group[0].get("venue") or "unknown"),
                "axis": axis, "axis_finish": next(
                    (r.get("finish") for r in group if str(r["horse_no"]) == axis), None),
                "axis_in_podium": axis_in_podium,
                "candidate_pool": list(pool),
                "selected_opponents": selected_opponents,
                "selected_pairs": [list(pair) for pair in pairs],
                "actual_podium": sorted(podium, key=int),
                "both_partners_in_pool": bool(both_in_pool),
                "both_partners_selected": bool(both_selected),
                "tickets": [list(ticket) for ticket in tickets],
                "hit": returned > 0, "return_jpy": int(returned),
            })
    n, stake = totals["races"], totals["stake_jpy"]
    totals.update({
        "selector": mode,
        "trio_weight": round(float(trio_weight), 2) if mode == "blend" else None,
        "candidate_pool_size": 6,
        "axis_win_rate_pct": round(100 * totals["axis_wins"] / n, 2) if n else None,
        "axis_podium_rate_pct": round(100 * totals["axis_podiums"] / n, 2) if n else None,
        "both_partners_in_pool_pct_of_axis_podiums": round(
            100 * totals["both_partners_in_pool"] / totals["axis_podiums"], 2
        ) if totals["axis_podiums"] else None,
        "both_partners_selected_pct_of_axis_podiums": round(
            100 * totals["both_partners_selected"] / totals["axis_podiums"], 2
        ) if totals["axis_podiums"] else None,
        "trio_hit_rate_pct": round(100 * totals["hits"] / n, 2) if n else None,
        "roi_pct": round(100 * totals["return_jpy"] / stake, 2) if stake else None,
        "profit_jpy": totals["return_jpy"] - stake,
        "tickets_per_race": round(totals["tickets"] / n, 2) if n else None,
        "stake_per_race_jpy": 600,
        "average_distinct_opponents_per_race": round(
            totals["distinct_opponents_selected"] / n, 2
        ) if n else None,
    })
    if include_diagnostics:
        totals["race_diagnostics"] = diagnostics
    return totals


def compare_direct_pair_tickets(dev_races, holdout_races):
    """Compare direct six-pair ticket selection against the four-opponent box design.
    Tune the trio/ability blend only on chronological development races.
    """
    dev = [item for item in dev_races if len(item["group"]) >= 9]
    holdout = [item for item in holdout_races if len(item["group"]) >= 9]
    if len(dev) < 10 or len(holdout) < 10:
        raise ValueError(
            "Direct-pair comparison requires at least 10 common-cohort races "
            f"with >=9 runners; development={len(dev)}, holdout={len(holdout)}"
        )
    weights = (0.0, 0.25, 0.50, 0.75, 1.0)
    development = {
        "ability_pair": evaluate_direct_pair_tickets(dev, "ability_pair"),
        "axis_trio_score": evaluate_direct_pair_tickets(dev, "trio"),
        "pair_compatibility": evaluate_direct_pair_tickets(dev, "pair"),
        "trio_ability_blends": {
            f"{weight:.2f}": evaluate_direct_pair_tickets(
                dev, "blend", trio_weight=weight
            ) for weight in weights
        },
    }
    def rank(result):
        return (
            result["trio_hit_rate_pct"] if result["trio_hit_rate_pct"] is not None else -1,
            result["roi_pct"] if result["roi_pct"] is not None else -1,
            result["profit_jpy"],
        )
    selected_weight = max(
        weights,
        key=lambda weight: rank(development["trio_ability_blends"][f"{weight:.2f}"]),
    )
    holdout_results = {
        "ability_pair": evaluate_direct_pair_tickets(
            holdout, "ability_pair", include_diagnostics=True
        ),
        "axis_trio_score": evaluate_direct_pair_tickets(
            holdout, "trio", include_diagnostics=True
        ),
        "pair_compatibility": evaluate_direct_pair_tickets(
            holdout, "pair", include_diagnostics=True
        ),
        "trio_ability_blends": {
            f"{weight:.2f}": evaluate_direct_pair_tickets(
                holdout, "blend", trio_weight=weight, include_diagnostics=True
            ) for weight in weights
        },
        "blend_selected_on_development": evaluate_direct_pair_tickets(
            holdout, "blend", trio_weight=selected_weight,
            include_diagnostics=True
        ),
    }
    return {
        "policy": (
            "Fixed winner-classifier axis and top-six LambdaRank ability candidate pool. "
            "Select six opponent pairs independently from all 15 possible pairs; unlike "
            "the four-opponent box, selected tickets may cover more than four opponents. "
            "Each ticket is one axis plus one selected pair; six tickets at 100 JPY each. "
            "Blend weight selected on chronological development races only."
        ),
        "common_cohort": {
            "development_races": len(dev), "holdout_races": len(holdout),
            "minimum_field_size": 9,
        },
        "development": development,
        "selected_trio_weight_on_development": selected_weight,
        "untouched_holdout": holdout_results,
        "research_only": True,
    }
