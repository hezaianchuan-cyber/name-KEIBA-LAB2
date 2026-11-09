"""Research-only separated axis and opponent scoring for NAR trio tickets.

The axis score is never modified by pair compatibility; pair compatibility only
re-ranks the opponents conditional on the selected axis.
"""
from __future__ import annotations

import itertools

import nar_backtest as nb


def _percentile(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values, key=lambda horse: (-float(values[horse]), int(horse)))
    if len(ordered) <= 1:
        return {horse: 0.5 for horse in ordered}
    result = {}
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and float(values[ordered[end]]) == float(values[ordered[index]]):
            end += 1
        average_rank = (index + end - 1) / 2
        score = 1.0 - average_rank / (len(ordered) - 1)
        for pos in range(index, end):
            result[ordered[pos]] = score
        index = end
    return result


def build_axis_conditioned_opponent_scores(group, axis, individual_scores, pair_scores, compatibility_weight):
    """Score opponents conditionally on the independently selected axis."""
    if not 0.0 <= float(compatibility_weight) <= 1.0:
        raise ValueError("compatibility_weight must be between 0 and 1")
    numbers = [str(row["horse_no"]) for row in group]
    if axis not in numbers:
        raise ValueError("axis must be a runner in group")
    individual = _percentile({h: float(individual_scores[h]) for h in numbers})
    result = {}
    for horse in numbers:
        if horse == axis:
            continue
        pair = tuple(sorted((axis, horse), key=int))
        if pair not in pair_scores:
            raise ValueError(f"Missing learned pair score for {pair}")
        compatibility = float(pair_scores[pair])
        result[horse] = (
            (1.0 - compatibility_weight) * individual[horse]
            + compatibility_weight * compatibility
        )
    return result


def build_axis_longshot_tickets(group, axis_scores, opponent_scores, market_win_probabilities):
    """Choose axis from axis_scores; choose all opponents conditionally from opponent_scores."""
    numbers = [str(row["horse_no"]) for row in group]
    if len(numbers) < 5:
        raise ValueError("At least five runners are required for the six-ticket baseline")
    if set(numbers) - set(axis_scores) or (set(numbers) - {str(axis)}) - set(opponent_scores):
        raise ValueError("Axis scores must cover every runner and opponent scores every non-axis runner")
    axis = sorted(numbers, key=lambda h: (-float(axis_scores[h]), int(h)))[0]
    opponents = sorted((h for h in numbers if h != axis),
                       key=lambda h: (-float(opponent_scores[h]), int(h)))
    ordinary = opponents[:4]
    baseline = [sorted((axis, a, b), key=int)
                for a, b in itertools.combinations(ordinary, 2)]
    base = {
        "axis": axis, "ordinary_opponents": ordinary,
        "baseline_tickets": baseline, "longshot": None,
        "ten_ticket_tickets": None, "longshot_gap": None,
        "longshot_market_probability": None,
    }
    if len(numbers) < 6:
        return {**base, "selection_status": "insufficient_runners_for_longshot"}
    if not set(numbers).issubset(market_win_probabilities):
        return {**base, "selection_status": "missing_preoff_market_probability"}

    model_pct = _percentile({h: float(opponent_scores[h]) for h in numbers if h != axis})
    market_pct = _percentile({h: float(market_win_probabilities[h]) for h in numbers if h != axis})
    ordinary_min_probability = min(float(market_win_probabilities[h]) for h in ordinary)
    candidates = []
    for horse in opponents[4:]:
        probability = float(market_win_probabilities[horse])
        gap = model_pct[horse] - market_pct[horse]
        if probability > 0 and probability < ordinary_min_probability and gap > 0:
            candidates.append((gap, probability, int(horse), horse))
    if not candidates:
        return {**base, "selection_status": "no_qualified_longshot"}
    gap, probability, _, longshot = sorted(
        candidates, key=lambda item: (-item[0], item[1], item[2])
    )[0]
    expanded = ordinary + [longshot]
    ten = [sorted((axis, a, b), key=int)
           for a, b in itertools.combinations(expanded, 2)]
    return {**base, "longshot": longshot, "ten_ticket_tickets": ten,
            "longshot_gap": round(gap, 8),
            "longshot_market_probability": round(probability, 8),
            "selection_status": "qualified"}


def _ticket_return(tickets, payout):
    return sum(nb.payout_for("trio", ticket, payout) or 0 for ticket in tickets)


def _summary(rows, ticket_key):
    races = len(rows)
    stake = sum(len(row[ticket_key]) * 100 for row in rows)
    returned = sum(row[f"{ticket_key}_return"] for row in rows)
    hits = sum(row[f"{ticket_key}_return"] > 0 for row in rows)
    return {
        "races": races, "hits": hits,
        "hit_rate_pct": round(100 * hits / races, 2) if races else None,
        "tickets": sum(len(row[ticket_key]) for row in rows),
        "stake_jpy": stake, "return_jpy": returned,
        "roi_pct": round(100 * returned / stake, 2) if stake else None,
        "profit_jpy": returned - stake,
    }


def evaluate_axis_longshot(races):
    evaluated = []
    axis_wins = axis_podiums = eligible = longshot_top3 = 0
    for item in races:
        if len(item["group"]) < 5:
            continue
        selection = build_axis_longshot_tickets(
            item["group"], item["axis_scores"], item["opponent_scores"],
            item["market_win_probabilities"])
        winners = {str(r["horse_no"]) for r in item["group"] if r.get("finish") == 1}
        podium = {str(r["horse_no"]) for r in item["group"] if r.get("finish") in (1, 2, 3)}
        if len(podium) != 3 or not winners:
            continue
        axis_wins += int(selection["axis"] in winners)
        axis_podiums += int(selection["axis"] in podium)
        baseline_return = _ticket_return(selection["baseline_tickets"], item["payout"])
        row = {"baseline_tickets": selection["baseline_tickets"],
               "baseline_tickets_return": baseline_return}
        if selection["selection_status"] == "qualified":
            eligible += 1
            is_top3 = selection["longshot"] in podium
            longshot_top3 += int(is_top3)
            ten_return = _ticket_return(selection["ten_ticket_tickets"], item["payout"])
            row.update({"ten_ticket_tickets": selection["ten_ticket_tickets"],
                        "ten_ticket_tickets_return": ten_return,
                        "longshot_top3": is_top3,
                        "incremental_return_jpy": ten_return - baseline_return})
        evaluated.append(row)
    eligible_rows = [row for row in evaluated if "ten_ticket_tickets" in row]
    baseline_all = _summary(evaluated, "baseline_tickets")
    baseline_eligible = _summary(eligible_rows, "baseline_tickets")
    ten_eligible = _summary(eligible_rows, "ten_ticket_tickets")
    n = len(evaluated)
    return {
        "research_only": True, "races": n,
        "axis_win_rate_pct": round(100 * axis_wins / n, 2) if n else None,
        "axis_podium_rate_pct": round(100 * axis_podiums / n, 2) if n else None,
        "longshot_eligible_races": eligible,
        "longshot_eligibility_rate_pct": round(100 * eligible / n, 2) if n else None,
        "longshot_top3_rate_pct": round(100 * longshot_top3 / eligible, 2) if eligible else None,
        "six_ticket_baseline_all_races": baseline_all,
        "six_ticket_baseline_eligible_races": baseline_eligible,
        "ten_ticket_longshot_eligible_races": ten_eligible,
        "incremental_vs_six_on_same_eligible_races": {
            "extra_stake_jpy": ten_eligible["stake_jpy"] - baseline_eligible["stake_jpy"],
            "extra_return_jpy": ten_eligible["return_jpy"] - baseline_eligible["return_jpy"],
            "extra_profit_jpy": ten_eligible["profit_jpy"] - baseline_eligible["profit_jpy"],
            "extra_hit_races": ten_eligible["hits"] - baseline_eligible["hits"],
        },
        "selection_policy": (
            "Axis is selected solely by the individual axis score. Opponents are "
            "ranked separately by a blend of individual merit percentile and the "
            "learned pair-compatibility score conditional on that axis. A longshot "
            "is considered only among remaining opponents with positive model-over-market "
            "percentile gap and lower pre-off win probability than all four ordinary opponents."
        ),
        "limitations": [
            "All selection inputs must be derived from valid pre-off data and historical training only.",
            "Pair compatibility is a relative score, not a calibrated trio probability or yen expected value.",
            "The 10-ticket comparison is reported on the same eligible races as its 6-ticket control.",
            "Official payouts are used only for retrospective scoring, never for horse selection.",
        ],
    }
