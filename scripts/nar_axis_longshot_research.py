"""Research-only axis reliability and odds-informed longshot trio strategy.

The six-ticket baseline uses the top-ranked axis plus individual ranks 2-5.
The ten-ticket variant adds exactly one runner ranked 6th or lower whose
model-vs-market percentile gap is positive and whose pre-off win probability
is lower than every ordinary opponent's. Selection uses pre-off inputs only.
"""
from __future__ import annotations

import itertools

import nar_backtest as nb


def _percentile(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values, key=lambda horse: (-float(values[horse]), int(horse)))
    if len(ordered) <= 1:
        return {horse: 0.5 for horse in ordered}
    return {
        horse: 1.0 - index / (len(ordered) - 1)
        for index, horse in enumerate(ordered)
    }


def build_axis_longshot_tickets(group, model_scores, market_win_probabilities):
    """Return axis, 6-ticket baseline, optional 10-ticket set, and audit fields."""
    ranked = sorted(
        group,
        key=lambda row: (-float(model_scores[str(row["horse_no"])]), int(row["horse_no"])),
    )
    numbers = [str(row["horse_no"]) for row in ranked]
    if len(numbers) < 5:
        raise ValueError("At least five runners are required for the six-ticket baseline")

    axis = numbers[0]
    ordinary = numbers[1:5]
    baseline = [
        sorted((axis, a, b), key=int)
        for a, b in itertools.combinations(ordinary, 2)
    ]

    # Five-runner races still support the six-ticket baseline, but cannot
    # supply a rank-6+ longshot. Keep them in axis/baseline evaluation.
    if len(numbers) < 6:
        return {
            "axis": axis, "ordinary_opponents": ordinary,
            "baseline_tickets": baseline, "longshot": None,
            "ten_ticket_tickets": None, "longshot_gap": None,
            "longshot_market_probability": None,
            "selection_status": "insufficient_runners_for_longshot",
        }

    common = set(numbers)
    if not common.issubset(market_win_probabilities):
        return {
            "axis": axis, "ordinary_opponents": ordinary,
            "baseline_tickets": baseline, "longshot": None,
            "ten_ticket_tickets": None, "longshot_gap": None,
            "longshot_market_probability": None,
            "selection_status": "missing_preoff_market_probability",
        }

    model_pct = _percentile({h: float(model_scores[h]) for h in numbers})
    market_pct = _percentile({h: float(market_win_probabilities[h]) for h in numbers})
    ordinary_min_probability = min(float(market_win_probabilities[h]) for h in ordinary)

    candidates = []
    for horse in numbers[5:]:
        probability = float(market_win_probabilities[horse])
        gap = model_pct[horse] - market_pct[horse]
        if probability > 0 and probability < ordinary_min_probability and gap > 0:
            candidates.append((gap, probability, int(horse), horse))

    if not candidates:
        return {
            "axis": axis, "ordinary_opponents": ordinary,
            "baseline_tickets": baseline, "longshot": None,
            "ten_ticket_tickets": None, "longshot_gap": None,
            "longshot_market_probability": None,
            "selection_status": "no_qualified_longshot",
        }

    # Largest model-over-market percentile gap; deterministic tie-breaks.
    gap, probability, _, longshot = sorted(
        candidates, key=lambda item: (-item[0], item[1], item[2])
    )[0]
    expanded = ordinary + [longshot]
    ten_tickets = [
        sorted((axis, a, b), key=int)
        for a, b in itertools.combinations(expanded, 2)
    ]
    return {
        "axis": axis, "ordinary_opponents": ordinary,
        "baseline_tickets": baseline, "longshot": longshot,
        "ten_ticket_tickets": ten_tickets,
        "longshot_gap": round(gap, 8),
        "longshot_market_probability": round(probability, 8),
        "selection_status": "qualified",
    }


def _ticket_return(tickets, payout):
    return sum(nb.payout_for("trio", ticket, payout) or 0 for ticket in tickets)


def _summary(rows, ticket_key):
    races = len(rows)
    stake = sum(len(row[ticket_key]) * 100 for row in rows)
    returned = sum(row[f"{ticket_key}_return"] for row in rows)
    hits = sum(row[f"{ticket_key}_return"] > 0 for row in rows)
    return {
        "races": races,
        "hits": hits,
        "hit_rate_pct": round(100 * hits / races, 2) if races else None,
        "tickets": sum(len(row[ticket_key]) for row in rows),
        "stake_jpy": stake,
        "return_jpy": returned,
        "roi_pct": round(100 * returned / stake, 2) if stake else None,
        "profit_jpy": returned - stake,
    }


def evaluate_axis_longshot(races):
    """Evaluate baseline on all races and 10-ticket increment on eligible races."""
    evaluated = []
    axis_wins = axis_podiums = 0
    eligible = 0
    longshot_top3 = 0
    for item in races:
        if len(item["group"]) < 5:
            continue
        selection = build_axis_longshot_tickets(
            item["group"], item["model_scores"], item["market_win_probabilities"]
        )
        winners = {str(r["horse_no"]) for r in item["group"] if r.get("finish") == 1}
        podium = {str(r["horse_no"]) for r in item["group"] if r.get("finish") in (1, 2, 3)}
        if len(podium) != 3 or not winners:
            continue
        axis_wins += int(selection["axis"] in winners)
        axis_podiums += int(selection["axis"] in podium)
        baseline_return = _ticket_return(selection["baseline_tickets"], item["payout"])
        row = {
            "baseline_tickets": selection["baseline_tickets"],
            "baseline_tickets_return": baseline_return,
        }
        if selection["selection_status"] == "qualified":
            eligible += 1
            is_top3 = selection["longshot"] in podium
            longshot_top3 += int(is_top3)
            ten_return = _ticket_return(selection["ten_ticket_tickets"], item["payout"])
            row.update({
                "ten_ticket_tickets": selection["ten_ticket_tickets"],
                "ten_ticket_tickets_return": ten_return,
                "longshot_top3": is_top3,
                "incremental_return_jpy": ten_return - baseline_return,
            })
            evaluated.append(row)
        else:
            evaluated.append(row)

    eligible_rows = [row for row in evaluated if "ten_ticket_tickets" in row]
    baseline_all = _summary(evaluated, "baseline_tickets")
    baseline_eligible = _summary(eligible_rows, "baseline_tickets")
    ten_eligible = _summary(eligible_rows, "ten_ticket_tickets")
    n = len(evaluated)
    return {
        "research_only": True,
        "races": n,
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
            "Axis and ordinary opponents are individual model ranks 1 and 2-5. "
            "One longshot is selected only among ranks 6+, only if its pre-off "
            "market win probability is lower than all four ordinary opponents "
            "and its model percentile exceeds its market percentile. Choose the "
            "largest positive percentile gap. Otherwise do not force a 10-ticket bet."
        ),
        "limitations": [
            "Market probabilities must come from a valid timestamped pre-off win-odds snapshot.",
            "This is a ranking-gap proxy, not calibrated trio probability or yen expected value.",
            "The 10-ticket comparison is reported on the same eligible races as its 6-ticket control.",
            "Official payouts are used only for retrospective scoring, never for horse selection.",
        ],
    }
