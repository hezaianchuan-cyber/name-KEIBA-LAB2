"""Research-only exacta (馬単) evaluation with a fixed six-ticket, 600-JPY budget."""
from __future__ import annotations

import itertools


def build_exacta_tickets(group, first_scores, second_scores, limit=6):
    """Top-2 first-place pool x top-4 second-place pool; rank ordered pairs by role score product."""
    horses = [str(row["horse_no"]) for row in group]
    if len(horses) < 5 or set(horses) != set(first_scores) or set(horses) != set(second_scores):
        raise ValueError("At least five runners and complete role scores are required")
    firsts = sorted(horses, key=lambda h: (-float(first_scores[h]), int(h)))[:2]
    seconds = sorted(horses, key=lambda h: (-float(second_scores[h]), int(h)))[:4]
    candidates = []
    for first, second in itertools.product(firsts, seconds):
        if first == second:
            continue
        score = float(first_scores[first]) * float(second_scores[second])
        candidates.append((score, int(first), int(second), [first, second]))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [item[3] for item in candidates[:limit]]


def evaluate(races):
    import nar_backtest as nb
    totals = {"races": 0, "hits": 0, "tickets": 0, "stake_jpy": 0, "return_jpy": 0,
              "first_pool_winner_hits": 0, "second_pool_runnerup_hits": 0}
    diagnostics = []
    for item in races:
        group = item["group"]
        first_scores = item["winner_model_probabilities"]
        second_scores = item["podium_model_probabilities"]
        tickets = build_exacta_tickets(group, first_scores, second_scores, limit=6)
        payout = item.get("payout") or {}
        winners = {str(r["horse_no"]) for r in group if r.get("finish") == 1}
        seconds = {str(r["horse_no"]) for r in group if r.get("finish") == 2}
        if len(winners) != 1 or len(seconds) != 1:
            continue
        first_pool = set(sorted(first_scores, key=lambda h: (-float(first_scores[h]), int(h)))[:2])
        second_pool = set(sorted(second_scores, key=lambda h: (-float(second_scores[h]), int(h)))[:4])
        returned_by_ticket = [nb.payout_for("exacta", ticket, payout) or 0 for ticket in tickets]
        returned = sum(returned_by_ticket)
        totals["races"] += 1
        totals["hits"] += int(returned > 0)
        totals["tickets"] += len(tickets)
        totals["stake_jpy"] += len(tickets) * 100
        totals["return_jpy"] += returned
        totals["first_pool_winner_hits"] += int(bool(winners & first_pool))
        totals["second_pool_runnerup_hits"] += int(bool(seconds & second_pool))
        diagnostics.append({
            "date": item.get("date"), "race_key": item.get("race_key"),
            "tickets": tickets, "hit": returned > 0,
            "return_jpy": int(returned),
            "winner_in_first_pool": bool(winners & first_pool),
            "runnerup_in_second_pool": bool(seconds & second_pool),
        })
    n, stake = totals["races"], totals["stake_jpy"]
    totals.update({
        "hit_rate_pct": round(100 * totals["hits"] / n, 2) if n else None,
        "roi_pct": round(100 * totals["return_jpy"] / stake, 2) if stake else None,
        "profit_jpy": totals["return_jpy"] - stake,
        "average_tickets_per_race": round(totals["tickets"] / n, 2) if n else None,
        "average_stake_per_race_jpy": round(stake / n, 2) if n else None,
        "first_pool_winner_capture_pct": round(100 * totals["first_pool_winner_hits"] / n, 2) if n else None,
        "second_pool_runnerup_capture_pct": round(100 * totals["second_pool_runnerup_hits"] / n, 2) if n else None,
        "race_diagnostics": diagnostics,
        "research_only": True,
        "policy": "Six exacta tickets per race, 100 JPY each; first-place top-2 winner classifier probabilities x second-place top-4 podium classifier probabilities, excluding same-horse pairs and ranking by probability product. Same chronological holdout race cohort as the trio comparison.",
    })
    return totals
