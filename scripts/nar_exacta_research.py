"""Research-only exacta (馬単) evaluation with a fixed six-ticket, 600-JPY budget."""
from __future__ import annotations

import itertools
import math


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


def has_exacta_payout_data(payout):
    """True only when the official archive contains at least one exacta payout row."""
    if not isinstance(payout, dict):
        return False
    rows = payout.get("exacta")
    if isinstance(rows, (list, tuple)) and rows:
        return True
    index = payout.get("_index")
    exacta_index = index.get("exacta") if isinstance(index, dict) else None
    return isinstance(exacta_index, dict) and bool(exacta_index)


def evaluate(races):
    import nar_backtest as nb
    totals = {
        "races": 0, "eligible_result_races": 0, "payout_data_missing_races": 0,
        "hits": 0, "tickets": 0, "stake_jpy": 0, "return_jpy": 0,
        "first_pool_winner_hits": 0, "second_pool_runnerup_hits": 0,
    }
    diagnostics = []
    missing_payout_races = []
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

        totals["eligible_result_races"] += 1
        if not has_exacta_payout_data(payout):
            totals["payout_data_missing_races"] += 1
            missing_payout_races.append({
                "date": item.get("date"), "race_key": item.get("race_key"),
            })
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
            "exacta_payout_data_present": True,
        })
    n, stake = totals["races"], totals["stake_jpy"]
    eligible = totals["eligible_result_races"]
    totals.update({
        "hit_rate_pct": round(100 * totals["hits"] / n, 2) if n else None,
        "roi_pct": round(100 * totals["return_jpy"] / stake, 2) if stake else None,
        "profit_jpy": totals["return_jpy"] - stake,
        "average_tickets_per_race": round(totals["tickets"] / n, 2) if n else None,
        "average_stake_per_race_jpy": round(stake / n, 2) if n else None,
        "payout_data_coverage_pct": round(100 * n / eligible, 2) if eligible else None,
        "first_pool_winner_capture_pct": round(100 * totals["first_pool_winner_hits"] / n, 2) if n else None,
        "second_pool_runnerup_capture_pct": round(100 * totals["second_pool_runnerup_hits"] / n, 2) if n else None,
        "race_diagnostics": diagnostics,
        "payout_data_missing_races_detail": missing_payout_races,
        "research_only": True,
        "policy": "Six exacta tickets per race, 100 JPY each; first-place top-2 winner classifier probabilities x second-place top-4 podium classifier probabilities, excluding same-horse pairs and ranking by probability product. ROI and hit rate use only races with an official exacta payout row; missing-payout races are reported and excluded, not silently treated as zero. Same chronological holdout race cohort as the trio comparison.",
    })
    return totals


def evaluate_conditional(races):
    """Evaluate conditional-second model with the same six-ticket / 600-JPY policy."""
    import nar_backtest as nb
    import nar_exacta_conditional_model as conditional
    totals = {
        "races": 0, "eligible_result_races": 0, "payout_data_missing_races": 0,
        "hits": 0, "tickets": 0, "stake_jpy": 0, "return_jpy": 0,
        "first_pool_winner_hits": 0, "second_pool_runnerup_hits": 0,
        "runnerup_eval_races": 0, "runnerup_top1_hits": 0,
        "runnerup_top2_hits": 0, "runnerup_top4_hits": 0,
    }
    diagnostics, missing = [], []
    calibration_probabilities, calibration_labels = [], []
    for item in races:
        group = item["group"]
        first_scores = item["winner_model_probabilities"]
        conditional_scores = item["conditional_second_scores"]
        tickets = conditional.build_conditional_exacta_tickets(
            group, first_scores, conditional_scores, limit=6
        )
        payout = item.get("payout") or {}
        winners = {str(r["horse_no"]) for r in group if r.get("finish") == 1}
        seconds = {str(r["horse_no"]) for r in group if r.get("finish") == 2}
        if len(winners) != 1 or len(seconds) != 1:
            continue
        totals["eligible_result_races"] += 1
        actual_winner = next(iter(winners))
        actual_second = next(iter(seconds))
        conditional_for_actual_winner = conditional_scores[actual_winner]
        runnerup_ranking = sorted(
            conditional_for_actual_winner,
            key=lambda h: (-float(conditional_for_actual_winner[h]), int(h)),
        )
        totals["runnerup_eval_races"] += 1
        totals["runnerup_top1_hits"] += int(runnerup_ranking[0] == actual_second)
        totals["runnerup_top2_hits"] += int(actual_second in runnerup_ranking[:2])
        totals["runnerup_top4_hits"] += int(actual_second in runnerup_ranking[:4])
        for horse, raw_probability in conditional_for_actual_winner.items():
            probability = min(max(float(raw_probability), 1e-7), 1 - 1e-7)
            calibration_probabilities.append(probability)
            calibration_labels.append(int(horse == actual_second))
        if not has_exacta_payout_data(payout):
            totals["payout_data_missing_races"] += 1
            missing.append({"date": item.get("date"), "race_key": item.get("race_key")})
            continue
        first_pool = set(sorted(first_scores, key=lambda h: (-float(first_scores[h]), int(h)))[:2])
        second_pool = set(runnerup_ranking[:4])
        returned = sum(nb.payout_for("exacta", ticket, payout) or 0 for ticket in tickets)
        totals["races"] += 1
        totals["hits"] += int(returned > 0)
        totals["tickets"] += len(tickets)
        totals["stake_jpy"] += len(tickets) * 100
        totals["return_jpy"] += returned
        totals["first_pool_winner_hits"] += int(bool(winners & first_pool))
        totals["second_pool_runnerup_hits"] += int(bool(seconds & second_pool))
        diagnostics.append({
            "date": item.get("date"), "race_key": item.get("race_key"),
            "tickets": tickets, "hit": returned > 0, "return_jpy": int(returned),
            "winner_in_first_pool": bool(winners & first_pool),
            "runnerup_in_conditional_top4": bool(seconds & second_pool),
            "exacta_payout_data_present": True,
        })
    n, stake = totals["races"], totals["stake_jpy"]
    eligible = totals["eligible_result_races"]
    runnerup_n = totals["runnerup_eval_races"]
    if calibration_probabilities:
        brier = sum((p - y) ** 2 for p, y in zip(calibration_probabilities, calibration_labels)) / len(calibration_labels)
        log_loss = -sum(y * math.log(p) + (1 - y) * math.log(1 - p)
                        for p, y in zip(calibration_probabilities, calibration_labels)) / len(calibration_labels)
        ece = 0.0
        for bin_index in range(10):
            lower, upper = bin_index / 10, (bin_index + 1) / 10
            indices = [i for i, p in enumerate(calibration_probabilities)
                       if lower <= p < upper or (bin_index == 9 and p == 1.0)]
            if indices:
                mean_p = sum(calibration_probabilities[i] for i in indices) / len(indices)
                mean_y = sum(calibration_labels[i] for i in indices) / len(indices)
                ece += len(indices) / len(calibration_labels) * abs(mean_p - mean_y)
    else:
        brier = log_loss = ece = None
    totals.update({
        "hit_rate_pct": round(100 * totals["hits"] / n, 2) if n else None,
        "roi_pct": round(100 * totals["return_jpy"] / stake, 2) if stake else None,
        "profit_jpy": totals["return_jpy"] - stake,
        "average_tickets_per_race": round(totals["tickets"] / n, 2) if n else None,
        "average_stake_per_race_jpy": round(stake / n, 2) if n else None,
        "payout_data_coverage_pct": round(100 * n / eligible, 2) if eligible else None,
        "runnerup_top1_accuracy_pct": round(100 * totals["runnerup_top1_hits"] / runnerup_n, 2) if runnerup_n else None,
        "runnerup_top2_accuracy_pct": round(100 * totals["runnerup_top2_hits"] / runnerup_n, 2) if runnerup_n else None,
        "runnerup_top4_accuracy_pct": round(100 * totals["runnerup_top4_hits"] / runnerup_n, 2) if runnerup_n else None,
        "runnerup_brier_score": round(brier, 6) if brier is not None else None,
        "runnerup_log_loss": round(log_loss, 6) if log_loss is not None else None,
        "runnerup_ece_10_bins": round(ece, 6) if ece is not None else None,
        "runnerup_probability_warning": "Raw class-balanced model scores; negative sampling changes class prevalence, so values are not calibrated conditional probabilities. Brier/log-loss/ECE are diagnostic only.",
        "first_pool_winner_capture_pct": round(100 * totals["first_pool_winner_hits"] / n, 2) if n else None,
        "second_pool_runnerup_capture_pct": round(100 * totals["second_pool_runnerup_hits"] / n, 2) if n else None,
        "race_diagnostics": diagnostics, "payout_data_missing_races_detail": missing,
        "research_only": True,
        "policy": "Six ordered exacta tickets at 100 JPY each; top-2 winner candidates multiplied by uncalibrated conditional runner-up score proxies for each possible first horse. Conditional model is trained only on races before the evaluation window. Runner-up top-k and calibration diagnostics condition on the actual winner only for retrospective evaluation. ROI excludes races without an official exacta payout row and reports coverage.",
    })
    return totals
