#!/usr/bin/env python3
"""Research-only 3連複 combo scoring using Plackett-Luce top-3 probabilities.

This evaluates combinations from each strategy's top-seven horse pool. It does
not alter the live model or live ticket generator. Model scores are converted
to win-strength probabilities with the validated G1 temperature; captured
win odds are normalized into market probabilities. The probability of an
unordered top-three is the sum of its six Plackett-Luce finish orders.
"""
from __future__ import annotations

import itertools
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "trio-combination-probability-backtest.json"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import nar_trio_narrowing_pilot as pilot  # noqa: E402

MARKET_WEIGHTS = (0.0, 0.25, 0.50, 0.75, 1.0)
TICKET_COUNTS = (10, 12, 19)
BUDGET_UNITS = 19
UNIT_JPY = 100
TOP_HORSES = 7


def _horse_no(value):
    digits = "".join(ch for ch in str(value or "") if ch.isdigit())
    return str(int(digits)) if digits else ""


def _normalize(values):
    clean = {str(k): float(v) for k, v in values.items()
             if math.isfinite(float(v)) and float(v) > 0}
    total = sum(clean.values())
    if total <= 0:
        raise ValueError("probability weights must have a positive finite sum")
    return {k: v / total for k, v in clean.items()}


def model_probabilities(predictions, temperature):
    """Convert saved raw scores with the already validated G1 temperature."""
    scores = {}
    for item in predictions or []:
        no = _horse_no(item.get("horse_no"))
        try:
            score = float(item.get("score"))
        except (TypeError, ValueError):
            continue
        if no and math.isfinite(score):
            scores[no] = score
    if len(scores) < 3:
        return {}
    temp = max(0.005, float(temperature))
    peak = max(scores.values())
    weights = {
        no: math.exp(max(-700.0, (score - peak) / temp))
        for no, score in scores.items()
    }
    return _normalize(weights)


def market_probabilities(odds):
    """Normalize inverse captured win odds; ignore place odds and invalid prices."""
    inverse = {}
    for item in odds or []:
        label = pilot.norm(item.get("type")).replace(" ", "")
        if "単勝" not in label:
            continue
        no = _horse_no(item.get("n1"))
        try:
            price = float(item.get("odds"))
        except (TypeError, ValueError):
            continue
        if no and math.isfinite(price) and price > 0:
            inverse[no] = 1.0 / price
    if len(inverse) < 3:
        return {}
    return _normalize(inverse)


def blend_probabilities(model, market, market_weight):
    """Blend normalized probabilities on the identical runner intersection."""
    common = sorted(set(model).intersection(market), key=int)
    if len(common) < 3:
        return {}
    model_common = _normalize({no: model[no] for no in common})
    market_common = _normalize({no: market[no] for no in common})
    weight = float(market_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("market_weight must be between 0 and 1")
    return _normalize({
        no: (1.0 - weight) * model_common[no] + weight * market_common[no]
        for no in common
    })


def plackett_luce_trio_probability(probabilities, combo):
    """Probability that an unordered trio occupies places 1-3.

    The input probabilities act as Plackett-Luce worth parameters. The six
    possible finishing orders are summed, so order within a 3連複 ticket
    does not matter.
    """
    p = _normalize(probabilities)
    horses = tuple(_horse_no(no) for no in combo)
    if len(horses) != 3 or len(set(horses)) != 3:
        raise ValueError("a trio must contain exactly three distinct horses")
    if any(no not in p for no in horses):
        return 0.0
    total = 0.0
    for first, second, third in itertools.permutations(horses):
        remaining_after_first = 1.0 - p[first]
        remaining_after_second = 1.0 - p[first] - p[second]
        if remaining_after_first <= 0 or remaining_after_second <= 0:
            continue
        total += (
            p[first]
            * p[second] / remaining_after_first
            * p[third] / remaining_after_second
        )
    return total


def rank_trio_combinations(probabilities, candidate_count=TOP_HORSES):
    """Score all C(7,3) combinations, then rank by exact trio probability."""
    p = _normalize(probabilities)
    ranked_horses = sorted(p, key=lambda no: (-p[no], int(no)))
    candidates = ranked_horses[:max(3, min(int(candidate_count), len(ranked_horses)))]
    scored = []
    for combo in itertools.combinations(candidates, 3):
        score = plackett_luce_trio_probability(p, combo)
        canonical = tuple(sorted(combo, key=int))
        scored.append((score, canonical))
    scored.sort(key=lambda item: (-item[0], tuple(map(int, item[1]))))
    return [
        {"combo": list(combo), "probability": score}
        for score, combo in scored
    ]


def market_trio_data(odds):
    """Return normalized direct 3連複 implied probabilities and captured prices."""
    inverse = {}
    prices = {}
    for item in odds or []:
        label = pilot.norm(item.get("type")).replace(" ", "")
        if "3連複" not in label and "三連複" not in label:
            continue
        raw_numbers = [
            _horse_no(item.get("n1")),
            _horse_no(item.get("n2")),
            _horse_no(item.get("n3")),
        ]
        if len(raw_numbers) != 3 or not all(raw_numbers) or len(set(raw_numbers)) != 3:
            continue
        numbers = tuple(sorted(raw_numbers, key=int))
        try:
            price = float(item.get("odds"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(price) or price <= 0:
            continue
        prices[numbers] = price
        inverse[numbers] = 1.0 / price
    if len(inverse) < 3:
        return {}, prices
    return _normalize({ "-".join(k): v for k, v in inverse.items() }), prices


def rank_fused_trio_combinations(
    model_probabilities_by_horse,
    market_trio_probabilities,
    market_trio_odds,
    candidate_horses,
    market_weight,
):
    """Rank tickets by fused trio hit probability; retain LAB EV as a separate signal."""
    weight = float(market_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("market_weight must be between 0 and 1")
    market_probs = {
        tuple(key.split("-")): value
        for key, value in market_trio_probabilities.items()
    }
    rows = []
    for combo in itertools.combinations(sorted(candidate_horses, key=int), 3):
        price = market_trio_odds.get(combo)
        market_prob = market_probs.get(combo)
        if not price or not market_prob:
            continue
        model_prob = plackett_luce_trio_probability(model_probabilities_by_horse, combo)
        fused_prob = (1.0 - weight) * model_prob + weight * market_prob
        rows.append({
            "combo": list(combo),
            "probability": fused_prob,
            "fused_probability": fused_prob,
            "model_probability": model_prob,
            "market_probability": market_prob,
            "odds": price,
            "lab_ev": model_prob * price,
        })
    rows.sort(key=lambda item: (
        -item["fused_probability"],
        -item["lab_ev"],
        tuple(map(int, item["combo"])),
    ))
    return rows


def select_top_tickets(ranked_combinations, count):
    return [tuple(item["combo"]) for item in ranked_combinations[:max(0, int(count))]]


def rank_existing_tickets(tickets, ranked_horses):
    """Preserve the existing 2-4-5 rank-sum method as a direct baseline."""
    rank = {no: i for i, no in enumerate(ranked_horses, start=1)}
    ordered = sorted(
        tickets,
        key=lambda combo: (
            sum(rank.get(no, 10_000) for no in combo),
            tuple(map(int, combo)),
        ),
    )
    return [{"combo": list(combo), "probability": None} for combo in ordered]


def allocate_budget_units(ticket_count, total_units=BUDGET_UNITS):
    if ticket_count <= 0:
        return []
    base, remainder = divmod(total_units, ticket_count)
    return [base + (1 if i < remainder else 0) for i in range(ticket_count)]


def evaluate_tickets(races, selector, ticket_count, candidate_strategy=None):
    hit_races = 0
    candidate_coverage_hits = 0
    stake = 0
    returned = 0
    total_tickets = 0
    for race in races:
        tickets = selector(race)[:ticket_count]
        if not tickets:
            continue
        payouts = race["payouts"]
        hit = any(tuple(combo) in payouts for combo in tickets)
        hit_races += int(hit)
        if candidate_strategy:
            candidate_horses = set(race["candidate_horses"][candidate_strategy])
            candidate_coverage_hits += int(any(
                set(combo).issubset(candidate_horses) for combo in payouts
            ))
        units = allocate_budget_units(len(tickets))
        stake += BUDGET_UNITS * UNIT_JPY
        total_tickets += len(tickets)
        returned += sum(
            int(payouts.get(tuple(combo), 0)) * units[i]
            for i, combo in enumerate(tickets)
        )
    n = len(races)
    return {
        "races": n,
        "hit_races": hit_races,
        "race_hit_rate_pct": round(100 * hit_races / n, 2) if n else None,
        "candidate_top3_coverage_hits": candidate_coverage_hits,
        "candidate_top3_coverage_pct": round(100 * candidate_coverage_hits / n, 2) if n and candidate_strategy else None,
        "tickets_per_race": ticket_count,
        "tickets_total": total_tickets,
        "stake_jpy": stake,
        "return_jpy": returned,
        "profit_jpy": returned - stake,
        "roi_pct": round(100 * returned / stake, 2) if stake else None,
        "budget_per_race_jpy": BUDGET_UNITS * UNIT_JPY,
    }


def build_race_strategies(snapshot, payouts, temperature):
    model = model_probabilities(snapshot.get("predictions"), temperature)
    market = market_probabilities(snapshot.get("odds"))
    if not model or not market:
        return None
    market_trio_probs, market_trio_odds = market_trio_data(snapshot.get("odds"))
    if len(market_trio_probs) < 3:
        return None

    model_rank = pilot.model_rank(snapshot.get("predictions"))
    market_rank = pilot.market_rank(snapshot.get("odds"))
    strategies = {
        "model_245_rank_sum": rank_existing_tickets(pilot.build_245(model_rank), model_rank),
        "market_245_rank_sum": rank_existing_tickets(pilot.build_245(market_rank), market_rank),
    }
    for weight in MARKET_WEIGHTS:
        horse_blend = blend_probabilities(model, market, weight)
        ranked_horses = sorted(horse_blend, key=lambda no: (-horse_blend[no], int(no)))
        candidate_horses = ranked_horses[:TOP_HORSES]
        strategies[f"blend_{weight:.2f}"] = rank_fused_trio_combinations(
            model, market_trio_probs, market_trio_odds, candidate_horses, weight
        )
    strategies["model"] = strategies["blend_0.00"]
    strategies["market"] = strategies["blend_1.00"]

    candidate_horses = {
        name: sorted({no for item in ranked for no in item["combo"]}, key=int)
        for name, ranked in strategies.items()
    }
    return {
        "race_key": snapshot["race_key"],
        "date": snapshot["race_key"].split("|")[1],
        "model_version": snapshot.get("model_version") or "unknown",
        "payouts": payouts,
        "strategies": strategies,
        "candidate_horses": candidate_horses,
    }


def main():
    g1 = json.loads((DATA / "model-g1-calibration-backtest.json").read_text(encoding="utf-8"))
    if g1.get("status") != "G1_improved_on_untouched_holdout":
        raise RuntimeError("Validated G1 temperature is required for this research backtest.")
    temperature = float(g1["selected_temperature"])
    outcomes = pilot.load_trio_outcomes()
    snapshots, loader_diagnostics = pilot.load_github_snapshots()
    selected, selection_diagnostics = pilot.select_latest_snapshots(snapshots)

    races = []
    missing_outcome = 0
    invalid_probability = 0
    for snapshot in selected:
        payouts = outcomes.get(snapshot["race_key"])
        if not payouts:
            missing_outcome += 1
            continue
        race = build_race_strategies(snapshot, payouts, temperature)
        if race is None:
            invalid_probability += 1
            continue
        races.append(race)

    results = {}
    strategy_names = ["model", "market", "model_245_rank_sum", "market_245_rank_sum"] + [f"blend_{w:.2f}" for w in MARKET_WEIGHTS]
    for name in strategy_names:
        results[name] = {
            str(count): evaluate_tickets(
                races,
                lambda race, strategy=name: select_top_tickets(race["strategies"][strategy], 100),
                count,
                candidate_strategy=name,
            )
            for count in TICKET_COUNTS
        }

    by_version = {}
    for version in sorted({race["model_version"] for race in races}):
        subset = [race for race in races if race["model_version"] == version]
        by_version[version] = {
            "races": len(subset),
            "results": {
                name: {
                    str(count): evaluate_tickets(
                        subset,
                        lambda race, strategy=name: select_top_tickets(race["strategies"][strategy], 100),
                        count,
                        candidate_strategy=name,
                    )
                    for count in TICKET_COUNTS
                }
                for name in strategy_names
            },
        }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "feature": "trio_combination_probability_backtest",
        "status": "exploratory_only",
        "method": {
            "candidate_pool": "top seven horses by each strategy's blended win probability",
            "candidate_combinations_scored": "all 35 unordered triples from the seven-horse pool",
            "combo_score": "blend of LABO Plackett-Luce trio probability and direct market-implied 3連複 probability",
            "ticket_order": "descending fused trio hit probability; LABO expected value is a separate display/ranking tie-break signal, never a hard EV cutoff",
            "lab_expected_value": "LABO trio probability multiplied by the captured 3連複 odds; gross-return index per 100-yen stake",
            "market_trio_probability": "normalized inverse captured official 3連複 odds across all available combinations",
            "candidate_coverage": "whether all three official top-three horses are contained in the strategy's top-seven candidate pool",
            "model_probability": "saved raw model scores recalibrated with validated G1 temperature",
            "horse_candidate_pool": "top seven horses from the corresponding model/market win-probability blend",
            "market_probability": "direct normalized inverse captured pre-off 3連複 odds for each unordered combination",
            "market_weights": list(MARKET_WEIGHTS),
            "ticket_counts": list(TICKET_COUNTS),
            "baseline": "existing 2-4-5 formation, narrowed by sum of the corresponding model or market horse ranks",
            "budget": "JPY1900 per race for every strategy and ticket count; integer 100-yen units redistributed across selected tickets",
            "snapshot_window_minutes_to_post": [2, 30],
            "snapshot_selection": "one closest-to-post valid pre-off snapshot per race",
            "outcome_source": "official 3連複 payouts only",
            "live_model_changed": False,
            "live_ticket_generator_changed": False,
        },
        "coverage": {
            "selected_snapshots": len(selected),
            "eligible_races_with_official_trio_payout": len(races),
            "races_missing_official_trio_payout": missing_outcome,
            "races_with_invalid_probability_inputs": invalid_probability,
            "model_versions": {
                version: sum(race["model_version"] == version for race in races)
                for version in sorted({race["model_version"] for race in races})
            },
            "loader_diagnostics": loader_diagnostics,
            "selection_diagnostics": selection_diagnostics,
        },
        "results": results,
        "results_by_model_version": by_version,
        "interpretation": "Research comparison only. The Plackett-Luce trio score is a ranking model derived from win-strength probabilities, not a separately trained or validated place-probability model. No production settings are changed.",
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
