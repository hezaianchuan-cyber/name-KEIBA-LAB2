#!/usr/bin/env python3
"""Research-only 3連複 bankroll allocation backtest.

Uses immutable, SHA-256-verified pre-off snapshots and official 3連複 payouts.
Ticket selection is held constant (top 12 by 50% LAB/market fused probability)
so this experiment isolates stake allocation from adaptive ticket-count policy.
No live ticket generation or model weights are changed.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = ROOT / "data" / "odds-snapshots"
PAYOUTS_FILE = ROOT / "data" / "daily-payouts.json"
OUT = ROOT / "data" / "trio-bankroll-allocation-backtest.json"
UNIT = 100
BUDGET = 1900
MAX_TICKETS = 12
MARKET_WEIGHT = 0.50
B_MAX_STAKE_SHARE = 0.30

sys.path.insert(0, str(Path(__file__).resolve().parent))
from publish_model_snapshots import verify_record  # noqa: E402


def norm(value):
    return unicodedata.normalize("NFKC", str(value or "")).strip()


def horse_no(value):
    digits = "".join(ch for ch in str(value or "") if ch.isdigit())
    return str(int(digits)) if digits else ""


def combo_key(numbers):
    values = [horse_no(value) for value in numbers]
    if len(values) != 3 or not all(values) or len(set(values)) != 3:
        return None
    return tuple(sorted(values, key=int))


def race_key(value):
    parts = str(value or "").split("|")
    if len(parts) != 3:
        return None
    venue = norm(parts[0])
    date = "".join(ch for ch in parts[1] if ch.isdigit())
    race = "".join(ch for ch in parts[2] if ch.isdigit())
    if not venue or len(date) != 8 or not race:
        return None
    return f"{venue}|{date}|{int(race)}"


def trio_odds_lookup(odds_rows):
    lookup = {}
    for item in odds_rows or []:
        kind = norm(item.get("type")).replace(" ", "")
        if kind not in ("3連複", "三連複"):
            continue
        key = combo_key([item.get("n1"), item.get("n2"), item.get("n3")])
        try:
            price = float(item.get("odds"))
        except (TypeError, ValueError):
            continue
        if key and math.isfinite(price) and price > 0:
            lookup[key] = price
    return lookup


def build_candidates(record, market_weight=MARKET_WEIGHT):
    """Join LAB trio probabilities to the exact 3連複 odds in the same lock."""
    odds = trio_odds_lookup(record.get("odds"))
    if not odds:
        return []
    inverse = {key: 1.0 / price for key, price in odds.items()}
    inverse_total = sum(inverse.values())
    if inverse_total <= 0:
        return []
    market_prob = {key: value / inverse_total for key, value in inverse.items()}
    rows = []
    seen = set()
    for item in record.get("candidate_combinations") or []:
        if item.get("bet_type") != "trio":
            continue
        key = combo_key(item.get("numbers") or [])
        if not key or key in seen or key not in odds or key not in market_prob:
            continue
        try:
            lab_prob = float(item.get("probability"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(lab_prob) or lab_prob <= 0 or lab_prob > 1:
            continue
        fused = (1.0 - market_weight) * lab_prob + market_weight * market_prob[key]
        price = odds[key]
        rows.append({
            "combo": key,
            "lab_probability": lab_prob,
            "market_probability": market_prob[key],
            "fused_probability": fused,
            "odds": price,
            "expected_value": fused * price,
        })
        seen.add(key)
    rows.sort(key=lambda row: (-row["fused_probability"], -row["expected_value"], row["combo"]))
    return rows


def load_payouts(path=PAYOUTS_FILE):
    if not path.is_file():
        raise RuntimeError(f"Official payout file is missing: {path}")
    source = json.loads(path.read_text(encoding="utf-8"))
    outcomes = {}
    for day in source.get("days", []):
        date = "".join(ch for ch in str(day.get("date") or "") if ch.isdigit())
        if len(date) != 8:
            continue
        for race in day.get("races", []):
            venue = norm(race.get("venue"))
            number = race.get("race")
            if not venue or number is None:
                continue
            key = f"{venue}|{date}|{int(number)}"
            trio_rows = (race.get("payouts") or {}).get("trio") or []
            payouts = {}
            for item in trio_rows:
                combo = combo_key(item.get("numbers") or [])
                try:
                    pay = int(item.get("pay_100") or 0)
                except (TypeError, ValueError):
                    continue
                if combo and pay > 0:
                    payouts[combo] = max(pay, payouts.get(combo, 0))
            if payouts:
                outcomes[key] = payouts
    return outcomes


def load_snapshots(snapshot_dir=SNAPSHOT_DIR):
    rows = []
    diagnostics = {
        "files": 0, "lines": 0, "valid_locked_preoff": 0,
        "invalid_lock_or_shape": 0, "missing_trio_candidates_or_odds": 0,
    }
    if not snapshot_dir.is_dir():
        return rows, diagnostics
    for path in sorted(snapshot_dir.glob("*.jsonl.gz")):
        diagnostics["files"] += 1
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                diagnostics["lines"] += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    diagnostics["invalid_lock_or_shape"] += 1
                    continue
                if not verify_record(record):
                    diagnostics["invalid_lock_or_shape"] += 1
                    continue
                key = race_key(record.get("race_key"))
                if not key:
                    diagnostics["invalid_lock_or_shape"] += 1
                    continue
                candidates = build_candidates(record)
                if not candidates:
                    diagnostics["missing_trio_candidates_or_odds"] += 1
                    continue
                diagnostics["valid_locked_preoff"] += 1
                rows.append({
                    "race_key": key,
                    "captured_at": record.get("captured_at"),
                    "minutes_to_post": float(record["minutes_to_post"]),
                    "model_version": record.get("model_version") or "unknown",
                    "candidates": candidates,
                })
    return rows, diagnostics


def select_one_snapshot_per_race(rows):
    """Use the latest available snapshot closest to post, within T-2 to T-30."""
    selected = {}
    for row in rows:
        minutes = row.get("minutes_to_post")
        if minutes is None or not 2.0 <= float(minutes) <= 30.0:
            continue
        key = row["race_key"]
        order = (float(minutes), str(row.get("captured_at") or ""))
        previous = selected.get(key)
        if previous is None:
            selected[key] = (order, row)
        else:
            old_order = previous[0]
            if order[0] < old_order[0] or (order[0] == old_order[0] and order[1] > old_order[1]):
                selected[key] = (order, row)
    return [value[1] for value in selected.values()]


def allocate_equal_units(ticket_count, budget_units=BUDGET // UNIT):
    if ticket_count <= 0:
        return []
    base, remainder = divmod(budget_units, ticket_count)
    return [base + (1 if index < remainder else 0) for index in range(ticket_count)]


def allocate_weighted_units(weights, budget_units=BUDGET // UNIT, max_share=1.0):
    """Allocate integer 100-yen units by weight, respecting a per-ticket cap.

    If every positive-edge ticket reaches its cap, the remaining budget is left
    unspent. A zero/non-positive edge never receives stake.
    """
    if not weights or budget_units <= 0:
        return []
    clean = [float(value) if math.isfinite(float(value)) and float(value) > 0 else 0.0 for value in weights]
    cap_units = max(0, int(math.floor((budget_units * max_share) + 1e-12)))
    allocated = [0] * len(clean)
    remaining = budget_units
    while remaining > 0:
        active = [i for i, weight in enumerate(clean) if weight > 0 and allocated[i] < cap_units]
        if not active:
            break
        total_weight = sum(clean[i] for i in active)
        quotas = {i: remaining * clean[i] / total_weight for i in active}
        capped = [i for i in active if quotas[i] >= cap_units - allocated[i]]
        if capped:
            for i in capped:
                amount = cap_units - allocated[i]
                allocated[i] += amount
                remaining -= amount
            continue
        base = {i: min(cap_units - allocated[i], int(math.floor(quotas[i]))) for i in active}
        used = sum(base.values())
        for i, amount in base.items():
            allocated[i] += amount
        remaining -= used
        if remaining <= 0:
            break
        fractions = sorted(
            active,
            key=lambda i: (-(quotas[i] - math.floor(quotas[i])), -clean[i], i),
        )
        progressed = False
        for i in fractions:
            if remaining <= 0:
                break
            if allocated[i] < cap_units:
                allocated[i] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    return allocated


def allocate_for_race(candidates, mode):
    """Return integer 100-yen units; all modes receive the same candidate set."""
    if mode == "equal":
        return allocate_equal_units(len(candidates))
    if mode == "ev_only":
        weights = [max(0.0, row["expected_value"] - 1.0) for row in candidates]
        return allocate_weighted_units(weights, max_share=0.30)
    if mode == "balanced":
        weights = [
            max(0.0, row["expected_value"] - 1.0) * math.sqrt(row["fused_probability"])
            for row in candidates
        ]
        return allocate_weighted_units(weights, max_share=B_MAX_STAKE_SHARE)
    raise ValueError(f"Unknown allocation mode: {mode}")


def evaluate_races(races, mode):
    per_race = []
    for race in races:
        candidates = race["candidates"][:MAX_TICKETS]
        units = allocate_for_race(candidates, mode)
        payouts = race["payouts"]
        stake = sum(units) * UNIT
        returned = sum(
            units[index] * payouts.get(row["combo"], 0)
            for index, row in enumerate(candidates)
        )
        hit = any(units[index] > 0 and row["combo"] in payouts
                  for index, row in enumerate(candidates))
        per_race.append({
            "race_key": race["race_key"],
            "hit": hit,
            "ticket_count": len(candidates),
            "staked_tickets": sum(1 for value in units if value > 0),
            "stake": stake,
            "return": returned,
            "profit": returned - stake,
        })
    total_stake = sum(row["stake"] for row in per_race)
    total_return = sum(row["return"] for row in per_race)
    hits = sum(1 for row in per_race if row["hit"])
    ordered = sorted(per_race, key=lambda row: row["race_key"])
    cumulative = peak = max_drawdown = 0
    losing_streak = current_losing_streak = 0
    for row in ordered:
        cumulative += row["profit"]
        peak = max(peak, cumulative)
        max_drawdown = max(max_drawdown, peak - cumulative)
        if row["profit"] < 0:
            current_losing_streak += 1
            losing_streak = max(losing_streak, current_losing_streak)
        else:
            current_losing_streak = 0
    return {
        "strategy": mode,
        "races": len(per_race),
        "hit_races": hits,
        "race_hit_rate_pct": round(100 * hits / len(per_race), 2) if per_race else None,
        "staked_tickets": sum(row["staked_tickets"] for row in per_race),
        "stake_jpy": total_stake,
        "return_jpy": total_return,
        "profit_jpy": total_return - total_stake,
        "roi_pct": round(100 * total_return / total_stake, 2) if total_stake else None,
        "average_stake_per_race_jpy": round(total_stake / len(per_race), 2) if per_race else None,
        "max_drawdown_jpy": max_drawdown,
        "max_losing_streak": losing_streak,
        "unspent_races": sum(1 for row in per_race if row["stake"] == 0),
    }


def build_matched_races(snapshot_rows, payouts):
    selected = select_one_snapshot_per_race(snapshot_rows)
    matched = []
    for row in selected:
        outcome = payouts.get(row["race_key"])
        if not outcome:
            continue
        matched.append({**row, "payouts": outcome})
    return matched


def run_backtest(snapshot_dir=SNAPSHOT_DIR, payouts_file=PAYOUTS_FILE, output_file=OUT):
    snapshots, diagnostics = load_snapshots(snapshot_dir)
    payouts = load_payouts(payouts_file)
    races = build_matched_races(snapshots, payouts)
    if not races:
        raise RuntimeError(
            "No races have both a verified pre-off model snapshot with 3連複 candidates "
            "and an official 3連複 payout. Do not report a performance result."
        )
    report = {
        "generated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "source": "SHA-256-verified NAR pre-off model snapshots + official NAR trio payouts",
        "market_weight": MARKET_WEIGHT,
        "ticket_selection": "same top 12 by fused probability for every allocation strategy; selection held fixed to isolate stake allocation",
        "budget_per_race_jpy": BUDGET,
        "unit_jpy": UNIT,
        "balanced_weight_formula": "max(fused_probability * preoff_trio_odds - 1, 0) * sqrt(fused_probability)",
        "balanced_max_ticket_share": B_MAX_STAKE_SHARE,
        "diagnostics": diagnostics,
        "matched_races": len(races),
        "model_versions": dict(sorted((version, sum(1 for row in races if row["model_version"] == version))
                                      for version in {row["model_version"] for row in races})),
        "strategies": [
            evaluate_races(races, "equal"),
            evaluate_races(races, "ev_only"),
            evaluate_races(races, "balanced"),
        ],
        "note": "Research only. A and B use the same selected tickets; variable ticket count and model weight tuning are separate experiments. Stakes are capped at JPY 1,900 per race in 100-yen units; unused budget is not forced into negative-edge tickets.",
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-dir", type=Path, default=SNAPSHOT_DIR)
    parser.add_argument("--payouts", type=Path, default=PAYOUTS_FILE)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    report = run_backtest(args.snapshot_dir, args.payouts, args.output)
    print(json.dumps({
        "matched_races": report["matched_races"],
        "strategies": report["strategies"],
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
