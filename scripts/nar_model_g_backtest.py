#!/usr/bin/env python3
"""Prospective, chronological evaluation of experimental NAR Model G snapshots."""
from __future__ import annotations

import datetime as dt
import gzip
import json
import math
import random
from pathlib import Path

import nar_backtest as nb
import nar_model_g as mg

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SNAPSHOT_DIR = DATA / "odds-snapshots"
OUT = DATA / "model-g-prospective-backtest.json"


def load_g_snapshots():
    records = {}
    if not SNAPSHOT_DIR.exists():
        return records
    for path in sorted(SNAPSHOT_DIR.glob("*.jsonl.gz")):
        try:
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if row.get("model_g_version") != mg.MODEL_VERSION:
                        continue
                    key = str(row.get("race_key") or "")
                    if key:
                        records[key] = row
        except (OSError, EOFError):
            continue
    return records


def is_valid_snapshot(record):
    try:
        captured = dt.datetime.fromisoformat(record["captured_at"])
        scheduled = dt.datetime.fromisoformat(record["scheduled_post_time"])
        minutes = float(record["minutes_to_post"])
    except (KeyError, TypeError, ValueError):
        return False
    return (
        captured.tzinfo is not None and scheduled.tzinfo is not None
        and captured < scheduled and 2 <= minutes <= 30
        and bool(record.get("model_g_predictions"))
        and bool(record.get("model_g_candidate_combinations"))
    )


def evaluate(records, payouts):
    races = tickets = hits = stake = returned = skipped = 0
    selected_types = {}
    race_results = []
    skip_reasons = {}
    for record in records:
        parts = str(record.get("race_key") or "").split("|")
        if len(parts) != 3:
            skipped += 1
            continue
        key = nb.race_key(parts[0], parts[1], parts[2])
        payout = payouts.get(key)
        if not payout or not payout.get("win"):
            skipped += 1
            continue
        strategy = mg.select_g_strategy(
            {"venue": record.get("venue"), "distance": None},
            record.get("model_g_predictions") or [],
            record.get("model_g_candidate_combinations") or [],
            int((record.get("model_g_strategy") or {}).get("calibration_races") or 0),
        )
        if not strategy.get("tickets"):
            skipped += 1
            reason = (strategy.get("reasons") or [strategy.get("status") or "skip"])[0]
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            continue
        race_stake = race_return = race_hits = 0
        for ticket in strategy["tickets"]:
            amount = int(ticket.get("stake_jpy") or 0)
            kind = ticket["bet_type"]
            numbers = [str(x) for x in ticket["numbers"]]
            pay = nb.payout_for(kind, numbers, payout) or 0
            race_stake += amount
            if pay:
                race_hits += 1
                race_return += int(pay * amount / 100)
        if race_stake > mg.RACE_CAP_JPY:
            raise AssertionError("Model G per-race cap exceeded")
        if race_stake:
            races += 1
            tickets += len(strategy["tickets"])
            hits += race_hits
            stake += race_stake
            returned += race_return
            selected_types[strategy["selected_bet_type"]] = selected_types.get(strategy["selected_bet_type"], 0) + 1
            race_results.append((race_stake, race_return))
    ci_low = ci_high = None
    if len(race_results) >= 30:
        rng = random.Random(20261104 + len(race_results))
        samples = []
        for _ in range(3000):
            sample = [race_results[rng.randrange(len(race_results))] for _ in race_results]
            sample_stake = sum(x[0] for x in sample)
            sample_return = sum(x[1] for x in sample)
            if sample_stake:
                samples.append(100 * sample_return / sample_stake)
        samples.sort()
        ci_low = round(samples[int(0.025 * (len(samples) - 1))], 2)
        ci_high = round(samples[int(0.975 * (len(samples) - 1))], 2)
    return {
        "races_with_bets": races, "skipped_races": skipped,
        "skip_reasons": skip_reasons, "tickets": tickets, "hits": hits,
        "hit_rate_percent": round(100 * hits / tickets, 2) if tickets else None,
        "stake_jpy": stake, "return_jpy": returned, "profit_jpy": returned - stake,
        "roi_percent": round(100 * returned / stake, 2) if stake else None,
        "roi_ci95_race_cluster_low": ci_low, "roi_ci95_race_cluster_high": ci_high,
        "selected_bet_type_race_counts": selected_types,
        "max_race_stake_jpy": max((x[0] for x in race_results), default=0),
        "race_budget_cap_jpy": mg.RACE_CAP_JPY,
        "minimum_sample_met": races >= 100 and tickets >= 300,
    }


def main():
    snapshots = [row for row in load_g_snapshots().values() if is_valid_snapshot(row)]
    dates = sorted({str(row.get("date") or "") for row in snapshots if row.get("date")})
    base = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model_version": mg.MODEL_VERSION,
        "snapshot_policy": "Only genuine official snapshots captured 2-30 minutes before scheduled post time; final odds are never substituted.",
        "strategy_policy": {
            "minimum_expected_value": mg.MIN_EXPECTED_VALUE,
            "ticket_unit_jpy": mg.TICKET_UNIT_JPY,
            "maximum_tickets": mg.MAX_TICKETS,
            "race_budget_cap_jpy": mg.RACE_CAP_JPY,
            "bet_type": "One bet type selected per race from qualified candidates; races without evidence are skipped.",
            "threshold_selection": "Fixed before prospective evaluation; no holdout tuning.",
        },
        "snapshot_races": len(snapshots),
        "snapshot_dates": len(dates),
        "status": "awaiting_preoff_snapshots",
        "development": None,
        "holdout": None,
        "profitable_candidate": False,
        "promotion_policy": "Promote only after at least 100 holdout races and 300 tickets, with race-cluster bootstrap 95% ROI lower bound above 100%.",
    }
    if len(dates) < 10:
        base["status"] = "awaiting_more_snapshot_dates" if snapshots else "awaiting_preoff_snapshots"
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(base, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(base, ensure_ascii=False), flush=True)
        return

    # Download official payouts only after snapshots exist. Historical data is
    # matched by race key; only the snapshot's contemporaneous G ticket is scored.
    _, _, payouts, downloaded = nb.load_archives()
    split = max(1, min(len(dates) - 1, int(len(dates) * 0.70)))
    development_dates = set(dates[:split])
    development = [r for r in snapshots if str(r.get("date") or "") in development_dates]
    holdout = [r for r in snapshots if str(r.get("date") or "") not in development_dates]
    dev_metrics = evaluate(development, payouts)
    holdout_metrics = evaluate(holdout, payouts)
    minimum = holdout_metrics["minimum_sample_met"]
    lower = holdout_metrics["roi_ci95_race_cluster_low"]
    profitable = bool(minimum and lower is not None and lower > 100)
    if profitable:
        status = "profitable_candidate"
    elif not minimum:
        status = "awaiting_holdout_sample"
    else:
        status = "not_validated_profitable"
    base.update({
        "status": status, "downloaded_payout_months": downloaded,
        "development_period": [dates[0], dates[split - 1]],
        "holdout_period": [dates[split], dates[-1]],
        "development": dev_metrics, "holdout": holdout_metrics,
        "profitable_candidate": profitable,
    })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(base, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(base, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
