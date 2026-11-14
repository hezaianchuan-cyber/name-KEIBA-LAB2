#!/usr/bin/env python3
"""Evaluate fixed A45 and matched market 3連複 ticket policies against official payouts."""
from __future__ import annotations

import datetime as dt
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "research-a45-result-source" / "latest-race.zip"
JOINED = ROOT / "data" / "research-a45-odds-joined.jsonl"
OUT = ROOT / "data" / "research-a45-trio-evaluation.json"
REPORT = ROOT / "data" / "research-a45-trio-evaluation-report.json"


def parse_official_results(target_ids):
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import nar_backtest as nb

    if not SOURCE.is_file():
        raise RuntimeError("Current official NAR daily result archive is missing")
    payouts = {}
    finishes = defaultdict(dict)
    with zipfile.ZipFile(SOURCE) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            kind = nb.classify(info.filename)
            if kind not in ("payouts", "horses"):
                continue
            for row in nb.rows_from_bytes(archive.read(info)):
                if len(row) < 4:
                    continue
                key = nb.race_key(row[0], row[1], row[2])
                if not key or "|".join(key) not in target_ids:
                    continue
                if kind == "payouts" and len(row) > 52:
                    nb.merge_payout_row(payouts, key, row)
                elif kind == "horses" and len(row) > 35:
                    no = re.sub(r"\\D", "", str(row[5] or ""))
                    finish = nb.rank_value(row[31])
                    if no and finish is not None:
                        finishes["|".join(key)][str(int(no))] = int(finish)
    nb.finalize_payout_indexes(payouts)
    return {"|".join(key): value for key, value in payouts.items()}, dict(finishes)


def score_strategy(strategy, payout_index):
    tickets = strategy.get("tickets") or []
    hits = 0
    returned = 0.0
    hit_tickets = []
    for ticket in tickets:
        numbers = tuple(sorted(str(x) for x in ticket.get("numbers", [])))
        amount = payout_index.get(numbers)
        if amount:
            hits += 1
            returned += float(amount)
            hit_tickets.append({"numbers": list(numbers), "payout_yen": float(amount)})
    stake = int(strategy.get("ticket_count", len(tickets))) * 100
    return {
        "ticket_count": len(tickets),
        "stake_yen": stake,
        "hit_tickets": hits,
        "race_hit": hits > 0,
        "return_yen": returned,
        "profit_yen": returned - stake,
        "roi_pct": round(100.0 * returned / stake, 3) if stake else None,
        "hit_details": hit_tickets,
        "odds_all_quoted": bool(strategy.get("all_tickets_quoted")),
    }


def main():
    if not JOINED.is_file():
        raise RuntimeError("A45 pre-off odds join file is missing")
    joined = [
        json.loads(line) for line in JOINED.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    joined = [item for item in joined if item.get("odds_status") == "joined"]
    target_ids = {item["race_id"] for item in joined}
    payouts, finishes = parse_official_results(target_ids)

    per_race = []
    totals = {
        model: {
            strategy: {"races": 0, "races_hit": 0, "tickets": 0, "hit_tickets": 0,
                       "stake_yen": 0.0, "return_yen": 0.0}
            for strategy in ("top3_single", "axis_plus4", "axis_plus5", "top5_box")
        }
        for model in ("a45", "market")
    }
    pending = []
    for item in joined:
        race_id = item["race_id"]
        payout = payouts.get(tuple(race_id.split("|")))
        if not payout or not payout.get("trio"):
            pending.append(race_id)
            per_race.append({
                "race_id": race_id,
                "status": "awaiting_official_trio_payout",
                "prediction_captured_at": item["prediction_captured_at"],
                "odds_snapshot_captured_at": item["odds_snapshot_captured_at"],
                "a45_top7": item.get("a45_top7"),
                "market_top7": item.get("market_top7"),
                "finish_order": None,
                "models": {},
            })
            continue

        index = (payout.get("_index") or {}).get("trio", {})
        models = {}
        for model_name, field_name in (("a45", "strategies"), ("market", "market_strategies")):
            model_result = {}
            for strategy_name, strategy in (item.get(field_name) or {}).items():
                scored = score_strategy(strategy, index)
                model_result[strategy_name] = scored
                bucket = totals[model_name][strategy_name]
                bucket["races"] += 1
                bucket["races_hit"] += int(scored["race_hit"])
                bucket["tickets"] += scored["ticket_count"]
                bucket["hit_tickets"] += scored["hit_tickets"]
                bucket["stake_yen"] += scored["stake_yen"]
                bucket["return_yen"] += scored["return_yen"]
            models[model_name] = model_result

        finish_map = finishes.get(race_id, {})
        finish_order = [
            no for no, rank in sorted(finish_map.items(), key=lambda pair: pair[1])
            if rank in (1, 2, 3)
        ]
        per_race.append({
            "race_id": race_id,
            "status": "scored",
            "prediction_captured_at": item["prediction_captured_at"],
            "odds_snapshot_captured_at": item["odds_snapshot_captured_at"],
            "minutes_to_post_at_odds_capture": item["minutes_to_post_at_odds_capture"],
            "a45_top7": item.get("a45_top7"),
            "market_top7": item.get("market_top7"),
            "finish_order": finish_order,
            "official_trio_payouts": [
                {"numbers": list(numbers), "payout_yen": float(amount)}
                for numbers, amount in payout.get("trio", [])
            ],
            "models": models,
        })

    aggregate = {}
    for model_name, strategies in totals.items():
        aggregate[model_name] = {}
        for strategy_name, bucket in strategies.items():
            stake = bucket["stake_yen"]
            races = bucket["races"]
            aggregate[model_name][strategy_name] = {
                **bucket,
                "race_hit_rate_pct": round(100 * bucket["races_hit"] / races, 3) if races else None,
                "ticket_hit_rate_pct": round(100 * bucket["hit_tickets"] / bucket["tickets"], 3) if bucket["tickets"] else None,
                "roi_pct": round(100 * bucket["return_yen"] / stake, 3) if stake else None,
                "profit_yen": round(bucket["return_yen"] - stake, 2),
            }

    scored_races = len(joined) - len(pending)
    output = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model_version": "lambdarank-a45-research-v1",
        "status": "waiting_for_results" if pending else "scored",
        "predictions_with_odds": len(joined),
        "races_scored": scored_races,
        "races_pending": pending,
        "evaluation_policy": {
            "bet_type": "3連複",
            "stake_per_ticket_yen": 100,
            "prediction": "A45 frozen ranking captured before odds snapshot",
            "market_baseline": "Win-odds ranking from the same pre-off odds snapshot",
            "payout_source": "Official NAR daily result archive",
            "payouts_used_for_prediction": False,
            "sample_size_warning": "Three races are descriptive only; no profitability conclusion or confidence claim.",
        },
        "aggregate": aggregate,
        "races": per_race,
    }
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
    report = {
        "generated_at_utc": output["generated_at_utc"],
        "status": output["status"],
        "predictions_with_odds": len(joined),
        "races_scored": scored_races,
        "races_pending": pending,
        "output": str(OUT.relative_to(ROOT)),
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
    print("A45_TRIO_EVALUATION_REPORT " + json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
