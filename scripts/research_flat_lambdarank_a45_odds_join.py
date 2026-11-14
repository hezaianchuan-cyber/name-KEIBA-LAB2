#!/usr/bin/env python3
"""Join frozen A45 predictions to strictly later, pre-off 3連複 odds snapshots.

This is a research-only join. It never changes predictions or model inputs and
never reads payouts/results. It selects the latest valid odds snapshot after
the A45 prediction timestamp and before scheduled post time.
"""
from __future__ import annotations

import datetime as dt
import gzip
import itertools
import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "data" / "research-a45-predictions.jsonl"
ODDS_DIR = ROOT / "data" / "research-a45-odds-source"
OUT = ROOT / "data" / "research-a45-odds-joined.jsonl"
REPORT = ROOT / "data" / "research-a45-odds-join-report.json"


def normalize_no(value):
    digits = re.sub(r"\\D", "", str(value or ""))
    return str(int(digits)) if digits else None


def normalize_trio_type(value):
    label = unicodedata.normalize("NFKC", str(value or "")).replace(" ", "")
    return label in {"3連複", "三連複"}


def load_predictions():
    if not PREDICTIONS.is_file():
        raise RuntimeError("A45 fixed prediction file is missing")
    records = []
    for line in PREDICTIONS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("model_version") != "lambdarank-a45-research-v1":
            continue
        records.append(item)
    return records


def load_odds_snapshots(date_text):
    path = ODDS_DIR / f"{date_text}.jsonl.gz"
    if not path.is_file():
        return []
    snapshots = []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                snapshots.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return snapshots


def select_snapshot(prediction, snapshots):
    predicted_at = dt.datetime.fromisoformat(prediction["captured_at"])
    post = dt.datetime.fromisoformat(prediction["scheduled_post_time"])
    candidates = []
    for snap in snapshots:
        if snap.get("race_key") != prediction.get("race_id"):
            continue
        try:
            captured = dt.datetime.fromisoformat(snap["captured_at"])
        except (KeyError, TypeError, ValueError):
            continue
        if captured < predicted_at or captured >= post:
            continue
        minutes_to_post = (post - captured).total_seconds() / 60.0
        if not (2.0 <= minutes_to_post <= 60.0):
            continue
        candidates.append((captured, minutes_to_post, snap))
    return max(candidates, key=lambda x: x[0]) if candidates else None


def trio_odds_lookup(snapshot):
    lookup = {}
    for item in snapshot.get("odds") or []:
        if not normalize_trio_type(item.get("type")):
            continue
        numbers = [normalize_no(item.get(k)) for k in ("n1", "n2", "n3")]
        if any(x is None for x in numbers) or len(set(numbers)) != 3:
            continue
        try:
            value = float(item.get("odds"))
        except (TypeError, ValueError):
            continue
        if value <= 0:
            continue
        lookup[tuple(sorted(numbers))] = value
    return lookup


def build_strategies(ranked_numbers, odds_lookup):
    axis = ranked_numbers[0]
    strategies = {
        "top3_single": [tuple(sorted(ranked_numbers[:3]))],
        "axis_plus4": [
            tuple(sorted((axis, a, b)))
            for a, b in itertools.combinations(ranked_numbers[1:5], 2)
        ],
        "axis_plus5": [
            tuple(sorted((axis, a, b)))
            for a, b in itertools.combinations(ranked_numbers[1:6], 2)
        ],
        "top5_box": [
            tuple(sorted(combo))
            for combo in itertools.combinations(ranked_numbers[:5], 3)
        ],
    }
    output = {}
    for name, tickets in strategies.items():
        quotes = [
            {"numbers": list(combo), "odds": odds_lookup.get(combo)}
            for combo in tickets
        ]
        quoted = sum(1 for item in quotes if item["odds"] is not None)
        output[name] = {
            "ticket_count": len(tickets),
            "stake_yen": len(tickets) * 100,
            "quoted_ticket_count": quoted,
            "missing_ticket_count": len(tickets) - quoted,
            "all_tickets_quoted": quoted == len(tickets),
            "tickets": quotes,
        }
    return output


def main():
    predictions = load_predictions()
    by_date = {}
    output = []
    no_snapshot = []
    for prediction in predictions:
        date_raw = str(prediction["race_date"])\n        date_text = dt.datetime.strptime(date_raw, "%Y%m%d").date().isoformat()
        if date_text not in by_date:
            by_date[date_text] = load_odds_snapshots(date_text)
        selected = select_snapshot(prediction, by_date[date_text])
        if selected is None:
            no_snapshot.append(prediction["race_id"])
            output.append({
                "race_id": prediction["race_id"],
                "model_version": prediction["model_version"],
                "prediction_captured_at": prediction["captured_at"],
                "scheduled_post_time": prediction["scheduled_post_time"],
                "odds_status": "no_valid_snapshot_after_prediction",
                "strategies": {},
            })
            continue

        odds_time, minutes_to_post, snapshot = selected
        odds_lookup = trio_odds_lookup(snapshot)
        ranked = sorted(prediction["predictions"], key=lambda x: int(x["rank"]))
        ranked_numbers = [normalize_no(x["horse_no"]) for x in ranked]
        strategies = build_strategies(ranked_numbers, odds_lookup)
        trio_count = len(odds_lookup)
        output.append({
            "race_id": prediction["race_id"],
            "model_version": prediction["model_version"],
            "prediction_captured_at": prediction["captured_at"],
            "odds_snapshot_captured_at": odds_time.isoformat(),
            "odds_snapshot_age_minutes": round((odds_time - dt.datetime.fromisoformat(prediction["captured_at"])).total_seconds() / 60.0, 3),
            "scheduled_post_time": prediction["scheduled_post_time"],
            "minutes_to_post_at_odds_capture": round(minutes_to_post, 3),
            "odds_status": "joined",
            "odds_source": f"main/data/odds-snapshots/{date_text}.jsonl.gz",
            "trio_combinations_quoted": trio_count,
            "a45_top7": ranked_numbers[:7],
            "strategies": strategies,
            "results_used": False,
            "payouts_used": False,
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        "".join(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n" for item in output),
        encoding="utf-8",
    )
    joined = sum(1 for item in output if item["odds_status"] == "joined")
    report = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model_version": "lambdarank-a45-research-v1",
        "predictions": len(predictions),
        "joined_races": joined,
        "no_valid_odds_snapshot": no_snapshot,
        "odds_policy": "Latest 3連複 snapshot captured after the fixed A45 prediction and 2-60 minutes before scheduled post.",
        "stake_assumption_yen_per_ticket": 100,
        "results_or_payouts_used": False,
        "output": str(OUT.relative_to(ROOT)),
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("A45_ODDS_JOIN_REPORT " + json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
