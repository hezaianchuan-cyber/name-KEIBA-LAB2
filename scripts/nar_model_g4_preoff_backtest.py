#!/usr/bin/env python3
"""G4 prospective test using only odds snapshots captured before post time.

No official final odds are read. The probability/odds rule is fixed in advance
from the earlier G2 baseline (model/market ratio >= 1.20 and p*odds >= 1.08).
Each race contributes at most one snapshot per time bucket and one ticket.
"""
from __future__ import annotations

import datetime as dt
import gzip
import json
import math
import random
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = ROOT / "data" / "odds-snapshots"
PAYOUTS_FILE = ROOT / "data" / "daily-payouts.json"
G1_FILE = ROOT / "data" / "model-g1-calibration-backtest.json"
OUT = ROOT / "data" / "model-g4-preoff-backtest.json"

MINUTES_MIN = 2.0
MINUTES_MAX = 30.0
MIN_ODDS_COVERAGE = 0.70
MIN_PROBABILITY_RATIO = 1.20
MIN_EXPECTED_VALUE = 1.08
STAKE = 100
BOOTSTRAP_REPS = 2000

# Disjoint buckets work with the existing 5-minute collector and improve as
# the one-minute Supabase collector starts filling the T-2 bucket.
BUCKETS = {
    "T10": {"target": 10.0, "low": 8.0, "high": 15.0},
    "T5": {"target": 5.0, "low": 4.0, "high": 8.0},
    "T2": {"target": 2.5, "low": 2.0, "high": 4.0},
}


def norm_venue(value):
    return unicodedata.normalize("NFKC", str(value or "")).strip()


def race_key(value):
    parts = str(value or "").split("|")
    if len(parts) != 3:
        return None
    venue = norm_venue(parts[0])
    date = "".join(ch for ch in parts[1] if ch.isdigit())
    race = "".join(ch for ch in parts[2] if ch.isdigit())
    if not venue or len(date) != 8 or not race:
        return None
    return venue + "|" + date + "|" + str(int(race))


def parse_time(value):
    try:
        return dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def load_outcomes():
    if not PAYOUTS_FILE.exists():
        raise RuntimeError("data/daily-payouts.json is missing; official results are required.")
    source = json.loads(PAYOUTS_FILE.read_text(encoding="utf-8"))
    outcomes = {}
    for day in source.get("days", []):
        date = "".join(ch for ch in str(day.get("date") or "") if ch.isdigit())
        if len(date) != 8:
            continue
        for race in day.get("races", []):
            venue = norm_venue(race.get("venue"))
            number = race.get("race")
            if not venue or number is None:
                continue
            key = venue + "|" + date + "|" + str(int(number))
            win_rows = (race.get("payouts") or {}).get("win") or []
            winners = set()
            payout_by_horse = {}
            for item in win_rows:
                nums = item.get("numbers") or []
                if not nums:
                    continue
                no = str(nums[0])
                winners.add(no)
                payout_by_horse[no] = int(item.get("pay_100") or 0)
            if winners:
                outcomes[key] = {"winners": winners, "payout_by_horse": payout_by_horse}
    return outcomes


def load_snapshots():
    rows = []
    diagnostics = {
        "files": 0, "lines": 0, "invalid_json": 0, "outside_window": 0,
        "missing_predictions": 0, "missing_odds": 0, "missing_result": 0,
        "temperature_mismatch": 0, "valid_snapshots": 0,
    }
    for path in sorted(SNAPSHOT_DIR.glob("*.jsonl.gz")):
        diagnostics["files"] += 1
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                diagnostics["lines"] += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    diagnostics["invalid_json"] += 1
                    continue
                key = race_key(record.get("race_key"))
                minutes = record.get("minutes_to_post")
                try:
                    minutes = float(minutes)
                except (TypeError, ValueError):
                    continue
                if not MINUTES_MIN <= minutes <= MINUTES_MAX:
                    diagnostics["outside_window"] += 1
                    continue
                predictions = record.get("predictions") or []
                if len(predictions) < 3:
                    diagnostics["missing_predictions"] += 1
                    continue
                odds = record.get("odds") or []
                if not odds:
                    diagnostics["missing_odds"] += 1
                    continue
                if not key:
                    continue
                rows.append({
                    "race_key": key,
                    "date": key.split("|")[1],
                    "venue": key.split("|")[0],
                    "captured_at": record.get("captured_at"),
                    "scheduled_post_time": record.get("scheduled_post_time"),
                    "minutes_to_post": minutes,
                    "model_temperature": record.get("model_temperature"),
                    "predictions": predictions,
                    "odds": odds,
                })
    return rows, diagnostics


def build_race_evaluation(snapshot, outcome, expected_temperature):
    scored_by_no = {}
    for item in snapshot["predictions"]:
        no = str(item.get("horse_no") or "").strip()
        try:
            score = float(item.get("score"))
        except (TypeError, ValueError):
            continue
        if no and math.isfinite(score):
            scored_by_no[no] = score
    if len(scored_by_no) < 3:
        return None, "missing_predictions"

    # Re-apply the G1 temperature to the raw scores saved at capture time.
    # This keeps the historical odds timestamp intact while using the
    # calibrated probability scale validated by the untouched G1 holdout.
    peak = max(scored_by_no.values())
    weights = {
        no: math.exp(max(-700.0, (score - peak) / max(0.005, expected_temperature)))
        for no, score in scored_by_no.items()
    }
    weight_total = sum(weights.values()) or 1.0
    prediction_by_no = {no: value / weight_total for no, value in weights.items()}

    win_odds = {}
    for item in snapshot["odds"]:
        label = unicodedata.normalize("NFKC", str(item.get("type") or "")).replace(" ", "")
        if "単勝" not in label:
            continue
        no = str(item.get("n1") or "").strip()
        try:
            price = float(item.get("odds"))
        except (TypeError, ValueError):
            continue
        if no and price > 0 and math.isfinite(price):
            win_odds[no] = price

    matched = [no for no in prediction_by_no if no in win_odds]
    coverage = len(matched) / len(prediction_by_no)
    if coverage < MIN_ODDS_COVERAGE:
        return None, "insufficient_odds_coverage"

    inverse = {no: 1.0 / win_odds[no] for no in matched}
    total = sum(inverse.values())
    if total <= 0:
        return None, "invalid_market"
    market = {no: value / total for no, value in inverse.items()}

    model_top = max(prediction_by_no, key=lambda no: (prediction_by_no[no], -int(no)))
    market_top = min(matched, key=lambda no: (win_odds[no], int(no)))
    candidates = []
    for no in matched:
        p = prediction_by_no[no]
        ratio = p / market[no] if market[no] > 0 else 0.0
        ev = p * win_odds[no]
        if ratio >= MIN_PROBABILITY_RATIO and ev >= MIN_EXPECTED_VALUE:
            candidates.append((ev, ratio, p, no))
    candidates.sort(reverse=True)
    selected = candidates[0] if candidates else None

    winners = outcome["winners"]
    payout_by_horse = outcome["payout_by_horse"]

    def ticket(no, label):
        hit = no in winners
        return {
            "label": label, "horse_no": no, "stake": STAKE,
            "return": payout_by_horse.get(no, 0) if hit else 0,
            "hit": int(hit),
        }

    value_ticket = None
    if selected:
        ev, ratio, p, no = selected
        value_ticket = ticket(no, "G4_preoff_value")
        value_ticket.update({
            "model_probability": round(p, 8),
            "market_probability": round(market[no], 8),
            "captured_odds": round(win_odds[no], 4),
            "probability_ratio": round(ratio, 4),
            "expected_value": round(ev, 4),
        })

    return {
        "race_key": snapshot["race_key"],
        "date": snapshot["date"],
        "minutes_to_post": snapshot["minutes_to_post"],
        "captured_at": snapshot["captured_at"],
        "odds_coverage": round(coverage, 4),
        "model_top": ticket(model_top, "G1_model_top"),
        "market_top": ticket(market_top, "market_favorite"),
        "value_ticket": value_ticket,
    }, None


def roi_interval(rows):
    if len(rows) < 30:
        return {"method": "race-level bootstrap", "replicates": 0, "roi_95ci": None}
    rng = random.Random(20261104 + len(rows))
    values = []
    n = len(rows)
    for _ in range(BOOTSTRAP_REPS):
        sample = [rows[rng.randrange(n)] for _ in range(n)]
        stake = sum(x["stake"] for x in sample)
        if stake:
            values.append(100.0 * sum(x["return"] for x in sample) / stake)
    values.sort()
    return {
        "method": "race-level bootstrap percentile interval",
        "replicates": BOOTSTRAP_REPS,
        "roi_95ci": [
            round(values[int(0.025 * (len(values) - 1))], 4),
            round(values[int(0.975 * (len(values) - 1))], 4),
        ] if values else None,
    }


def summarize(rows):
    stake = sum(x["stake"] for x in rows)
    returned = sum(x["return"] for x in rows)
    hits = sum(x["hit"] for x in rows)
    return {
        "races_bet": len(rows),
        "hits": hits,
        "hit_rate_pct": round(100 * hits / len(rows), 2) if rows else None,
        "stake_jpy": stake,
        "return_jpy": returned,
        "profit_jpy": returned - stake,
        "roi_pct": round(100 * returned / stake, 2) if stake else None,
        "roi_95ci": roi_interval(rows),
    }


def choose_bucket(rows, name):
    config = BUCKETS[name]
    eligible = [r for r in rows if config["low"] <= r["minutes_to_post"] <= config["high"]]
    by_race = {}
    for row in eligible:
        key = row["race_key"]
        previous = by_race.get(key)
        rank = (abs(row["minutes_to_post"] - config["target"]),
                -((parse_time(row["captured_at"]) or dt.datetime.min.replace(tzinfo=dt.timezone.utc)).timestamp()))
        if previous is None or rank < previous[0]:
            by_race[key] = (rank, row)
    return [item[1] for item in by_race.values()]


def main():
    g1 = json.loads(G1_FILE.read_text(encoding="utf-8"))
    if g1.get("status") != "G1_improved_on_untouched_holdout":
        raise RuntimeError("G1 calibrated baseline is not validated; G4 will not use an unvalidated model.")
    temperature = float(g1["selected_temperature"])
    outcomes = load_outcomes()
    snapshots, diagnostics = load_snapshots()
    diagnostics["snapshots_with_official_result"] = 0
    diagnostics["snapshots_missing_official_result"] = 0
    diagnostics["recalibrated_with_g1_temperature"] = 0
    diagnostics["insufficient_odds_coverage"] = 0

    joined = []
    for snapshot in snapshots:
        outcome = outcomes.get(snapshot["race_key"])
        if not outcome:
            diagnostics["snapshots_missing_official_result"] += 1
            continue
        diagnostics["snapshots_with_official_result"] += 1
        evaluated, reason = build_race_evaluation(snapshot, outcome, temperature)
        if reason:
            diagnostics[reason] = diagnostics.get(reason, 0) + 1
            continue
        joined.append(evaluated)
        diagnostics["recalibrated_with_g1_temperature"] += 1
    diagnostics["valid_snapshots"] = len(joined)

    dates = sorted({row["date"] for row in joined})
    if len(dates) >= 2:
        split_index = max(1, min(len(dates) - 1, int(len(dates) * 0.70)))
        dev_end = dates[split_index - 1]
        holdout_start = dates[split_index]
    else:
        dev_end = dates[-1] if dates else None
        holdout_start = None

    buckets_report = {}
    for name in BUCKETS:
        selected = choose_bucket(joined, name)
        development = [r for r in selected if dev_end and r["date"] <= dev_end]
        holdout = [r for r in selected if holdout_start and r["date"] >= holdout_start]
        dev_value = [r["value_ticket"] for r in development if r.get("value_ticket")]
        hold_value = [r["value_ticket"] for r in holdout if r.get("value_ticket")]
        buckets_report[name] = {
            "minutes_to_post_range": [BUCKETS[name]["low"], BUCKETS[name]["high"]],
            "unique_races_with_snapshot": len(selected),
            "development_races": len(development),
            "holdout_races": len(holdout),
            "development": {
                "model_top": summarize([r["model_top"] for r in development]),
                "market_favorite": summarize([r["market_top"] for r in development]),
                "value_rule": summarize(dev_value),
                "value_rule_skip_rate_pct": round(100 * (len(development) - len(dev_value)) / len(development), 2) if development else None,
            },
            "holdout": {
                "model_top": summarize([r["model_top"] for r in holdout]),
                "market_favorite": summarize([r["market_top"] for r in holdout]),
                "value_rule": summarize(hold_value),
                "value_rule_skip_rate_pct": round(100 * (len(holdout) - len(hold_value)) / len(holdout), 2) if holdout else None,
            },
        }

    latest_by_race = {}
    for row in joined:
        previous = latest_by_race.get(row["race_key"])
        if previous is None or row["minutes_to_post"] < previous["minutes_to_post"]:
            latest_by_race[row["race_key"]] = row
    latest_rows = list(latest_by_race.values())
    latest_dates = sorted({row["date"] for row in latest_rows})
    if len(latest_dates) >= 2:
        split_index = max(1, min(len(latest_dates) - 1, int(len(latest_dates) * 0.70)))
        latest_dev_end = latest_dates[split_index - 1]
        latest_holdout_start = latest_dates[split_index]
    else:
        latest_dev_end = latest_dates[-1] if latest_dates else None
        latest_holdout_start = None
    latest_dev = [r for r in latest_rows if latest_dev_end and r["date"] <= latest_dev_end]
    latest_hold = [r for r in latest_rows if latest_holdout_start and r["date"] >= latest_holdout_start]

    value_dev = [r["value_ticket"] for r in latest_dev if r.get("value_ticket")]
    value_hold = [r["value_ticket"] for r in latest_hold if r.get("value_ticket")]
    hold_ci = roi_interval(value_hold).get("roi_95ci")
    enough = len(latest_hold) >= 100 and len(value_hold) >= 100
    g4_pass = bool(enough and hold_ci and hold_ci[0] > 100.0)

    report = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "feature": "G4_real_preoff_odds_evaluation",
        "status": "G4_passed" if g4_pass else (
            "insufficient_prospective_sample" if len(latest_rows) < 100 or len(value_hold) < 30
            else "evaluated_not_profitable_or_not_confirmed"
        ),
        "g1_temperature": temperature,
        "probability_source": "raw per-horse scores saved at capture time, recalibrated with the validated G1 temperature",
        "evaluation_period": [latest_dates[0], latest_dates[-1]] if latest_dates else None,
        "development_period": [latest_dates[0], latest_dev_end] if latest_dates and latest_dev_end else None,
        "holdout_period": [latest_holdout_start, latest_dates[-1]] if latest_holdout_start and latest_dates else None,
        "data_source": "KEIBA LAB timestamped official NAR pre-off odds snapshots only; official final odds are not read.",
        "pre_registered_rule": {
            "minimum_odds_coverage": MIN_ODDS_COVERAGE,
            "minimum_model_to_market_probability_ratio": MIN_PROBABILITY_RATIO,
            "minimum_model_probability_times_captured_odds": MIN_EXPECTED_VALUE,
            "selection": "At most one highest expected-value qualifying win horse per race; otherwise skip; JPY100 diagnostic stake.",
            "thresholds": "Fixed before G4 evaluation; no tuning on holdout.",
        },
        "diagnostics": diagnostics,
        "time_buckets": buckets_report,
        "closest_to_post": {
            "unique_races": len(latest_rows),
            "development_races": len(latest_dev),
            "holdout_races": len(latest_hold),
            "development": {
                "model_top": summarize([r["model_top"] for r in latest_dev]),
                "market_favorite": summarize([r["market_top"] for r in latest_dev]),
                "value_rule": summarize(value_dev),
                "skip_rate_pct": round(100 * (len(latest_dev) - len(value_dev)) / len(latest_dev), 2) if latest_dev else None,
            },
            "holdout": {
                "model_top": summarize([r["model_top"] for r in latest_hold]),
                "market_favorite": summarize([r["market_top"] for r in latest_hold]),
                "value_rule": summarize(value_hold),
                "skip_rate_pct": round(100 * (len(latest_hold) - len(value_hold)) / len(latest_hold), 2) if latest_hold else None,
            },
        },
        "g4_pass": g4_pass,
        "live_model_changed": False,
        "profitable_claim": g4_pass,
        "interpretation": "G4 uses only captured pre-off odds. Results remain exploratory until at least 100 unique holdout races and 100 qualifying value bets are available and the race-level bootstrap 95% ROI lower bound exceeds 100%. The 1-minute collector improves time-bucket coverage; it does not backfill earlier odds.",
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
