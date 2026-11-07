#!/usr/bin/env python3
"""G4 prospective test using only odds snapshots captured before post time.

No official final odds are read. The probability/odds rule is fixed in advance
from the earlier G2 baseline (model/market ratio >= 1.20 and p*odds >= 1.08).
Each race contributes at most one snapshot per time bucket and one ticket.
"""
from __future__ import annotations

import datetime as dt
import bisect
import gzip
import json
import math
import os
import random
import urllib.parse
import urllib.request
import unicodedata
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = ROOT / "data" / "odds-snapshots"
PAYOUTS_FILE = ROOT / "data" / "daily-payouts.json"
G1_FILE = ROOT / "data" / "model-g1-calibration-backtest.json"
OUT = ROOT / "data" / "model-g4-preoff-backtest.json"
SUPABASE_URL = os.environ.get("KEIBA_SUPABASE_URL", "https://mwiinzmnoludksnfnjaa.supabase.co").rstrip("/")
SUPABASE_KEY = os.environ.get("KEIBA_SUPABASE_KEY", "sb_publishable_uCeqBv6QPzHo6lHPBzkYiQ_i_dcuc_2")
MAX_MODEL_AGE_MINUTES = 15.0

MINUTES_MIN = 2.0
MINUTES_MAX = 60.0
MIN_ODDS_COVERAGE = 0.70
MIN_PROBABILITY_RATIO = 1.20
MIN_EXPECTED_VALUE = 1.08
STAKE = 100
BOOTSTRAP_REPS = 2000
BLEND_MARKET_WEIGHTS = (0.0, 0.25, 0.50, 0.75, 1.0)

# Disjoint buckets work with the existing 5-minute collector and improve as
# the one-minute Supabase collector starts filling the T-2 bucket.
BUCKETS = {
    "T10": {"target": 10.0, "low": 8.0, "high": 15.0},
    "T5": {"target": 5.0, "low": 4.0, "high": 8.0},
    "T2": {"target": 2.5, "low": 2.0, "high": 4.0},
}


def norm_venue(value):
    normalized = unicodedata.normalize("NFKC", str(value or "")).strip()
    # Repair venue names that were previously stored after a UTF-8/CP932
    # round-trip error in the one-minute Supabase collector.
    repairs = {
        "髢\u0080蛻・": "門別",
        "驥第イ「": "金沢",
    }
    return repairs.get(normalized, normalized)


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

def supabase_odds_capture_time(item):
    """Prefer the collector's source timestamp over the database insert timestamp."""
    odds_data = item.get("odds_data") or {}
    return parse_time(odds_data.get("captured_at")) or parse_time(item.get("captured_at"))


def supabase_odds_minutes_to_post(item, captured):
    """Derive T-minus from the same timestamp used to identify the odds snapshot."""
    odds_data = item.get("odds_data") or {}
    post = parse_time(odds_data.get("scheduled_post_time"))
    if post is not None and captured is not None:
        return (post - captured).total_seconds() / 60.0
    try:
        return float(odds_data.get("minutes_to_post"))
    except (TypeError, ValueError):
        return None


OUTCOME_DIAGNOSTICS = {}


def load_outcomes():
    """Load official payouts from the rolling archive and fill recent gaps from Supabase results."""
    if not PAYOUTS_FILE.exists():
        raise RuntimeError("data/daily-payouts.json is missing; official results are required.")
    source = json.loads(PAYOUTS_FILE.read_text(encoding="utf-8"))
    records = []
    for day in source.get("days", []):
        date = "".join(ch for ch in str(day.get("date") or "") if ch.isdigit())
        if len(date) != 8:
            continue
        for race in day.get("races", []):
            records.append((race.get("venue"), date, race.get("race"), race.get("payouts") or {}))

    # The rolling official payout archive can lag the current race day. Use the
    # immutable result table as a second official source for every snapshot date.
    snapshot_dates = {
        path.name.split(".")[0].replace("-", "")
        for path in SNAPSHOT_DIR.glob("*.jsonl.gz")
        if len(path.name.split(".")[0].replace("-", "")) == 8
    }
    result_rows = 0
    result_errors = []
    if SUPABASE_URL and SUPABASE_KEY:
        for date in sorted(snapshot_dates):
            date_iso = date[:4] + "-" + date[4:6] + "-" + date[6:8]
            offset = 0
            while True:
                query = urllib.parse.urlencode([
                    ("select", "race_id,race_date,venue,race_number,result_data"),
                    ("race_date", "eq." + date_iso),
                    ("order", "venue.asc,race_number.asc"),
                    ("limit", "1000"),
                    ("offset", str(offset)),
                ])
                request = urllib.request.Request(
                    SUPABASE_URL + "/rest/v1/nar_race_results?" + query,
                    headers={
                        "apikey": SUPABASE_KEY,
                        "Authorization": "Bearer " + SUPABASE_KEY,
                        "Accept": "application/json",
                        "User-Agent": "KEIBA-LAB-G4/1.0",
                    },
                )
                try:
                    with urllib.request.urlopen(request, timeout=45) as response:
                        page = json.loads(response.read().decode("utf-8"))
                    if not isinstance(page, list):
                        raise RuntimeError("nar_race_results response was not a JSON array")
                except Exception as exc:
                    result_errors.append(date_iso + ": " + str(exc))
                    break
                for row in page:
                    data = row.get("result_data") or {}
                    records.append((row.get("venue"), date, row.get("race_number"), data.get("payouts") or {}))
                    result_rows += 1
                if len(page) < 1000:
                    break
                offset += 1000

    outcomes = {}
    for venue_value, date, number, payouts in records:
        venue = norm_venue(venue_value)
        if not venue or number is None:
            continue
        key = venue + "|" + date + "|" + str(int(number))
        winners = set()
        payout_by_horse = {}
        for item in payouts.get("win") or []:
            nums = item.get("numbers") or []
            if nums:
                no = str(nums[0])
                winners.add(no)
                payout_by_horse[no] = int(item.get("pay_100") or 0)
        trio_payouts = {}
        for item in payouts.get("trio") or []:
            nums = []
            for value in item.get("numbers") or []:
                digits = "".join(ch for ch in str(value or "") if ch.isdigit())
                if not digits:
                    nums = []
                    break
                nums.append(str(int(digits)))
            if len(nums) != 3 or len(set(nums)) != 3:
                continue
            try:
                pay = int(item.get("pay_100") or 0)
            except (TypeError, ValueError):
                continue
            if pay > 0:
                combo = tuple(sorted(nums, key=int))
                trio_payouts["-".join(combo)] = max(pay, trio_payouts.get("-".join(combo), 0))
        if winners or trio_payouts:
            existing = outcomes.setdefault(key, {
                "winners": set(), "payout_by_horse": {}, "trio_payouts": {},
            })
            existing["winners"].update(winners)
            existing["payout_by_horse"].update(payout_by_horse)
            for combo, pay in trio_payouts.items():
                existing["trio_payouts"][combo] = max(pay, existing["trio_payouts"].get(combo, 0))

    OUTCOME_DIAGNOSTICS.clear()
    OUTCOME_DIAGNOSTICS.update({
        "supabase_result_rows": result_rows,
        "supabase_result_dates": sorted(snapshot_dates),
        "supabase_result_errors": result_errors,
        "official_outcome_races": len(outcomes),
    })
    return outcomes


def merge_model_snapshot_sources(github_rows, supabase_rows):
    """Merge snapshot sources by race and capture time; Supabase wins exact duplicates."""
    merged = {}
    for source_rows in (github_rows or [], supabase_rows or []):
        for source in source_rows:
            key = race_key(source.get("race_key") or source.get("race_id"))
            captured = parse_time(source.get("captured_at"))
            if not key or captured is None:
                continue
            row = dict(source)
            row["race_key"] = key
            date = "".join(ch for ch in str(row.get("date") or row.get("race_date") or "") if ch.isdigit())
            if date:
                row["date"] = date
            merged[(key, captured)] = row
    by_race = {}
    for (key, captured), row in merged.items():
        by_race.setdefault(key, []).append((captured, row))
    for values in by_race.values():
        values.sort(key=lambda item: item[0])
    return by_race


def load_supabase_snapshots(model_rows):
    # The GitHub gzip archive remains a safe fallback during rollout. Once the
    # immutable Supabase table is populated, use it as the authoritative model
    # snapshot source so the odds collector and evaluator share the same rows.
    models_by_race = merge_model_snapshot_sources(model_rows, [])
    github_model_keys = set(models_by_race)
    github_model_rows = sum(len(values) for values in models_by_race.values())

    diagnostics = {"supabase_query_rows": 0, "supabase_snapshots_joined": 0, "supabase_model_races": len(models_by_race), "supabase_race_key_matches": 0, "supabase_no_prior_model": 0, "supabase_model_too_old": 0, "supabase_empty_odds": 0, "supabase_model_dates": sorted({row.get("date") for row in model_rows if row.get("date")}), "supabase_model_key_samples": sorted(models_by_race.keys())[:8], "supabase_odds_key_samples": [], "supabase_odds_dates": [], "supabase_model_venues": sorted({key.split("|")[0] for key in models_by_race}), "supabase_odds_venues": [], "supabase_common_venues": [], "model_snapshot_table_rows": 0, "model_snapshot_table_status": "github_fallback"}
    if not SUPABASE_URL or not SUPABASE_KEY:
        diagnostics["supabase_status"] = "credentials_unavailable"
        return [], diagnostics

    dates = [str(row["date"]).replace("-", "") for row in model_rows if row.get("date")]
    today = dt.datetime.now(dt.timezone.utc).date().isoformat().replace("-", "")
    start_date = min(dates) if dates else today
    end_date = max(dates) if dates else today
    start_date = start_date[:4] + "-" + start_date[4:6] + "-" + start_date[6:8]
    end_date = end_date[:4] + "-" + end_date[4:6] + "-" + end_date[6:8]

    output = []
    offset = 0
    # Keep JSONB response pages small and match the composite
    # (race_date, captured_at DESC, id DESC) index to avoid API timeouts.
    page_size = 10
    try:
        model_table_rows = []
        model_offset = 0
        model_page_size = 100
        try:
            while True:
                model_query = urllib.parse.urlencode([
                    ("select", "race_id,race_date,captured_at,model_temperature,model_version,predictions,candidate_combinations"),
                    ("race_date", "gte." + start_date),
                    ("race_date", "lte." + end_date),
                    ("order", "race_date.asc,captured_at.desc"),
                    ("limit", str(model_page_size)),
                    ("offset", str(model_offset)),
                ])
                model_request = urllib.request.Request(
                    SUPABASE_URL + "/rest/v1/model_prediction_snapshots?" + model_query,
                    headers={
                        "apikey": SUPABASE_KEY,
                        "Authorization": "Bearer " + SUPABASE_KEY,
                        "Accept": "application/json",
                        "User-Agent": "KEIBA-LAB-G4/1.0",
                    },
                )
                with urllib.request.urlopen(model_request, timeout=30) as model_response:
                    model_page = json.loads(model_response.read().decode("utf-8"))
                if not isinstance(model_page, list):
                    raise RuntimeError("Supabase model_prediction_snapshots response was not a JSON array")
                model_table_rows.extend(model_page)
                if len(model_page) < model_page_size:
                    break
                model_offset += model_page_size
            supabase_model_rows = []
            for item in model_table_rows:
                key = race_key(item.get("race_id"))
                captured = parse_time(item.get("captured_at"))
                if not key or captured is None:
                    continue
                date = "".join(ch for ch in str(item.get("race_date") or "") if ch.isdigit())
                supabase_model_rows.append({
                    "race_key": key,
                    "date": date,
                    "captured_at": item.get("captured_at"),
                    "model_temperature": item.get("model_temperature"),
                    "model_version": item.get("model_version"),
                    "predictions": item.get("predictions") or [],
                    "candidate_combinations": item.get("candidate_combinations") or [],
                })
            # Supabase is authoritative for an identical (race, capture time),
            # but must not erase GitHub archive-only races. During rollout the
            # database can be only partially populated, including newcomer races.
            models_by_race = merge_model_snapshot_sources(model_rows, supabase_model_rows)
            supabase_model_keys = {
                key for key in (race_key(row.get("race_key")) for row in supabase_model_rows) if key
            }
            diagnostics["github_model_snapshot_rows"] = github_model_rows
            diagnostics["github_model_snapshot_races"] = len(github_model_keys)
            diagnostics["supabase_model_snapshot_rows"] = len(supabase_model_rows)
            diagnostics["supabase_model_snapshot_races"] = len(supabase_model_keys)
            diagnostics["github_fallback_only_races"] = len(github_model_keys - supabase_model_keys)
            diagnostics["merged_model_snapshot_rows"] = sum(len(values) for values in models_by_race.values())
            if model_table_rows:
                diagnostics["model_snapshot_table_status"] = "loaded_merged_with_github_fallback"
                diagnostics["model_snapshot_table_rows"] = len(model_table_rows)
            else:
                diagnostics["model_snapshot_table_status"] = "empty_using_github_fallback"
        except Exception as model_exc:
            diagnostics["model_snapshot_table_status"] = "unavailable_using_github_fallback"
            diagnostics["model_snapshot_table_error"] = str(model_exc)[:300]
        for values in models_by_race.values():
            values.sort(key=lambda item: item[0])
        # Precompute capture timestamps once. The former per-odds-row list
        # comprehension scanned every model snapshot; bisect makes the
        # latest-prior-model lookup logarithmic without changing selection.
        model_times_by_race = {
            key: [item[0] for item in values]
            for key, values in models_by_race.items()
        }
        diagnostics["supabase_model_races"] = len(models_by_race)
        diagnostics["supabase_model_key_samples"] = sorted(models_by_race.keys())[:8]
        diagnostics["supabase_model_venues"] = sorted({key.split("|")[0] for key in models_by_race})

        # Read one race_date at a time, but fetch independent dates concurrently.
        # JSONB odds_data is large (often 100-300KB per row), so keep each page
        # small and each day's pagination sequential. Four date workers reduce
        # network wait without creating an unbounded request burst or buffering
        # every raw JSONB page in memory.
        day_cursor = dt.date.fromisoformat(start_date)
        last_day = dt.date.fromisoformat(end_date)
        day_list = []
        while day_cursor <= last_day:
            day_list.append(day_cursor.isoformat())
            day_cursor += dt.timedelta(days=1)

        def fetch_and_join_day(day):
            day_output = []
            day_diagnostics = {
                "supabase_query_rows": 0,
                "supabase_snapshots_joined": 0,
                "supabase_race_key_matches": 0,
                "supabase_no_prior_model": 0,
                "supabase_model_too_old": 0,
                "supabase_empty_odds": 0,
                "supabase_odds_key_samples": [],
                "supabase_odds_dates": [],
                "supabase_odds_venues": [],
            }
            offset = 0
            while True:
                query = urllib.parse.urlencode([
                    ("select", "id,race_id,race_date,captured_at,odds_data"),
                    ("race_date", "eq." + day),
                    ("order", "captured_at.desc,id.desc"),
                    ("limit", str(page_size)),
                    ("offset", str(offset)),
                ])
                request = urllib.request.Request(
                    SUPABASE_URL + "/rest/v1/odds_snapshots?" + query,
                    headers={
                        "apikey": SUPABASE_KEY,
                        "Authorization": "Bearer " + SUPABASE_KEY,
                        "Accept": "application/json",
                        "User-Agent": "KEIBA-LAB-G4/1.0",
                    },
                )
                with urllib.request.urlopen(request, timeout=45) as response:
                    page = json.loads(response.read().decode("utf-8"))
                if not isinstance(page, list):
                    raise RuntimeError("Supabase odds_snapshots response was not a JSON array")
                day_diagnostics["supabase_query_rows"] += len(page)
                for item in page:
                    key = race_key(item.get("race_id"))
                    odds_data = item.get("odds_data") or {}
                    captured = supabase_odds_capture_time(item)
                    if not key or captured is None:
                        continue
                    if len(day_diagnostics["supabase_odds_key_samples"]) < 8 and key not in day_diagnostics["supabase_odds_key_samples"]:
                        day_diagnostics["supabase_odds_key_samples"].append(key)
                    race_date = key.split("|")[1]
                    if len(day_diagnostics["supabase_odds_dates"]) < 10 and race_date not in day_diagnostics["supabase_odds_dates"]:
                        day_diagnostics["supabase_odds_dates"].append(race_date)
                    venue = key.split("|")[0]
                    if venue not in day_diagnostics["supabase_odds_venues"]:
                        day_diagnostics["supabase_odds_venues"].append(venue)
                    model_candidates = models_by_race.get(key, [])
                    if model_candidates:
                        day_diagnostics["supabase_race_key_matches"] += 1
                    model_times = model_times_by_race.get(key, [])
                    prior_index = bisect.bisect_right(model_times, captured) - 1
                    if prior_index < 0:
                        day_diagnostics["supabase_no_prior_model"] += 1
                        continue
                    model_time, model = model_candidates[prior_index]
                    age_minutes = (captured - model_time).total_seconds() / 60.0
                    if age_minutes < 0 or age_minutes > MAX_MODEL_AGE_MINUTES:
                        day_diagnostics["supabase_model_too_old"] += 1
                        continue
                    odds = odds_data.get("odds") or []
                    if not odds:
                        day_diagnostics["supabase_empty_odds"] += 1
                        continue
                    minutes = supabase_odds_minutes_to_post(item, captured)
                    if minutes is None:
                        continue
                    day_output.append({
                        "race_key": key,
                        "date": race_date,
                        "venue": venue,
                        "captured_at": captured.isoformat(),
                        "scheduled_post_time": odds_data.get("scheduled_post_time"),
                        "minutes_to_post": minutes,
                        "model_temperature": model.get("model_temperature"),
                        "predictions": model.get("predictions") or [],
                        "candidate_combinations": model.get("candidate_combinations") or [],
                        "odds": odds,
                        "capture_source": "supabase_one_minute",
                        "prediction_age_minutes": round(age_minutes, 3),
                    })
                    day_diagnostics["supabase_snapshots_joined"] += 1
                if len(page) < page_size:
                    break
                offset += page_size
            return day_output, day_diagnostics

        from concurrent.futures import ThreadPoolExecutor
        worker_count = min(2, max(1, len(day_list)))
        diagnostics["supabase_odds_days_requested"] = len(day_list)
        diagnostics["supabase_odds_fetch_parallelism"] = worker_count
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            # executor.map preserves day order, keeping report generation stable.
            for day_output, day_diagnostics in executor.map(fetch_and_join_day, day_list):
                output.extend(day_output)
                for name in (
                    "supabase_query_rows", "supabase_snapshots_joined",
                    "supabase_race_key_matches", "supabase_no_prior_model",
                    "supabase_model_too_old", "supabase_empty_odds",
                ):
                    diagnostics[name] += day_diagnostics[name]
                for name in ("supabase_odds_key_samples", "supabase_odds_dates", "supabase_odds_venues"):
                    for value in day_diagnostics[name]:
                        if value not in diagnostics[name] and (
                            name != "supabase_odds_key_samples" or len(diagnostics[name]) < 8
                        ) and (
                            name != "supabase_odds_dates" or len(diagnostics[name]) < 10
                        ):
                            diagnostics[name].append(value)
        output.sort(key=lambda row: (row["captured_at"], row["race_key"]))
        diagnostics["supabase_status"] = "loaded"
    except Exception as exc:
        diagnostics["supabase_status"] = "unavailable"
        diagnostics["supabase_error"] = str(exc)[:300]
    diagnostics["supabase_common_venues"] = sorted(set(diagnostics["supabase_model_venues"]).intersection(diagnostics["supabase_odds_venues"]))
    return output, diagnostics


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
                    "candidate_combinations": record.get("candidate_combinations") or [],
                    "odds": odds,
                    "capture_source": "github_five_minute",
                    "prediction_age_minutes": 0.0,
                })
    supabase_rows, supabase_diagnostics = load_supabase_snapshots(rows)
    rows.extend(supabase_rows)
    diagnostics.update(supabase_diagnostics)
    diagnostics["snapshots_from_github_five_minute"] = sum(1 for row in rows if row.get("capture_source") == "github_five_minute")
    diagnostics["snapshots_from_supabase_one_minute"] = len(supabase_rows)
    return rows, diagnostics


def build_245_tickets(ranked):
    """Build unique 2-4-5 formation combinations from a ranked horse list."""
    unique = []
    seen = set()
    for value in ranked or []:
        digits = "".join(ch for ch in str(value or "") if ch.isdigit())
        if not digits:
            continue
        no = str(int(digits))
        if no not in seen:
            unique.append(no)
            seen.add(no)
    firsts, seconds, thirds = unique[:2], unique[:4], unique[2:7]
    tickets, seen_tickets = [], set()
    for first in firsts:
        for second in seconds:
            if second == first:
                continue
            for third in thirds:
                if third in (first, second):
                    continue
                combo = tuple(sorted((first, second, third), key=int))
                if combo not in seen_tickets:
                    seen_tickets.add(combo)
                    tickets.append(combo)
    return tickets


def narrow_trio_tickets(tickets, ranked, count=12):
    """Keep the lowest rank-sum combinations, with a deterministic tie-break."""
    rank = {str(no): i for i, no in enumerate(ranked or [], start=1)}
    ordered = sorted(
        tickets,
        key=lambda combo: (
            sum(rank.get(str(no), 10_000) for no in combo),
            tuple(int(no) for no in combo),
        ),
    )
    return ordered[:max(0, min(int(count), len(ordered)))]


def summarize_podium_coverage(rows):
    """Measure whether a ranking contains a complete official 3-horse podium."""
    def metric(items, hit_key, captured_key):
        hits = sum(int(row.get(hit_key, 0)) for row in items)
        captured = [int(row.get(captured_key, 0)) for row in items]
        return {
            "races": len(items),
            "full_podium_hits": hits,
            "full_podium_coverage_pct": round(100 * hits / len(items), 2) if items else None,
            "mean_podium_horses_captured": round(sum(captured) / len(captured), 3) if captured else None,
        }

    report = {
        "model": metric(rows, "model_top7_podium_hit", "model_top7_podium_captured"),
        "market": metric(rows, "market_top7_podium_hit", "market_top7_podium_captured"),
        "blend_candidates": {},
    }
    for weight in BLEND_MARKET_WEIGHTS:
        key = f"{weight:.2f}"
        items = [row.get("blend_candidates", {}).get(key, {}) for row in rows]
        hits = sum(int(item.get("top7_podium_hit", 0)) for item in items)
        captured = [int(item.get("top7_podium_captured", 0)) for item in items]
        report["blend_candidates"][key] = {
            "races": len(items),
            "full_podium_hits": hits,
            "full_podium_coverage_pct": round(100 * hits / len(items), 2) if items else None,
            "mean_podium_horses_captured": round(sum(captured) / len(captured), 3) if captured else None,
        }
    return report


def summarize_trio_ticket_strategies(rows, ticket_map_field="trio_ticket_candidates"):
    """Evaluate 3/6/9/12 ticket partial-buy choices against official trio payouts."""
    report = {}
    for count in (3, 6, 9, 12):
        race_count = hit_races = ticket_count = stake = returned = 0
        for row in rows:
            ticket_map = row.get(ticket_map_field) or {}
            tickets = ticket_map.get(str(count)) or []
            if not tickets:
                continue
            race_count += 1
            payouts = row.get("trio_payouts") or {}
            hit = False
            race_return = 0
            for combo in tickets:
                key = "-".join(str(no) for no in combo)
                pay = int(payouts.get(key, 0) or 0)
                if pay > 0:
                    hit = True
                    race_return += pay
            hit_races += int(hit)
            ticket_count += len(tickets)
            stake += len(tickets) * STAKE
            returned += race_return
        report[str(count)] = {
            "races": race_count,
            "hit_races": hit_races,
            "race_hit_rate_pct": round(100 * hit_races / race_count, 2) if race_count else None,
            "tickets": ticket_count,
            "stake_jpy": stake,
            "return_jpy": returned,
            "profit_jpy": returned - stake,
            "roi_pct": round(100 * returned / stake, 2) if stake else None,
            "average_tickets_per_race": round(ticket_count / race_count, 2) if race_count else None,
        }
    return report


def summarize_trio_ticket_strategies_by_blend(rows):
    """Compare the same 2-4-5 ticket budget across predeclared market weights."""
    return {
        f"{weight:.2f}": summarize_trio_ticket_strategies(
            rows, ticket_map_field="trio_ticket_candidates_by_blend_" + f"{weight:.2f}"
        )
        for weight in BLEND_MARKET_WEIGHTS
    }


def build_trio_combo_candidates(candidate_combinations, odds_rows):
    """Join saved trio probabilities to the exact captured pre-off trio odds."""
    def horse_no(value):
        digits = "".join(ch for ch in str(value or "") if ch.isdigit())
        return str(int(digits)) if digits else ""

    odds_by_combo = {}
    for item in odds_rows or []:
        label = unicodedata.normalize("NFKC", str(item.get("type") or "")).replace(" ", "")
        if label != "3連複":
            continue
        nums = [horse_no(item.get(key)) for key in ("n1", "n2", "n3")]
        if len(nums) != 3 or not all(nums) or len(set(nums)) != 3:
            continue
        try:
            price = float(item.get("odds"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(price) and price > 0:
            key = "-".join(sorted(nums, key=int))
            odds_by_combo[key] = price

    by_combo = {}
    for item in candidate_combinations or []:
        bet_type = unicodedata.normalize("NFKC", str(item.get("bet_type") or "")).strip().lower()
        if bet_type not in ("trio", "3連複"):
            continue
        nums = [horse_no(value) for value in (item.get("numbers") or [])]
        if len(nums) != 3 or not all(nums) or len(set(nums)) != 3:
            continue
        key = "-".join(sorted(nums, key=int))
        price = odds_by_combo.get(key)
        try:
            probability = float(item.get("probability"))
        except (TypeError, ValueError):
            continue
        if price is None or not math.isfinite(probability) or probability <= 0:
            continue
        expected_value = probability * price
        candidate = {
            "key": key,
            "numbers": list(sorted(nums, key=int)),
            "probability": probability,
            "captured_odds": price,
            "expected_value": expected_value,
        }
        previous = by_combo.get(key)
        if previous is None or candidate["probability"] > previous["probability"]:
            by_combo[key] = candidate
    return list(by_combo.values())


def rank_trio_combo_candidates(candidates):
    """Create probability, EV, and equal-weight rank-fusion orderings."""
    probability_order = sorted(
        candidates,
        key=lambda row: (-row["probability"], -row["expected_value"], row["key"]),
    )
    ev_order = sorted(
        candidates,
        key=lambda row: (-row["expected_value"], -row["probability"], row["key"]),
    )
    probability_rank = {row["key"]: rank for rank, row in enumerate(probability_order, 1)}
    ev_rank = {row["key"]: rank for rank, row in enumerate(ev_order, 1)}
    balanced_order = sorted(
        candidates,
        key=lambda row: (
            probability_rank[row["key"]] + ev_rank[row["key"]],
            probability_rank[row["key"]],
            ev_rank[row["key"]],
            -row["probability"],
            -row["expected_value"],
            row["key"],
        ),
    )
    return {
        "probability": [row["key"] for row in probability_order],
        "expected_value": [row["key"] for row in ev_order],
        "balanced_50_50": [row["key"] for row in balanced_order],
    }


def build_trio_candidate_ticket_sets(candidates):
    rankings = rank_trio_combo_candidates(candidates)
    return {
        strategy: {
            str(count): keys[:count]
            for count in (3, 6, 9, 12)
        }
        for strategy, keys in rankings.items()
    }


def summarize_trio_candidate_strategies(rows):
    """Evaluate partial buys using saved combo probabilities and captured trio odds."""
    report = {}
    for strategy in ("probability", "expected_value", "balanced_50_50"):
        report[strategy] = {}
        for count in (3, 6, 9, 12):
            race_count = hit_races = ticket_count = stake = returned = 0
            for row in rows:
                ticket_map = row.get("trio_candidate_ticket_candidates") or {}
                tickets = (ticket_map.get(strategy) or {}).get(str(count)) or []
                if not tickets:
                    continue
                race_count += 1
                payouts = row.get("trio_payouts") or {}
                hit = False
                race_return = 0
                for key in tickets:
                    pay = int(payouts.get(key, 0) or 0)
                    if pay > 0:
                        hit = True
                        race_return += pay
                hit_races += int(hit)
                ticket_count += len(tickets)
                stake += len(tickets) * STAKE
                returned += race_return
            report[strategy][str(count)] = {
                "races": race_count,
                "hit_races": hit_races,
                "race_hit_rate_pct": round(100 * hit_races / race_count, 2) if race_count else None,
                "tickets": ticket_count,
                "stake_jpy": stake,
                "return_jpy": returned,
                "profit_jpy": returned - stake,
                "roi_pct": round(100 * returned / stake, 2) if stake else None,
                "average_tickets_per_race": round(ticket_count / race_count, 2) if race_count else None,
            }
    return report


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

    # Fixed 50/50 blend: compare model and market on the same odds-covered field.
    model_total = sum(prediction_by_no[no] for no in matched)
    if model_total <= 0:
        return None, "invalid_model_probability"
    model_common = {no: prediction_by_no[no] / model_total for no in matched}
    blend_market_weight = 0.50
    blended = {no: 0.50 * model_common[no] + 0.50 * market[no] for no in matched}
    model_top = max(model_common, key=lambda no: (model_common[no], -int(no)))
    market_top = min(matched, key=lambda no: (win_odds[no], int(no)))
    blend_top = max(blended, key=lambda no: (blended[no], -int(no)))
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

    model_order = sorted(matched, key=lambda no: (-model_common[no], int(no)))
    market_order = sorted(matched, key=lambda no: (-market[no], int(no)))
    model_rank = {no: rank + 1 for rank, no in enumerate(model_order)}
    market_rank = {no: rank + 1 for rank, no in enumerate(market_order)}
    runner_diagnostics = [
        {
            "horse_no": no,
            "model_rank": model_rank[no],
            "market_rank": market_rank[no],
            "model_probability": round(model_common[no], 8),
            "market_probability": round(market[no], 8),
            "captured_odds": round(win_odds[no], 4),
            "is_winner": int(no in winners),
        }
        for no in matched
    ]

    trio_payouts = outcome.get("trio_payouts") or {}
    podium_combos = [tuple(key.split("-")) for key in trio_payouts]
    def podium_capture(order):
        top7 = set(order[:7])
        captured_counts = [len(top7.intersection(combo)) for combo in podium_combos]
        return (
            int(any(count == 3 for count in captured_counts)),
            max(captured_counts, default=0),
        )

    model_podium_hit, model_podium_captured = podium_capture(model_order)
    market_podium_hit, market_podium_captured = podium_capture(market_order)

    blend_candidates = {}
    for market_weight in BLEND_MARKET_WEIGHTS:
        candidate_probabilities = {
            no: (1.0 - market_weight) * model_common[no] + market_weight * market[no]
            for no in matched
        }
        candidate_order = sorted(
            matched, key=lambda no: (-candidate_probabilities[no], int(no))
        )
        candidate_top = ticket(candidate_order[0], f"blend_market_{market_weight:.2f}")
        podium_hit, podium_captured = podium_capture(candidate_order)
        blend_candidates[f"{market_weight:.2f}"] = {
            "top": candidate_top,
            "top3": candidate_order[:3],
            "top3_hit": int(bool(set(candidate_order[:3]).intersection(winners))),
            "top7_podium_hit": podium_hit,
            "top7_podium_captured": podium_captured,
        }

    blended_order = sorted(matched, key=lambda no: (-blended[no], int(no)))
    trio_ticket_candidates_by_blend = {}
    for market_weight in BLEND_MARKET_WEIGHTS:
        weight_key = f"{market_weight:.2f}"
        candidate_probabilities = {
            no: (1.0 - market_weight) * model_common[no] + market_weight * market[no]
            for no in matched
        }
        candidate_order = sorted(
            matched, key=lambda no: (-candidate_probabilities[no], int(no))
        )
        all_tickets = narrow_trio_tickets(
            build_245_tickets(candidate_order), candidate_order, count=1000
        )
        trio_ticket_candidates_by_blend[weight_key] = {
            str(count): [list(combo) for combo in all_tickets[:count]]
            for count in (3, 6, 9, 12)
        }
    trio_ticket_candidates = trio_ticket_candidates_by_blend.get("0.50", {})
    trio_combo_candidates = build_trio_combo_candidates(
        snapshot.get("candidate_combinations") or [], snapshot.get("odds") or []
    )
    trio_candidate_ticket_candidates = build_trio_candidate_ticket_sets(trio_combo_candidates)

    return {
        "race_key": snapshot["race_key"],
        "date": snapshot["date"],
        "minutes_to_post": snapshot["minutes_to_post"],
        "captured_at": snapshot["captured_at"],
        "odds_coverage": round(coverage, 4),
        "model_top": ticket(model_top, "G1_model_top"),
        "market_top": ticket(market_top, "market_favorite"),
        "blend_top": ticket(blend_top, "G4_model_market_blend"),
        "blend_market_weight": blend_market_weight,
        "blend_candidates": blend_candidates,
        "runner_diagnostics": runner_diagnostics,
        "model_market_top_agree": int(model_top == market_top),
        "model_top_market_rank": market_rank[model_top],
        "market_favorite_model_rank": model_rank[market_top],
        "model_top3_hit": int(bool(set(model_order[:3]).intersection(winners))),
        "market_top3_hit": int(bool(set(market_order[:3]).intersection(winners))),
        "model_top7_podium_hit": model_podium_hit,
        "model_top7_podium_captured": model_podium_captured,
        "market_top7_podium_hit": market_podium_hit,
        "market_top7_podium_captured": market_podium_captured,
        "trio_payouts": trio_payouts,
        "trio_ticket_candidates": trio_ticket_candidates,
        "trio_ticket_candidates_by_blend": trio_ticket_candidates_by_blend,
        **{
            "trio_ticket_candidates_by_blend_" + key: value
            for key, value in trio_ticket_candidates_by_blend.items()
        },
        "trio_combo_candidates": trio_combo_candidates,
        "trio_candidate_ticket_candidates": trio_candidate_ticket_candidates,
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


def summarize_blend_candidates(rows):
    """Compare predeclared model/market probability blends on identical races."""
    report = {}
    for weight in BLEND_MARKET_WEIGHTS:
        key = f"{weight:.2f}"
        candidates = [
            row.get("blend_candidates", {}).get(key)
            for row in rows
            if row.get("blend_candidates", {}).get(key)
        ]
        top_rows = [item["top"] for item in candidates]
        report[key] = {
            "market_probability_weight_pct": round(weight * 100, 1),
            "races": len(candidates),
            "top1": summarize(top_rows),
            "top3_coverage_hits": sum(item["top3_hit"] for item in candidates),
            "top3_coverage_pct": round(
                100 * sum(item["top3_hit"] for item in candidates) / len(candidates), 2
            ) if candidates else None,
        }
    return report


def summarize_rank_disagreement(rows):
    """Describe horses whose model rank sharply disagrees with the market."""
    runners = [
        runner
        for row in rows
        for runner in row.get("runner_diagnostics", [])
    ]
    groups = {
        "model_ranks_5plus_above_market": [
            (row, item) for row in rows for item in row.get("runner_diagnostics", [])
            if item["market_rank"] - item["model_rank"] >= 5
        ],
        "market_ranks_5plus_above_model": [
            (row, item) for row in rows for item in row.get("runner_diagnostics", [])
            if item["model_rank"] - item["market_rank"] >= 5
        ],
    }
    result = {}
    for name, pairs in groups.items():
        items = [item for _, item in pairs]
        result[name] = {
            "horses": len(items),
            "races": len({row["race_key"] for row, _ in pairs}),
            "winners": sum(item["is_winner"] for item in items),
            "horse_level_win_rate_pct": round(
                100 * sum(item["is_winner"] for item in items) / len(items), 2
            ) if items else None,
        }
    return result


def summarize_market_edge_conditions(rows):
    """Find race contexts where the model narrows or widens the market gap."""
    def metric(items):
        blend = {}
        for key in ("0.25", "0.50", "0.75"):
            tickets = [
                row.get("blend_candidates", {}).get(key, {}).get("top")
                for row in items
                if row.get("blend_candidates", {}).get(key, {}).get("top")
            ]
            blend[key] = summarize(tickets)
        return {
            "races": len(items),
            "model_top1_hit_rate_pct": summarize([row["model_top"] for row in items])["hit_rate_pct"],
            "market_favorite_hit_rate_pct": summarize([row["market_top"] for row in items])["hit_rate_pct"],
            "model_top3_coverage_pct": round(100 * sum(row.get("model_top3_hit", 0) for row in items) / len(items), 2) if items else None,
            "market_top3_coverage_pct": round(100 * sum(row.get("market_top3_hit", 0) for row in items) / len(items), 2) if items else None,
            "blend_top1": blend,
        }

    groups = {
        "model_agrees_with_market_favorite": [row for row in rows if row.get("model_market_top_agree") == 1],
        "model_disagrees_with_market_favorite": [row for row in rows if row.get("model_market_top_agree") == 0],
        "model_top_is_market_rank_1_to_3": [row for row in rows if 1 <= row.get("model_top_market_rank", 0) <= 3],
        "model_top_is_market_rank_4_to_6": [row for row in rows if 4 <= row.get("model_top_market_rank", 0) <= 6],
        "model_top_is_market_rank_7_plus": [row for row in rows if row.get("model_top_market_rank", 0) >= 7],
        "market_favorite_is_model_rank_1_to_3": [row for row in rows if 1 <= row.get("market_favorite_model_rank", 0) <= 3],
        "market_favorite_is_model_rank_4_to_6": [row for row in rows if 4 <= row.get("market_favorite_model_rank", 0) <= 6],
        "market_favorite_is_model_rank_7_plus": [row for row in rows if row.get("market_favorite_model_rank", 0) >= 7],
    }
    return {name: metric(items) for name, items in groups.items()}


def summarize_bucket_movement(joined):
    """Compare captured market ranks/odds from T10 to T5 to T2."""
    selected = {name: choose_bucket(joined, name) for name in BUCKETS}

    def compare(left_name, right_name):
        left = {row["race_key"]: row for row in selected[left_name]}
        right = {row["race_key"]: row for row in selected[right_name]}
        common = sorted(set(left).intersection(right))
        rank_changes = []
        odds_changes = []
        winner_odds_changes = []
        nonwinner_odds_changes = []
        favorite_changed = 0
        for race_key in common:
            left_row, right_row = left[race_key], right[race_key]
            left_runners = {
                item["horse_no"]: item
                for item in left_row.get("runner_diagnostics", [])
            }
            right_runners = {
                item["horse_no"]: item
                for item in right_row.get("runner_diagnostics", [])
            }
            left_favorite = min(left_runners.values(), key=lambda x: x["market_rank"]) if left_runners else None
            right_favorite = min(right_runners.values(), key=lambda x: x["market_rank"]) if right_runners else None
            if left_favorite and right_favorite and left_favorite["horse_no"] != right_favorite["horse_no"]:
                favorite_changed += 1
            for no in set(left_runners).intersection(right_runners):
                a, b = left_runners[no], right_runners[no]
                rank_changes.append(abs(b["market_rank"] - a["market_rank"]))
                if a["captured_odds"] > 0:
                    change = 100 * (b["captured_odds"] / a["captured_odds"] - 1)
                    odds_changes.append(change)
                    if a["is_winner"] or b["is_winner"]:
                        winner_odds_changes.append(change)
                    else:
                        nonwinner_odds_changes.append(change)
        return {
            "matched_races": len(common),
            "shared_horses": len(rank_changes),
            "favorite_changed_races": favorite_changed,
            "favorite_changed_pct": round(100 * favorite_changed / len(common), 2) if common else None,
            "mean_absolute_market_rank_change": round(sum(rank_changes) / len(rank_changes), 3) if rank_changes else None,
            "horses_with_shortened_odds_pct": round(100 * sum(value < 0 for value in odds_changes) / len(odds_changes), 2) if odds_changes else None,
            "winner_horses_with_shortened_odds": sum(value < 0 for value in winner_odds_changes),
            "winner_horses_compared": len(winner_odds_changes),
            "winner_odds_shortened_pct": round(100 * sum(value < 0 for value in winner_odds_changes) / len(winner_odds_changes), 2) if winner_odds_changes else None,
            "nonwinner_horses_with_shortened_odds": sum(value < 0 for value in nonwinner_odds_changes),
            "nonwinner_horses_compared": len(nonwinner_odds_changes),
            "nonwinner_odds_shortened_pct": round(100 * sum(value < 0 for value in nonwinner_odds_changes) / len(nonwinner_odds_changes), 2) if nonwinner_odds_changes else None,
            "median_winner_odds_change_pct": round(sorted(winner_odds_changes)[len(winner_odds_changes) // 2], 2) if winner_odds_changes else None,
        }

    return {
        "selected_races_by_bucket": {name: len(rows) for name, rows in selected.items()},
        "T10_to_T5": compare("T10", "T5"),
        "T5_to_T2": compare("T5", "T2"),
        "T10_to_T2": compare("T10", "T2"),
        "note": "Odds change is (later captured odds / earlier captured odds - 1); negative means odds shortened. Only horses present in both snapshots are compared.",
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
    run_started = time.perf_counter()
    stage_timings = {}
    stage_started = time.perf_counter()
    g1 = json.loads(G1_FILE.read_text(encoding="utf-8"))
    if g1.get("status") != "G1_improved_on_untouched_holdout":
        raise RuntimeError("G1 calibrated baseline is not validated; G4 will not use an unvalidated model.")
    temperature = float(g1["selected_temperature"])
    stage_timings["g1_load_seconds"] = round(time.perf_counter() - stage_started, 3)
    stage_started = time.perf_counter()
    outcomes = load_outcomes()
    stage_timings["official_outcomes_load_seconds"] = round(time.perf_counter() - stage_started, 3)
    stage_started = time.perf_counter()
    snapshots, diagnostics = load_snapshots()
    stage_timings["snapshots_load_and_supabase_join_seconds"] = round(time.perf_counter() - stage_started, 3)
    diagnostics.update(OUTCOME_DIAGNOSTICS)
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
    stage_timings["snapshot_outcome_evaluation_seconds"] = round(time.perf_counter() - stage_started, 3)
    stage_started = time.perf_counter()

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
                "fixed_50_50_model_market_blend": summarize([r["blend_top"] for r in development]),
                "blend_candidate_comparison": summarize_blend_candidates(development),
                "rank_disagreement": summarize_rank_disagreement(development),
                "value_rule": summarize(dev_value),
                "value_rule_skip_rate_pct": round(100 * (len(development) - len(dev_value)) / len(development), 2) if development else None,
            },
            "holdout": {
                "model_top": summarize([r["model_top"] for r in holdout]),
                "market_favorite": summarize([r["market_top"] for r in holdout]),
                "fixed_50_50_model_market_blend": summarize([r["blend_top"] for r in holdout]),
                "blend_candidate_comparison": summarize_blend_candidates(holdout),
                "rank_disagreement": summarize_rank_disagreement(holdout),
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
    daily_rows = {}
    for row in latest_rows:
        daily_rows.setdefault(row["date"], []).append(row)
    daily_results = []
    for date in sorted(daily_rows):
        day_rows = daily_rows[date]
        day_value = [row["value_ticket"] for row in day_rows if row.get("value_ticket")]
        daily_results.append({
            "date": date,
            "unique_races": len(day_rows),
            "model_top": summarize([row["model_top"] for row in day_rows]),
            "market_favorite": summarize([row["market_top"] for row in day_rows]),
            "fixed_50_50_model_market_blend": summarize([row["blend_top"] for row in day_rows]),
            "blend_candidate_comparison": summarize_blend_candidates(day_rows),
            "value_rule": summarize(day_value),
            "value_rule_bets": len(day_value),
            "note": "One closest-to-post captured pre-off snapshot per race; official result joined after the race.",
        })
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
    trio_rows = [row for row in latest_rows if row.get("trio_payouts")]
    trio_dev = [row for row in trio_rows if latest_dev_end and row["date"] <= latest_dev_end]
    trio_hold = [row for row in trio_rows if latest_holdout_start and row["date"] >= latest_holdout_start]
    trio_evaluation = {
        "races_with_official_trio_result": len(trio_rows),
        "candidate_combo_coverage": {
            "races_with_matched_candidate_odds": sum(bool(row.get("trio_combo_candidates")) for row in trio_rows),
            "mean_matched_candidates_per_race": round(
                sum(len(row.get("trio_combo_candidates") or []) for row in trio_rows) / len(trio_rows), 2
            ) if trio_rows else None,
        },
        "development": {
            "races": len(trio_dev),
            "top7_podium_coverage": summarize_podium_coverage(trio_dev),
            "ticket_strategies": summarize_trio_ticket_strategies(trio_dev),
            "ticket_strategies_by_blend": summarize_trio_ticket_strategies_by_blend(trio_dev),
            "candidate_combo_strategies": summarize_trio_candidate_strategies(trio_dev),
        },
        "holdout": {
            "races": len(trio_hold),
            "top7_podium_coverage": summarize_podium_coverage(trio_hold),
            "ticket_strategies": summarize_trio_ticket_strategies(trio_hold),
            "ticket_strategies_by_blend": summarize_trio_ticket_strategies_by_blend(trio_hold),
            "candidate_combo_strategies": summarize_trio_candidate_strategies(trio_hold),
        },
        "method": "Horse-rank 2-4-5 baseline plus saved trio-combination probabilities joined to the exact selected pre-off 3連複 odds. Combo EV is recalculated as saved probability × captured odds. Partial buys of top 3/6/9/12 are compared by probability rank, EV rank, and 50:50 rank fusion; official NAR trio payouts are outcomes only.",
        "ticket_limit": 12,
    }
    enough = len(latest_hold) >= 100 and len(value_hold) >= 100
    g4_pass = bool(enough and hold_ci and hold_ci[0] > 100.0)

    stage_timings["report_aggregation_seconds"] = round(time.perf_counter() - stage_started, 3)
    stage_timings["total_seconds_before_write"] = round(time.perf_counter() - run_started, 3)
    report = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "performance": {
            "stage_seconds": stage_timings,
            "total_seconds": round(time.perf_counter() - run_started, 3),
            "optimization": "bisect_right latest-prior model lookup plus bounded two-worker per-day odds retrieval with 10-row JSONB pages; outputs and selection rules unchanged",
        },
        "feature": "G4_real_preoff_odds_evaluation",
        "status": "G4_passed" if g4_pass else (
            "insufficient_prospective_sample" if len(latest_rows) < 100 or len(value_hold) < 30
            else "evaluated_not_profitable_or_not_confirmed"
        ),
        "g1_temperature": temperature,
        "probability_source": "raw per-horse scores saved at capture time, recalibrated with the validated G1 temperature",
        "evaluation_period": [latest_dates[0], latest_dates[-1]] if latest_dates else None,
        "daily_results": daily_results,
        "development_period": [latest_dates[0], latest_dev_end] if latest_dates and latest_dev_end else None,
        "holdout_period": [latest_holdout_start, latest_dates[-1]] if latest_holdout_start and latest_dates else None,
        "data_source": "KEIBA LAB timestamped official NAR pre-off odds snapshots from five-minute GitHub capture and one-minute Supabase capture; official final odds are not read.",
        "pre_registered_rule": {
            "blend": "Fixed 50% model probability + 50% normalized captured-market probability; same odds-covered runners; no holdout tuning.",
            "minimum_odds_coverage": MIN_ODDS_COVERAGE,
            "minimum_model_to_market_probability_ratio": MIN_PROBABILITY_RATIO,
            "minimum_model_probability_times_captured_odds": MIN_EXPECTED_VALUE,
            "selection": "At most one highest expected-value qualifying win horse per race; otherwise skip; JPY100 diagnostic stake.",
            "thresholds": "Fixed before G4 evaluation; no tuning on holdout.",
        },
        "diagnostics": diagnostics,
        "time_buckets": buckets_report,
        "market_rank_disagreement": {
            "development": summarize_rank_disagreement(latest_dev),
            "holdout": summarize_rank_disagreement(latest_hold),
        },
        "market_edge_conditions": {
            "development": summarize_market_edge_conditions(latest_dev),
            "holdout": summarize_market_edge_conditions(latest_hold),
        },
        "time_bucket_market_movement": summarize_bucket_movement(joined),
        "trio_evaluation": trio_evaluation,
        "closest_to_post": {
            "unique_races": len(latest_rows),
            "development_races": len(latest_dev),
            "holdout_races": len(latest_hold),
            "development": {
                "model_top": summarize([r["model_top"] for r in latest_dev]),
                "market_favorite": summarize([r["market_top"] for r in latest_dev]),
                "fixed_50_50_model_market_blend": summarize([r["blend_top"] for r in latest_dev]),
                "blend_candidate_comparison": summarize_blend_candidates(latest_dev),
                "rank_disagreement": summarize_rank_disagreement(latest_dev),
                "value_rule": summarize(value_dev),
                "skip_rate_pct": round(100 * (len(latest_dev) - len(value_dev)) / len(latest_dev), 2) if latest_dev else None,
            },
            "holdout": {
                "model_top": summarize([r["model_top"] for r in latest_hold]),
                "market_favorite": summarize([r["market_top"] for r in latest_hold]),
                "fixed_50_50_model_market_blend": summarize([r["blend_top"] for r in latest_hold]),
                "blend_candidate_comparison": summarize_blend_candidates(latest_hold),
                "rank_disagreement": summarize_rank_disagreement(latest_hold),
                "value_rule": summarize(value_hold),
                "skip_rate_pct": round(100 * (len(latest_hold) - len(value_hold)) / len(latest_hold), 2) if latest_hold else None,
            },
        },
        "g4_pass": g4_pass,
        "live_model_changed": False,
        "profitable_claim": g4_pass,
        "interpretation": "G4 joins one-minute Supabase odds to the most recent prior model-score snapshot (maximum age 15 minutes) and also retains five-minute combined snapshots. It uses only captured pre-off odds. Results remain exploratory until at least 100 unique holdout races and 100 qualifying value bets are available and the race-level bootstrap 95% ROI lower bound exceeds 100%. The 1-minute collector improves time-bucket coverage; it does not backfill earlier odds.",
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
