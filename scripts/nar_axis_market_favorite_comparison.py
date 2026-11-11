#!/usr/bin/env python3
"""Same-race comparison of the fixed LAB axis and pre-off market favorite.

Uses the closest valid 30-to-2-minute pre-off odds snapshot and official NAR
finish results. The market favorite is identified from the same snapshot's
win-odds popularity (lowest win odds as a fallback). Offline research only.
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import nar_backtest as nb
from nar_axis_pair_walkforward_backtest import collect_fixed_axes

OUT = ROOT / "data" / "axis-market-favorite-comparison.json"
BOOTSTRAP_REPS = 2000


def numeric(value):
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def market_favorite(snapshot):
    win_rows = []
    for item in snapshot.get("odds") or []:
        if "単勝" not in str(item.get("type") or ""):
            continue
        no = re.sub(r"\\D", "", str(item.get("n1") or ""))
        odds = numeric(item.get("odds"))
        if not no or odds is None or odds <= 0:
            continue
        popularity = numeric(item.get("popularity"))
        win_rows.append((str(int(no)), odds, popularity))
    if not win_rows:
        return None, "missing"
    ranked_by_popularity = [row for row in win_rows if row[2] is not None and row[2] > 0]
    if ranked_by_popularity:
        return min(ranked_by_popularity, key=lambda row: (row[2], row[1], int(row[0])))[0], "snapshot_popularity"
    return min(win_rows, key=lambda row: (row[1], int(row[0])))[0], "lowest_win_odds_fallback"


def distance_band(value):
    distance = numeric(value)
    if distance is None:
        return "unknown"
    if distance <= 1200:
        return "<=1200m"
    if distance <= 1600:
        return "1300-1600m"
    if distance <= 2000:
        return "1700-2000m"
    return ">2000m"


def field_band(value):
    size = numeric(value)
    if size is None:
        return "unknown"
    if size <= 8:
        return "<=8"
    if size <= 12:
        return "9-12"
    return ">=13"


def summarize(rows):
    n = len(rows)
    if not n:
        return {"races": 0}
    axis_hits = sum(row["axis_top3"] for row in rows)
    market_hits = sum(row["market_top3"] for row in rows)
    podium_total = sum(len(row["podium"]) for row in rows)
    covered_horses = sum(len(row["podium"] & row["top7"]) for row in rows)
    full_top7 = sum(len(row["podium"]) == 3 and row["podium"].issubset(row["top7"]) for row in rows)
    both = sum(row["axis_top3"] and row["market_top3"] for row in rows)
    neither = sum(not row["axis_top3"] and not row["market_top3"] for row in rows)
    rng = random.Random(20261111)
    deltas = []
    if n >= 30:
        for _ in range(BOOTSTRAP_REPS):
            sample = [rows[rng.randrange(n)] for _ in rows]
            deltas.append(100 * sum(int(x["axis_top3"]) - int(x["market_top3"]) for x in sample) / n)
        deltas.sort()
    return {
        "races": n,
        "lab_axis_top3": axis_hits,
        "lab_axis_top3_rate_pct": round(100 * axis_hits / n, 2),
        "market_favorite_top3": market_hits,
        "market_favorite_top3_rate_pct": round(100 * market_hits / n, 2),
        "axis_minus_market_top3_pp": round(100 * (axis_hits - market_hits) / n, 2),
        "paired_bootstrap_95ci_axis_minus_market_pp": [
            round(deltas[int(0.025 * (len(deltas)-1))], 2),
            round(deltas[int(0.975 * (len(deltas)-1))], 2),
        ] if deltas else None,
        "both_top3": both,
        "neither_top3": neither,
        "top7_podium_horse_coverage_pct": round(100 * covered_horses / podium_total, 2) if podium_total else None,
        "top7_full_podium_races": full_top7,
        "top7_full_podium_rate_pct": round(100 * full_top7 / n, 2),
        "podium_horses": podium_total,
    }


def main():
    # Snapshot collection can extend into the current month, while the canonical
    # archive backtest intentionally stops at the previous month. Extend only
    # this research process's archive window to the latest snapshot date so
    # completed current-month races can be joined to their official results.
    snapshots = nb.load_odds_snapshots()
    snapshot_dates = []
    for snapshot in snapshots.values():
        raw_date = str(snapshot.get("date") or snapshot.get("race_key") or "")
        parts = raw_date.split("|")
        date_text = parts[1] if len(parts) == 3 else raw_date
        digits = re.sub(r"\\D", "", date_text)
        if len(digits) == 8:
            try:
                snapshot_dates.append(datetime.strptime(digits, "%Y%m%d").date())
            except ValueError:
                pass
    if snapshot_dates:
        nb.END = max(nb.END, max(snapshot_dates))
        nb.START = nb.END - nb.datetime.timedelta(days=1460)
    print(
        f"MARKET_COMPARE_WINDOW start={nb.START} end={nb.END} "
        f"snapshot_races={len(snapshots)} snapshot_dates="
        f"{min(snapshot_dates) if snapshot_dates else None}..{max(snapshot_dates) if snapshot_dates else None}",
        flush=True,
    )
    races, horses, payouts, months = nb.load_archives()
    print(f"MARKET_COMPARE_ARCHIVES races={len(races)} horse_races={len(horses)} months={len(months)}", flush=True)
    axis_by_key = collect_fixed_axes(races, horses, payouts)
    print(f"MARKET_COMPARE_AXES count={len(axis_by_key)}", flush=True)
    rows = []
    exclusion = defaultdict(int)
    favorite_sources = defaultdict(int)

    for key, snapshot in snapshots.items():
        try:
            captured = datetime.fromisoformat(str(snapshot["captured_at"]))
            scheduled = datetime.fromisoformat(str(snapshot["scheduled_post_time"]))
            minutes = numeric(snapshot.get("minutes_to_post"))
        except (KeyError, TypeError, ValueError):
            exclusion["invalid_timestamp"] += 1
            continue
        if captured.tzinfo is None or scheduled.tzinfo is None or captured >= scheduled or minutes is None or not 2 <= minutes <= 30:
            exclusion["outside_valid_preoff_window"] += 1
            continue

        race_key = nb.race_key(key[0], key[1], key[2])
        race = races.get(race_key)
        field = horses.get(race_key) or []
        axis_info = axis_by_key.get("|".join(map(str, race_key)))
        if not race or not field or not axis_info:
            exclusion["missing_race_or_axis"] += 1
            continue
        favorite, source = market_favorite(snapshot)
        favorite_sources[source] += 1
        if favorite is None:
            exclusion["missing_market_favorite"] += 1
            continue

        finish_by_no = {
            str(int(h["no"])): int(h["finish"])
            for h in field
            if h.get("no") and h.get("finish") is not None and 1 <= int(h["finish"]) <= 3
        }
        podium = set(finish_by_no)
        axis = str(axis_info["axis"])
        top7 = set(str(x) for x in axis_info["top7"])
        if len(podium) != 3:
            exclusion["not_three_classified_podium"] += 1
            continue
        if axis not in {str(h["no"]) for h in field} or favorite not in {str(h["no"]) for h in field}:
            exclusion["axis_or_favorite_not_in_field"] += 1
            continue

        rows.append({
            "date": str(snapshot.get("date") or key[1]),
            "venue": str(race.get("venue") or key[0]),
            "distance_band": distance_band(race.get("distance")),
            "field_size_band": field_band(race.get("field_size") or len(field)),
            "axis_top3": axis in podium,
            "market_top3": favorite in podium,
            "podium": podium,
            "top7": top7,
        })

    rows.sort(key=lambda row: (row["date"], row["venue"]))
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed" if rows else "no_matched_races",
        "comparison": "fixed LAB pre-off axis vs same-race pre-off market win favorite",
        "odds_policy": "closest valid snapshot from 30 to 2 minutes before post; explicit win popularity first, lowest win odds fallback",
        "outcome_source": "official NAR monthly race archive",
        "sample": {
            "matched_races": len(rows),
            "first_date": min((x["date"] for x in rows), default=None),
            "last_date": max((x["date"] for x in rows), default=None),
            "favorite_identification": dict(favorite_sources),
            "exclusions": dict(exclusion),
        },
        "overall": summarize(rows),
        "by_venue": {key: summarize(value) for key, value in sorted(
            ((k, [x for x in rows if x["venue"] == k]) for k in {x["venue"] for x in rows}),
            key=lambda item: item[0])},
        "by_distance_band": {key: summarize(value) for key, value in sorted(
            ((k, [x for x in rows if x["distance_band"] == k]) for k in {x["distance_band"] for x in rows}),
            key=lambda item: item[0])},
        "by_field_size_band": {key: summarize(value) for key, value in sorted(
            ((k, [x for x in rows if x["field_size_band"] == k]) for k in {x["field_size_band"] for x in rows}),
            key=lambda item: item[0])},
        "limitations": [
            "Only races with a valid timestamped pre-off win-odds snapshot and three classified official finishers are included.",
            "Top-7 coverage is the share of all three podium horses contained in LAB's fixed pre-off top seven.",
            "Market favorite is a benchmark only; odds are not used as LAB model inputs.",
            "Venue/distance/field-size segments are descriptive; small segment samples should not be overinterpreted.",
            "Offline virtual evaluation only; no production model or ticket logic is changed."
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "sample": report["sample"], "overall": report["overall"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
