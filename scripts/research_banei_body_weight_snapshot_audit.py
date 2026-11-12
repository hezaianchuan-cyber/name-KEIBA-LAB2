#!/usr/bin/env python3
"""Research-only audit of timestamped Banei body-weight snapshots against final NAR archives.

This checks whether the values captured before scheduled post time agree with the
official monthly result archive. It does not train or alter any model.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

import nar_backtest as nb

FIELDS = ("carried_weight", "body_weight", "body_weight_change")
OUT = nb.ROOT / "data" / "banei-body-weight-snapshot-audit.json"


def fetch_snapshots():
    base = os.environ["SUPABASE_URL"].rstrip("/")
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    start = os.environ.get("NAR_BANEI_WEIGHT_AUDIT_START", "2026-11-01")
    end = os.environ.get("NAR_BANEI_WEIGHT_AUDIT_END", dt.date.today().isoformat())
    rows, offset, page_size = [], 0, 1000
    while True:
        params = {
            "select": "race_id,race_date,venue,race_number,horse_number,horse_name,carried_weight,body_weight,body_weight_change,captured_at,scheduled_post_time,minutes_to_post,source",
            "venue": "eq.帯広",
            "race_date": f"gte.{start}",
            "order": "race_date.asc,captured_at.asc",
            "limit": str(page_size),
            "offset": str(offset),
        }
        query = urllib.parse.urlencode(params)
        request = urllib.request.Request(
            f"{base}/rest/v1/banei_body_weight_snapshots?{query}",
            headers={"apikey": key, "Authorization": f"Bearer {key}", "Accept": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=45) as response:
            batch = json.loads(response.read().decode("utf-8"))
        if not batch:
            break
        rows.extend(row for row in batch if str(row.get("race_date") or "") <= end)
        if len(batch) < page_size:
            break
        offset += page_size
    return rows, start, end


def parse_time(value):
    if not value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return None


def main():
    snapshots, start, end = fetch_snapshots()
    months = sorted({(int(row["race_date"][:4]), int(row["race_date"][5:7])) for row in snapshots})
    official = {}
    archive_errors = []
    for year, month in months:
        try:
            _races, horses, _payouts, metadata = nb.load_parsed_month((year, month))
            for race_key, runners in horses.items():
                if "帯広" not in race_key:
                    continue
                for runner in runners:
                    official[("|".join(race_key), str(runner.get("no") or ""))] = runner
        except Exception as exc:
            archive_errors.append({"month": f"{year}-{month:02d}", "error": str(exc)[:500]})

    groups = defaultdict(list)
    preoff_rows = []
    invalid_clock = 0
    for row in snapshots:
        captured = parse_time(row.get("captured_at"))
        post = parse_time(row.get("scheduled_post_time"))
        if captured is None or post is None or captured >= post:
            invalid_clock += 1
            continue
        minutes = (post - captured).total_seconds() / 60
        if not 2 <= minutes <= 60:
            continue
        preoff_rows.append(row)
        groups[(row["race_id"], str(row["horse_number"]))].append(row)

    field_results = {field: {"compared": 0, "matched": 0, "mismatched": 0, "missing_snapshot_value": 0, "missing_final_value": 0} for field in FIELDS}
    missing_official_runner = 0
    first_seen_minutes = []
    race_keys = set()
    for key, rows in groups.items():
        runner = official.get(key)
        if runner is None:
            missing_official_runner += 1
            continue
        race_keys.add(key[0])
        rows.sort(key=lambda x: str(x.get("captured_at") or ""))
        first = rows[0]
        captured, post = parse_time(first.get("captured_at")), parse_time(first.get("scheduled_post_time"))
        if captured and post:
            first_seen_minutes.append((post - captured).total_seconds() / 60)
        for field in FIELDS:
            source_key = {"carried_weight": "carried", "body_weight": "body_weight", "body_weight_change": "body_weight_change"}[field]
            final_value = nb.number(runner.get(source_key))
            observed = [nb.number(row.get(field)) for row in rows if nb.number(row.get(field)) is not None]
            stats = field_results[field]
            if not observed:
                stats["missing_snapshot_value"] += 1
                continue
            if final_value is None:
                stats["missing_final_value"] += 1
                continue
            stats["compared"] += 1
            if observed[-1] == final_value:
                stats["matched"] += 1
            else:
                stats["mismatched"] += 1

    def pct(n, d):
        return round(100 * n / d, 3) if d else None

    report = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "research_only_snapshot_vs_final_archive_audit",
        "period": {"start": start, "end": end},
        "snapshot_rows_loaded": len(snapshots),
        "preoff_rows_in_2_to_60_min_window": len(preoff_rows),
        "unique_races_with_preoff_rows": len(race_keys),
        "unique_race_horse_pairs": len(groups),
        "invalid_or_postoff_timestamps": invalid_clock,
        "missing_official_runner_pairs": missing_official_runner,
        "first_observed_minutes_before_post": {
            "n": len(first_seen_minutes),
            "median": round(sorted(first_seen_minutes)[len(first_seen_minutes)//2], 2) if first_seen_minutes else None,
            "min": round(min(first_seen_minutes), 2) if first_seen_minutes else None,
            "max": round(max(first_seen_minutes), 2) if first_seen_minutes else None,
            "note": "This is first observation by this collector, not the official public announcement time."
        },
        "field_comparison": {
            field: {
                **stats,
                "exact_match_pct": pct(stats["matched"], stats["compared"]),
            } for field, stats in field_results.items()
        },
        "archive_errors": archive_errors,
        "guardrails": [
            "Only observations with captured_at earlier than scheduled_post_time are eligible for pre-race use.",
            "Final monthly archive values are used only as post-race verification targets, never as predictors.",
            "A match with the final archive does not by itself prove when the information became publicly available.",
            "No model is trained or modified by this audit.",
            "Banei rows remain separate from flat-racing data."
        ]
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("BANEI_WEIGHT_SNAPSHOT_AUDIT " + json.dumps({
        "snapshot_rows": len(snapshots), "preoff_rows": len(preoff_rows),
        "races": len(race_keys), "field_comparison": report["field_comparison"],
        "report": str(OUT)
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
