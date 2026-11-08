#!/usr/bin/env python3
"""Publish immutable, pre-off model snapshots to Supabase through the trusted Edge Function."""
from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = ROOT / "data" / "odds-snapshots"
LOCK_FIELDS = (
    "race_key", "venue", "date", "race", "scheduled_post_time", "captured_at",
    "minutes_to_post", "capture_policy", "model_version", "model_temperature",
    "predictions", "combination_predictions", "candidate_combinations",
    "model_g_version", "model_g_predictions", "model_g_candidate_combinations",
    "model_g_strategy", "candidate_universe_policy", "race_readiness", "odds",
)


def canonical_timestamp(value: str) -> str:
    """Normalize timestamptz values so equivalent UTC/JST strings compare identically."""
    parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc).isoformat()


def lock_digest(record: dict) -> str:
    payload = {key: record.get(key) for key in LOCK_FIELDS}
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def verify_record(record: dict) -> bool:
    lock = record.get("prediction_lock") or {}
    required = (
        "race_key", "scheduled_post_time", "captured_at", "model_version",
        "predictions", "odds",
    )
    if any(not record.get(key) for key in required):
        return False
    if lock.get("version") != "nar-prediction-lock-v1" or lock.get("algorithm") != "sha256":
        return False
    if lock.get("locked_at") != record.get("captured_at"):
        return False
    if lock.get("digest") != lock_digest(record):
        return False
    try:
        captured = dt.datetime.fromisoformat(str(record["captured_at"]).replace("Z", "+00:00"))
        post = dt.datetime.fromisoformat(str(record["scheduled_post_time"]).replace("Z", "+00:00"))
        minutes = (post - captured).total_seconds() / 60.0
    except (TypeError, ValueError):
        return False
    return 2.0 <= minutes <= 60.0 and isinstance(record.get("predictions"), list) and len(record["predictions"]) >= 3


def record_to_row(record: dict) -> dict | None:
    if not verify_record(record):
        return None
    race_id = str(record.get("race_key") or "").strip()
    parts = race_id.split("|")
    date_digits = re.sub(r"\D", "", str(record.get("date") or (parts[1] if len(parts) == 3 else "")))
    if len(parts) != 3 or not parts[0] or len(date_digits) != 8:
        return None
    race_date = f"{date_digits[:4]}-{date_digits[4:6]}-{date_digits[6:8]}"
    return {
        "race_id": race_id,
        "race_date": race_date,
        "captured_at": record["captured_at"],
        "scheduled_post_time": record["scheduled_post_time"],
        "model_version": str(record["model_version"]),
        "model_temperature": record.get("model_temperature"),
        "predictions": record["predictions"],
        "candidate_combinations": record.get("candidate_combinations") or [],
        "model_g_version": record.get("model_g_version"),
        "model_g_predictions": record.get("model_g_predictions") or [],
        "model_g_candidate_combinations": record.get("model_g_candidate_combinations") or [],
        "prediction_lock": record["prediction_lock"],
        "locked_payload": {key: record.get(key) for key in LOCK_FIELDS},
    }


def load_rows() -> tuple[list[dict], dict]:
    rows_by_key = {}
    stats = {"files": 0, "lines": 0, "valid": 0, "invalid_lock": 0, "invalid_shape": 0}
    # Scan the newest two daily archives. Existing database keys are filtered
    # later so each scheduled capture publishes only snapshots not yet archived.
    archive_files = sorted(SNAPSHOT_DIR.glob("*.jsonl.gz"))[-2:]
    for path in archive_files:
        stats["files"] += 1
        try:
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                for line in stream:
                    stats["lines"] += 1
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        stats["invalid_shape"] += 1
                        continue
                    if not verify_record(record):
                        stats["invalid_lock"] += 1
                        continue
                    row = record_to_row(record)
                    if row is None:
                        stats["invalid_shape"] += 1
                        continue
                    rows_by_key[(row["race_id"], row["captured_at"])] = row
        except OSError:
            stats["invalid_shape"] += 1
    stats["valid"] = len(rows_by_key)
    return list(rows_by_key.values()), stats



def fetch_existing_keys(
    rows: list[dict], url: str, anon_key: str, target_new_rows: int = 5, batch_size: int = 5
) -> set[tuple[str, str]]:
    """Check exact indexed race/time keys, stopping after enough new rows are found."""
    ordered = sorted(
        rows,
        key=lambda row: (str(row.get("captured_at") or ""), str(row.get("race_id") or "")),
        reverse=True,
    )
    existing: set[tuple[str, str]] = set()
    found_new = 0
    for start in range(0, len(ordered), max(1, batch_size)):
        batch = ordered[start:start + max(1, batch_size)]
        query_pairs = [
            (str(row.get("race_id") or ""), str(row.get("captured_at") or ""))
            for row in batch
        ]
        query_pairs = [(race_id, captured_at) for race_id, captured_at in query_pairs if race_id and captured_at]
        if not query_pairs:
            continue
        normalized_pairs = [
            (race_id, canonical_timestamp(captured_at))
            for race_id, captured_at in query_pairs
        ]
        clauses = ",".join(
            f"and(race_id.eq.{race_id},captured_at.eq.{captured_at})"
            for race_id, captured_at in query_pairs
        )
        query = urllib.parse.urlencode([
            ("select", "race_id,captured_at"),
            ("or", "(" + clauses + ")"),
            ("limit", str(len(query_pairs))),
        ])
        request = urllib.request.Request(
            url.rstrip("/") + "/rest/v1/model_prediction_snapshots?" + query,
            headers={
                "apikey": anon_key,
                "Authorization": "Bearer " + anon_key,
                "Accept": "application/json",
                "User-Agent": "KEIBA-LAB-model-snapshot-publisher/1.0",
            },
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            page = json.loads(response.read().decode("utf-8"))
        if not isinstance(page, list):
            raise RuntimeError("Existing model snapshot lookup did not return a JSON array")
        for item in page:
            race_id = str(item.get("race_id") or "")
            captured_at = str(item.get("captured_at") or "")
            if race_id and captured_at:
                existing.add((race_id, canonical_timestamp(captured_at)))
        found_new += sum(1 for pair in normalized_pairs if pair not in existing)
        if found_new >= max(1, target_new_rows):
            break
    return existing


def filter_unpublished_rows(rows: list[dict], existing_keys: set[tuple[str, str]]) -> list[dict]:
    return [
        row for row in rows
        if (str(row.get("race_id") or ""), canonical_timestamp(str(row.get("captured_at") or ""))) not in existing_keys
    ]



def limit_pending_rows(rows: list[dict], max_rows: int = 5) -> list[dict]:
    """Prioritize the newest snapshots; GitHub gzip remains the complete archive."""
    ordered = sorted(
        rows,
        key=lambda row: (str(row.get("captured_at") or ""), str(row.get("race_id") or "")),
        reverse=True,
    )
    return ordered[:max(0, int(max_rows))]


def publish(rows: list[dict], endpoint: str, token: str, anon_key: str, batch_size: int = 5) -> int:
    published = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        request = urllib.request.Request(
            endpoint,
            data=json.dumps({"action": "archive_model_snapshots", "snapshots": batch},
                            ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
            headers={
                "Authorization": "Bearer " + token,
                "apikey": anon_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        # The archive endpoint uses idempotent upserts keyed by race/time.
        # A transient 5xx can happen after the database committed the batch but
        # before the gateway returned its response, so retrying is safe.
        for attempt in range(2):
            try:
                with urllib.request.urlopen(request, timeout=25) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                    if (response.status < 200 or response.status >= 300 or not payload.get("ok")
                            or "upserted" not in payload or "received" not in payload):
                        raise RuntimeError("Model snapshot archive endpoint did not confirm the archive action: " + str(payload)[:500])
                    published += int(payload.get("upserted", 0))
                break
            except urllib.error.HTTPError as exc:
                if exc.code not in (408, 425, 429, 500, 502, 503, 504) or attempt == 1:
                    raise
                delay = min(2 ** (attempt + 1), 12)
                print(f"Transient archive HTTP {exc.code}; retrying batch in {delay}s (attempt {attempt + 2}/2)", flush=True)
                time.sleep(delay)
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                if attempt == 1:
                    raise
                delay = min(2 ** (attempt + 1), 12)
                print(f"Transient archive connection error; retrying batch in {delay}s (attempt {attempt + 2}/4): {exc}", flush=True)
                time.sleep(delay)
    return published


def record_publish_status(status: str, stats: dict, published: int = 0, error: str | None = None, saved_race_ids: set[str] | None = None) -> None:
    """Persist Supabase archive outcome beside the per-race capture audit."""
    path = ROOT / "data" / "odds-capture-status.json"
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        report = {}
    archive_race_ids = set()
    try:
        archive_race_ids = {str(row.get("race_id")) for row in load_rows()[0]}
    except Exception:
        archive_race_ids = set()
    confirmed_race_ids = set(saved_race_ids or []) if status == "success" else set()
    attempted_at = dt.datetime.now(dt.timezone.utc).isoformat()
    audit = report.get("race_audit") or []
    for item in audit:
        if item.get("race_id") in confirmed_race_ids and status == "success":
            item["supabase_saved"] = True
            item["supabase_saved_at"] = attempted_at
            item.pop("supabase_error", None)
        elif status in ("failed", "deferred") and item.get("snapshot_saved"):
            # A transient failure must not erase a previously confirmed save.
            item["supabase_saved"] = bool(item.get("supabase_saved"))
            item["supabase_error"] = (error or "Supabase publish failed")[:500]
    report["supabase_publish"] = {
        "attempted_at": attempted_at,
        "status": status,
        "valid_snapshot_rows": int(stats.get("valid", 0)),
        "published_rows": int(published),
        "race_count_in_archive": len(archive_race_ids),
        "race_count_confirmed_saved": len(confirmed_race_ids),
        "error": (error or "")[:500] if error else None,
    }
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    url = os.environ.get("SUPABASE_URL", "https://mwiinzmnoludksnfnjaa.supabase.co").rstrip("/")
    token = os.environ.get("NAR_CAPTURE_TOKEN", "").strip()
    anon_key = os.environ.get("SUPABASE_ANON_KEY", "").strip()
    if not token:
        raise RuntimeError("NAR_CAPTURE_TOKEN is required to publish model snapshots")
    if not anon_key:
        raise RuntimeError("SUPABASE_ANON_KEY is required to publish model snapshots")
    rows, stats = load_rows()
    if not rows:
        record_publish_status("success", stats, 0, saved_race_ids=set())
        print(json.dumps({"ok": True, **stats, "published": 0, "rows_sent": 0}, ensure_ascii=False), flush=True)
        return
    try:
        existing_keys = fetch_existing_keys(rows, url, anon_key, target_new_rows=25, batch_size=25)
        pending_all = filter_unpublished_rows(rows, existing_keys)
        saved_race_ids = {race_id for race_id, _captured_at in existing_keys}
        stats["already_in_supabase"] = len(rows) - len(pending_all)
        stats["pending_publish_total"] = len(pending_all)
        stats["existing_key_lookup"] = "success"
    except Exception as lookup_exc:
        # A failed read indicates database pressure. Do not immediately add
        # more writes; the complete gzip archive remains safely committed.
        stats["existing_key_lookup"] = "failed_deferred"
        stats["existing_key_lookup_error"] = str(lookup_exc)[:300]
        stats["pending_publish_total"] = None
        record_publish_status("deferred", stats, 0, str(lookup_exc))
        print(json.dumps({
            "ok": True, **stats, "published": 0, "rows_sent": 0,
            "deferred": True, "reason": "Supabase lookup unavailable; local archive preserved",
        }, ensure_ascii=False), flush=True)
        return
    pending_rows = limit_pending_rows(pending_all, max_rows=25)
    stats["pending_publish"] = len(pending_rows)
    stats["pending_publish_deferred"] = max(0, len(pending_all) - len(pending_rows))
    print(
        f"Model snapshot archive delta: total={len(rows)} already_saved={stats.get('already_in_supabase', 0)} "
        f"pending_total={len(pending_all)} sending_now={len(pending_rows)}",
        flush=True,
    )
    if not pending_rows:
        record_publish_status("success", stats, 0, saved_race_ids=saved_race_ids)
        print(json.dumps({"ok": True, **stats, "published": 0, "rows_sent": 0}, ensure_ascii=False), flush=True)
        return
    try:
        published = publish(pending_rows, url + "/functions/v1/nar-model-snapshot-archive", token, anon_key, batch_size=5)
    except Exception as exc:
        record_publish_status("failed", stats, 0, str(exc))
        raise
    saved_race_ids.update(str(row.get("race_id")) for row in pending_rows if row.get("race_id"))
    record_publish_status("success", stats, published, saved_race_ids=saved_race_ids)
    print(json.dumps({"ok": True, **stats, "published": published, "rows_sent": len(pending_rows)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
