#!/usr/bin/env python3
"""Publish immutable, pre-off model snapshots to Supabase through the trusted Edge Function."""
from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import json
import os
import re
import urllib.request
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
    return 2.0 <= minutes <= 30.0 and isinstance(record.get("predictions"), list) and len(record["predictions"]) >= 3


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
    }


def load_rows() -> tuple[list[dict], dict]:
    rows_by_key = {}
    stats = {"files": 0, "lines": 0, "valid": 0, "invalid_lock": 0, "invalid_shape": 0}
    # Re-publish the newest two daily archives on every run. Upserts make
    # retries safe while keeping the five-minute workflow bounded.
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


def publish(rows: list[dict], endpoint: str, token: str, batch_size: int = 40) -> int:
    published = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        request = urllib.request.Request(
            endpoint,
            data=json.dumps({"action": "archive_model_snapshots", "snapshots": batch},
                            ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
            headers={
                "Authorization": "Bearer " + token,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=45) as response:
            payload = json.loads(response.read().decode("utf-8"))
            if (response.status < 200 or response.status >= 300 or not payload.get("ok")
                    or "upserted" not in payload or "received" not in payload):
                raise RuntimeError("Model snapshot archive endpoint did not confirm the archive action: " + str(payload)[:500])
            published += int(payload.get("upserted", 0))
    return published


def main() -> None:
    url = os.environ.get("SUPABASE_URL", "https://mwiinzmnoludksnfnjaa.supabase.co").rstrip("/")
    token = os.environ.get("NAR_CAPTURE_TOKEN", "").strip()
    if not token:
        raise RuntimeError("NAR_CAPTURE_TOKEN is required to publish model snapshots")
    rows, stats = load_rows()
    if not rows:
        print(json.dumps({"ok": True, **stats, "published": 0}, ensure_ascii=False), flush=True)
        return
    published = publish(rows, url + "/functions/v1/nar-odds-capture", token)
    print(json.dumps({"ok": True, **stats, "published": published}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
