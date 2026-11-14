#!/usr/bin/env python3
"""Idempotently import frozen A45 predictions and joined 3-combo quotes into SQLite."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sqlite3
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "research_db" / "sqlite_schema_v1.sql"


def stable_id(*parts: object) -> str:
    raw = "|".join(str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def positive_or_none(value):
    """Normalize source placeholders such as 0/-1 to SQL NULL for positive-only fields."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def read_jsonl(path: Path):
    if not path.exists():
        return []
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def import_files(db: sqlite3.Connection, predictions_path: Path, odds_path: Path):
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    batch_id = str(uuid.uuid4())
    db.execute(
        """INSERT INTO ingestion_batches
        (batch_id,source,started_at,status,notes)
        VALUES (?,?,?,?,?)""",
        (batch_id, "A45 research JSONL", now, "running",
         "Idempotent import; only frozen predictions and joined strategy quotes"),
    )
    read_count = inserted = rejected = 0
    try:
        predictions = read_jsonl(predictions_path)
        odds_rows = read_jsonl(odds_path)
        read_count = len(predictions) + len(odds_rows)

        for item in predictions:
            race_id = str(item["race_id"])
            date_raw = str(item.get("race_date") or race_id.split("|")[1])
            race_date = dt.datetime.strptime(date_raw, "%Y%m%d").date().isoformat()
            venue = str(item.get("venue") or race_id.split("|")[0])
            race_number = int(item.get("race_number") or race_id.split("|")[2])
            scheduled = item.get("scheduled_post_time")
            captured = str(item["captured_at"])
            first_features = next(
                (p.get("features") or {} for p in item.get("predictions", [])), {}
            )
            field_size = max(
                (int(p.get("features", {}).get("field_size") or 0)
                 for p in item.get("predictions", [])),
                default=0,
            )
            db.execute(
                """INSERT INTO races
                (race_id,race_date,venue,race_type,race_number,race_name,surface,distance_m,
                 going,scheduled_post_at,field_size,source,first_seen_at,last_seen_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(race_id) DO UPDATE SET last_seen_at=excluded.last_seen_at""",
                (race_id,race_date,venue,"banei" if "帯広" in venue else "flat",race_number,item.get("race_name"),
                 first_features.get("surface"),positive_or_none(first_features.get("distance_m")),
                 first_features.get("going"),scheduled,field_size,
                 str(item.get("source") or "research_a45_prediction"),
                 captured,captured),
            )
            schema_version = stable_id(*item.get("feature_schema", []))[:16]
            for horse in item.get("predictions", []):
                horse_no = int(horse["horse_no"])
                features = horse.get("features") or {}
                db.execute(
                    """INSERT INTO runners
                    (race_id,horse_no,horse_name,frame_no,age,jockey,carried_weight,
                     sire,source,first_seen_at,last_seen_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(race_id,horse_no) DO UPDATE SET
                      horse_name=excluded.horse_name,last_seen_at=excluded.last_seen_at""",
                    (race_id,horse_no,str(horse.get("name") or f"horse-{horse_no}"),
                     features.get("frame_no"),features.get("age"),
                     features.get("jockey"),positive_or_none(features.get("carried_weight")),
                     features.get("sire"),"research_a45_prediction",captured,captured),
                )
                prediction_id = stable_id(
                    race_id,horse_no,item.get("model_version"),captured
                )
                cur = db.execute(
                    """INSERT OR IGNORE INTO model_predictions
                    (prediction_id,race_id,horse_no,model_name,model_version,
                     feature_schema_version,rank_position,score,features_json,
                     predicted_at,scheduled_post_at,minutes_to_post,source,batch_id)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (prediction_id,race_id,horse_no,"LambdaRank",
                     str(item.get("model_version") or "unknown"),schema_version,
                     int(horse["rank"]),float(horse["score"]),
                     json.dumps(features,ensure_ascii=False,sort_keys=True),
                     captured,scheduled,item.get("minutes_to_post"),
                     "research_a45_frozen_prediction",batch_id),
                )
                inserted += cur.rowcount

        for item in odds_rows:
            if item.get("odds_status") != "joined":
                continue
            race_id = str(item["race_id"])
            captured = str(item["odds_snapshot_captured_at"])
            for group_name in ("strategies", "market_strategies"):
                for strategy_name, strategy in (item.get(group_name) or {}).items():
                    for ticket in strategy.get("tickets", []):
                        value = ticket.get("odds")
                        if value is None:
                            continue
                        numbers = sorted(str(n) for n in ticket.get("numbers", []))
                        if len(numbers) != 3 or len(set(numbers)) != 3:
                            rejected += 1
                            continue
                        combo = "-".join(numbers)
                        snapshot_id = stable_id(
                            race_id,"trio",combo,captured,group_name,strategy_name
                        )
                        cur = db.execute(
                            """INSERT OR IGNORE INTO odds_snapshots
                            (snapshot_id,race_id,bet_type,combination_key,odds,
                             captured_at,source,batch_id)
                            VALUES (?,?,?,?,?,?,?,?)""",
                            (snapshot_id,race_id,"trio",combo,float(value),
                             captured,f"A45_{group_name}_{strategy_name}",batch_id),
                        )
                        inserted += cur.rowcount

        db.execute(
            """UPDATE ingestion_batches SET completed_at=?,status='completed',
            rows_read=?,rows_inserted=?,rows_rejected=? WHERE batch_id=?""",
            (dt.datetime.now(dt.timezone.utc).isoformat(),read_count,
             inserted,rejected,batch_id),
        )
        db.commit()
        return {"batch_id":batch_id,"rows_read":read_count,
                "rows_inserted":inserted,"rows_rejected":rejected}
    except Exception:
        db.rollback()
        db.execute(
            """UPDATE ingestion_batches SET completed_at=?,status='failed'
            WHERE batch_id=?""",
            (dt.datetime.now(dt.timezone.utc).isoformat(),batch_id),
        )
        db.commit()
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=ROOT / "data" / "keibalab-research.sqlite")
    parser.add_argument("--predictions", type=Path,
                        default=ROOT / "data" / "research-a45-predictions.jsonl")
    parser.add_argument("--odds", type=Path,
                        default=ROOT / "data" / "research-a45-odds-joined.jsonl")
    args = parser.parse_args()

    args.db.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(args.db)
    try:
        db.execute("PRAGMA foreign_keys = ON")
        db.execute("PRAGMA journal_mode = WAL")
        db.executescript(SCHEMA.read_text(encoding="utf-8"))
        result = import_files(db,args.predictions,args.odds)
        counts = {
            table: db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("races","runners","model_predictions","odds_snapshots")
        }
        print(json.dumps({"status":"ok",**result,"table_counts":counts},
                         ensure_ascii=False,indent=2))
    finally:
        db.close()


if __name__ == "__main__":
    main()
