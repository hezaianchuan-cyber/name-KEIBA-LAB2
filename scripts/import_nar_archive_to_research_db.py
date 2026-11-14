#!/usr/bin/env python3
"""Import official NAR monthly race archives into the research SQLite database.

This stores historical race cards, runners, official finishing results and payouts.
It does not fabricate pre-race odds or model predictions. The source archive only
contains final race data; keep it separate from frozen pre-race captures.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_DEFAULT = ROOT / "data" / "keibalab-research.sqlite"
sys.path.insert(0, str(ROOT / "scripts"))
import nar_backtest as nar  # Reuse the existing official NAR archive parser.


def parse_date(value: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD") from exc


def canonical_race_id(key) -> str:
    venue, date_raw, race_no = key
    return f"{venue}|{date_raw}|{int(race_no)}"


def canonical_payout_key(kind: str, numbers) -> str:
    normalized = [str(int(str(n))) for n in numbers if str(n).strip()]
    if kind in {"place", "win", "quinella", "wide", "trio"}:
        normalized.sort(key=int)
    return "-".join(normalized)


def payout_rows(payout):
    """Yield official payout rows; trio is the primary KEIBALAB target."""
    for kind in ("win", "place", "quinella", "exacta", "wide", "trio", "trifecta"):
        for numbers, amount in (payout or {}).get(kind, []):
            key = canonical_payout_key(kind, numbers)
            if key and amount is not None and float(amount) > 0:
                yield kind, key, int(round(float(amount)))


def import_month(db: sqlite3.Connection, year: int, month: int, imported_at: str):
    archive, source_url = nar.download_month(year, month)
    archive_sha = hashlib.sha256(archive).hexdigest()
    races, horses_by_key, payouts = nar.parse_month_archive(archive, year, month)
    inserted = {"races": 0, "runners": 0, "official_results": 0, "payouts": 0}
    rejected = 0

    batch_id = hashlib.sha256(
        f"nar-monthly|{year:04d}-{month:02d}|{archive_sha}".encode("utf-8")
    ).hexdigest()
    db.execute(
        """INSERT OR IGNORE INTO ingestion_batches
        (batch_id,source,started_at,status,checksum,notes)
        VALUES (?,?,?,?,?,?)""",
        (batch_id, "official_nar_monthly_archive", imported_at, "running",
         archive_sha, f"{year:04d}-{month:02d}; source={source_url}"),
    )

    for key, race in races.items():
        race_id = canonical_race_id(key)
        race_date = dt.datetime.strptime(key[1], "%Y%m%d").date().isoformat()
        field = horses_by_key.get(key, [])
        field_size = int(race.get("field_size") or len(field) or 0)
        db.execute(
            """INSERT INTO races
            (race_id,race_date,venue,race_type,race_number,race_name,surface,distance_m,
             going,scheduled_post_at,field_size,source,first_seen_at,last_seen_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(race_id) DO UPDATE SET
              race_name=excluded.race_name,surface=excluded.surface,
              distance_m=excluded.distance_m,going=excluded.going,
              field_size=excluded.field_size,last_seen_at=excluded.last_seen_at""",
            (race_id, race_date, str(race.get("venue") or key[0]),
             "banei" if "帯広" in str(race.get("venue") or key[0]) else "flat", int(key[2]),
             race.get("name") or None, race.get("surface") or None,
             int(race["distance"]) if race.get("distance") else None,
             race.get("going") or None, None, field_size,
             "official_nar_monthly_archive", imported_at, imported_at),
        )
        inserted["races"] += db.execute("SELECT changes()").fetchone()[0]

        for horse in field:
            try:
                horse_no = int(horse["no"])
                if horse_no <= 0:
                    raise ValueError("non-positive horse number")
            except (KeyError, TypeError, ValueError):
                rejected += 1
                continue
            db.execute(
                """INSERT INTO runners
                (race_id,horse_no,horse_name,frame_no,sex,age,jockey,carried_weight,
                 sire,dam,source,first_seen_at,last_seen_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(race_id,horse_no) DO UPDATE SET
                  horse_name=excluded.horse_name,frame_no=excluded.frame_no,
                  sex=excluded.sex,age=excluded.age,jockey=excluded.jockey,
                  carried_weight=excluded.carried_weight,sire=excluded.sire,
                  dam=excluded.dam,last_seen_at=excluded.last_seen_at""",
                (race_id, horse_no, str(horse.get("name") or f"horse-{horse_no}"),
                 horse.get("frame_no"), horse.get("sex") or None,
                 int(horse["age"]) if horse.get("age") is not None else None,
                 horse.get("jockey") or None,
                 float(horse["carried"]) if horse.get("carried") and float(horse["carried"]) > 0 else None,
                 horse.get("sire") or None, horse.get("dam") or None,
                 "official_nar_monthly_archive", imported_at, imported_at),
            )
            inserted["runners"] += db.execute("SELECT changes()").fetchone()[0]

            finish = horse.get("finish")
            if finish is not None:
                db.execute(
                    """INSERT INTO official_results
                    (race_id,horse_no,finish_position,finish_time_seconds,last3f_seconds,
                     dead_heat_group,confirmed_at,source,batch_id)
                    VALUES (?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(race_id,horse_no) DO UPDATE SET
                      finish_position=excluded.finish_position,
                      finish_time_seconds=excluded.finish_time_seconds,
                      last3f_seconds=excluded.last3f_seconds,
                      confirmed_at=excluded.confirmed_at,source=excluded.source,
                      batch_id=excluded.batch_id""",
                    (race_id, horse_no, int(finish), horse.get("time_seconds"),
                     horse.get("last3f"), None, race_date,
                     "official_nar_monthly_archive", batch_id),
                )
                inserted["official_results"] += db.execute("SELECT changes()").fetchone()[0]

        for kind, combination, amount in payout_rows(payouts.get(key)):
            db.execute(
                """INSERT INTO payouts
                (race_id,bet_type,combination_key,payout_yen,stake_unit_yen,
                 confirmed_at,source,batch_id)
                VALUES (?,?,?,?,?,?,?,?)
                ON CONFLICT(race_id,bet_type,combination_key) DO UPDATE SET
                  payout_yen=excluded.payout_yen,confirmed_at=excluded.confirmed_at,
                  source=excluded.source,batch_id=excluded.batch_id""",
                (race_id, kind, combination, amount, 100, race_date,
                 "official_nar_monthly_archive", batch_id),
            )
            inserted["payouts"] += db.execute("SELECT changes()").fetchone()[0]

    db.execute(
        """UPDATE ingestion_batches SET completed_at=?,status='completed',
        rows_read=?,rows_inserted=?,rows_rejected=? WHERE batch_id=?""",
        (imported_at, len(races) + sum(map(len, horses_by_key.values())),
         sum(inserted.values()), rejected, batch_id),
    )
    return {
        "month": f"{year:04d}-{month:02d}",
        "archive_bytes": len(archive),
        "archive_sha256": archive_sha,
        "parsed_races": len(races),
        "parsed_runner_rows": sum(map(len, horses_by_key.values())),
        "inserted_or_updated": inserted,
        "rejected_runner_rows": rejected,
        "batch_id": batch_id,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", type=parse_date, required=True)
    parser.add_argument("--end-date", type=parse_date, required=True)
    parser.add_argument("--database", type=Path, default=DB_DEFAULT)
    parser.add_argument("--max-months", type=int, default=0,
                        help="optional safety limit; 1 is useful for a first-month test")
    args = parser.parse_args()
    if args.end_date < args.start_date:
        parser.error("--end-date must be on or after --start-date")
    if args.start_date < dt.date(2000, 1, 1):
        parser.error("start date is outside the supported archive range")

    months = list(nar.month_iter(args.start_date, args.end_date))
    if args.max_months:
        if args.max_months < 1:
            parser.error("--max-months must be positive")
        months = months[:args.max_months]

    nar.START, nar.END = args.start_date, args.end_date
    args.database.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(args.database)
    db.execute("PRAGMA foreign_keys=ON")
    db.executescript((ROOT / "research_db" / "sqlite_schema_v1.sql").read_text(encoding="utf-8"))
    imported_at = dt.datetime.now(dt.timezone.utc).isoformat()
    reports = []
    try:
        for year, month in months:
            report = import_month(db, year, month, imported_at)
            db.commit()
            reports.append(report)
            print(json.dumps(report, ensure_ascii=False), flush=True)
        integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
        violations = db.execute("PRAGMA foreign_key_check").fetchall()
        counts = {
            table: db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("races", "runners", "official_results", "payouts")
        }
        summary = {
            "date_range": [args.start_date.isoformat(), args.end_date.isoformat()],
            "months_processed": len(reports),
            "integrity_check": integrity,
            "foreign_key_violations": violations,
            "counts": counts,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if integrity != "ok" or violations:
            raise RuntimeError("SQLite integrity or foreign-key verification failed")
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()
