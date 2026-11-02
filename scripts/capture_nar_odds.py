#!/usr/bin/env python3
"""Capture one official pre-off odds snapshot per NAR race shortly before post time.

The monthly NAR odds archive contains confirmed odds and must not be used as
pre-race information. This collector uses only the frequently refreshed daily
odds ZIP and records the capture timestamp alongside every race snapshot.
"""
from __future__ import annotations

import csv
import gzip
import datetime as dt
import io
import json
import re
import unicodedata
import zipfile
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SNAPSHOT_DIR = DATA / "odds-snapshots"
JST = ZoneInfo("Asia/Tokyo")
CAPTURE_WINDOW_MINUTES = 7


def decode(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "cp932", "shift_jis"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def csv_rows(raw: bytes):
    return list(csv.reader(io.StringIO(decode(raw))))


def kind_for(filename: str) -> str | None:
    base = unicodedata.normalize("NFKC", filename.replace("\\", "/").split("/")[-1]).lower()
    if re.search(r"racelist|race.?list|レース一覧", base):
        return "races"
    if re.search(r"odds|オッズ", base):
        return "odds"
    return None


def race_id(venue: str, date: str, race: str) -> tuple[str, str, str] | None:
    date = re.sub(r"\D", "", str(date or ""))
    race = re.sub(r"\D", "", str(race or ""))
    if not venue or len(date) != 8 or not race:
        return None
    return venue.strip(), date, str(int(race))


def post_time(value: str) -> dt.time | None:
    digits = re.sub(r"\D", "", str(value or ""))
    if len(digits) not in (3, 4):
        return None
    digits = digits.zfill(4)
    hour, minute = int(digits[:2]), int(digits[2:])
    if hour > 23 or minute > 59:
        return None
    return dt.time(hour, minute)


def read_archives():
    race_rows = {}
    odds_rows = {}
    race_zip = DATA / "latest-race.zip"
    odds_zip = DATA / "latest-odds.zip"
    if not race_zip.exists() or not odds_zip.exists():
        return race_rows, odds_rows
    with zipfile.ZipFile(race_zip) as archive:
        for name in archive.namelist():
            if kind_for(name) != "races":
                continue
            for row in csv_rows(archive.read(name)):
                if len(row) < 4:
                    continue
                key = race_id(row[0], row[1], row[2])
                if key:
                    race_rows[key] = {"venue": key[0], "date": key[1], "race": key[2], "time": row[3]}
    with zipfile.ZipFile(odds_zip) as archive:
        for name in archive.namelist():
            if kind_for(name) != "odds":
                continue
            for row in csv_rows(archive.read(name)):
                if len(row) < 10:
                    continue
                key = race_id(row[0], row[1], row[2])
                if not key:
                    continue
                try:
                    odds = float(str(row[7]).replace(",", "").strip())
                except (ValueError, TypeError):
                    continue
                if odds <= 0:
                    continue
                item = {
                    "type": str(row[3]).strip(),
                    "n1": str(row[4]).strip(),
                    "n2": str(row[5]).strip(),
                    "n3": str(row[6]).strip(),
                    "odds": odds,
                    "max_odds": str(row[8]).strip(),
                    "popularity": str(row[9]).strip(),
                }
                odds_rows.setdefault(key, []).append(item)
    return race_rows, odds_rows


def append_snapshot(path: Path, record: dict) -> bool:
    """Append a race once as a concatenated gzip member; return whether it was new."""
    race_key_text = record.get("race_key")
    if path.exists():
        try:
            with gzip.open(path, "rt", encoding="utf-8") as existing:
                for line in existing:
                    try:
                        if json.loads(line).get("race_key") == race_key_text:
                            return False
                    except json.JSONDecodeError:
                        continue
        except OSError:
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "ab", compresslevel=9) as output:
        output.write((json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\\n").encode("utf-8"))
    return True


def main():
    now = dt.datetime.now(JST)
    races, odds = read_archives()
    if not races or not odds:
        print("No readable official daily race/odds data; no snapshot written.")
        return
    additions = 0
    for key, race in races.items():
        try:
            race_date = dt.datetime.strptime(race["date"], "%Y%m%d").date()
        except ValueError:
            continue
        start = post_time(race["time"])
        if not start:
            continue
        post = dt.datetime.combine(race_date, start, tzinfo=JST)
        minutes = (post - now).total_seconds() / 60
        if not (0 <= minutes <= CAPTURE_WINDOW_MINUTES):
            continue
        entries = odds.get(key, [])
        if not entries:
            continue
        date_name = race_date.isoformat()
        path = SNAPSHOT_DIR / f"{date_name}.jsonl.gz"
        record = {
            "race_key": "|".join(key),
            "venue": race["venue"],
            "date": race["date"],
            "race": race["race"],
            "scheduled_post_time": post.isoformat(),
            "captured_at": now.isoformat(),
            "minutes_to_post": round(minutes, 2),
            "odds": entries,
        }
        if append_snapshot(path, record):
            additions += 1
    print(f"Captured {additions} pre-off race snapshots at {now.isoformat()}")


if __name__ == "__main__":
    main()
