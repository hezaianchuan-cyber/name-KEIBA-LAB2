#!/usr/bin/env python3
"""Capture official NAR odds snapshots from 15 to 5 minutes before post time.

The monthly NAR odds archive contains confirmed odds and must not be used as
pre-race information. This collector uses only the frequently refreshed daily
odds ZIP and records the capture timestamp alongside every race snapshot.
"""
from __future__ import annotations

import csv
from collections import defaultdict
import gzip
import datetime as dt
import io
import json
import re
import unicodedata
import zipfile
from pathlib import Path
from zoneinfo import ZoneInfo

import nar_backtest as nb

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SNAPSHOT_DIR = DATA / "odds-snapshots"
JST = ZoneInfo("Asia/Tokyo")
CAPTURE_WINDOW_MINUTES = 15
STOP_BEFORE_POST_MINUTES = 5


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
    if re.search(r"horselist|horse.?list|出馬表", base):
        return "horses"
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
    horse_rows = {}
    odds_rows = {}
    race_zip = DATA / "latest-race.zip"
    odds_zip = DATA / "latest-odds.zip"
    if not race_zip.exists() or not odds_zip.exists():
        return race_rows, horse_rows, odds_rows
    with zipfile.ZipFile(race_zip) as archive:
        for name in archive.namelist():
            kind = kind_for(name)
            if kind == "races":
                for row in csv_rows(archive.read(name)):
                    if len(row) < 27:
                        continue
                    key = race_id(row[0], row[1], row[2])
                    if key:
                        race_rows[key] = {
                            "venue": key[0], "date": key[1], "race": key[2], "time": row[3],
                            "name": row[5] if len(row) > 5 else "",
                            "surface": row[21] if len(row) > 21 else "",
                            "distance": nb.number(row[23]) if len(row) > 23 else None,
                            "going": row[25] if len(row) > 25 else "",
                            "field_size": nb.number(row[26]) if len(row) > 26 else None,
                        }
            elif kind == "horses":
                for row in csv_rows(archive.read(name)):
                    if len(row) < 15:
                        continue
                    key = race_id(row[0], row[1], row[2])
                    if not key:
                        continue
                    no = re.sub(r"\D", "", str(row[5] or ""))
                    horse_name = str(row[6] or "").strip()
                    if not no or not horse_name:
                        continue
                    horse_rows.setdefault(key, []).append({
                        "no": str(int(no)),
                        "name": horse_name,
                        "age": nb.number(row[8]) if len(row) > 8 else None,
                        "sire": str(row[11] or "").strip() if len(row) > 11 else "",
                        "jockey": str(row[14] or "").strip() if len(row) > 14 else "",
                    })
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
                    odds_value = float(str(row[7]).replace(",", "").strip())
                except (ValueError, TypeError):
                    continue
                if odds_value <= 0:
                    continue
                item = {
                    "type": str(row[3]).strip(),
                    "n1": str(row[4]).strip(),
                    "n2": str(row[5]).strip(),
                    "n3": str(row[6]).strip(),
                    "odds": odds_value,
                    "max_odds": str(row[8]).strip(),
                    "popularity": str(row[9]).strip(),
                }
                odds_rows.setdefault(key, []).append(item)
    return race_rows, horse_rows, odds_rows


def load_model_context():
    model_path = DATA / "live-model.json"
    history_path = DATA / "live-history.json"
    if not model_path.exists() or not history_path.exists():
        return None
    model = json.loads(model_path.read_text(encoding="utf-8"))
    history_data = json.loads(history_path.read_text(encoding="utf-8"))
    history = defaultdict(list, history_data.get("horse_history") or {})
    sire_stats = defaultdict(lambda: {"runs": 0, "top3": 0}, history_data.get("sire_stats") or {})
    jockey_stats = defaultdict(lambda: {"runs": 0, "top3": 0}, history_data.get("jockey_stats") or {})
    weights = model.get("weights") or {}
    if not all(key in weights for key in ("form", "venue", "distance", "sire", "jockey", "age")):
        return None
    return model, history, sire_stats, jockey_stats, weights


def predict_horses(race, horses, odds_rows, model_context):
    if not model_context or len(horses) < 3:
        return []
    model, history, sire_stats, jockey_stats, weights = model_context
    scored = [nb.score_runner(horse, history, sire_stats, jockey_stats, race, weights)[0]
              for horse in horses]
    temperature = float(model.get("temperature") or 0.18)
    probabilities = nb.softmax_probabilities(scored, temperature)
    win_odds = {}
    for item in odds_rows:
        if "単勝" not in unicodedata.normalize("NFKC", str(item.get("type") or "")):
            continue
        no = re.sub(r"\D", "", str(item.get("n1") or ""))
        if no:
            win_odds[str(int(no))] = item.get("odds")
    predictions = []
    for horse, score, probability in zip(horses, scored, probabilities):
        odds_value = win_odds.get(horse["no"])
        ev = probability * float(odds_value) if odds_value and float(odds_value) > 0 else None
        predictions.append({
            "horse_no": horse["no"],
            "name": horse["name"],
            "score": round(score, 6),
            "win_probability": round(probability, 8),
            "win_odds": odds_value,
            "expected_value": round(ev, 6) if ev is not None else None,
        })
    return predictions


def append_snapshot(path: Path, record: dict) -> bool:
    """Append a race once as a concatenated gzip member; return whether it was new."""
    race_key_text = record.get("race_key")
    capture_time = record.get("captured_at")
    if path.exists():
        try:
            with gzip.open(path, "rt", encoding="utf-8") as existing:
                for line in existing:
                    try:
                        previous = json.loads(line)
                        if previous.get("race_key") == race_key_text and previous.get("captured_at") == capture_time:
                            return False
                    except json.JSONDecodeError:
                        continue
        except OSError:
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "ab", compresslevel=9) as output:
        output.write((json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))
    return True


def main():
    now = dt.datetime.now(JST)
    races, horses_by_race, odds = read_archives()
    if not races or not odds:
        print(f"No current official race/odds rows; race_rows={len(races)} odds_races={len(odds)}")
        return
    candidates = []
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
        if not (STOP_BEFORE_POST_MINUTES <= minutes <= CAPTURE_WINDOW_MINUTES):
            continue
        entries = odds.get(key, [])
        field = horses_by_race.get(key, [])
        if not entries or len(field) < 3:
            continue
        candidates.append((key, race, race_date, post, minutes, entries, field))
    if not candidates:
        print(f"No race inside {CAPTURE_WINDOW_MINUTES}-to-{STOP_BEFORE_POST_MINUTES}-minute pre-off window; race_rows={len(races)} odds_races={len(odds)}")
        return
    model_context = load_model_context()
    additions = 0
    for key, race, race_date, post, minutes, entries, field in candidates:
        predictions = predict_horses(race, field, entries, model_context)
        path = SNAPSHOT_DIR / f"{race_date.isoformat()}.jsonl.gz"
        model = model_context[0] if model_context else {}
        record = {
            "race_key": "|".join(key),
            "venue": race["venue"],
            "date": race["date"],
            "race": race["race"],
            "scheduled_post_time": post.isoformat(),
            "captured_at": now.isoformat(),
            "minutes_to_post": round(minutes, 2),
            "capture_policy": {"window_start_minutes": CAPTURE_WINDOW_MINUTES, "stop_before_post_minutes": STOP_BEFORE_POST_MINUTES},
            "model_version": model.get("model_version"),
            "model_temperature": model.get("temperature"),
            "predictions": predictions,
            "odds": entries,
        }
        if append_snapshot(path, record):
            additions += 1
    print(f"Captured {additions} pre-off race snapshots at {now.isoformat()}; candidates={len(candidates)} window={CAPTURE_WINDOW_MINUTES}-{STOP_BEFORE_POST_MINUTES}min model_available={model_context is not None}")


if __name__ == "__main__":
    main()
