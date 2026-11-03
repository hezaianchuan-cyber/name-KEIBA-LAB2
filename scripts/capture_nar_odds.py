#!/usr/bin/env python3
"""Capture official NAR odds snapshots from 30 to 2 minutes before post time.

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
CAPTURE_WINDOW_MINUTES = 30
STOP_BEFORE_POST_MINUTES = 2


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
    odds_diagnostics = {"rows": 0, "positive_rows": 0, "zero_rows": 0, "races_with_positive_odds": 0}
    race_zip = DATA / "latest-race.zip"
    odds_zip = DATA / "latest-odds.zip"
    if not race_zip.exists() or not odds_zip.exists():
        return race_rows, horse_rows, odds_rows
    with zipfile.ZipFile(race_zip) as archive:
        print("Race archive members:", archive.namelist()[:30], flush=True)
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
        print("Odds archive members:", archive.namelist()[:50], flush=True)
        for name in archive.namelist():
            kind = kind_for(name)
            raw_rows = csv_rows(archive.read(name)) if not name.endswith("/") else []
            if raw_rows:
                print(f"Odds member {name}: classified={kind}, rows={len(raw_rows)}, first_row_columns={len(raw_rows[0])}", flush=True)
                print("Odds sample rows:", [row[:10] for row in raw_rows[:4]], flush=True)
            if kind != "odds":
                continue
            for row in raw_rows:
                if len(row) < 10:
                    continue
                key = race_id(row[0], row[1], row[2])
                if not key:
                    continue
                try:
                    odds_value = float(str(row[7]).replace(",", "").strip())
                except (ValueError, TypeError):
                    continue
                odds_diagnostics["rows"] += 1
                if odds_value <= 0:
                    odds_diagnostics["zero_rows"] += 1
                    continue
                odds_diagnostics["positive_rows"] += 1
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
    odds_diagnostics["races_with_positive_odds"] = len(odds_rows)
    print("Odds diagnostics:", json.dumps(odds_diagnostics, ensure_ascii=False), flush=True)
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


def ordered_finish_probability(order, probabilities):
    """Plackett-Luce probability for an exact ordered top-k finish."""
    remaining = sum(probabilities.values())
    result = 1.0
    used = set()
    for raw_no in order:
        no = str(int(re.sub(r"\D", "", str(raw_no or "")) or 0))
        if no in used or no not in probabilities or remaining <= 0:
            return 0.0
        weight = probabilities[no]
        result *= weight / remaining
        remaining -= weight
        used.add(no)
    return result


def combination_probability(kind, numbers, probabilities):
    """Probability for a ticket under a Plackett-Luce top-three finish model."""
    import itertools
    numbers = [str(int(re.sub(r"\D", "", str(x or "")) or 0)) for x in numbers]
    if kind == "win" and len(numbers) == 1:
        return probabilities.get(numbers[0], 0.0)
    if kind == "place" and len(numbers) == 1:
        target = numbers[0]
        if target not in probabilities:
            return 0.0
        others = [x for x in probabilities if x != target]
        total = 0.0
        for a in others:
            total += ordered_finish_probability([target, a], probabilities)
            total += ordered_finish_probability([a, target], probabilities)
            for b in others:
                if b != a:
                    total += ordered_finish_probability([a, b, target], probabilities)
        return min(1.0, total)
    if kind in ("quinella", "wide") and len(numbers) == 2:
        a, b = numbers
        if a not in probabilities or b not in probabilities or a == b:
            return 0.0
        if kind == "quinella":
            return ordered_finish_probability([a, b], probabilities) + ordered_finish_probability([b, a], probabilities)
        total = 0.0
        for third in probabilities:
            if third not in (a, b):
                for order in ((a,b,third),(a,third,b),(b,a,third),(b,third,a),(third,a,b),(third,b,a)):
                    total += ordered_finish_probability(order, probabilities)
        return min(1.0, total)
    if kind == "exacta" and len(numbers) == 2:
        return ordered_finish_probability(numbers, probabilities)
    if kind == "trio" and len(numbers) == 3:
        return sum(ordered_finish_probability(order, probabilities) for order in itertools.permutations(numbers))
    if kind == "trifecta" and len(numbers) == 3:
        return ordered_finish_probability(numbers, probabilities)
    return 0.0


def predict_combinations(odds_rows, horse_predictions):
    """Generate the full rank-based ticket portfolio and attach available official odds."""
    import itertools
    probabilities = {str(p["horse_no"]): float(p["win_probability"]) for p in horse_predictions}
    ranked = sorted(horse_predictions, key=lambda p: (
        float(p.get("win_probability") or 0), float(p.get("score") or 0)), reverse=True)
    ticket_map = nb.build_fixed_tickets([{"no":str(p["horse_no"])} for p in ranked])
    type_map = {
        "単勝":"win", "複勝":"place", "ワイド":"wide", "馬連":"quinella",
        "馬単":"exacta", "3連複":"trio", "三連複":"trio",
        "3連単":"trifecta", "三連単":"trifecta",
    }
    unordered = {"place","wide","quinella","trio"}
    odds_lookup = {}
    for item in odds_rows:
        label = unicodedata.normalize("NFKC", str(item.get("type") or "")).replace(" ", "")
        kind = next((value for key,value in type_map.items() if key in label), None)
        if not kind:
            continue
        numbers = [str(int(re.sub(r"\D", "", str(item.get(key) or "")) or 0))
                   for key in ("n1","n2","n3") if re.sub(r"\D", "", str(item.get(key) or ""))]
        required = {"win":1,"place":1,"wide":2,"quinella":2,"exacta":2,"trio":3,"trifecta":3}[kind]
        if len(numbers) != required:
            continue
        try:
            odds_value = float(item.get("odds"))
        except (TypeError, ValueError):
            continue
        if odds_value <= 0:
            continue
        key_numbers = tuple(sorted(numbers)) if kind in unordered else tuple(numbers)
        odds_lookup[(kind,key_numbers)] = odds_value

    output = []
    for kind,tickets in ticket_map.items():
        for numbers in tickets:
            key_numbers = tuple(sorted(numbers)) if kind in unordered else tuple(numbers)
            odds_value = odds_lookup.get((kind,key_numbers))
            probability = combination_probability(kind,numbers,probabilities)
            output.append({
                "bet_type":kind,
                "numbers":numbers,
                "odds":odds_value,
                "probability":round(probability,10),
                "expected_value":round(probability*odds_value,6) if odds_value is not None else None,
                "odds_status":"取得済み" if odds_value is not None else "オッズ待ち",
            })
    return output

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


def write_capture_status(now, races, horses_by_race, odds, **extra):
    status = {
        "updated_at": now.isoformat(),
        "capture_window_minutes": [CAPTURE_WINDOW_MINUTES, STOP_BEFORE_POST_MINUTES],
        "race_rows": len(races),
        "horse_rows": sum(len(rows) for rows in horses_by_race.values()),
        "odds_rows": sum(len(rows) for rows in odds.values()),
        "races_with_horses": sum(1 for rows in horses_by_race.values() if len(rows) >= 3),
        "races_with_odds": len(odds),
        **extra,
    }
    path = DATA / "odds-capture-status.json"
    path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Capture status:", json.dumps(status, ensure_ascii=False), flush=True)


def main():
    now = dt.datetime.now(JST)
    races, horses_by_race, odds = read_archives()
    if not races:
        write_capture_status(now, races, horses_by_race, odds, result="no_race_rows", captured_races=0)
        print(f"No current official race rows; race_rows={len(races)}")
        return
    candidates = []
    window_races = 0
    window_with_field = 0
    window_with_positive_odds = 0
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
        window_races += 1
        entries = odds.get(key, [])
        field = horses_by_race.get(key, [])
        if len(field) >= 3:
            window_with_field += 1
        if entries and len(field) >= 3:
            window_with_positive_odds += 1
            candidates.append((key, race, race_date, post, minutes, entries, field))
    print(
        "Pre-off eligibility:",
        json.dumps({
            "window_minutes": [CAPTURE_WINDOW_MINUTES, STOP_BEFORE_POST_MINUTES],
            "races_in_window": window_races,
            "window_races_with_field": window_with_field,
            "window_races_with_positive_odds": window_with_positive_odds,
            "odds_races_with_positive_quotes": len(odds),
        }, ensure_ascii=False),
        flush=True,
    )
    if not candidates:
        write_capture_status(
            now, races, horses_by_race, odds,
            result="no_eligible_preoff_races",
            races_in_window=window_races,
            window_races_with_field=window_with_field,
            window_races_with_positive_odds=window_with_positive_odds,
            captured_races=0,
        )
        print("No eligible pre-off races yet; the next scheduled capture will retry.")
        return
    model_context = load_model_context()
    additions = 0
    for key, race, race_date, post, minutes, entries, field in candidates:
        predictions = predict_horses(race, field, entries, model_context)
        combination_predictions = predict_combinations(entries, predictions)
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
            "combination_predictions": combination_predictions,
            "odds": entries,
        }
        if append_snapshot(path, record):
            additions += 1
    write_capture_status(
        now, races, horses_by_race, odds,
        result="captured" if additions else "already_recorded",
        races_in_window=window_races,
        window_races_with_field=window_with_field,
        window_races_with_positive_odds=window_with_positive_odds,
        candidate_races=len(candidates),
        captured_races=additions,
        model_available=model_context is not None,
    )
    print(f"Captured {additions} pre-off race snapshots at {now.isoformat()}; candidates={len(candidates)} window={CAPTURE_WINDOW_MINUTES}-{STOP_BEFORE_POST_MINUTES}min model_available={model_context is not None}")


if __name__ == "__main__":
    main()
