#!/usr/bin/env python3
"""Research-only A45 pre-off ranking capture.

Builds the frozen 45-feature LambdaRank baseline from completed monthly NAR
archives, scores only current official daily racecards 2-60 minutes before
scheduled post time, and appends immutable prediction snapshots. No odds,
payouts, current-race outcomes, production weights, or ticket logic are used.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import nar_backtest as nb
import nar_ml_challenger_backtest as suite
import capture_nar_odds as capture
import research_flat_lambdarank_feature_ablation as base

JST = ZoneInfo("Asia/Tokyo")
MODEL_VERSION = "lambdarank-a45-research-v1"
OUT = ROOT / "data" / "research-a45-predictions.jsonl"
REPORT = ROOT / "data" / "research-a45-capture-report.json"
EXISTING_BODY_WEIGHT = set(base.EXCLUDED_BODY_WEIGHT)


def parse_post_time(value: str) -> dt.time | None:
    digits = re.sub(r"\D", "", str(value or ""))
    if len(digits) not in (3, 4):
        return None
    digits = digits.zfill(4)
    hour, minute = int(digits[:2]), int(digits[2:])
    if hour > 23 or minute > 59:
        return None
    return dt.time(hour, minute)


def read_daily_racecards(path: Path):
    """Parse the official daily ZIP using the same CSV column contract as monthly archives."""
    races, horses = {}, defaultdict(list)
    if not path.is_file():
        raise RuntimeError(f"Official daily racecard ZIP is missing: {path}")
    import zipfile
    with zipfile.ZipFile(path) as archive:
        for member in archive.namelist():
            kind = capture.kind_for(member)
            if kind not in ("races", "horses") or member.endswith("/"):
                continue
            for row in capture.csv_rows(archive.read(member)):
                if kind == "races":
                    if len(row) < 27:
                        continue
                    key = nb.race_key(row[0], row[1], row[2])
                    if not key:
                        continue
                    races[key] = {
                        "key": key,
                        "date": dt.datetime.strptime(key[1], "%Y%m%d").date(),
                        "venue": key[0],
                        "race": key[2],
                        "name": row[5] if len(row) > 5 else "",
                        "surface": row[21] if len(row) > 21 else "",
                        "distance": nb.number(row[23]) if len(row) > 23 else None,
                        "going": row[25] if len(row) > 25 else "",
                        "field_size": nb.number(row[26]) if len(row) > 26 else None,
                        "scheduled_time_raw": row[3] if len(row) > 3 else "",
                    }
                else:
                    if len(row) <= 34:
                        continue
                    key = nb.race_key(row[0], row[1], row[2])
                    if not key:
                        continue
                    number = re.sub(r"\D", "", str(row[5] or ""))
                    name = nb.norm(row[6])
                    if not number or not name:
                        continue
                    finish = nb.rank_value(row[31]) if len(row) > 31 else None
                    horses[key].append({
                        "no": str(int(number)),
                        "name": name,
                        "race_key": key,
                        "frame_no": nb.rank_value(row[3]) if len(row) > 3 else None,
                        "sex": nb.norm(row[7]) if len(row) > 7 else "",
                        "age": nb.number(row[8]) if len(row) > 8 else None,
                        "birth_date": re.sub(r"\D", "", str(row[10] or "")) if len(row) > 10 else "",
                        "sire": nb.norm(row[11]) if len(row) > 11 else "",
                        "dam": nb.norm(row[12]) if len(row) > 12 else "",
                        "damsire": nb.norm(row[13]) if len(row) > 13 else "",
                        "jockey": nb.norm(row[14]) if len(row) > 14 else "",
                        "carried": nb.number(row[16]) if len(row) > 16 else None,
                        # Current-race body weight/change are intentionally omitted.
                        "finish": finish,
                        "time_seconds": None,
                        "last3f": None,
                    })
    for key, race in races.items():
        race["field_size"] = int(race.get("field_size") or len(horses.get(key, [])))
    return races, dict(horses)


def eligible_races(races, horses, now, min_minutes=2.0, max_minutes=60.0):
    eligible = {}
    for key, race in races.items():
        venue = str(race.get("venue") or "")
        if base.BANEI_VENUE.search(venue):
            continue
        field = horses.get(key, [])
        if len(field) < 3:
            continue
        post = parse_post_time(race.get("scheduled_time_raw"))
        if post is None:
            continue
        scheduled = dt.datetime.combine(race["date"], post, tzinfo=JST)
        minutes = (scheduled - now).total_seconds() / 60.0
        if min_minutes <= minutes <= max_minutes:
            eligible[key] = (scheduled, minutes)
    return eligible


def category_maps(rows):
    return {
        key: {value: i for i, value in enumerate(sorted({row[key] for row in rows}))}
        for key in suite.CATEGORICAL
    }


def feature_matrix(rows, features, maps):
    import numpy as np
    return np.asarray([
        [row[key] if key not in suite.CATEGORICAL else maps[key].get(row[key], -1)
         for key in features]
        for row in rows
    ], dtype=float)


def train_frozen_a45(rows):
    from lightgbm import LGBMRanker
    dates = sorted({dt.date.fromisoformat(row["date"]) for row in rows})
    if len(dates) < 180:
        raise RuntimeError(f"Insufficient historical dates for A45 training: {len(dates)}")
    cut = int(len(dates) * 0.90)
    fit_dates, early_dates = set(dates[:cut]), set(dates[cut:])
    fit_rows = [row for row in rows if dt.date.fromisoformat(row["date"]) in fit_dates]
    early_rows = [row for row in rows if dt.date.fromisoformat(row["date"]) in early_dates]
    fit_groups = suite.group_rows(fit_rows)
    early_groups = suite.group_rows(early_rows)
    all_groups = suite.group_rows(rows)
    if not fit_groups or not early_groups or not all_groups:
        raise RuntimeError("A45 chronological training split is empty")

    fit_maps = category_maps(fit_rows)
    early_model = base.fit_ranker(
        fit_groups, early_groups, list(suite.RAW_FEATURES), fit_maps
    )
    best_iteration = int(getattr(early_model, "best_iteration_", 0) or base.PARAMS["n_estimators"])
    final_maps = category_maps(rows)
    x_train, y_train, group_sizes = base.flatten(
        all_groups, list(suite.RAW_FEATURES), final_maps
    )
    params = dict(base.PARAMS)
    params["n_estimators"] = max(1, best_iteration)
    final_model = LGBMRanker(**params)
    final_model.fit(x_train, y_train, group=group_sizes)
    return final_model, final_maps, {
        "training_races": len(all_groups),
        "training_runners": len(rows),
        "training_dates": len(dates),
        "training_start": dates[0].isoformat(),
        "training_end": dates[-1].isoformat(),
        "early_stopping_dates": len(early_dates),
        "best_iteration": best_iteration,
        "feature_count": len(suite.RAW_FEATURES),
        "feature_names": list(suite.RAW_FEATURES),
        "parameters": params,
    }


def main():
    now = dt.datetime.now(JST)
    # Freeze the already evaluated schema: exclude current-race bodyweight fields.
    suite.RAW_NUMERIC = [f for f in suite.RAW_NUMERIC if f not in EXISTING_BODY_WEIGHT]
    suite.RAW_FEATURES = suite.RAW_NUMERIC + suite.CATEGORICAL
    features = list(suite.RAW_FEATURES)
    if len(features) != 45:
        raise RuntimeError(f"A45 feature schema changed unexpectedly: {len(features)}")

    daily_races, daily_horses = read_daily_racecards(ROOT / "data" / "latest-race.zip")
    eligible = eligible_races(daily_races, daily_horses, now)
    if not eligible:
        report = {
            "generated_at": now.isoformat(),
            "status": "no_races_in_capture_window",
            "capture_window_minutes": [2, 60],
            "daily_racecard_races": len(daily_races),
            "eligible_races": 0,
            "predictions_written": 0,
            "model_version": MODEL_VERSION,
        }
        REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("A45_CAPTURE_REPORT " + json.dumps(report, ensure_ascii=False), flush=True)
        return

    historical_races, historical_horses, scope = base.load_flat()
    # The monthly archive ends at the last completed month; today's official
    # daily card is added only as an outcome-free scoring target.
    for key, race in daily_races.items():
        if not base.BANEI_VENUE.search(str(race.get("venue") or "")):
            historical_races[key] = race
            historical_horses[key] = daily_horses.get(key, [])

    all_rows, all_dates = suite.build_rows(
        historical_races, historical_horses, include_unfinished=True
    )
    training_rows = [
        row for row in all_rows
        if row.get("finish") is not None
        and dt.date.fromisoformat(row["date"]) <= nb.END
    ]
    if not training_rows:
        raise RuntimeError("No completed historical rows available for A45 training")
    target_keys = {"|".join(key) for key in eligible}
    target_rows = [
        row for row in all_rows
        if row["race_key"] in target_keys and row.get("finish") is None
    ]
    target_by_race = defaultdict(list)
    for row in target_rows:
        target_by_race[row["race_key"]].append(row)
    target_by_race = {
        key: rows for key, rows in target_by_race.items() if len(rows) >= 3
    }
    if not target_by_race:
        raise RuntimeError(
            "Eligible races have no outcome-free runner rows; check official racecard finish/status columns"
        )

    # Do not overwrite an earlier immutable prediction for the same race.
    existing = set()
    if OUT.exists():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
                if item.get("model_version") == MODEL_VERSION:
                    existing.add(item.get("race_id"))
            except json.JSONDecodeError:
                continue
    candidate_pending = set(target_by_race) - existing
    if not candidate_pending:
        report = {
            "generated_at": dt.datetime.now(JST).isoformat(),
            "status": "all_eligible_races_already_locked",
            "eligible_races": len(eligible),
            "already_locked": len(target_by_race),
            "predictions_written": 0,
            "model_version": MODEL_VERSION,
        }
        REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("A45_CAPTURE_REPORT " + json.dumps(report, ensure_ascii=False), flush=True)
        return

    model, maps, training = train_frozen_a45(training_rows)
    # Training can take long enough for a race to start. Re-check the capture
    # window after fitting; never backdate a prediction to the job start time.
    capture_now = dt.datetime.now(JST)
    eligible_at_capture = eligible_races(daily_races, daily_horses, capture_now)
    pending_keys = candidate_pending & set(target_by_race) & set(eligible_at_capture)
    if not pending_keys:
        report = {
            "generated_at": capture_now.isoformat(),
            "status": "capture_window_elapsed_during_training",
            "initially_eligible_races": len(eligible),
            "eligible_after_training": len(eligible_at_capture),
            "predictions_written": 0,
            "model_version": MODEL_VERSION,
            "training": training,
        }
        REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("A45_CAPTURE_REPORT " + json.dumps(report, ensure_ascii=False), flush=True)
        return

    records = []
    for race_key in sorted(pending_keys):
        rows = target_by_race[race_key]
        matrix = feature_matrix(rows, features, maps)
        scores = model.predict(matrix)
        order = sorted(
            range(len(rows)),
            key=lambda i: (-float(scores[i]), int(rows[i]["horse_no"]))
        )
        rank_by_index = {index: rank + 1 for rank, index in enumerate(order)}
        race_tuple = tuple(race_key.split("|"))
        scheduled, minutes = eligible_at_capture[race_tuple]
        runners = []
        for index, row in enumerate(rows):
            runners.append({
                "horse_no": str(row["horse_no"]),
                "name": next((h.get("name") for h in daily_horses.get(race_tuple, [])
                              if str(h.get("no")) == str(row["horse_no"])), ""),
                "rank": rank_by_index[index],
                "score": float(scores[index]),
                "features": {key: (float(row[key]) if key not in suite.CATEGORICAL else str(row[key]))
                             for key in features},
            })
        runners.sort(key=lambda item: item["rank"])
        record = {
            "race_id": race_key,
            "race_date": race_tuple[1],
            "venue": race_tuple[0],
            "race_number": race_tuple[2],
            "captured_at": None,
            "scheduled_post_time": scheduled.isoformat(),
            "minutes_to_post": round(minutes, 3),
            "model_version": MODEL_VERSION,
            "feature_schema": features,
            "training": training,
            "source": "official_nar_daily_racecard",
            "prediction_policy": {
                "odds_used_as_features": False,
                "payouts_used_as_features": False,
                "current_race_results_used_as_features": False,
                "current_race_bodyweight_used_as_features": False,
                "same_day_results_withheld_until_day_complete": True,
                "capture_window_minutes": [2, 60],
            },
            "predictions": runners,
        }
        records.append(record)

    # Timestamp after scoring is conservative: it cannot make a late
    # prediction appear earlier than it really was.
    prediction_completed_at = dt.datetime.now(JST)
    valid_records = []
    for record in records:
        post = dt.datetime.fromisoformat(record["scheduled_post_time"])
        minutes = (post - prediction_completed_at).total_seconds() / 60.0
        if 2.0 <= minutes <= 60.0:
            record["captured_at"] = prediction_completed_at.isoformat()
            record["minutes_to_post"] = round(minutes, 3)
            valid_records.append(record)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a", encoding="utf-8") as stream:
        for record in valid_records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    report = {
        "generated_at": prediction_completed_at.isoformat(),
        "status": "success" if valid_records else "capture_window_elapsed_during_scoring",
        "model_version": MODEL_VERSION,
        "daily_racecard_races": len(daily_races),
        "eligible_races": len(eligible),
        "already_locked": len(existing),
        "predictions_written": len(valid_records),
        "predictions_discarded_after_time_recheck": len(records) - len(valid_records),
        "training": training,
        "historical_scope": scope,
        "feature_schema": features,
        "output": str(OUT.relative_to(ROOT)),
        "limitations": [
            "Research only; no production prediction, ticket generation, or Supabase write.",
            "Prediction capture is immutable per race and first eligible capture wins.",
            "Pre-off odds are joined only in a separate evaluation step.",
            "Historical model identity remains limited by the monthly archive's available horse identifiers.",
        ],
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("A45_CAPTURE_REPORT " + json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
