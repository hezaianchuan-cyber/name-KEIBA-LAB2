#!/usr/bin/env python3
"""Research-only CatBoost/LightGBM challenger against a frozen LAB baseline.

Builds runner-level features strictly chronologically from official NAR archives.
The final 20% of dates is an untouched holdout; no production model is modified.
"""
from __future__ import annotations
import datetime as dt
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import nar_backtest as nb

OUT = ROOT / "data" / "ml-challenger-backtest.json"
BASE_WEIGHTS = {"form": 0.55, "venue": 0.12, "distance": 0.12,
                "sire": 0.10, "jockey": 0.07, "age": 0.04}
NUMERIC = ["lab_score", "form", "venue_score", "distance_score", "sire_score",
           "jockey_score", "age_score", "field_size", "distance_m"]
CATEGORICAL = ["venue", "surface", "going"]
FEATURES = NUMERIC + CATEGORICAL


def build_rows(races, horses):
    history = defaultdict(list)
    sire_stats = defaultdict(lambda: {"runs": 0, "top3": 0})
    jockey_stats = defaultdict(lambda: {"runs": 0, "top3": 0})
    rows, race_rows = [], []
    ordered = sorted(races.values(), key=lambda r: (r["date"], r["venue"], int(r["race"])))
    dates = sorted({r["date"] for r in ordered})
    progress = max(1, len(ordered) // 10)
    for i, race in enumerate(ordered, 1):
        if i == 1 or i % progress == 0 or i == len(ordered):
            print(f"ML_DATA_PROGRESS {i}/{len(ordered)} date={race['date']}", flush=True)
        field = horses.get(race["key"], [])
        if len(field) < 3:
            continue
        pending = []
        for h in field:
            _, parts = nb.score_runner(h, history, sire_stats, jockey_stats, race, BASE_WEIGHTS)
            score = sum(BASE_WEIGHTS[k] * parts[k] for k in BASE_WEIGHTS)
            finish = nb.rank_value(h.get("finish"))
            if finish is None:
                continue
            row = {
                "race_key": "|".join(map(str, race["key"])),
                "date": race["date"].isoformat(),
                "horse_no": str(h.get("no", "")),
                "finish": finish,
                "winner": int(finish == 1),
                "top3": int(finish <= 3),
                "lab_score": float(score),
                "form": float(parts["form"]),
                "venue_score": float(parts["venue"]),
                "distance_score": float(parts["distance"]),
                "sire_score": float(parts["sire"]),
                "jockey_score": float(parts["jockey"]),
                "age_score": float(parts["age"]),
                "field_size": float(race.get("field_size") or len(field)),
                "distance_m": float(race.get("distance") or 0),
                "venue": str(race.get("venue") or "unknown"),
                "surface": str(race.get("surface") or "unknown"),
                "going": str(race.get("going") or "unknown"),
            }
            rows.append(row)
            pending.append(h)
        # Same-date races must not learn from each other's outcomes.
        race_rows.append((race["date"], pending, race))
        if i == len(ordered) or ordered[i]["date"] != race["date"]:
            for _, day_field, day_race in race_rows:
                performance = nb.field_performance_features(day_field)
                for runner in day_field:
                    if not runner.get("finish"):
                        continue
                    history[runner["name"]].append({
                        **performance.get(runner["no"], {}),
                        "date": day_race["date"], "venue": day_race["venue"],
                        "distance": day_race.get("distance"),
                        "going": day_race.get("going"), "surface": day_race.get("surface"),
                        "field_size": day_race.get("field_size"),
                        "finish": runner["finish"],
                        "margin": nb.margin_score(runner.get("margin")),
                        "last3f": runner.get("last3f"),
                        "going_quality": nb.rank_quality(runner["finish"], day_race.get("field_size")) / 100.0,
                        "time_seconds": runner.get("time_seconds"),
                        "time_per_km": (runner.get("time_seconds") * 1000 / day_race["distance"]
                                        if runner.get("time_seconds") and day_race.get("distance") else None),
                        "frame_no": runner.get("frame_no"),
                        **nb.corner_history_for_runner(runner, day_race),
                    })
                    if runner.get("sire"):
                        sire_stats[runner["sire"]]["runs"] += 1
                        sire_stats[runner["sire"]]["top3"] += int(runner["finish"] <= 3)
                    if runner.get("jockey"):
                        jockey_stats[runner["jockey"]]["runs"] += 1
                        jockey_stats[runner["jockey"]]["top3"] += int(runner["finish"] <= 3)
            race_rows = []
    return rows, dates


def summarize(name, model, test, categorical=False):
    from collections import defaultdict
    grouped = defaultdict(list)
    for row in test:
        grouped[row["race_key"]].append(row)
    top1 = top3 = races = 0
    for group in grouped.values():
        if len(group) < 3:
            continue
        if categorical:
            x = [[r[k] if k in CATEGORICAL else r[k] for k in FEATURES] for r in group]
        else:
            x = [[r[k] for k in FEATURES] for r in group]
        probs = model.predict_proba(x)[:, 1]
        order = sorted(range(len(group)), key=lambda z: (-float(probs[z]), group[z]["horse_no"]))
        winner_idx = next((i for i, r in enumerate(group) if r["winner"]), None)
        if winner_idx is None:
            continue
        races += 1
        top1 += int(order[0] == winner_idx)
        top3 += int(winner_idx in order[:3])
    return {"name": name, "races": races,
            "top1_rate": round(100 * top1 / races, 3) if races else None,
            "top3_capture_rate": round(100 * top3 / races, 3) if races else None,
            "top1_hits": top1, "top3_hits": top3}


def main():
    races, horses, payouts, downloads = nb.load_archives()
    rows, dates = build_rows(races, horses)
    if len(dates) < 100:
        raise RuntimeError(f"Insufficient chronological date coverage: {len(dates)}")
    cut_train = max(1, int(len(dates) * 0.70))
    cut_dev = max(cut_train + 1, int(len(dates) * 0.80))
    train_dates, dev_dates, test_dates = set(dates[:cut_train]), set(dates[cut_train:cut_dev]), set(dates[cut_dev:])
    train = [r for r in rows if dt.date.fromisoformat(r["date"]) in train_dates]
    dev = [r for r in rows if dt.date.fromisoformat(r["date"]) in dev_dates]
    test = [r for r in rows if dt.date.fromisoformat(r["date"]) in test_dates]
    if not train or not dev or not test:
        raise RuntimeError("Chronological train/development/holdout split is empty")
    import numpy as np
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    # Development period is reported separately; parameters are fixed in advance.
    category_maps = {
        key: {value: index for index, value in enumerate(sorted({r[key] for r in train}))}
        for key in CATEGORICAL
    }
    def matrix(data, cat=False):
        if cat:
            return [[r[k] if k in CATEGORICAL else r[k] for k in FEATURES] for r in data]
        # Train-only category dictionaries prevent unseen holdout categories from
        # influencing the feature vocabulary; unknown categories map to -1.
        return np.asarray([
            [r[k] if k in NUMERIC else category_maps[k].get(r[k], -1) for k in FEATURES]
            for r in data
        ], dtype=float)
    y_train = np.asarray([r["winner"] for r in train], dtype=int)
    cat = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.04,
        loss_function="Logloss", eval_metric="AUC", verbose=False,
        random_seed=42, allow_writing_files=False, thread_count=4)
    cat.fit(matrix(train, True), y_train, cat_features=list(range(len(NUMERIC), len(FEATURES))),
            eval_set=(matrix(dev, True), np.asarray([r["winner"] for r in dev], dtype=int)),
            early_stopping_rounds=50, verbose=False)
    lgb = LGBMClassifier(n_estimators=500, learning_rate=0.04, num_leaves=15,
        max_depth=-1, min_child_samples=100, reg_lambda=2.0,
        subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=4,
        verbosity=-1)
    lgb.fit(matrix(train), y_train, eval_set=[(matrix(dev), np.asarray([r["winner"] for r in dev], dtype=int))],
            callbacks=[])
    # Frozen LAB baseline ranking on the identical untouched holdout.
    lab = {"name": "LAB_frozen_baseline", "races": 0, "top1_hits": 0, "top3_hits": 0}
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in test:
        grouped[r["race_key"]].append(r)
    for group in grouped.values():
        if len(group) < 3 or not any(r["winner"] for r in group):
            continue
        order = sorted(range(len(group)), key=lambda i: (-group[i]["lab_score"], group[i]["horse_no"]))
        winner = next(i for i, r in enumerate(group) if r["winner"])
        lab["races"] += 1
        lab["top1_hits"] += int(order[0] == winner)
        lab["top3_hits"] += int(winner in order[:3])
    for key, hits in (("top1_rate", "top1_hits"), ("top3_capture_rate", "top3_hits")):
        lab[key] = round(100 * lab[hits] / lab["races"], 3) if lab["races"] else None
    results = [lab, summarize("CatBoost", cat, test, True), summarize("LightGBM", lgb, test)]
    report = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "evaluated",
        "period": {"start": dates[0].isoformat(), "end": dates[-1].isoformat()},
        "split": {"train_end": dates[cut_train-1].isoformat(),
                  "development_start": dates[cut_train].isoformat(),
                  "development_end": dates[cut_dev-1].isoformat(),
                  "holdout_start": dates[cut_dev].isoformat(), "holdout_end": dates[-1].isoformat(),
                  "policy": "70% earliest dates train, next 10% development for early stopping, final 20% untouched holdout"},
        "data": {"races": len(races), "runner_rows": len(rows), "dates": len(dates),
                 "downloaded_months": downloads},
        "features": FEATURES,
        "models": results,
        "limitations": [
            "Research-only runner ranking; no production code or tickets changed.",
            "Winner classification is trained per runner; top-k is evaluated within race.",
            "LAB reference is a frozen baseline feature score, not the adaptive production B/C model.",
            "No profitability claim: this first experiment tests ranking skill only.",
            "CatBoost and LightGBM parameters were fixed before holdout; no tuning on holdout."
        ]
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "data": report["data"],
                      "split": report["split"], "models": report["models"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
