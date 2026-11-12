#!/usr/bin/env python3
"""Leakage-safe Banei historical body-weight profile experiment.

Current-race body weight/body-weight-change are deliberately NOT predictors:
the monthly result archive does not prove those values were available before
the start. Only values from earlier starts are used to build horse profiles.
Current carried weight is a separate, pre-race field-card feature.
"""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import nar_backtest as nb

OUT = nb.ROOT / "data" / "banei-historical-body-weight-profile.json"
SEED = 20261113
BOOTSTRAPS = 2000


def is_banei(venue):
    value = nb.norm(venue)
    return "帯広" in value or "ばんえい" in value.lower()


def safe_mean(values, default=None):
    vals = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return sum(vals) / len(vals) if vals else default


def safe_sd(values, default=0.0):
    vals = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    if len(vals) < 2:
        return default
    mean = sum(vals) / len(vals)
    return math.sqrt(sum((x - mean) ** 2 for x in vals) / len(vals))


def build_rows(races, horses):
    history = defaultdict(list)
    records = []
    ordered = sorted(races.items(), key=lambda item: (item[1]["date"], item[0]))
    for key, race in ordered:
        if not is_banei(race.get("venue")):
            continue
        field = horses.get(key, [])
        if len(field) < 5:
            continue

        carried = [nb.number(h.get("carried")) for h in field]
        valid_carried = [x for x in carried if x is not None]
        if len(valid_carried) >= max(3, math.ceil(len(field) * 0.70)):
            mean_carried = safe_mean(valid_carried, 0.0)
            sd_carried = safe_sd(valid_carried, 1.0)
            if sd_carried < 1e-9:
                sd_carried = 1.0
        else:
            mean_carried, sd_carried = 0.0, 1.0

        for horse, current_carried in zip(field, carried):
            name = horse.get("name")
            past = history[name]
            finishes = [nb.number(x.get("finish")) for x in past]
            finishes = [x for x in finishes if x is not None and x > 0]
            recent_finishes = finishes[-5:]
            last_finish = finishes[-1] if finishes else None
            last_field = nb.number(past[-1].get("field_size")) if past else None
            age = nb.number(horse.get("age"))

            # These body-weight measurements come only from completed earlier
            # starts. Never read horse['body_weight'] for the current race here.
            past_body = [nb.number(x.get("body_weight")) for x in past]
            past_body = [x for x in past_body if x is not None and x > 0]
            past_changes = [nb.number(x.get("body_weight_change")) for x in past]
            past_changes = [x for x in past_changes if x is not None]
            past_carried = [nb.number(x.get("carried")) for x in past]
            past_carried = [x for x in past_carried if x is not None]
            past_ratios = [
                100.0 * nb.number(x.get("carried")) / nb.number(x.get("body_weight"))
                for x in past
                if nb.number(x.get("carried")) is not None
                and nb.number(x.get("body_weight")) is not None
                and nb.number(x.get("body_weight")) > 0
            ]

            base = [
                math.log1p(len(finishes)),
                (sum(x <= 3 for x in finishes) + 1.5) / (len(finishes) + 5.0),
                (sum(x <= 3 for x in recent_finishes) + 1.0) / (len(recent_finishes) + 3.0),
                (1.0 - (last_finish - 1.0) / (last_field - 1.0))
                if last_finish is not None and last_field and last_field > 1 else 0.5,
                age if age is not None else 5.0,
            ]
            body_profile = [
                past_body[-1] if past_body else None,
                safe_mean(past_body[-3:]),
                (past_body[-1] - past_body[-2]) if len(past_body) >= 2 else None,
                safe_sd(past_body[-5:]) if len(past_body) >= 2 else None,
                past_changes[-1] if past_changes else None,
                safe_mean(past_changes[-3:]),
                safe_mean(past_ratios[-3:]),
                safe_mean(past_carried[-3:]),
                len(past_body),
            ]
            carried_z = (
                (current_carried - mean_carried) / sd_carried
                if current_carried is not None and len(valid_carried) >= 3 else None
            )

            finish = nb.number(horse.get("finish"))
            records.append({
                "date": race["date"].isoformat(),
                "race_key": "|".join(str(x) for x in key),
                "horse_no": str(horse.get("no")),
                "base": base,
                "body_profile": body_profile,
                "carried_z": carried_z,
                "finish": finish,
                "label_top3": int(finish is not None and 0 < finish <= 3),
                "label_win": int(finish is not None and finish == 1),
                "field_size": len(field),
                "history_starts": len(finishes),
                "has_prior_body_weight": bool(past_body),
            })

        # Update history only after all runners in this race have been featurized.
        for horse in field:
            history[horse.get("name")].append({
                "finish": horse.get("finish"),
                "field_size": len(field),
                "body_weight": horse.get("body_weight"),
                "body_weight_change": horse.get("body_weight_change"),
                "carried": horse.get("carried"),
            })
    return records


def fit_predict(train, test, feature_set):
    def vector(row):
        if feature_set == "base":
            return row["base"]
        if feature_set == "base_plus_past_body":
            return row["base"] + row["body_profile"]
        if feature_set == "base_plus_past_body_and_carried":
            return row["base"] + row["body_profile"] + [row["carried_z"]]
        raise ValueError(feature_set)

    x_train = np.asarray([vector(r) for r in train], dtype=float)
    y_train = np.asarray([r["label_top3"] for r in train], dtype=int)
    x_test = np.asarray([vector(r) for r in test], dtype=float)
    model = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        LogisticRegression(C=0.5, max_iter=1500, random_state=SEED),
    )
    model.fit(x_train, y_train)
    return model.predict_proba(x_test)[:, 1]


def race_metrics(records, probabilities):
    grouped = defaultdict(list)
    for row, prob in zip(records, probabilities):
        grouped[row["race_key"]].append((float(prob), row))
    result = {}
    all_y, all_p = [], []
    for race_key, entries in grouped.items():
        entries.sort(key=lambda x: (-x[0], int(x[1]["horse_no"]) if x[1]["horse_no"].isdigit() else x[1]["horse_no"]))
        actual_top3 = {r["horse_no"] for _, r in entries if r["label_top3"]}
        if len(actual_top3) != 3:
            continue
        predicted_top3 = {r["horse_no"] for _, r in entries[:3]}
        axis = entries[0][1]["horse_no"]
        winner = next((r["horse_no"] for _, r in entries if r["label_win"]), None)
        result[race_key] = {
            "axis_top3": float(axis in actual_top3),
            "axis_win": float(axis == winner),
            "top3_coverage": len(predicted_top3 & actual_top3) / 3.0,
            "full_top3": float(actual_top3.issubset(predicted_top3)),
            "winner_in_top3": float(winner in predicted_top3) if winner else 0.0,
        }
        all_y.extend([r["label_top3"] for _, r in entries])
        all_p.extend([p for p, _ in entries])
    if all_y:
        y, p = np.asarray(all_y, dtype=float), np.clip(np.asarray(all_p), 1e-8, 1-1e-8)
        brier = float(np.mean((p-y)**2))
        logloss = float(np.mean(-(y*np.log(p)+(1-y)*np.log(1-p))))
    else:
        brier = logloss = None
    return result, {
        "races": len(result),
        "runner_rows": len(all_y),
        "axis_top3_pct": round(100*np.mean([x["axis_top3"] for x in result.values()]), 3) if result else None,
        "axis_win_pct": round(100*np.mean([x["axis_win"] for x in result.values()]), 3) if result else None,
        "predicted_top3_podium_coverage_pct": round(100*np.mean([x["top3_coverage"] for x in result.values()]), 3) if result else None,
        "full_podium_in_predicted_top3_pct": round(100*np.mean([x["full_top3"] for x in result.values()]), 3) if result else None,
        "winner_in_predicted_top3_pct": round(100*np.mean([x["winner_in_top3"] for x in result.values()]), 3) if result else None,
        "runner_top3_brier": round(brier, 6) if brier is not None else None,
        "runner_top3_logloss": round(logloss, 6) if logloss is not None else None,
    }


def paired_bootstrap(base, candidate, n=BOOTSTRAPS):
    keys = sorted(set(base) & set(candidate))
    if not keys:
        return {"races": 0, "metrics": {}}
    rng = random.Random(SEED)
    names = ("axis_top3", "axis_win", "top3_coverage", "full_top3", "winner_in_top3")
    deltas = {name: [] for name in names}
    for _ in range(n):
        sample = [keys[rng.randrange(len(keys))] for _ in keys]
        for name in names:
            deltas[name].append(sum(candidate[k][name]-base[k][name] for k in sample)/len(sample))
    return {
        "races": len(keys),
        "metrics": {
            name: {
                "mean_delta_pp": round(100*float(np.mean(values)), 3),
                "ci95_low_pp": round(100*float(np.quantile(values, 0.025)), 3),
                "ci95_high_pp": round(100*float(np.quantile(values, 0.975)), 3),
            } for name, values in deltas.items()
        },
    }


def main():
    races, horses, _payouts, metadata = nb.load_archives()
    all_rows = build_rows(races, horses)
    usable = [r for r in all_rows if r["finish"] is not None and r["field_size"] >= 5]
    dates = sorted({r["date"] for r in usable})
    if len(dates) < 100:
        raise RuntimeError(f"Insufficient Banei dates: {len(dates)}")

    train_cut = dates[int(len(dates)*0.70)-1]
    dev_cut = dates[int(len(dates)*0.85)-1]
    train = [r for r in usable if r["date"] <= train_cut]
    holdout_raw = [r for r in usable if r["date"] > dev_cut]
    podium_counts = defaultdict(int)
    for row in holdout_raw:
        podium_counts[row["race_key"]] += row["label_top3"]
    holdout_races = {key for key, count in podium_counts.items() if count == 3}
    holdout = [r for r in holdout_raw if r["race_key"] in holdout_races]
    if not train or not holdout:
        raise RuntimeError("Empty chronological train or holdout")

    models = {}
    per_race = {}
    for name in ("base", "base_plus_past_body", "base_plus_past_body_and_carried"):
        probabilities = fit_predict(train, holdout, name)
        per_race[name], summary = race_metrics(holdout, probabilities)
        models[name] = summary

    body_coverage = sum(r["has_prior_body_weight"] for r in holdout) / len(holdout)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "research_only_banei_past_body_weight_profile",
        "scope": "Banei/Obihiro only; no flat-racing rows; no production changes.",
        "source": "Existing NAR monthly archives already used by KEIBA-LAB; no new historical download format.",
        "period": {"start": nb.START.isoformat(), "end": nb.END.isoformat()},
        "sample": {
            "all_banei_runner_rows": len(all_rows),
            "usable_runner_rows": len(usable),
            "unique_banei_dates": len(dates),
            "train_end": train_cut,
            "development_gap_end": dev_cut,
            "train_races": len({r["race_key"] for r in train}),
            "train_runner_rows": len(train),
            "holdout_races_before_podium_filter": len({r["race_key"] for r in holdout_raw}),
            "holdout_races": len(holdout_races),
            "holdout_runner_rows": len(holdout),
            "holdout_start": min(r["date"] for r in holdout),
            "holdout_end": max(r["date"] for r in holdout),
            "holdout_prior_body_weight_coverage_pct": round(100*body_coverage, 3),
        },
        "method": {
            "split": "Chronological 70% train / 15% development gap / final 15% holdout",
            "target": "Top-three finish; win rate separately reported",
            "baseline_features": ["past_start_count", "smoothed_career_top3", "smoothed_recent5_top3", "last_finish_relative_quality", "age"],
            "historical_body_features": ["last_prior_body_weight", "mean_prior_body_weight_last3", "body_weight_change_between_prior_starts", "prior_body_weight_sd_last5", "last_prior_recorded_body_weight_change", "mean_prior_body_weight_change_last3", "mean_prior_carried_to_body_ratio_last3", "mean_prior_carried_weight_last3", "number_of_prior_body_weight_records"],
            "carried_feature": "current race carried weight relative to the current field; used only in the third model",
            "leakage_controls": [
                "Current-race body_weight and body_weight_change are never predictors.",
                "All historical body-weight profile values are taken only from earlier completed starts.",
                "Current carried weight is treated as a pre-race race-card field.",
                "All runners in a race are featurized before any results from that race update history.",
                "The 15% development period is excluded from fitting and final holdout evaluation.",
                "Banei is isolated from flat-racing rows."
            ],
        },
        "models": models,
        "paired_deltas": {
            "past_body_vs_base": paired_bootstrap(per_race["base"], per_race["base_plus_past_body"]),
            "past_body_plus_carried_vs_past_body": paired_bootstrap(per_race["base_plus_past_body"], per_race["base_plus_past_body_and_carried"]),
            "past_body_plus_carried_vs_base": paired_bootstrap(per_race["base"], per_race["base_plus_past_body_and_carried"]),
        },
        "monthly_archive_metadata": metadata,
        "guardrails": [
            "This is a historical feature experiment, not a live pre-off capture validation.",
            "Historical prior-start body weight is used; current-race post-result body weight is excluded.",
            "No ROI claim is made because this experiment evaluates horse ranking, not a fixed betting strategy.",
            "A positive result requires paired confidence intervals and a separate independent future holdout.",
            "No production model, prediction screen, or ticket generator is changed."
        ]
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("BANEI_HISTORICAL_BODY_PROFILE " + json.dumps({
        "status": report["status"], "sample": report["sample"],
        "models": models, "paired_deltas": report["paired_deltas"],
        "report": str(OUT)
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
