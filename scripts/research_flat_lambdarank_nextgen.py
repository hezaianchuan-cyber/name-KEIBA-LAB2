#!/usr/bin/env python3
"""Research-only next-generation NAR flat LambdaRank benchmark.

Trains several predeclared LambdaRank feature/regularization candidates.
Candidate selection uses the second half of chronological development data;
the final chronological holdout is evaluated once and never used for tuning.
Banei/Obihiro and unverified current-race body-weight fields are excluded.
No production model or ticket generator is modified.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import nar_backtest as nb  # noqa: E402
import nar_ml_challenger_backtest as suite  # noqa: E402

OUT = ROOT / "data" / "flat-lambdarank-nextgen.json"
PREDICTIONS = ROOT / "data" / "flat-lambdarank-nextgen-predictions.csv"
EXCLUDED_BODY_WEIGHT_FEATURES = {
    "body_weight", "body_weight_change", "body_weight_delta_pct",
    "race_rel_body_weight_delta_pct",
}
BANEI_PATTERN = re.compile(r"帯広|ばんえい", re.IGNORECASE)


def is_banei(race: dict) -> bool:
    return bool(BANEI_PATTERN.search(str(race.get("venue") or "") + " " +
                                     str(race.get("name") or "")))


def load_flat_archives():
    races, horses, payouts, downloads = nb.load_archives()
    excluded = {key for key, race in races.items() if is_banei(race)}
    flat = {key: race for key, race in races.items() if key not in excluded}
    flat_horses = {key: value for key, value in horses.items() if key in flat}
    flat_payouts = {key: value for key, value in payouts.items() if key in flat}
    if not flat:
        raise RuntimeError("Flat-only filter returned zero races")
    if any(is_banei(race) for race in flat.values()):
        raise RuntimeError("Banei/Obihiro race remained after filtering")
    audit = {
        "source_races": len(races),
        "excluded_banei_races": len(excluded),
        "flat_races": len(flat),
        "flat_dates": len({race["date"] for race in flat.values()}),
        "excluded_venues": sorted({str(race.get("venue") or "") for race in races.values()
                                   if is_banei(race)}),
        "filter": "Venue or race name contains 帯広 or ばんえい; applied before feature generation.",
    }
    print("FLAT_SCOPE_AUDIT " + json.dumps(audit, ensure_ascii=False), flush=True)
    return flat, flat_horses, flat_payouts, downloads, audit


def make_feature_profiles(full_features):
    drop_jockey = {
        "jockey", "jockey_runs", "jockey_top3_rate", "race_rel_jockey_top3_rate",
        "jockey_score",
    }
    drop_breeding_age = {"sire", "sire_runs", "sire_top3_rate", "sire_score", "age", "age_score"}
    form_distance = {
        "prior_runs", "career_win_rate", "career_top3_rate", "recent_finish_pct",
        "recent_top3_rate", "recent_margin_quality", "recent_last3f_rank",
        "recent_time_per_km", "recent_time_trend", "recent_opponent_strength",
        "recent_first_corner_pct", "recent_last_corner_pct", "recent_position_gain",
        "frame_no", "carried_weight", "predicted_front_count", "front_crowding_ratio",
        "race_rel_recent_finish_pct", "race_rel_recent_top3_rate",
        "race_rel_recent_last_corner_pct", "race_rel_recent_last3f_rank",
        "race_rel_recent_opponent_strength", "race_rel_recent_time_per_km",
        "venue_runs", "venue_top3_rate", "distance_runs", "distance_top3_rate",
        "going_runs", "going_top3_rate", "age", "field_size", "distance_m",
        "venue", "surface", "going",
    }
    return {
        "full_independent_raw": list(full_features),
        "without_jockey_family": [f for f in full_features if f not in drop_jockey],
        "without_breeding_age_family": [f for f in full_features if f not in drop_breeding_age],
        "form_venue_distance_core": [f for f in full_features if f in form_distance],
    }


def selection_score(metrics):
    # Ranking-only objective: prioritize winner-in-top3, podium coverage in top7,
    # and winner-at-rank1. It is not an ROI or betting objective.
    return round(
        0.45 * float(metrics["top3_capture_rate"] or 0.0)
        + 0.35 * float(metrics["top7_podium_coverage_pct"] or 0.0)
        + 0.20 * float(metrics["top1_rate"] or 0.0), 6
    )


def predict_groups(model, groups, features, category_maps):
    outputs = []
    for group in groups:
        x = [[r[k] if k not in suite.CATEGORICAL else category_maps[k].get(r[k], -1)
              for k in features] for r in group]
        outputs.append((group, model.predict(x)))
    return outputs


def export_predictions(path: Path, scored_groups):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "race_key", "date", "horse_no", "finish", "predicted_score", "predicted_rank"
        ])
        writer.writeheader()
        for group, scores in scored_groups:
            order = sorted(range(len(group)),
                           key=lambda i: (-float(scores[i]), int(group[i]["horse_no"])))
            ranks = {idx: rank + 1 for rank, idx in enumerate(order)}
            for idx, row in enumerate(group):
                writer.writerow({
                    "race_key": row["race_key"], "date": row["date"],
                    "horse_no": row["horse_no"], "finish": row["finish"],
                    "predicted_score": round(float(scores[idx]), 10),
                    "predicted_rank": ranks[idx],
                })


def main():
    # Exclude current-race body-weight values; retain prior-start average only.
    suite.RAW_NUMERIC = [f for f in suite.RAW_NUMERIC
                         if f not in EXCLUDED_BODY_WEIGHT_FEATURES]
    suite.RAW_FEATURES = suite.RAW_NUMERIC + suite.CATEGORICAL

    races, horses, payouts, downloads, scope_audit = load_flat_archives()
    rows, dates = suite.build_rows(races, horses)
    if len(dates) < 100:
        raise RuntimeError(f"Insufficient chronological coverage: {len(dates)}")
    train_cut = int(len(dates) * 0.70)
    dev_cut = int(len(dates) * 0.80)
    train_dates, dev_dates, holdout_dates = set(dates[:train_cut]), set(dates[train_cut:dev_cut]), set(dates[dev_cut:])
    train_rows = [r for r in rows if dt.date.fromisoformat(r["date"]) in train_dates]
    dev_rows = [r for r in rows if dt.date.fromisoformat(r["date"]) in dev_dates]
    holdout_rows = [r for r in rows if dt.date.fromisoformat(r["date"]) in holdout_dates]
    train_groups = suite.group_rows(train_rows)
    dev_groups = suite.group_rows(dev_rows)
    holdout_groups = suite.group_rows(holdout_rows)
    if not train_groups or len(dev_groups) < 20 or not holdout_groups:
        raise RuntimeError("Train/development/holdout split is insufficient")

    category_maps = {
        key: {value: i for i, value in enumerate(sorted({r[key] for r in train_rows}))}
        for key in suite.CATEGORICAL
    }
    dev_mid = len(dev_groups) // 2
    early_stop_groups, selection_groups = dev_groups[:dev_mid], dev_groups[dev_mid:]
    if not early_stop_groups or not selection_groups:
        raise RuntimeError("Development split could not be divided chronologically")

    profiles = make_feature_profiles(suite.RAW_FEATURES)
    candidates = [
        ("full_default", "full_independent_raw", {}),
        ("full_regularized", "full_independent_raw",
         {"learning_rate": 0.025, "num_leaves": 15, "min_child_samples": 120, "reg_lambda": 4.0}),
        ("full_compact", "full_independent_raw",
         {"learning_rate": 0.030, "num_leaves": 7, "min_child_samples": 120, "reg_lambda": 4.0}),
        ("full_wide", "full_independent_raw",
         {"learning_rate": 0.025, "num_leaves": 31, "min_child_samples": 120, "reg_lambda": 4.0}),
        ("without_jockey_family", "without_jockey_family",
         {"learning_rate": 0.030, "num_leaves": 15, "min_child_samples": 100, "reg_lambda": 3.0}),
        ("without_breeding_age_family", "without_breeding_age_family",
         {"learning_rate": 0.030, "num_leaves": 15, "min_child_samples": 100, "reg_lambda": 3.0}),
        ("form_venue_distance_core", "form_venue_distance_core",
         {"learning_rate": 0.030, "num_leaves": 15, "min_child_samples": 100, "reg_lambda": 3.0}),
    ]
    reports, fitted = [], {}
    for name, profile_name, params in candidates:
        features = profiles[profile_name]
        if len(features) < 8:
            raise RuntimeError(f"Feature profile too small: {profile_name}")
        model = suite.fit_lambdarank(
            train_groups, early_stop_groups, features, category_maps, params=params
        )
        scored = predict_groups(model, selection_groups, features, category_maps)
        metrics = suite.evaluate_scores(name, scored)
        item = {
            "name": name, "feature_profile": profile_name, "feature_count": len(features),
            "params": params or {"default": True},
            "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
            "development_selection_metrics": metrics,
            "development_selection_score": selection_score(metrics),
        }
        reports.append(item)
        fitted[name] = (model, features)
        print("LAMBDA_CANDIDATE " + json.dumps(item, ensure_ascii=False), flush=True)

    selected = max(reports, key=lambda item: (
        item["development_selection_score"],
        item["development_selection_metrics"]["top7_podium_coverage_pct"],
        item["development_selection_metrics"]["top3_capture_rate"],
    ))
    selected_model, selected_features = fitted[selected["name"]]

    # A fixed default LambdaRank is retained as the same-sample comparator.
    baseline_model, baseline_features = fitted["full_default"]
    baseline_holdout = predict_groups(baseline_model, holdout_groups, baseline_features, category_maps)
    selected_holdout = predict_groups(selected_model, holdout_groups, selected_features, category_maps)
    baseline_metrics = suite.evaluate_scores("LambdaRank_full_default", baseline_holdout)
    selected_metrics = suite.evaluate_scores("LambdaRank_selected_candidate", selected_holdout)
    export_predictions(PREDICTIONS, selected_holdout)

    report = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "evaluated",
        "scope": "NAR flat racing only; Banei/Obihiro excluded before feature generation.",
        "scope_audit": scope_audit,
        "data": {
            "races": len(races), "runner_rows": len(rows), "dates": len(dates),
            "train_races": len(train_groups), "development_races": len(dev_groups),
            "development_early_stop_races": len(early_stop_groups),
            "development_selection_races": len(selection_groups),
            "holdout_races": len(holdout_groups),
        },
        "period": {"start": dates[0].isoformat(), "end": dates[-1].isoformat()},
        "split": {
            "train_end": dates[train_cut - 1].isoformat(),
            "development_start": dates[train_cut].isoformat(),
            "development_end": dates[dev_cut - 1].isoformat(),
            "holdout_start": dates[dev_cut].isoformat(),
            "holdout_end": dates[-1].isoformat(),
            "policy": "Chronological 70/10/20; development split in half for early stopping and candidate selection; holdout not used for selection.",
        },
        "feature_policy": {
            "excluded_current_race_features": sorted(EXCLUDED_BODY_WEIGHT_FEATURES),
            "retained_historical_feature": "prior_body_weight_avg (prior starts only)",
            "profiles": {name: {"count": len(features), "features": features}
                         for name, features in profiles.items()},
        },
        "candidate_selection": {
            "candidates": reports,
            "selected": selected["name"],
            "score": "0.45*winner_in_top3 + 0.35*top7_podium_coverage + 0.20*top1_accuracy; ranking only, no payout/odds inputs.",
            "selection_data": "Second chronological half of development only.",
        },
        "holdout": {
            "baseline": baseline_metrics,
            "selected": selected_metrics,
            "selected_minus_baseline_pp": {
                "top1_rate": round(selected_metrics["top1_rate"] - baseline_metrics["top1_rate"], 3),
                "top3_capture_rate": round(selected_metrics["top3_capture_rate"] - baseline_metrics["top3_capture_rate"], 3),
                "top7_podium_coverage_pct": round(
                    selected_metrics["top7_podium_coverage_pct"] - baseline_metrics["top7_podium_coverage_pct"], 3
                ),
                "top7_full_podium_rate": round(
                    selected_metrics["top7_full_podium_rate"] - baseline_metrics["top7_full_podium_rate"], 3
                ),
            },
            "policy": "One final evaluation after candidate selection; no tuning on holdout.",
        },
        "prediction_export": str(PREDICTIONS.relative_to(ROOT)),
        "limitations": [
            "Research-only; no production model, ticket generator, or canonical report changed.",
            "Ranking metrics do not establish profitability.",
            "No final odds or payout fields are used as model features or selection targets.",
            "Candidate selection can overfit the development selection block; independent future-period confirmation remains necessary.",
            "Horse identity uses the upstream conservative composite identity because monthly NAR CSV does not provide a permanent horse ID.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("LAMBDA_NEXTGEN_RESULT " + json.dumps({
        "status": report["status"], "data": report["data"],
        "selected": selected["name"], "holdout": report["holdout"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
