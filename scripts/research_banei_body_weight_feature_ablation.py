#!/usr/bin/env python3
"""Banei-only ablation of historical body-weight feature groups.

Uses the existing NAR monthly archive, chronological split, and leakage-safe
row construction. Each group is tested independently against the same baseline
on the same final holdout. Research only; no production integration.
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import research_banei_historical_body_weight_profile as profile

OUT = profile.nb.ROOT / "data" / "banei-body-weight-feature-ablation.json"
GROUPS = {
    "body_weight_level": [0, 1],
    "body_weight_trend_and_variation": [2, 3, 4, 5],
    "past_carried_to_body_ratio": [6],
    "past_carried_weight_level": [7],
    "body_weight_record_count": [8],
}
SEED = 20261113

def predict(train, test, group_name=None, include_current_carried=False):
    def vec(row):
        values = list(row["base"])
        if group_name:
            values += [row["body_profile"][i] for i in GROUPS[group_name]]
        if include_current_carried:
            values += [row["carried_z"]]
        return values
    x_train = np.asarray([vec(r) for r in train], dtype=float)
    y_train = np.asarray([r["label_top3"] for r in train], dtype=int)
    x_test = np.asarray([vec(r) for r in test], dtype=float)
    model = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        LogisticRegression(C=0.5, max_iter=1500, random_state=SEED),
    )
    model.fit(x_train, y_train)
    return model.predict_proba(x_test)[:, 1]

def main():
    nb = profile.nb
    races, horses, _payouts, metadata = nb.load_archives()
    rows = profile.build_rows(races, horses)
    usable = [r for r in rows if r["finish"] is not None and r["field_size"] >= 5]
    dates = sorted({r["date"] for r in usable})
    if len(dates) < 100:
        raise RuntimeError(f"Insufficient Banei dates: {len(dates)}")
    train_cut = dates[int(len(dates)*0.70)-1]
    dev_cut = dates[int(len(dates)*0.85)-1]
    train = [r for r in usable if r["date"] <= train_cut]
    holdout_raw = [r for r in usable if r["date"] > dev_cut]
    podium_counts = {}
    for r in holdout_raw:
        podium_counts[r["race_key"]] = podium_counts.get(r["race_key"], 0) + r["label_top3"]
    valid_races = {k for k, n in podium_counts.items() if n >= 3}
    holdout = [r for r in holdout_raw if r["race_key"] in valid_races]

    names = ["baseline", *GROUPS.keys(), "all_past_body", "all_past_body_plus_current_carried"]
    probs, per_race, summaries = {}, {}, {}
    for name in names:
        if name == "baseline":
            p = predict(train, holdout)
        elif name == "all_past_body":
            p = predict(train, holdout, "all")
        elif name == "all_past_body_plus_current_carried":
            p = predict(train, holdout, "all", True)
        else:
            p = predict(train, holdout, name)
        probs[name] = p
        per_race[name], summaries[name] = profile.race_metrics(holdout, p)

    # "all" is handled as the complete 9-feature body profile.
    def predict_all(train_rows, test_rows, carried=False):
        def vec(r):
            v = list(r["base"]) + list(r["body_profile"])
            if carried: v.append(r["carried_z"])
            return v
        xtr = np.asarray([vec(r) for r in train_rows], dtype=float)
        ytr = np.asarray([r["label_top3"] for r in train_rows], dtype=int)
        xte = np.asarray([vec(r) for r in test_rows], dtype=float)
        model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(), LogisticRegression(C=0.5, max_iter=1500, random_state=SEED))
        model.fit(xtr, ytr)
        return model.predict_proba(xte)[:, 1]

    for name, carried in [("all_past_body", False), ("all_past_body_plus_current_carried", True)]:
        p = predict_all(train, holdout, carried)
        probs[name] = p
        per_race[name], summaries[name] = profile.race_metrics(holdout, p)

    deltas = {}
    for name in names[1:]:
        deltas[name] = profile.paired_bootstrap(per_race["baseline"], per_race[name])

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "research_only_banei_body_weight_feature_ablation",
        "scope": "Banei/Obihiro only; no flat rows; no production changes",
        "sample": {
            "train_races": len({r["race_key"] for r in train}),
            "holdout_races": len(valid_races),
            "holdout_runner_rows": len(holdout),
            "train_end": train_cut, "development_gap_end": dev_cut,
            "holdout_start": min(r["date"] for r in holdout),
            "holdout_end": max(r["date"] for r in holdout),
        },
        "feature_groups": GROUPS,
        "models": summaries,
        "paired_vs_baseline": deltas,
        "method": "Each body-weight feature group is added separately to the same baseline; same chronological holdout and paired race bootstrap. Current-race body weight excluded. Current carried weight is only tested in the final combined model.",
        "source_metadata": metadata,
        "guardrails": [
            "This is feature diagnosis, not proof of predictive improvement.",
            "All current-race body-weight values are excluded.",
            "No ROI claim; no fixed betting strategy evaluated.",
            "No production model, prediction screen, or ticket generator changed.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("BANEI_BODY_WEIGHT_ABLATION " + json.dumps({
        "sample": report["sample"], "models": summaries, "paired_vs_baseline": deltas,
        "report": str(OUT)
    }, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
