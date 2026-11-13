#!/usr/bin/env python3
"""Research-only Banei winner-profile feature-group ablation."""
from __future__ import annotations
import json, math, random
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import nar_backtest as nb
from research_banei_historical_body_weight_profile import (
    build_rows, fit_predict, race_metrics, paired_bootstrap
)
from research_banei_winner_profile_similarity import ranking_metrics

OUT = nb.ROOT / "data" / "banei-winner-profile-ablation.json"
SEED = 20261113
BOOTSTRAPS = 2000

# Vector layout: base[0:5], body profile[5:14], current carried relative[14].
GROUPS = {
    "基礎成績・年齢": list(range(0, 5)),
    "過去馬体重の水準": [5, 6],
    "過去馬体重の変化・ばらつき": [7, 8, 9, 10],
    "過去の斤量／馬体重比": [11],
    "過去の斤量水準": [12],
    "馬体重記録数": [13],
    "今回斤量の相対値": [14],
}
ALL_FEATURES = list(range(15))


def vector(row):
    return row["base"] + row["body_profile"] + [row["carried_z"]]


def score_similarity(train, test, indices):
    x_train = np.asarray([[vector(r)[i] for i in indices] for r in train], dtype=float)
    x_test = np.asarray([[vector(r)[i] for i in indices] for r in test], dtype=float)
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    x_train = imputer.fit_transform(x_train)
    x_test = imputer.transform(x_test)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    y = np.asarray([r["label_win"] for r in train], dtype=int)
    winners, nonwinners = x_train[y == 1], x_train[y == 0]
    if len(winners) < 20 or len(nonwinners) < 100:
        raise RuntimeError("学習データ不足")
    wc, nc = np.median(winners, axis=0), np.median(nonwinners, axis=0)
    ws = np.maximum(np.sqrt(np.mean((winners - wc) ** 2, axis=0)), 0.35)
    ns = np.maximum(np.sqrt(np.mean((nonwinners - nc) ** 2, axis=0)), 0.35)
    dw = np.mean(((x_test - wc) / ws) ** 2, axis=1)
    dn = np.mean(((x_test - nc) / ns) ** 2, axis=1)
    return dn - dw


def main():
    races, horses, _payouts, metadata = nb.load_archives()
    rows = build_rows(races, horses)
    usable = [r for r in rows if r["finish"] is not None and r["field_size"] >= 5]
    dates = sorted({r["date"] for r in usable})
    if len(dates) < 100:
        raise RuntimeError(f"ばんえい開催日数不足: {len(dates)}")
    train_cut = dates[int(len(dates) * 0.70) - 1]
    dev_cut = dates[int(len(dates) * 0.85) - 1]
    train = [r for r in usable if r["date"] <= train_cut]
    holdout_raw = [r for r in usable if r["date"] > dev_cut]
    podium_counts = defaultdict(int)
    for r in holdout_raw:
        podium_counts[r["race_key"]] += r["label_top3"]
    holdout_keys = {k for k, n in podium_counts.items() if n >= 3}
    holdout = [r for r in holdout_raw if r["race_key"] in holdout_keys]
    if not train or not holdout:
        raise RuntimeError("学習または検証データが空です")

    full_scores = score_similarity(train, holdout, ALL_FEATURES)
    full_races, full_summary = ranking_metrics(holdout, full_scores)
    variants = {"full": {"summary": full_summary, "scores": full_scores, "races": full_races}}
    # Drop-one-group: reveals whether a feature group contributes to the full model.
    for name, idx in GROUPS.items():
        keep = [i for i in ALL_FEATURES if i not in idx]
        scores = score_similarity(train, holdout, keep)
        per_race, summary = ranking_metrics(holdout, scores)
        variants["drop:" + name] = {"summary": summary, "scores": scores, "races": per_race}
    # Group-only: reveals whether a group has standalone ranking signal.
    for name, idx in GROUPS.items():
        scores = score_similarity(train, holdout, idx)
        per_race, summary = ranking_metrics(holdout, scores)
        variants["only:" + name] = {"summary": summary, "scores": scores, "races": per_race}

    baseline_probs = fit_predict(train, holdout, "base_plus_past_body_and_carried")
    baseline_races, baseline_summary = race_metrics(holdout, baseline_probs)

    comparisons = {}
    for name, value in variants.items():
        if name == "full":
            continue
        comparisons[name + "_vs_full"] = paired_bootstrap(full_races, value["races"])
    comparisons["full_vs_existing_top3_model"] = paired_bootstrap(baseline_races, full_races)

    # Field-size subgroup diagnostics for the full similarity model and comparator.
    subgroup = {}
    for label, predicate in [
        ("5至7頭", lambda n: n <= 7),
        ("8至10頭", lambda n: 8 <= n <= 10),
        ("11頭以上", lambda n: n >= 11),
    ]:
        selected = [i for i, r in enumerate(holdout) if predicate(r["field_size"])]
        subset = [holdout[i] for i in selected]
        if not subset:
            continue
        _, sim = ranking_metrics(subset, full_scores[selected])
        _, base = race_metrics(subset, baseline_probs[selected])
        subgroup[label] = {"races": sim["races"], "winner_similarity": sim, "existing_model": base}

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "research_only_banei_winner_profile_feature_ablation",
        "scope": "ばんえい・帯広のみ。平地データ混在なし。本番変更なし。",
        "sample": {
            "train_races": len({r["race_key"] for r in train}),
            "train_runner_rows": len(train),
            "train_winner_rows": sum(r["label_win"] for r in train),
            "holdout_races": len(holdout_keys),
            "holdout_runner_rows": len(holdout),
            "train_end": train_cut,
            "development_gap_end": dev_cut,
            "holdout_start": min(r["date"] for r in holdout),
            "holdout_end": max(r["date"] for r in holdout),
        },
        "feature_groups": GROUPS,
        "full_model": full_summary,
        "existing_model": baseline_summary,
        "variants": {k: v["summary"] for k, v in variants.items()},
        "paired_comparisons": comparisons,
        "field_size_subgroups": subgroup,
        "interpretation_rules": [
            "dropで成績が下がる特徴群は、全体モデルへの寄与候補。ただし信頼区間を確認する。",
            "onlyは単独の識別力を示すが、他特徴との相互作用は評価しない。",
            "複数の特徴群を同時に比較する探索分析であり、多重比較の影響がある。",
            "勝率改善は回収率改善を意味しない。固定買い目で別途検証する。",
            "研究用。実運用・画面・買い目生成には反映しない。",
        ],
        "archive_months": len(metadata),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("BANEI_WINNER_PROFILE_ABLATION " + json.dumps({
        "status": report["status"], "sample": report["sample"],
        "full_model": full_summary, "existing_model": baseline_summary,
        "paired_comparisons": comparisons, "report": str(OUT)
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
