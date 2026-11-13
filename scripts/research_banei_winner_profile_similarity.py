#!/usr/bin/env python3
"""Research-only Banei winner-profile similarity model.

Learns a pre-race winner prototype from earlier Banei winners and ranks each
current field by relative similarity to winner vs non-winner profiles.
No current-race body weight, result, odds, or post-off data is used.
"""
from __future__ import annotations
import json, math, random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import nar_backtest as nb
from research_banei_historical_body_weight_profile import build_rows, is_banei, fit_predict, race_metrics, paired_bootstrap

OUT = nb.ROOT / "data" / "banei-winner-profile-similarity.json"
SEED = 20261113
BOOTSTRAPS = 2000


def vector(row):
    # Use only the established Banei pre-race profile. Current-race body weight
    # is intentionally absent; carried_z is available from the race card.
    return row["base"] + row["body_profile"] + [row["carried_z"]]


def fit_winner_similarity(train, test):
    x_train = np.asarray([vector(r) for r in train], dtype=float)
    x_test = np.asarray([vector(r) for r in test], dtype=float)
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    x_train = imputer.fit_transform(x_train)
    x_test = imputer.transform(x_test)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    y = np.asarray([r["label_win"] for r in train], dtype=int)
    winners = x_train[y == 1]
    nonwinners = x_train[y == 0]
    if len(winners) < 20 or len(nonwinners) < 100:
        raise RuntimeError(f"Insufficient training profiles: winners={len(winners)}, nonwinners={len(nonwinners)}")
    # Robust prototype: median profile reduces the influence of extreme winners.
    winner_center = np.median(winners, axis=0)
    nonwinner_center = np.median(nonwinners, axis=0)
    winner_scale = np.sqrt(np.mean((winners - winner_center) ** 2, axis=0))
    winner_scale = np.maximum(winner_scale, 0.35)
    nonwinner_scale = np.sqrt(np.mean((nonwinners - nonwinner_center) ** 2, axis=0))
    nonwinner_scale = np.maximum(nonwinner_scale, 0.35)
    d_winner = np.mean(((x_test - winner_center) / winner_scale) ** 2, axis=1)
    d_nonwinner = np.mean(((x_test - nonwinner_center) / nonwinner_scale) ** 2, axis=1)
    # Higher means more winner-like and less non-winner-like.
    return d_nonwinner - d_winner, {
        "training_winner_rows": int(len(winners)),
        "training_nonwinner_rows": int(len(nonwinners)),
        "feature_count_after_missing_indicators": int(x_train.shape[1]),
        "prototype": "standardized median centroid; score = nonwinner distance - winner distance",
    }


def ranking_metrics(records, scores):
    grouped = defaultdict(list)
    for row, score in zip(records, scores):
        grouped[row["race_key"]].append((float(score), row))
    per_race = {}
    for key, entries in grouped.items():
        entries.sort(key=lambda x: (-x[0], int(x[1]["horse_no"]) if x[1]["horse_no"].isdigit() else x[1]["horse_no"]))
        podium = {r["horse_no"] for _, r in entries if r["label_top3"]}
        if len(podium) < 3:
            continue
        predicted = {r["horse_no"] for _, r in entries[:3]}
        winner = next((r["horse_no"] for _, r in entries if r["label_win"]), None)
        first = entries[0][1]["horse_no"]
        per_race[key] = {
            "axis_top3": float(first in podium),
            "axis_win": float(first == winner),
            "top3_coverage": len(predicted & podium) / len(podium),
            "full_top3": float(podium.issubset(predicted)),
            "winner_in_top3": float(winner in predicted) if winner else 0.0,
        }
    names = ("axis_top3", "axis_win", "top3_coverage", "full_top3", "winner_in_top3")
    summary = {"races": len(per_race)}
    for name in names:
        summary[name + "_pct"] = round(100*np.mean([v[name] for v in per_race.values()]), 3) if per_race else None
    return per_race, summary


def main():
    races, horses, _payouts, metadata = nb.load_archives()
    rows = build_rows(races, horses)
    usable = [r for r in rows if r["finish"] is not None and r["field_size"] >= 5]
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
    holdout_keys = {k for k, count in podium_counts.items() if count >= 3}
    holdout = [r for r in holdout_raw if r["race_key"] in holdout_keys]
    if not train or not holdout:
        raise RuntimeError("Empty chronological train or holdout")

    similarity_scores, prototype_info = fit_winner_similarity(train, holdout)
    similarity_races, similarity_summary = ranking_metrics(holdout, similarity_scores)

    # Fair same-race comparator: existing top-three probability model with
    # historical body profile and current carried-weight-relative-to-field.
    baseline_scores = fit_predict(train, holdout, "base_plus_past_body_and_carried")
    baseline_races, baseline_summary = race_metrics(holdout, baseline_scores)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "research_only_banei_winner_profile_similarity",
        "scope": "Banei/Obihiro only; no flat-racing rows; no production changes.",
        "sample": {
            "train_races": len({r["race_key"] for r in train}),
            "train_runner_rows": len(train),
            "train_winner_rows": sum(r["label_win"] for r in train),
            "holdout_races_before_podium_filter": len({r["race_key"] for r in holdout_raw}),
            "holdout_races": len(holdout_keys),
            "holdout_runner_rows": len(holdout),
            "holdout_start": min(r["date"] for r in holdout),
            "holdout_end": max(r["date"] for r in holdout),
            "train_end": train_cut,
            "development_gap_end": dev_cut,
        },
        "method": {
            "description": "過去の勝ち馬の特徴の中央値を勝ち馬像として作り、各出走馬が勝ち馬像にどれだけ近く、非勝ち馬像からどれだけ離れているかを採点する。",
            "features": ["過去出走数", "通算・近5走の3着内率", "前走相対着順", "年齢", "過去走の馬体重水準・変化・ばらつき", "過去の斤量と馬体重の比率", "今回斤量の出走馬内相対値"],
            "split": "時系列70%学習 / 15%未使用期間 / 最終15%検証",
            "leakage_controls": [
                "ばんえいのみ。平地データは混ぜない。",
                "今回の馬体重・馬体重増減は予測に使用しない。",
                "馬体重履歴は当該レースより前の出走分だけを使用。",
                "今回斤量は出馬表項目として使用。",
                "同一レース全馬の特徴量作成後に履歴を更新。",
                "検証期間の結果を学習に使用しない。"
            ],
            "prototype_details": prototype_info,
        },
        "models": {
            "existing_top3_model": baseline_summary,
            "winner_profile_similarity": similarity_summary,
        },
        "paired_deltas_similarity_minus_existing": paired_bootstrap(baseline_races, similarity_races),
        "archive_metadata": metadata,
        "guardrails": [
            "これは勝ち馬特徴の照合方式を試す研究用検証であり、実運用の予測には未反映。",
            "勝率・軸3着内率・上位3頭の3着内カバー率を同一レースで比較。",
            "買い目生成・回収率は別途固定買い目検証が必要。",
            "信頼区間がゼロをまたぐ場合は改善と断定しない。",
            "本番モデル、画面、買い目生成は変更しない。"
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("BANEI_WINNER_PROFILE_SIMILARITY " + json.dumps({
        "status": report["status"], "sample": report["sample"],
        "models": report["models"], "paired_deltas": report["paired_deltas_similarity_minus_existing"],
        "report": str(OUT)
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
