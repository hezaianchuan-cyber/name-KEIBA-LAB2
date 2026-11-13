#!/usr/bin/env python3
"""Research-only blend of Banei winner-profile similarity and a win specialist.

The axis score is tested at 100/0, 75/25, 50/50, 25/75 and 0/100
similarity/win-specialist weights. Partner scores remain fixed to the
previously selected second/third-place HGB model so the axis effect is isolated.
No production or live-ticket changes.
"""
from __future__ import annotations
import json
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import nar_backtest as nb
from research_banei_historical_body_weight_profile import is_banei, paired_bootstrap
from research_banei_winner_profile_similarity import fit_winner_similarity
from research_banei_axis_partner_split import (
    build_rows_stable, partner_probabilities, blend_partner_scores, split_role_rank,
)

OUT = nb.ROOT / "data" / "banei-axis-win-blend.json"
WEIGHTS = (1.0, 0.75, 0.50, 0.25, 0.0)


def valid_races(rows):
    counts = defaultdict(int)
    for row in rows:
        counts[row["race_key"]] += row["label_top3"]
    keys = {key for key, n in counts.items() if n >= 3}
    return [row for row in rows if row["race_key"] in keys], keys


def main():
    races, horses, _payouts, _metadata = nb.load_archives()
    rows, identity_audit = build_rows_stable(races, horses)
    usable = [r for r in rows if r["finish"] is not None and r["field_size"] >= 5]
    dates = sorted({r["date"] for r in usable})
    if len(dates) < 100:
        raise RuntimeError(f"ばんえい開催日数不足: {len(dates)}")
    train_cut = dates[int(len(dates) * 0.70) - 1]
    dev_cut = dates[int(len(dates) * 0.85) - 1]
    train = [r for r in usable if r["date"] <= train_cut]
    gap_raw = [r for r in usable if train_cut < r["date"] <= dev_cut]
    holdout_raw = [r for r in usable if r["date"] > dev_cut]
    gap, gap_keys = valid_races(gap_raw)
    holdout, holdout_keys = valid_races(holdout_raw)
    if not train or not gap or not holdout:
        raise RuntimeError("学習・中間確認・最終検証のいずれかが空です")

    # Train only on chronological training races. No current bodyweight or
    # post-off information is used; label_win is the explicit specialist target.
    def evaluate(test):
        similarity, _ = fit_winner_similarity(train, test)
        win_specialist = partner_probabilities(train, test, "label_win", model_kind="hist_gradient_boosting")
        partners = partner_probabilities(
            train, test, "label_nonwinner_podium",
            model_kind="hist_gradient_boosting",
            feature_indices=[0, 1, 2, 3, 4, 5, 6, 11, 12],
        )
        results = {}
        race_maps = {}
        for weight in WEIGHTS:
            axis = blend_partner_scores(test, similarity, win_specialist, weight)
            race_map, summary = split_role_rank(test, axis, partners)
            key = f"similarity_{int(weight*100)}_win_{int((1-weight)*100)}"
            results[key] = summary
            race_maps[key] = race_map
        return results, race_maps

    holdout_results, holdout_maps = evaluate(holdout)
    gap_results, gap_maps = evaluate(gap)
    baseline = "similarity_100_win_0"
    comparisons = {}
    for weight in WEIGHTS[1:]:
        key = f"similarity_{int(weight*100)}_win_{int((1-weight)*100)}"
        comparisons[key] = {
            "holdout_vs_similarity_only": paired_bootstrap(holdout_maps[baseline], holdout_maps[key]),
            "gap_vs_similarity_only": paired_bootstrap(gap_maps[baseline], gap_maps[key]),
        }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "research_only_banei_axis_win_blend",
        "scope": "ばんえい・帯広のみ。平地混在なし。本番モデル・買い目生成は未変更。",
        "sample": {
            "train_races": len({r["race_key"] for r in train}),
            "train_runner_rows": len(train),
            "development_gap_races": len(gap_keys),
            "development_gap_runner_rows": len(gap),
            "holdout_races": len(holdout_keys),
            "holdout_runner_rows": len(holdout),
            "train_end": train_cut,
            "development_gap_end": dev_cut,
            "holdout_start": min(r["date"] for r in holdout),
            "holdout_end": max(r["date"] for r in holdout),
        },
        "models": {
            "holdout": holdout_results,
            "development_gap": gap_results,
        },
        "paired_comparisons": comparisons,
        "method": {
            "axis_blend": "レース内順位百分位に変換した勝ち馬特徴照合スコアと1着専用HGBスコアの混合",
            "weights_are_similarity_vs_win_specialist": list(WEIGHTS),
            "partner_model": "2着・3着専用HGB。軸の混合効果を分離するため全方式で固定",
            "win_specialist_target": "label_win",
            "current_race_body_weight_or_change_used": False,
            "post_off_data_or_odds_used": False,
            "same_race_history_update_delayed_until_all_features_built": True,
            "identity_audit": identity_audit,
        },
        "guardrails": [
            "研究用の独立ブランチのみ。本番反映・マイグレーション・デプロイなし。",
            "学習70%、中間15%は学習に使わず頑健性確認、最終15%をホールドアウトとして固定。",
            "パートナー順位は固定し、軸スコアの混合だけを変更。",
            "1着率・軸3着内率・3頭カバー率・3頭全頭カバー率を同時評価。",
            "回収率は固定買い目と発売時点オッズを使う別検証が必要。",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("BANEI_AXIS_WIN_BLEND " + json.dumps({
        "sample": report["sample"], "holdout": holdout_results,
        "gap": gap_results, "comparisons": comparisons,
        "identity_audit": identity_audit, "report": str(OUT),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
