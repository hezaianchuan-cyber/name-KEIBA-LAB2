#!/usr/bin/env python3
"""Fine-grained research-only Banei axis blend around 50:50.

The development gap is the tuning/selection set. The previously inspected final
holdout is reported only as a descriptive repeatability check, not used to
choose a ratio. Partners remain fixed to isolate axis blend effects.
"""
from __future__ import annotations
import json
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import nar_backtest as nb
from research_banei_historical_body_weight_profile import paired_bootstrap
from research_banei_winner_profile_similarity import fit_winner_similarity
from research_banei_axis_partner_split import (
    build_rows_stable, partner_probabilities, blend_partner_scores, split_role_rank,
)

OUT = nb.ROOT / "data" / "banei-axis-win-blend-fine.json"
# Similarity share; compare 60:40, 55:45, 50:50, 45:55, 40:60.
WEIGHTS = (0.60, 0.55, 0.50, 0.45, 0.40)


def valid_races(rows):
    counts = defaultdict(int)
    for row in rows:
        counts[row["race_key"]] += row["label_top3"]
    keys = {key for key, n in counts.items() if n >= 3}
    return [row for row in rows if row["race_key"] in keys], keys


def evaluate(train, test):
    similarity, _ = fit_winner_similarity(train, test)
    win_specialist = partner_probabilities(train, test, "label_win", model_kind="hist_gradient_boosting")
    partners = partner_probabilities(
        train, test, "label_nonwinner_podium",
        model_kind="hist_gradient_boosting",
        feature_indices=[0, 1, 2, 3, 4, 5, 6, 11, 12],
    )
    results, race_maps = {}, {}
    for weight in WEIGHTS:
        axis = blend_partner_scores(test, similarity, win_specialist, weight)
        race_map, summary = split_role_rank(test, axis, partners)
        key = f"similarity_{int(weight*100)}_win_{100-int(weight*100)}"
        results[key], race_maps[key] = summary, race_map
    return results, race_maps


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
    gap, gap_keys = valid_races([r for r in usable if train_cut < r["date"] <= dev_cut])
    holdout, holdout_keys = valid_races([r for r in usable if r["date"] > dev_cut])
    if not train or not gap or not holdout:
        raise RuntimeError("学習・中間確認・最終検証のいずれかが空です")

    # The 50:50 result is the comparison anchor; the development gap is used
    # for any ratio selection. The already-inspected holdout is not a fresh test.
    dev_results, dev_maps = evaluate(train, gap)
    hold_results, hold_maps = evaluate(train, holdout)
    anchor = "similarity_50_win_50"
    comparisons = {}
    for weight in WEIGHTS:
        key = f"similarity_{int(weight*100)}_win_{100-int(weight*100)}"
        comparisons[key] = {
            "development_gap_vs_50_50": paired_bootstrap(dev_maps[anchor], dev_maps[key]),
            "previously_inspected_holdout_vs_50_50_descriptive_only": paired_bootstrap(hold_maps[anchor], hold_maps[key]),
        }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "research_only_banei_axis_win_blend_fine",
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
        "models": {"development_gap": dev_results, "previously_inspected_holdout_descriptive_only": hold_results},
        "paired_comparisons": comparisons,
        "method": {
            "weights_are_similarity_vs_win_specialist": list(WEIGHTS),
            "selection_set": "時系列の中間15%（development gap）のみ",
            "holdout_usage": "過去に確認済みの最終15%。比率選択には使わず、記述的な再現性確認のみ",
            "partner_model": "2着・3着専用HGB。全方式で固定",
            "current_race_body_weight_or_change_used": False,
            "post_off_data_or_odds_used": False,
            "same_race_history_update_delayed_until_all_features_built": True,
            "identity_audit": identity_audit,
        },
        "guardrails": [
            "研究用の独立ブランチのみ。本番反映・マイグレーション・デプロイなし。",
            "学習70%、中間15%、最終15%の時系列分割を維持。",
            "比率の選択は中間15%のみ。最終15%は既に過去に確認済みであり、新規ホールドアウトとは扱わない。",
            "パートナー順位は固定し、軸スコアの混合だけを変更。",
            "回収率は固定買い目と発売時点オッズを使う別検証が必要。",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("BANEI_AXIS_WIN_BLEND_FINE " + json.dumps({
        "sample": report["sample"], "development_gap": dev_results,
        "holdout_descriptive_only": hold_results, "comparisons": comparisons,
        "identity_audit": identity_audit, "report": str(OUT),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
