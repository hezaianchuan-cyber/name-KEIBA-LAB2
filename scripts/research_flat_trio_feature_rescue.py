#!/usr/bin/env python3
"""Research-only feature rescue test for flat NAR trio partner selection.

Keeps the 40:60 model's rank-1 axis fixed and a fixed six-ticket/JPY600 budget.
Tests whether pre-race form, closing-speed and course-fit signals can improve
selection among partner candidates ranked 2-7. No odds are used for selection.
No production code, live workflow, database, or ticket generator is changed.
"""
from __future__ import annotations

import datetime as dt
import itertools
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

import nar_backtest as nb
import research_flat_axis_win_blend as axis_blend

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "flat-trio-feature-rescue.json"
BASE_MODEL = "similarity_40_win_60"
TICKET_COST = 100
TICKETS = 6
BOOTSTRAP_ITERATIONS = 3000

FEATURE_GROUPS = {
    "recent_form": [
        ("recent_finish_pct", True),
        ("recent_top3_rate", True),
        ("recent_margin_quality", True),
        ("career_top3_rate", True),
    ],
    "closing_and_progress": [
        ("recent_last3f_rank", True),
        ("recent_time_trend", True),
        ("recent_position_gain", True),
    ],
    "course_fit": [
        ("venue_top3_rate", True),
        ("distance_top3_rate", True),
        ("going_top3_rate", True),
        ("jockey_top3_rate", True),
    ],
}


def grouped_by_period(rows):
    groups = axis_blend.grouped(rows)
    return groups


def race_percentiles(group, feature, higher_is_better=True):
    values = []
    for row in group:
        try:
            value = float(row.get(feature))
            if not np.isfinite(value):
                value = 0.5
        except (TypeError, ValueError):
            value = 0.5
        values.append(value if higher_is_better else -value)
    return axis_blend.percentile(values)


def feature_rescue_scores(group):
    component_scores = {}
    for component, features in FEATURE_GROUPS.items():
        per_feature = [
            race_percentiles(group, feature, higher)
            for feature, higher in features
        ]
        component_scores[component] = np.mean(np.vstack(per_feature), axis=0)
    combined = np.mean(np.vstack(list(component_scores.values())), axis=0)
    return combined, component_scores


def ticket_pairs(group, order, partner_scores, candidate_count=6):
    """Return six axis-anchored pairs chosen from base ranks 2..candidate_count+1."""
    if len(order) < candidate_count + 1:
        return None
    axis_idx = order[0]
    candidates = order[1:candidate_count + 1]
    scored_pairs = []
    for a, b in itertools.combinations(candidates, 2):
        score = float(partner_scores[a]) + float(partner_scores[b])
        nos = tuple(sorted(
            (str(group[axis_idx]["horse_no"]),
             str(group[a]["horse_no"]),
             str(group[b]["horse_no"])),
            key=int,
        ))
        scored_pairs.append((score, nos))
    scored_pairs.sort(key=lambda item: (-item[0], tuple(int(x) for x in item[1])))
    chosen = [pair for _, pair in scored_pairs[:TICKETS]]
    return chosen if len(chosen) == TICKETS else None


def make_ticket_sets(group, base_scores):
    order = sorted(
        range(len(group)),
        key=lambda i: (-float(base_scores[i]), int(group[i]["horse_no"])),
    )
    if len(order) < 7:
        return None
    base_pct = axis_blend.percentile(base_scores)
    rescue, components = feature_rescue_scores(group)

    # Baseline: all six combinations from ranks 2-5.
    axis = order[0]
    baseline = [
        tuple(sorted(
            (str(group[axis]["horse_no"]), str(group[a]["horse_no"]),
             str(group[b]["horse_no"])), key=int
        ))
        for a, b in itertools.combinations(order[1:5], 2)
    ]
    sets = {"baseline_top5_all6": baseline}

    # Six-ticket top-pair selection from ranks 2-7 using model rank alone,
    # then with progressively stronger pre-race feature-rescue signals.
    for weight in (0.0, 0.15, 0.30, 0.45):
        partner_score = (1.0 - weight) * base_pct + weight * rescue
        name = f"pool6_feature_{int(weight * 100):02d}"
        sets[name] = ticket_pairs(group, order, partner_score, candidate_count=6)

    # Diagnostic component-only tests show which family contributes signal.
    for component in FEATURE_GROUPS:
        sets[f"pool6_{component}_only"] = ticket_pairs(
            group, order, components[component], candidate_count=6
        )
    return sets, order, rescue, components


def race_payout(group, payouts):
    parts = str(group[0]["race_key"]).split("|")
    if len(parts) != 3:
        return None
    return payouts.get(nb.race_key(parts[0], parts[1], parts[2]))


def evaluate_split(scored_groups, payouts):
    records = []
    skipped = defaultdict(int)
    for group, strategies in scored_groups:
        built = make_ticket_sets(group, strategies[BASE_MODEL])
        if built is None:
            skipped["field_too_small_for_pool6"] += 1
            continue
        sets, order, rescue, components = built
        payout = race_payout(group, payouts)
        if not payout:
            skipped["missing_payout"] += 1
            continue
        podium = [
            str(row["horse_no"]) for row in group
            if row.get("finish") is not None and int(row["finish"]) in (1, 2, 3)
        ]
        if len(podium) != 3:
            skipped["invalid_podium"] += 1
            continue
        axis_no = str(group[order[0]]["horse_no"])
        axis_finish = next(
            (int(row["finish"]) for row in group
             if str(row["horse_no"]) == axis_no and row.get("finish") is not None),
            None,
        )
        row_result = {
            "race_key": group[0]["race_key"],
            "date": str(group[0]["date"]),
            "field_size": len(group),
            "field_size_band": (
                "7-8頭" if len(group) <= 8 else
                "9-11頭" if len(group) <= 11 else
                "12-14頭" if len(group) <= 14 else "15頭以上"
            ),
            "axis_top3": int(axis_finish is not None and axis_finish <= 3),
            "axis_win": int(axis_finish == 1),
            "podium_rank6_7": sum(
                1 for rank, idx in enumerate(order, start=1)
                if rank in (6, 7) and str(group[idx]["horse_no"]) in podium
            ),
            "strategies": {},
        }
        for name, tickets in sets.items():
            if not tickets or len(tickets) != TICKETS:
                skipped[f"invalid_ticket_set_{name}"] += 1
                continue
            returned = sum(
                nb.payout_for("trio", list(ticket), payout) or 0
                for ticket in tickets
            )
            rank6_7_tickets = sum(
                any(
                    str(group[idx]["horse_no"]) in ticket
                    for rank, idx in enumerate(order, start=1)
                    if rank in (6, 7)
                )
                for ticket in tickets
            )
            row_result["strategies"][name] = {
                "hit": int(returned > 0),
                "stake": TICKETS * TICKET_COST,
                "returned": returned,
                "rank6_7_tickets": rank6_7_tickets,
            }
        if "baseline_top5_all6" in row_result["strategies"]:
            records.append(row_result)
    return records, dict(skipped)


def summarize(records, strategy):
    rows = [r for r in records if strategy in r["strategies"]]
    n = len(rows)
    stake = sum(r["strategies"][strategy]["stake"] for r in rows)
    returned = sum(r["strategies"][strategy]["returned"] for r in rows)
    hits = sum(r["strategies"][strategy]["hit"] for r in rows)
    rank67 = sum(r["strategies"][strategy]["rank6_7_tickets"] for r in rows)
    return {
        "races": n,
        "hit_rate_pct": round(100 * hits / n, 3) if n else None,
        "roi_pct": round(100 * returned / stake, 3) if stake else None,
        "stake_jpy": stake,
        "return_jpy": returned,
        "profit_jpy": returned - stake,
        "rank6_7_ticket_share_pct": round(100 * rank67 / (n * TICKETS), 3) if n else None,
        "axis_top3_rate_pct": round(100 * sum(r["axis_top3"] for r in rows) / n, 3) if n else None,
    }


def paired_roi_ci(records, candidate, baseline="baseline_top5_all6", seed=20261113):
    rows = [
        r for r in records
        if candidate in r["strategies"] and baseline in r["strategies"]
    ]
    n = len(rows)
    if n < 30:
        return {"races": n, "status": "30レース未満のため区間推定なし"}
    rng = random.Random(seed + n)
    deltas = []
    for _ in range(BOOTSTRAP_ITERATIONS):
        sample = [rows[rng.randrange(n)] for _ in range(n)]
        a_stake = sum(x["strategies"][baseline]["stake"] for x in sample)
        b_stake = sum(x["strategies"][candidate]["stake"] for x in sample)
        a_return = sum(x["strategies"][baseline]["returned"] for x in sample)
        b_return = sum(x["strategies"][candidate]["returned"] for x in sample)
        deltas.append(100 * b_return / b_stake - 100 * a_return / a_stake)
    deltas.sort()
    return {
        "races": n,
        "roi_delta_pp_vs_baseline": round(float(np.mean(deltas)), 3),
        "ci95_low": round(float(deltas[int(.025 * (n and len(deltas)-1))]), 3),
        "ci95_high": round(float(deltas[int(.975 * (len(deltas)-1))]), 3),
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
    }


def summarize_split(records, skipped):
    names = sorted({
        name for row in records for name in row["strategies"]
    })
    result = {
        name: summarize(records, name)
        for name in names
    }
    result["paired_roi_ci_vs_baseline"] = {
        name: paired_roi_ci(records, name)
        for name in names if name != "baseline_top5_all6"
    }
    by_size = {}
    for band in sorted({r["field_size_band"] for r in records}):
        subset = [r for r in records if r["field_size_band"] == band]
        by_size[band] = {
            name: summarize(subset, name)
            for name in names
        }
    return {
        "overall": result,
        "by_field_size": by_size,
        "skipped": skipped,
    }


def main():
    races, horses, payouts, months = nb.load_archives()
    races, excluded_banei = axis_blend.filter_flat_races(races)
    rows, _ = axis_blend.challenger.build_rows(races, horses)
    dates = sorted({str(row["date"]) for row in rows})
    if len(dates) < 100:
        raise RuntimeError(f"学習・検証に必要な開催日数が不足: {len(dates)}")
    train_end = dates[int(len(dates) * 0.70) - 1]
    dev_end = dates[int(len(dates) * 0.85) - 1]
    train_rows = [r for r in rows if str(r["date"]) <= train_end]
    dev_rows = [r for r in rows if train_end < str(r["date"]) <= dev_end]
    later_rows = [r for r in rows if str(r["date"]) > dev_end]
    train_groups = grouped_by_period(train_rows)
    dev_groups = grouped_by_period(dev_rows)
    later_groups = grouped_by_period(later_rows)
    if min(len(train_groups), len(dev_groups), len(later_groups)) < 100:
        raise RuntimeError("学習・開発・後半期間のいずれかで100レース未満です")

    sim_params = axis_blend.fit_similarity(train_rows)
    win_model = axis_blend.fit_win_specialist(train_rows)
    dev_scored = axis_blend.score_groups(dev_groups, sim_params, win_model)
    later_scored = axis_blend.score_groups(later_groups, sim_params, win_model)

    dev_records, dev_skipped = evaluate_split(dev_scored, payouts)
    later_records, later_skipped = evaluate_split(later_scored, payouts)
    if min(len(dev_records), len(later_records)) < 100:
        raise RuntimeError("払戻照合後の評価レースが100未満です")

    report = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "research_only_flat_trio_feature_rescue",
        "scope": "地方競馬の平地のみ。ばんえい除外。研究用で本番未変更。",
        "excluded_banei_race_count": excluded_banei,
        "included_venues": sorted({str(r.get("venue") or "不明") for r in races.values()}),
        "data": {
            "archive_months": months,
            "training_end": train_end,
            "development_end": dev_end,
            "later_period_start": min(str(r["date"]) for r in later_rows),
            "later_period_end": max(str(r["date"]) for r in later_rows),
            "train_races": len(train_groups),
            "development_races": len(dev_groups),
            "later_period_races": len(later_groups),
            "development_evaluated": len(dev_records),
            "later_evaluated": len(later_records),
        },
        "fixed_rules": {
            "axis": "40:60モデルの順位1位を固定",
            "candidate_pool": "同モデル順位2〜7位",
            "tickets_per_race": TICKETS,
            "stake_per_ticket_jpy": TICKET_COST,
            "budget_per_race_jpy": TICKETS * TICKET_COST,
            "base_score": "40:60モデルのレース内順位パーセンタイル",
            "feature_score": "近走成績・末脚/位置上昇・競馬場/距離/馬場/騎手適性の各特徴をレース内順位化し、グループ平均",
            "tested_weights": [0, 0.15, 0.30, 0.45],
        },
        "feature_groups": FEATURE_GROUPS,
        "development": summarize_split(dev_records, dev_skipped),
        "later_period_descriptive_only": summarize_split(later_records, later_skipped),
        "guardrails": [
            "モデルは最初の70%時系列期間だけで学習し、次の15%で比較、最後の15%は時系列確認用の記述結果。",
            "最後の15%は同じ過去アーカイブ内の期間であり、将来の独立ホールドアウトとは呼ばない。",
            "順位・着順・払戻・確定オッズを予測や候補選択に使用しない。",
            "購入前オッズは本実験の候補選択に使用しない。回収率は公式払戻による過去の事後評価。",
            "現レース馬体重・馬体重増減は特徴に使用しない。ばんえいは除外。",
            "全方式で同じ軸・6点・600円。候補順位6〜7位を含む組合せだけを比較。",
            "研究ブランチと成果物のみ。本番モデル・買い目生成・DB・ライブ処理は変更しない。",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("FLAT_TRIO_FEATURE_RESCUE " + json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
