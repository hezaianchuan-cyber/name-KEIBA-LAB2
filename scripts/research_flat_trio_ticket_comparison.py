#!/usr/bin/env python3
"""Research-only flat-racing 3連複 comparison with a fixed six-ticket policy.

The same six-ticket construction is applied to LAB, 40:60 and 50:50 rankings:
axis = rank 1; opponents = ranks 2-5; buy all six unordered pairs with the axis.
Each ticket costs 100 JPY (600 JPY/race). Official payouts are outcomes only.
No production files, model weights or live ticket generation are changed.
"""
from __future__ import annotations

import datetime as dt
import itertools
import json
import random
from pathlib import Path

import nar_backtest as nb
import research_flat_axis_win_blend as axis_blend

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "flat-trio-ticket-comparison.json"
STRATEGIES = (
    "LAB_existing",
    "similarity_40_win_60",
    "similarity_50_win_50",
)
TICKET_COST = 100
OPPONENT_COUNT = 4
BOOTSTRAP_ITERATIONS = 4000


def payout_for_group(group, payouts):
    parts = str(group[0]["race_key"]).split("|")
    if len(parts) != 3:
        return None
    return payouts.get(nb.race_key(parts[0], parts[1], parts[2]))


def build_axis_plus_four_tickets(group, scores):
    """Six unordered trios: top-ranked axis × every pair among ranks 2-5."""
    order = sorted(
        range(len(group)),
        key=lambda i: (-float(scores[i]), int(group[i]["horse_no"])),
    )
    if len(order) < OPPONENT_COUNT + 1:
        return []
    axis = str(group[order[0]]["horse_no"])
    opponents = [str(group[i]["horse_no"]) for i in order[1:OPPONENT_COUNT + 1]]
    return [
        sorted((axis, a, b), key=int)
        for a, b in itertools.combinations(opponents, 2)
    ]


def evaluate(scored, payouts):
    records = {name: [] for name in STRATEGIES}
    missing_payout = 0
    ineligible_races = 0
    for group, strategy_scores in scored:
        payout = payout_for_group(group, payouts)
        if not payout:
            missing_payout += 1
            continue
        if len(group) < 5 or sum(int(r.get("finish") == 1) for r in group) != 1:
            ineligible_races += 1
            continue
        podium = {str(r["horse_no"]) for r in group
                  if r.get("finish") is not None and int(r["finish"]) in (1, 2, 3)}
        if len(podium) != 3:
            ineligible_races += 1
            continue
        for name in STRATEGIES:
            scores = strategy_scores[name]
            order = sorted(
                range(len(group)),
                key=lambda i: (-float(scores[i]), int(group[i]["horse_no"])),
            )
            axis = str(group[order[0]]["horse_no"])
            tickets = build_axis_plus_four_tickets(group, scores)
            returned = sum(nb.payout_for("trio", ticket, payout) or 0 for ticket in tickets)
            records[name].append({
                "race_key": group[0]["race_key"],
                "axis_win": int(any(str(r["horse_no"]) == axis and int(r["finish"]) == 1 for r in group)),
                "axis_top3": int(axis in podium),
                "hit": int(returned > 0),
                "tickets": len(tickets),
                "stake": len(tickets) * TICKET_COST,
                "returned": returned,
            })
    return records, {
        "races_missing_official_payout": missing_payout,
        "races_ineligible_for_trio": ineligible_races,
    }


def percentile(values, q):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * q
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    frac = position - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def summarize(items, seed):
    n = len(items)
    stake = sum(x["stake"] for x in items)
    returned = sum(x["returned"] for x in items)
    return {
        "races": n,
        "axis_win_rate_pct": round(100 * sum(x["axis_win"] for x in items) / n, 3) if n else None,
        "axis_top3_rate_pct": round(100 * sum(x["axis_top3"] for x in items) / n, 3) if n else None,
        "trio_hit_races": sum(x["hit"] for x in items),
        "trio_hit_rate_pct": round(100 * sum(x["hit"] for x in items) / n, 3) if n else None,
        "tickets_per_race": round(sum(x["tickets"] for x in items) / n, 3) if n else None,
        "stake_jpy": stake,
        "return_jpy": returned,
        "profit_jpy": returned - stake,
        "roi_pct": round(100 * returned / stake, 3) if stake else None,
        "roi_ci95": bootstrap_roi(items, seed),
    }


def bootstrap_roi(items, seed):
    if len(items) < 30:
        return {"low": None, "high": None, "reason": "30レース未満"}
    rng = random.Random(seed)
    n = len(items)
    estimates = []
    for _ in range(BOOTSTRAP_ITERATIONS):
        sample = [items[rng.randrange(n)] for _ in range(n)]
        stake = sum(x["stake"] for x in sample)
        if stake:
            estimates.append(100 * sum(x["returned"] for x in sample) / stake)
    return {
        "low": round(percentile(estimates, 0.025), 3),
        "high": round(percentile(estimates, 0.975), 3),
        "method": "レース単位の対応あり再標本化",
        "iterations": BOOTSTRAP_ITERATIONS,
    }


def paired_comparison(a_items, b_items, seed):
    """Paired race bootstrap: report B minus A hit-rate and ROI differences."""
    by_a = {x["race_key"]: x for x in a_items}
    by_b = {x["race_key"]: x for x in b_items}
    keys = sorted(set(by_a) & set(by_b))
    if len(keys) < 30:
        return {"races": len(keys), "reason": "比較可能レースが30未満"}
    rng = random.Random(seed)
    n = len(keys)
    hit_deltas, roi_deltas = [], []
    for _ in range(BOOTSTRAP_ITERATIONS):
        sample = [keys[rng.randrange(n)] for _ in range(n)]
        a = [by_a[k] for k in sample]
        b = [by_b[k] for k in sample]
        hit_deltas.append(100 * (
            sum(x["hit"] for x in b) - sum(x["hit"] for x in a)
        ) / n)
        a_stake, b_stake = sum(x["stake"] for x in a), sum(x["stake"] for x in b)
        if a_stake and b_stake:
            a_roi = 100 * sum(x["returned"] for x in a) / a_stake
            b_roi = 100 * sum(x["returned"] for x in b) / b_stake
            roi_deltas.append(b_roi - a_roi)
    return {
        "races": n,
        "direction": "後者 − 前者",
        "trio_hit_rate_delta_pp": {
            "estimate": round(sum(hit_deltas) / len(hit_deltas), 3),
            "ci95_low": round(percentile(hit_deltas, 0.025), 3),
            "ci95_high": round(percentile(hit_deltas, 0.975), 3),
        },
        "roi_delta_pp": {
            "estimate": round(sum(roi_deltas) / len(roi_deltas), 3),
            "ci95_low": round(percentile(roi_deltas, 0.025), 3),
            "ci95_high": round(percentile(roi_deltas, 0.975), 3),
        },
    }


def main():
    races, horses, payouts, downloaded_months = nb.load_archives()
    rows, _ = axis_blend.challenger.build_rows(races, horses)
    dates = sorted({str(row["date"]) for row in rows})
    train_end = dates[int(len(dates) * 0.70) - 1]
    dev_end = dates[int(len(dates) * 0.85) - 1]
    train_rows = [r for r in rows if str(r["date"]) <= train_end]
    dev_rows = [r for r in rows if train_end < str(r["date"]) <= dev_end]
    hold_rows = [r for r in rows if str(r["date"]) > dev_end]
    train_groups = axis_blend.grouped(train_rows)
    dev_groups = axis_blend.grouped(dev_rows)
    hold_groups = axis_blend.grouped(hold_rows)

    similarity_params = axis_blend.fit_similarity(train_rows)
    win_model = axis_blend.fit_win_specialist(train_rows)
    dev_scored = axis_blend.score_groups(dev_groups, similarity_params, win_model)
    hold_scored = axis_blend.score_groups(hold_groups, similarity_params, win_model)

    dev_records, dev_coverage = evaluate(dev_scored, payouts)
    hold_records, hold_coverage = evaluate(hold_scored, payouts)
    dev_results = {
        name: summarize(items, 20261113 + index)
        for index, (name, items) in enumerate(dev_records.items())
    }
    hold_results = {
        name: summarize(items, 20261213 + index)
        for index, (name, items) in enumerate(hold_records.items())
    }
    comparisons = {}
    for index, candidate in enumerate(("similarity_40_win_60", "similarity_50_win_50")):
        comparisons[candidate] = {
            "development_vs_LAB": paired_comparison(
                dev_records["LAB_existing"], dev_records[candidate], 20261130 + index
            ),
            "descriptive_final_vs_LAB": paired_comparison(
                hold_records["LAB_existing"], hold_records[candidate], 20261230 + index
            ),
        }

    report = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "research_only_fixed_six_ticket_flat_trio_comparison",
        "scope": "地方競馬の平地のみ。ばんえい除外。本番モデル・本番買い目生成は未変更。",
        "data": {
            "downloaded_months": downloaded_months,
            "train_races": len(train_groups),
            "development_races_before_payout_match": len(dev_groups),
            "descriptive_final_races_before_payout_match": len(hold_groups),
            "development_payout_coverage": dev_coverage,
            "descriptive_final_payout_coverage": hold_coverage,
            "development_end": dev_end,
            "descriptive_final_start": min(str(r["date"]) for r in hold_rows),
            "descriptive_final_end": max(str(r["date"]) for r in hold_rows),
        },
        "ticket_policy": {
            "construction": "各モデルの順位1位を軸にし、順位2〜5位の4頭から全6組のペアを作って3連複を6点購入",
            "ticket_count_per_race": 6,
            "cost_per_ticket_jpy": TICKET_COST,
            "budget_per_race_jpy": 6 * TICKET_COST,
            "same_races_and_budget": True,
            "only_axis_ranking_changes": True,
        },
        "development": dev_results,
        "descriptive_final_only": hold_results,
        "paired_comparisons": comparisons,
        "guardrails": [
            "払戻は公式結果としてのみ使用し、モデル学習・順位・買い目選定には使用しない。",
            "最終15%期間は過去データの記述的確認であり、未使用の将来ホールドアウトではない。",
            "現在の馬体重・馬体重増減・発走後情報・確定オッズは予測特徴量に使用しない。",
            "本番モデル・本番買い目生成・DBは変更しない。",
            "ROIは過去の払戻を使った固定条件の後方検証であり、将来の利益を保証しない。",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("FLAT_TRIO_TICKET_COMPARISON " + json.dumps({
        "data": report["data"],
        "development": dev_results,
        "descriptive_final_only": hold_results,
        "paired_comparisons": comparisons,
        "report": str(OUT),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
