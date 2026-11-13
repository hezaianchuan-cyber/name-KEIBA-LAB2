#!/usr/bin/env python3
"""Research-only 3連複 partner-pool test for flat NAR racing.

Keep the 40:60 ranking fixed and compare partner pools through ranks 5, 6 and 7.
Every policy buys exactly six tickets (600 JPY) per eligible race. No production
model, ticket generator, database or live workflow is changed.
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
OUT = ROOT / "data" / "flat-trio-partner-pool-test.json"
MODEL = "similarity_40_win_60"
POOL_SIZES = (5, 6, 7)
COST = 100
TICKETS_PER_RACE = 6
BOOTSTRAP_ITERATIONS = 4000


def race_payout(group, payouts):
    parts = str(group[0]["race_key"]).split("|")
    if len(parts) != 3:
        return None
    return payouts.get(nb.race_key(parts[0], parts[1], parts[2]))


def make_tickets(group, scores, pool_size):
    """Fix rank-1 axis; choose the six highest-scoring pairs from ranks 2..K."""
    order = sorted(range(len(group)),
                   key=lambda i: (-float(scores[i]), int(group[i]["horse_no"])))
    if len(order) < pool_size:
        return []
    axis = order[0]
    candidates = []
    for a, b in itertools.combinations(order[1:pool_size], 2):
        pair_score = float(scores[a]) + float(scores[b])
        horse_pair = tuple(sorted((int(group[a]["horse_no"]), int(group[b]["horse_no"]))))
        candidates.append((pair_score, horse_pair, a, b))
    candidates.sort(key=lambda x: (-x[0], x[1]))
    chosen = candidates[:TICKETS_PER_RACE]
    if len(chosen) != TICKETS_PER_RACE:
        return []
    tickets = []
    for _, _, a, b in chosen:
        tickets.append(tuple(sorted(
            (str(group[axis]["horse_no"]), str(group[a]["horse_no"]),
             str(group[b]["horse_no"])), key=int)))
    return tickets


def evaluate_one(group, scores, payout, pool_size):
    order = sorted(range(len(group)),
                   key=lambda i: (-float(scores[i]), int(group[i]["horse_no"])))
    tickets = make_tickets(group, scores, pool_size)
    returned = sum(nb.payout_for("trio", list(t), payout) or 0 for t in tickets)
    finish = {str(r["horse_no"]): r.get("finish") for r in group}
    podium = {no for no, value in finish.items()
              if value is not None and int(value) in (1, 2, 3)}
    rank_by_no = {str(group[i]["horse_no"]): rank + 1
                  for rank, i in enumerate(order)}
    outside_5 = [no for ticket in tickets for no in ticket
                 if rank_by_no.get(no, 99) > 5]
    axis = str(group[order[0]]["horse_no"])
    return {
        "race_key": group[0]["race_key"],
        "hit": int(returned > 0),
        "stake": len(tickets) * COST,
        "returned": returned,
        "axis_top3": int(axis in podium),
        "podium_ranks": sorted(rank_by_no[x] for x in podium if x in rank_by_no),
        "tickets": len(tickets),
        "tickets_using_outside_top5": sum(
            any(rank_by_no.get(no, 99) > 5 for no in ticket) for ticket in tickets
        ),
        "tickets_using_rank6_or_lower": sum(
            any(rank_by_no.get(no, 99) >= 6 for no in ticket) for ticket in tickets
        ),
    }


def summarize(items):
    n = len(items)
    stake = sum(x["stake"] for x in items)
    returned = sum(x["returned"] for x in items)
    return {
        "races": n,
        "tickets_per_race": round(sum(x["tickets"] for x in items) / n, 3) if n else None,
        "trio_hit_rate_pct": round(100 * sum(x["hit"] for x in items) / n, 3) if n else None,
        "roi_pct": round(100 * returned / stake, 3) if stake else None,
        "stake_jpy": stake,
        "return_jpy": returned,
        "profit_jpy": returned - stake,
        "axis_top3_rate_pct": round(100 * sum(x["axis_top3"] for x in items) / n, 3) if n else None,
        "races_with_any_ticket_using_outside_top5_pct": round(
            100 * sum(x["tickets_using_outside_top5"] > 0 for x in items) / n, 3
        ) if n else None,
        "tickets_using_rank6_or_lower_pct": round(
            100 * sum(x["tickets_using_rank6_or_lower"] for x in items)
            / max(sum(x["tickets"] for x in items), 1), 3
        ) if n else None,
        "podium_with_rank6_or_lower_pct": round(
            100 * sum(any(rank >= 6 for rank in x["podium_ranks"]) for x in items) / n, 3
        ) if n else None,
    }


def percentile(values, q):
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] * (1 - (pos - lo)) + ordered[hi] * (pos - lo)


def paired_bootstrap(base, candidate, seed):
    a = {x["race_key"]: x for x in base}
    b = {x["race_key"]: x for x in candidate}
    keys = sorted(set(a) & set(b))
    if len(keys) < 30:
        return {"races": len(keys), "status": "比較レース不足"}
    rng = random.Random(seed)
    n = len(keys)
    hit_delta, roi_delta = [], []
    for _ in range(BOOTSTRAP_ITERATIONS):
        sample = [keys[rng.randrange(n)] for _ in range(n)]
        aa, bb = [a[k] for k in sample], [b[k] for k in sample]
        hit_delta.append(100 * (sum(x["hit"] for x in bb)
                                - sum(x["hit"] for x in aa)) / n)
        sa, sb = sum(x["stake"] for x in aa), sum(x["stake"] for x in bb)
        ra, rb = sum(x["returned"] for x in aa), sum(x["returned"] for x in bb)
        if sa and sb:
            roi_delta.append(100 * rb / sb - 100 * ra / sa)
    return {
        "races": n,
        "direction": "候補プール拡大 − 5頭基準",
        "hit_rate_delta_pp": {
            "estimate": round(sum(hit_delta) / len(hit_delta), 3),
            "ci95_low": round(percentile(hit_delta, .025), 3),
            "ci95_high": round(percentile(hit_delta, .975), 3),
        },
        "roi_delta_pp": {
            "estimate": round(sum(roi_delta) / len(roi_delta), 3),
            "ci95_low": round(percentile(roi_delta, .025), 3),
            "ci95_high": round(percentile(roi_delta, .975), 3),
        },
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
    }


def evaluate_period(scored, payouts):
    records = {k: [] for k in POOL_SIZES}
    skipped = {"missing_payout": 0, "invalid_result": 0, "field_too_small": 0}
    for group, scores_by_model in scored:
        payout = race_payout(group, payouts)
        if not payout:
            skipped["missing_payout"] += 1
            continue
        if len(group) < 7:
            skipped["field_too_small"] += 1
            continue
        if sum(int(r.get("finish") == 1) for r in group) != 1:
            skipped["invalid_result"] += 1
            continue
        podium = [r for r in group if r.get("finish") is not None
                  and int(r["finish"]) in (1, 2, 3)]
        if len(podium) != 3:
            skipped["invalid_result"] += 1
            continue
        scores = scores_by_model[MODEL]
        for size in POOL_SIZES:
            result = evaluate_one(group, scores, payout, size)
            if result["tickets"] == TICKETS_PER_RACE:
                records[size].append(result)
    return records, skipped


def main():
    races, horses, payouts, months = nb.load_archives()
    races, excluded_banei = axis_blend.filter_flat_races(races)
    rows, _ = axis_blend.challenger.build_rows(races, horses)
    dates = sorted({str(r["date"]) for r in rows})
    train_end = dates[int(len(dates) * .70) - 1]
    dev_end = dates[int(len(dates) * .85) - 1]
    train = [r for r in rows if str(r["date"]) <= train_end]
    dev = [r for r in rows if train_end < str(r["date"]) <= dev_end]
    later = [r for r in rows if str(r["date"]) > dev_end]
    sim = axis_blend.fit_similarity(train)
    win = axis_blend.fit_win_specialist(train)
    dev_scored = axis_blend.score_groups(axis_blend.grouped(dev), sim, win)
    later_scored = axis_blend.score_groups(axis_blend.grouped(later), sim, win)

    dev_records, dev_skipped = evaluate_period(dev_scored, payouts)
    later_records, later_skipped = evaluate_period(later_scored, payouts)

    def period_report(records, skipped, seed):
        summaries = {str(k): summarize(v) for k, v in records.items()}
        comparisons = {
            str(k): paired_bootstrap(records[5], records[k], seed + k)
            for k in (6, 7)
        }
        return {"summary": summaries, "paired_vs_top5": comparisons,
                "skipped": skipped}

    report = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "research_only_flat_trio_partner_pool_test",
        "scope": "地方競馬の平地のみ。ばんえい除外。40:60モデル固定。",
        "excluded_banei_race_count": excluded_banei,
        "included_venues": sorted({str(r.get("venue") or "不明") for r in races.values()}),
        "data": {
            "downloaded_months": months,
            "train_end": train_end,
            "development_end": dev_end,
            "descriptive_later_start": min(str(r["date"]) for r in later),
            "descriptive_later_end": max(str(r["date"]) for r in later),
            "model": MODEL,
        },
        "ticket_rule": {
            "axis": "順位1位で固定",
            "partner_pool": "順位2位から5位/6位/7位まで",
            "pair_selection": "候補ペア全組をモデル得点の合計順に並べ、上位6組を採用",
            "tickets_per_race": TICKETS_PER_RACE,
            "cost_per_ticket_jpy": COST,
            "budget_per_race_jpy": COST * TICKETS_PER_RACE,
            "same_races_and_budget": True,
        },
        "development": period_report(dev_records, dev_skipped, 20261113),
        "descriptive_later": period_report(later_records, later_skipped, 20261213),
        "guardrails": [
            "順位予測は学習期間だけで学習し、払戻・着順は買い目選定に使わない。",
            "当該レースの馬体重と増減は除外。発走後情報・確定オッズは不使用。",
            "後半期間は過去データの記述的確認であり、将来の未使用ホールドアウトではない。",
            "全方式6点・600円で比較。本番モデル・買い目生成・DBは変更しない。",
            "ROIは過去の払戻を用いた後方検証であり、将来利益を保証しない。",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("FLAT_TRIO_PARTNER_POOL_TEST " + json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
