#!/usr/bin/env python3
"""Research-only diagnostics for flat NAR trio model and ticket policies."""
from __future__ import annotations
import datetime as dt
import itertools
import json
from collections import defaultdict
from pathlib import Path

import nar_backtest as nb
import research_flat_axis_win_blend as axis_blend

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "flat-trio-deep-diagnostics.json"
MODELS = ("LAB_existing", "similarity_40_win_60", "similarity_50_win_50")
COST = 100
BUDGET = 600


def race_key_parts(group):
    parts = str(group[0]["race_key"]).split("|")
    return parts if len(parts) == 3 else None


def get_payout(group, payouts):
    parts = race_key_parts(group)
    return payouts.get(nb.race_key(parts[0], parts[1], parts[2])) if parts else None


def ranked_indices(group, scores):
    return sorted(range(len(group)), key=lambda i: (-float(scores[i]), int(group[i]["horse_no"])))


def tickets_for(group, scores, policy):
    order = ranked_indices(group, scores)
    if len(order) < 6:
        return []
    if policy == "axis1_top4_pairs":
        axis = order[0]
        return [tuple(sorted((str(group[axis]["horse_no"]), str(group[a]["horse_no"]), str(group[b]["horse_no"])), key=int))
                for a, b in itertools.combinations(order[1:5], 2)]
    if policy == "axis2_plus_one":
        axes = order[:2]
        return [tuple(sorted((str(group[axes[0]]["horse_no"]), str(group[axes[1]]["horse_no"]),
                              str(group[k]["horse_no"])), key=int)) for k in order[2:8]]
    if policy == "top6_score_combos":
        # Select six triples from the top six by summed normalized model scores.
        top = order[:6]
        combos = list(itertools.combinations(top, 3))
        combos.sort(key=lambda c: (-sum(float(scores[i]) for i in c), tuple(sorted(int(group[i]["horse_no"]) for i in c))))
        return [tuple(sorted((str(group[i]["horse_no"]) for i in c), key=int)) for c in combos[:6]]
    raise ValueError(policy)


def payout_for_ticket(ticket, payout):
    return nb.payout_for("trio", list(ticket), payout) or 0


def race_result(group, scores, payout, policy):
    tickets = tickets_for(group, scores, policy)
    returned = sum(payout_for_ticket(t, payout) for t in tickets)
    finish_by_no = {str(r["horse_no"]): r.get("finish") for r in group}
    podium = {no for no, finish in finish_by_no.items() if finish is not None and int(finish) in (1, 2, 3)}
    ranks = ranked_indices(group, scores)
    rank_by_no = {str(group[i]["horse_no"]): rank + 1 for rank, i in enumerate(ranks)}
    podium_ranks = sorted(rank_by_no[x] for x in podium if x in rank_by_no)
    axis = str(group[ranks[0]]["horse_no"])
    return {
        "hit": int(returned > 0), "stake": len(tickets) * COST, "returned": returned,
        "axis_top3": int(axis in podium),
        "axis_win": int(finish_by_no.get(axis) == 1),
        "podium_ranks": podium_ranks,
        "top3_all_in_top5": int(len(podium_ranks) == 3 and max(podium_ranks) <= 5),
        "winning_ticket_payout": returned,
        "ticket_count": len(tickets),
    }


def summarize(records):
    n = len(records)
    stake = sum(x["stake"] for x in records)
    returned = sum(x["returned"] for x in records)
    return {
        "races": n,
        "trio_hit_rate_pct": round(100 * sum(x["hit"] for x in records) / n, 3) if n else None,
        "roi_pct": round(100 * returned / stake, 3) if stake else None,
        "profit_jpy": returned - stake,
        "axis_top3_rate_pct": round(100 * sum(x["axis_top3"] for x in records) / n, 3) if n else None,
        "top3_all_in_top5_pct": round(100 * sum(x["top3_all_in_top5"] for x in records) / n, 3) if n else None,
        "podium_rank_distribution": {
            str(rank): sum(rank in x["podium_ranks"] for x in records) for rank in range(1, 11)
        },
    }


def segment_report(scored, payouts, dimension):
    buckets = defaultdict(lambda: {m: [] for m in MODELS})
    for group, scores_by_model in scored:
        payout = get_payout(group, payouts)
        if not payout or len(group) < 6:
            continue
        if sum(int(r.get("finish") == 1) for r in group) != 1:
            continue
        podium = [r for r in group if r.get("finish") is not None and int(r["finish"]) in (1, 2, 3)]
        if len(podium) != 3:
            continue
        race = group[0]
        if dimension == "venue":
            segment = str(race.get("venue") or "不明")
        elif dimension == "distance_band":
            d = float(race.get("distance_m") or 0)
            segment = "<1200m" if d < 1200 else "1200-1599m" if d < 1600 else "1600-1999m" if d < 2000 else "2000m+"
        elif dimension == "field_size":
            size = float(race.get("field_size") or len(group))
            segment = "8頭以下" if size <= 8 else "9-12頭" if size <= 12 else "13頭以上"
        else:
            raise ValueError(dimension)
        for model in MODELS:
            buckets[segment][model].append(race_result(group, scores_by_model[model], payout, "axis1_top4_pairs"))
    return {seg: {model: summarize(items) for model, items in models.items()}
            for seg, models in sorted(buckets.items())}


def policy_report(scored, payouts):
    output = {}
    for model in MODELS:
        for policy in ("axis1_top4_pairs", "axis2_plus_one", "top6_score_combos"):
            records = []
            for group, scores_by_model in scored:
                payout = get_payout(group, payouts)
                if not payout or len(group) < 8:
                    continue
                if sum(int(r.get("finish") == 1) for r in group) != 1:
                    continue
                podium = [r for r in group if r.get("finish") is not None and int(r["finish"]) in (1, 2, 3)]
                if len(podium) != 3:
                    continue
                scores = scores_by_model[model]
                tickets = tickets_for(group, scores, policy)
                if len(tickets) != 6:
                    continue
                records.append(race_result(group, scores, payout, policy))
            output[f"{model}__{policy}"] = summarize(records)
    return output


def payout_diagnostics(scored, payouts):
    rows = []
    for group, scores_by_model in scored:
        payout = get_payout(group, payouts)
        if not payout or len(group) < 6:
            continue
        podium = [r for r in group if r.get("finish") is not None and int(r["finish"]) in (1, 2, 3)]
        if len(podium) != 3:
            continue
        # Compare 40:60's six-ticket miss pattern to the actual podium ranks.
        result = race_result(group, scores_by_model["similarity_40_win_60"], payout, "axis1_top4_pairs")
        if result["hit"]:
            continue
        ranks = result["podium_ranks"]
        rows.append({
            "payout_100yen": payout.get("trio") if isinstance(payout, dict) else None,
            "podium_ranks": ranks,
            "outside_top5_count": sum(r > 5 for r in ranks),
            "axis_in_podium": result["axis_top3"],
        })
    return {
        "missed_races": len(rows),
        "misses_with_axis_in_podium_pct": round(100 * sum(x["axis_in_podium"] for x in rows) / len(rows), 3) if rows else None,
        "misses_with_at_least_one_podium_outside_top5_pct": round(100 * sum(x["outside_top5_count"] > 0 for x in rows) / len(rows), 3) if rows else None,
        "podium_rank_patterns_top10": {
            "all_three_in_top5": sum(all(r <= 5 for r in x["podium_ranks"]) for x in rows),
            "one_or_more_outside_top5": sum(any(r > 5 for r in x["podium_ranks"]) for x in rows),
            "axis_missed_podium": sum(not x["axis_in_podium"] for x in rows),
        },
        "note": "Missed-race diagnosis is descriptive; no result/payout field is used as a predictor."
    }



def filter_flat_races(races):
    """Exclude Banei/Obihiro races before building chronological features."""
    flat = {
        key: race for key, race in races.items()
        if not any(token in str(race.get("venue") or "") for token in ("帯広", "ばんえい"))
    }
    excluded = len(races) - len(flat)
    if not flat:
        raise RuntimeError("平地レースが0件です。会場フィルターを確認してください")
    return flat, excluded

def main():
    races, horses, payouts, months = nb.load_archives()
    races, excluded_banei = axis_blend.filter_flat_races(races)
    rows, _ = axis_blend.challenger.build_rows(races, horses)
    dates = sorted({str(r["date"]) for r in rows})
    train_end = dates[int(len(dates) * .70) - 1]
    dev_end = dates[int(len(dates) * .85) - 1]
    train = [r for r in rows if str(r["date"]) <= train_end]
    dev = [r for r in rows if train_end < str(r["date"]) <= dev_end]
    final = [r for r in rows if str(r["date"]) > dev_end]
    sim = axis_blend.fit_similarity(train)
    win = axis_blend.fit_win_specialist(train)
    dev_scored = axis_blend.score_groups(axis_blend.grouped(dev), sim, win)
    final_scored = axis_blend.score_groups(axis_blend.grouped(final), sim, win)
    report = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": "平地のみ・ばんえい除外・研究専用・本番未変更",
        "excluded_banei_race_count": excluded_banei,
        "included_venues": sorted({str(r.get("venue") or "不明") for r in races.values()}),
        "periods": {"train_end": train_end, "development_end": dev_end,
                    "descriptive_final_start": min(str(r["date"]) for r in final),
                    "descriptive_final_end": max(str(r["date"]) for r in final)},
        "venue": segment_report(dev_scored, payouts, "venue"),
        "distance_band": segment_report(dev_scored, payouts, "distance_band"),
        "field_size": segment_report(dev_scored, payouts, "field_size"),
        "payout_miss_diagnostics_development_40_60": payout_diagnostics(dev_scored, payouts),
        "ticket_policy_development": policy_report(dev_scored, payouts),
        "descriptive_final_ticket_policy": policy_report(final_scored, payouts),
        "guardrails": [
            "勝ち馬・順位・払戻・当該レース結果は予測特徴量に含めない。",
            "当該レースの馬体重と増減は除外。",
            "最終期間は記述的確認であり将来ホールドアウトではない。",
            "買い目は全方式6点・600円で比較。本番変更なし。"
        ]
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("FLAT_TRIO_DEEP_DIAGNOSTICS " + json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
