#!/usr/bin/env python3
"""Research-only 40:60 ROI segmentation on timestamped pre-off NAR snapshots.

Uses only flat racing, pre-off snapshots captured 2-30 minutes before post,
and a model trained strictly before the first evaluation date. The same
rank-1 axis x pairs from ranks 2-5 (six tickets, JPY 600) is used throughout.
No production model, ticket logic, database, or live workflow is changed.
"""
from __future__ import annotations

import datetime as dt
import itertools
import json
import math
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import nar_backtest as nb
import research_flat_axis_win_blend as axis_blend

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "flat-trio-roi-segments.json"
MODEL = "similarity_40_win_60"
TICKET_COST = 100
TICKETS_PER_RACE = 6
MIN_SEGMENT_RACES = 20
BOOTSTRAP_ITERATIONS = 2000


def normalized_race_key(value):
    parts = str(value or "").split("|")
    if len(parts) != 3:
        return None
    date = re.sub(r"\D", "", parts[1])
    race = re.sub(r"\D", "", parts[2])
    if len(date) != 8 or not race:
        return None
    return nb.race_key(parts[0], date, race)


def snapshot_is_valid(record):
    try:
        captured = dt.datetime.fromisoformat(str(record["captured_at"]).replace("Z", "+00:00"))
        scheduled = dt.datetime.fromisoformat(str(record["scheduled_post_time"]).replace("Z", "+00:00"))
        minutes = float(record["minutes_to_post"])
    except (KeyError, TypeError, ValueError):
        return False
    return (captured.tzinfo is not None and scheduled.tzinfo is not None
            and captured < scheduled and 2 <= minutes <= 30)


def trio_odds_lookup(odds_rows):
    output = {}
    for item in odds_rows or []:
        label = unicodedata.normalize("NFKC", str(item.get("type") or "")).replace(" ", "")
        if "3連複" not in label and "三連複" not in label:
            continue
        numbers = [re.sub(r"\D", "", str(item.get(k) or "")) for k in ("n1", "n2", "n3")]
        if not all(numbers) or len(set(numbers)) != 3:
            continue
        try:
            price = float(item.get("odds"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(price) or price <= 0:
            continue
        key = tuple(sorted((str(int(x)) for x in numbers), key=int))
        output[key] = price
    return output


def make_tickets(group, scores):
    order = sorted(range(len(group)),
                   key=lambda i: (-float(scores[i]), int(group[i]["horse_no"])))
    if len(order) < 5:
        return []
    axis = order[0]
    pairs = []
    for a, b in itertools.combinations(order[1:5], 2):
        combo = tuple(sorted((str(group[axis]["horse_no"]),
                              str(group[a]["horse_no"]),
                              str(group[b]["horse_no"])), key=int))
        pairs.append(combo)
    return pairs if len(pairs) == TICKETS_PER_RACE else []


def payout_for_group(group, payouts):
    parts = str(group[0]["race_key"]).split("|")
    if len(parts) != 3:
        return None
    return payouts.get(nb.race_key(parts[0], parts[1], parts[2]))


def distance_band(value):
    try:
        distance = float(value)
    except (TypeError, ValueError):
        return "距離不明"
    if distance < 1200:
        return "1200m未満"
    if distance < 1600:
        return "1200-1599m"
    if distance < 2000:
        return "1600-1999m"
    return "2000m以上"


def odds_band(value):
    if value < 10:
        return "10倍未満"
    if value < 20:
        return "10-19.9倍"
    if value < 50:
        return "20-49.9倍"
    if value < 100:
        return "50-99.9倍"
    if value < 200:
        return "100-199.9倍"
    return "200倍以上"


def field_size_band(value):
    size = int(value)
    if size <= 8:
        return "8頭以下"
    if size <= 11:
        return "9-11頭"
    if size <= 14:
        return "12-14頭"
    return "15頭以上"


def is_newcomer_race(name):
    normalized = unicodedata.normalize("NFKC", str(name or "")).replace(" ", "")
    return ("新馬" in normalized) or ("メイクデビュー" in normalized)


def filter_scenario(records, exclude_small=False, exclude_newcomer=False):
    kept = [
        row for row in records
        if not (exclude_small and row["field_size"] <= 8)
        and not (exclude_newcomer and row["is_newcomer"])
    ]
    return {
        "excluded_small_field": exclude_small,
        "excluded_newcomer": exclude_newcomer,
        "remaining": summarize(kept),
        "roi_ci95": bootstrap_roi_ci(kept, 20261113 + len(kept)),
    }


def evaluate_group(group, scores, snapshot, payouts):
    tickets = make_tickets(group, scores)
    if len(tickets) != TICKETS_PER_RACE:
        return None, "field_too_small"
    odds = trio_odds_lookup(snapshot.get("odds"))
    ticket_prices = [odds.get(ticket) for ticket in tickets]
    if any(price is None for price in ticket_prices):
        return None, "incomplete_ticket_odds"
    payout = payout_for_group(group, payouts)
    if not payout:
        return None, "missing_payout"
    finish = [r.get("finish") for r in group]
    podium = [x for x in finish if x is not None and int(x) in (1, 2, 3)]
    if len(podium) != 3:
        return None, "invalid_result"
    returned = sum(nb.payout_for("trio", list(ticket), payout) or 0 for ticket in tickets)
    med_odds = sorted(ticket_prices)[len(ticket_prices) // 2 - 1:len(ticket_prices) // 2 + 1]
    median_odds = sum(med_odds) / len(med_odds)
    parts = str(group[0]["race_key"]).split("|")
    return {
        "race_key": group[0]["race_key"],
        "date": parts[1] if len(parts) == 3 else "",
        "venue": str(group[0].get("venue") or "不明"),
        "distance_band": distance_band(group[0].get("distance_m")),
        "field_size": len(group),
        "field_size_band": field_size_band(len(group)),
        "ticket_median_odds": median_odds,
        "ticket_mean_odds": sum(ticket_prices) / len(ticket_prices),
        "odds_band": odds_band(median_odds),
        "hit": int(returned > 0),
        "stake": len(tickets) * TICKET_COST,
        "returned": returned,
        "ticket_odds_min": min(ticket_prices),
        "ticket_odds_max": max(ticket_prices),
        "runner_diagnostics": [
            {
                "horse_no": int(row["horse_no"]),
                "rank": rank + 1,
                "finish": int(row["finish"]) if row.get("finish") is not None else None,
                "is_axis": rank == 0,
                "is_ticket_partner": 1 <= rank <= 4,
            }
            for rank, idx in enumerate(
                sorted(
                    range(len(group)),
                    key=lambda i: (-float(scores[i]), int(group[i]["horse_no"])),
                )
            )
            for row in [group[idx]]
        ],
    }, None


def diagnose_field_size(records, target_band="9-11頭"):
    selected = [r for r in records if r["field_size_band"] == target_band]
    n = len(selected)
    if not n:
        return {"races": 0, "status": "no_races"}
    axis_finish = [next((x["finish"] for x in r["runner_diagnostics"] if x["is_axis"]), None)
                   for r in selected]
    axis_top3 = sum(x is not None and x <= 3 for x in axis_finish)
    axis_wins = sum(x == 1 for x in axis_finish)
    axis_out = n - axis_top3
    podium_rank_counts = defaultdict(int)
    podium_runner_total = 0
    podium_all_in_top5 = 0
    podium_all_in_top7 = 0
    races_with_podium_outside_top5 = 0
    partner_candidates_top5 = 0
    partner_candidates_top6_7 = 0
    for r in selected:
        podium = [x for x in r["runner_diagnostics"] if x["finish"] in (1, 2, 3)]
        ranks = [x["rank"] for x in podium]
        podium_runner_total += len(podium)
        for rank in ranks:
            bucket = "1-5位" if rank <= 5 else ("6-7位" if rank <= 7 else "8位以下")
            podium_rank_counts[bucket] += 1
        podium_all_in_top5 += int(len(ranks) == 3 and max(ranks) <= 5)
        podium_all_in_top7 += int(len(ranks) == 3 and max(ranks) <= 7)
        races_with_podium_outside_top5 += int(any(rank > 5 for rank in ranks))
        partner_candidates_top5 += sum(1 for rank in ranks if 2 <= rank <= 5)
        partner_candidates_top6_7 += sum(1 for rank in ranks if 6 <= rank <= 7)
    hit_records = [r for r in selected if r["hit"]]
    miss_records = [r for r in selected if not r["hit"]]
    return {
        "races": n,
        "axis_top3_rate_pct": round(100 * axis_top3 / n, 3),
        "axis_win_rate_pct": round(100 * axis_wins / n, 3),
        "axis_outside_top3_races": axis_out,
        "axis_outside_top3_rate_pct": round(100 * axis_out / n, 3),
        "podium_rank_counts": dict(podium_rank_counts),
        "podium_runners_total": podium_runner_total,
        "podium_runner_share_by_rank_bucket_pct": {
            k: round(100 * v / podium_runner_total, 3)
            for k, v in podium_rank_counts.items()
        },
        "races_with_any_podium_outside_model_top5": races_with_podium_outside_top5,
        "races_with_all_podium_inside_top5": podium_all_in_top5,
        "races_with_all_podium_inside_top7": podium_all_in_top7,
        "hit_group": summarize(hit_records),
        "miss_group": summarize(miss_records),
        "hit_vs_miss_axis_top3": {
            "hit_races": len(hit_records),
            "miss_races": len(miss_records),
            "hit_axis_top3": sum(
                any(x["is_axis"] and x["finish"] is not None and x["finish"] <= 3
                    for x in r["runner_diagnostics"]) for r in hit_records
            ),
            "miss_axis_top3": sum(
                any(x["is_axis"] and x["finish"] is not None and x["finish"] <= 3
                    for x in r["runner_diagnostics"]) for r in miss_records
            ),
        },
        "interpretation_note": "モデル順位の上位5頭外に3着以内馬がいるか、軸馬自体が3着外かを分離した記述統計。因果・将来成績を示さない。",
    }


def summarize(records):
    n = len(records)
    stake = sum(x["stake"] for x in records)
    returned = sum(x["returned"] for x in records)
    return {
        "races": n,
        "trio_hit_rate_pct": round(100 * sum(x["hit"] for x in records) / n, 3) if n else None,
        "roi_pct": round(100 * returned / stake, 3) if stake else None,
        "stake_jpy": stake,
        "return_jpy": returned,
        "profit_jpy": returned - stake,
        "average_ticket_median_odds": round(
            sum(x["ticket_median_odds"] for x in records) / n, 3
        ) if n else None,
        "sample_warning": "参考値（20レース未満）" if 0 < n < MIN_SEGMENT_RACES else None,
    }


def grouped_summary(records, key):
    groups = defaultdict(list)
    for row in records:
        groups[str(row[key])].append(row)
    return {name: summarize(items) for name, items in sorted(groups.items())}


def bootstrap_roi_ci(records, seed):
    if len(records) < 30:
        return {"races": len(records), "status": "30レース未満のため区間推定なし"}
    rng = random.Random(seed)
    n = len(records)
    estimates = []
    for _ in range(BOOTSTRAP_ITERATIONS):
        sample = [records[rng.randrange(n)] for _ in range(n)]
        stake = sum(x["stake"] for x in sample)
        returned = sum(x["returned"] for x in sample)
        if stake:
            estimates.append(100 * returned / stake)
    estimates.sort()
    return {
        "races": n,
        "ci95_low": round(estimates[int(.025 * (len(estimates) - 1))], 3),
        "ci95_high": round(estimates[int(.975 * (len(estimates) - 1))], 3),
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
    }


def main():
    snapshots = nb.load_odds_snapshots()
    flat_snapshots = {}
    for key, record in snapshots.items():
        venue = str(key[0])
        if any(token in venue for token in ("帯広", "ばんえい")):
            continue
        if not snapshot_is_valid(record):
            continue
        normalized = nb.race_key(key[0], key[1], key[2])
        if normalized and trio_odds_lookup(record.get("odds")):
            flat_snapshots[normalized] = record
    if not flat_snapshots:
        raise RuntimeError("平地の有効な発走前3連複オッズが見つかりません")

    snapshot_dates = sorted({
        re.sub(r"\D", "", str(record.get("date") or key.split("|")[1]))
        for key, record in flat_snapshots.items()
    })
    snapshot_dates = [x for x in snapshot_dates if len(x) == 8]
    if not snapshot_dates:
        raise RuntimeError("オッズ記録の日付を確認できません")
    first_day = dt.datetime.strptime(min(snapshot_dates), "%Y%m%d").date()
    last_day = dt.datetime.strptime(max(snapshot_dates), "%Y%m%d").date()

    # Extend archive through the latest snapshot date so the matching race
    # results and official payouts can be evaluated after the fact.
    nb.END = last_day
    nb.START = first_day - dt.timedelta(days=1095)
    races, horses, payouts, months = nb.load_archives()
    races, excluded_banei = axis_blend.filter_flat_races(races)
    rows, _ = axis_blend.challenger.build_rows(races, horses)

    train_rows = [row for row in rows if str(row["date"]) < first_day.isoformat()]
    if len(train_rows) < 1000:
        raise RuntimeError(f"発走前学習データが不足しています: {len(train_rows)}行")
    train_end = max(str(row["date"]) for row in train_rows)
    sim_params = axis_blend.fit_similarity(train_rows)
    win_model = axis_blend.fit_win_specialist(train_rows)

    eval_rows = [row for row in rows if str(row["date"]).replace("-", "") in set(snapshot_dates)]
    eval_groups = axis_blend.grouped(eval_rows)
    snapshot_groups = []
    for group in eval_groups:
        parts = str(group[0]["race_key"]).split("|")
        if len(parts) != 3:
            continue
        key = nb.race_key(parts[0], parts[1], parts[2])
        if key in flat_snapshots:
            snapshot_groups.append((group, flat_snapshots[key]))
    if not snapshot_groups:
        raise RuntimeError("スナップショットと平地レース結果を照合できません")

    scored = axis_blend.score_groups([x[0] for x in snapshot_groups], sim_params, win_model)
    records = []
    skipped = defaultdict(int)
    for (group, snapshot), (_, strategies) in zip(snapshot_groups, scored):
        result, reason = evaluate_group(group, strategies[MODEL], snapshot, payouts)
        if result:
            parts = str(group[0]["race_key"]).split("|")
            race_meta = races.get(nb.race_key(parts[0], parts[1], parts[2])) if len(parts) == 3 else None
            race_name = str((race_meta or {}).get("name") or "")
            result["race_name"] = race_name
            result["is_newcomer"] = is_newcomer_race(race_name)
            result["race_type"] = "新馬戦" if result["is_newcomer"] else "既走馬戦"
            records.append(result)
        else:
            skipped[reason] += 1

    if not records:
        raise RuntimeError("6点すべての発走前オッズと払戻を照合できたレースがありません")

    report = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "research_only_flat_trio_roi_segments",
        "scope": "地方競馬の平地のみ。ばんえい除外。40:60モデル固定。",
        "model": MODEL,
        "data": {
            "archive_months": months,
            "training_start": min(str(row["date"]) for row in train_rows),
            "training_end": train_end,
            "snapshot_start": min(snapshot_dates),
            "snapshot_end": max(snapshot_dates),
            "snapshot_dates": len(set(snapshot_dates)),
            "snapshot_races_with_trio_odds": len(flat_snapshots),
            "matched_races": len(snapshot_groups),
            "evaluated_races": len(records),
            "excluded_banei_archive_races": excluded_banei,
            "included_venues": sorted({str(r.get("venue") or "不明") for r in races.values()}),
        },
        "fixed_ticket_rule": {
            "axis": "40:60モデル順位1位",
            "partners": "同順位2位から5位の4頭",
            "combinations": "軸1頭と相手4頭から作る6組すべて",
            "tickets_per_race": TICKETS_PER_RACE,
            "cost_per_ticket_jpy": TICKET_COST,
            "budget_per_race_jpy": TICKETS_PER_RACE * TICKET_COST,
        },
        "overall": summarize(records),
        "roi_ci95": bootstrap_roi_ci(records, 20261113),
        "by_ticket_odds_band": grouped_summary(records, "odds_band"),
        "by_field_size": grouped_summary(records, "field_size_band"),
        "nine_to_eleven_runner_diagnostics": diagnose_field_size(records, "9-11頭"),
        "by_newcomer_status": grouped_summary(records, "race_type"),
        "by_field_size_and_newcomer": {
            f"{size} / {race_type}": summarize([
                row for row in records
                if row["field_size_band"] == size and row["race_type"] == race_type
            ])
            for size in sorted({row["field_size_band"] for row in records})
            for race_type in ("新馬戦", "既走馬戦")
        },
        "filter_scenarios": {
            "baseline_all": filter_scenario(records),
            "exclude_field_8_or_less": filter_scenario(records, exclude_small=True),
            "exclude_newcomer": filter_scenario(records, exclude_newcomer=True),
            "exclude_small_or_newcomer": filter_scenario(
                records, exclude_small=True, exclude_newcomer=True
            ),
        },
        "by_venue": grouped_summary(records, "venue"),
        "by_distance": grouped_summary(records, "distance_band"),
        "skipped": dict(sorted(skipped.items())),
        "guardrails": [
            "予測モデルは最初の発走前オッズ日より前のデータだけで学習。",
            "オッズ帯は購入予定6点の発走前3連複オッズ中央値で分類。",
            "全レース6点・600円で統一。6点すべてのオッズが取得できたレースのみ分析。",
            "発走2〜30分前の記録だけを使用。確定オッズ・着順・払戻は予測や買い目選定に不使用。",
            "後方検証は直近スナップショット期間の探索的分析。少頭数・新馬戦の区分と除外シナリオは事前定義した条件で比較。",
            "少頭数は出走馬8頭以下、新馬戦はレース名に新馬またはメイクデビューを含むものとして定義。",
            "見送りシナリオは発走前に判定できる出走頭数・レース名のみを使用。小標本では結論を保留。",
            "本番モデル・買い目生成・DB・ライブ処理は変更しない。",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("FLAT_TRIO_ROI_SEGMENTS " + json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
