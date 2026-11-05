#!/usr/bin/env python3
"""Three-day exploratory pilot for narrowing 3連複 2-4-5 tickets.

Uses only timestamped pre-off model scores and win-odds market ranks.
Official 3連複 payouts are used only as outcomes. This pilot is descriptive;
it does not tune thresholds or change the live model.
"""
from __future__ import annotations

import gzip
import json
import math
import sys
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "trio-narrowing-pilot.json"
PAYOUTS_FILE = DATA / "daily-payouts.json"
PILOT_DATES = {"20261103", "20261104", "20261105"}
BASE_TICKETS = 19
UNIT = 100
OVERLAP_NARROW_THRESHOLD = 5
NARROW_TICKETS = 10

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nar_model_g4_preoff_backtest as g4  # noqa: E402


def norm(value):
    return unicodedata.normalize("NFKC", str(value or "")).strip()


def horse_number(value):
    digits = "".join(ch for ch in str(value or "") if ch.isdigit())
    return str(int(digits)) if digits else ""


def build_245(ranked):
    unique = []
    seen = set()
    for horse in ranked or []:
        no = horse_number(horse)
        if no and no not in seen:
            unique.append(no)
            seen.add(no)
    firsts, seconds, thirds = unique[:2], unique[:4], unique[2:7]
    tickets, seen_tickets = [], set()
    for first in firsts:
        for second in seconds:
            if second == first:
                continue
            for third in thirds:
                if third in (first, second):
                    continue
                combo = tuple(sorted((first, second, third), key=int))
                if combo in seen_tickets:
                    continue
                seen_tickets.add(combo)
                tickets.append(combo)
    return tickets


def narrow_by_rank_sum(tickets, ranked, count=NARROW_TICKETS):
    rank = {horse_number(no): i for i, no in enumerate(ranked, start=1)}
    ordered = sorted(
        tickets,
        key=lambda combo: (
            sum(rank.get(no, 10_000) for no in combo),
            tuple(int(no) for no in combo),
        ),
    )
    return ordered[: max(0, min(int(count), len(ordered)))]


def market_rank(odds):
    rows = []
    for item in odds or []:
        if norm(item.get("type")).replace(" ", "") not in {"単勝", "単勝オッズ"}:
            continue
        no = horse_number(item.get("n1"))
        try:
            price = float(item.get("odds"))
        except (TypeError, ValueError):
            continue
        if not no or not math.isfinite(price) or price <= 0:
            continue
        try:
            popularity = float(item.get("popularity"))
        except (TypeError, ValueError):
            popularity = math.inf
        rows.append((popularity, price, int(no), no))
    # Popularity is the market's explicit ranking; odds break missing/tied ranks.
    rows.sort()
    result, seen = [], set()
    for _pop, _price, _num, no in rows:
        if no not in seen:
            result.append(no)
            seen.add(no)
    return result


def model_rank(predictions):
    rows = []
    for item in predictions or []:
        no = horse_number(item.get("horse_no"))
        try:
            score = float(item.get("score"))
        except (TypeError, ValueError):
            continue
        if no and math.isfinite(score):
            rows.append((score, int(no), no))
    rows.sort(key=lambda x: (-x[0], x[1]))
    result, seen = [], set()
    for _score, _num, no in rows:
        if no not in seen:
            result.append(no)
            seen.add(no)
    return result


def load_trio_outcomes():
    source = json.loads(PAYOUTS_FILE.read_text(encoding="utf-8"))
    outcomes = {}
    for day in source.get("days", []):
        date = "".join(ch for ch in str(day.get("date") or "") if ch.isdigit())
        if date not in PILOT_DATES:
            continue
        for race in day.get("races", []):
            venue = norm(race.get("venue"))
            number = race.get("race")
            if not venue or number is None:
                continue
            key = f"{venue}|{date}|{int(number)}"
            rows = (race.get("payouts") or {}).get("trio") or []
            payouts = {}
            for item in rows:
                nums = [horse_number(x) for x in (item.get("numbers") or [])]
                if len(nums) != 3 or not all(nums) or len(set(nums)) != 3:
                    continue
                combo = tuple(sorted(nums, key=int))
                try:
                    pay = int(item.get("pay_100") or 0)
                except (TypeError, ValueError):
                    continue
                if pay > 0:
                    payouts[combo] = max(pay, payouts.get(combo, 0))
            if payouts:
                outcomes[key] = payouts
    return outcomes


def snapshot_key(row):
    return g4.race_key(row.get("race_key"))


def select_latest_snapshots(snapshots):
    selected = {}
    diagnostics = {
        "snapshot_rows_in_three_day_window": 0,
        "snapshot_rows_outside_preoff_window": 0,
        "snapshot_rows_missing_model_or_market": 0,
    }
    for row in snapshots:
        key = snapshot_key(row)
        date = key.split("|")[1] if key else ""
        if date not in PILOT_DATES:
            continue
        try:
            minutes = float(row.get("minutes_to_post"))
        except (TypeError, ValueError):
            continue
        if not 2.0 <= minutes <= 30.0:
            diagnostics["snapshot_rows_outside_preoff_window"] += 1
            continue
        diagnostics["snapshot_rows_in_three_day_window"] += 1
        model = model_rank(row.get("predictions"))
        market = market_rank(row.get("odds"))
        if len(model) < 7 or len(market) < 7:
            diagnostics["snapshot_rows_missing_model_or_market"] += 1
            continue
        candidate = dict(row)
        candidate["race_key"] = key
        candidate["_model_rank"] = model
        candidate["_market_rank"] = market
        try:
            stamp = g4.parse_time(row.get("captured_at"))
            stamp_value = stamp.timestamp() if stamp else 0
        except (OverflowError, OSError, ValueError):
            stamp_value = 0
        candidate["_selection_order"] = (minutes, -stamp_value)
        previous = selected.get(key)
        if previous is None or candidate["_selection_order"] < previous["_selection_order"]:
            selected[key] = candidate
    return list(selected.values()), diagnostics


def allocate_units(ticket_count, budget_units=BASE_TICKETS):
    if ticket_count <= 0:
        return []
    base, remainder = divmod(budget_units, ticket_count)
    return [base + (1 if i < remainder else 0) for i in range(ticket_count)]


def summarize(rows):
    races = len(rows)
    hits = sum(row["hit"] for row in rows)
    stake = sum(row["stake"] for row in rows)
    returned = sum(row["return"] for row in rows)
    tickets = sum(row["tickets"] for row in rows)
    return {
        "races": races,
        "hit_races": hits,
        "race_hit_rate_pct": round(100 * hits / races, 2) if races else None,
        "tickets": tickets,
        "stake_jpy": stake,
        "return_jpy": returned,
        "profit_jpy": returned - stake,
        "roi_pct": round(100 * returned / stake, 2) if stake else None,
        "skip_rate_pct": None,
    }


def evaluate_strategy(races, strategy_name, ticket_selector):
    per_race_100, fixed_budget = [], []
    for race in races:
        tickets = ticket_selector(race)
        if not tickets:
            continue
        payouts = race["payouts"]
        hit = int(any(combo in payouts for combo in tickets))
        returned_100 = sum(payouts.get(combo, 0) for combo in tickets)
        per_race_100.append({
            "hit": hit,
            "tickets": len(tickets),
            "stake": len(tickets) * UNIT,
            "return": returned_100,
        })
        units = allocate_units(len(tickets))
        returned_fixed = sum(
            payouts.get(combo, 0) * units[i]
            for i, combo in enumerate(tickets)
        )
        fixed_budget.append({
            "hit": hit,
            "tickets": len(tickets),
            "stake": BASE_TICKETS * UNIT,
            "return": returned_fixed,
        })
    return {
        "strategy": strategy_name,
        "per_ticket_100_yen": summarize(per_race_100),
        "same_budget_1900_yen_per_race": summarize(fixed_budget),
    }


def load_github_snapshots():
    rows = []
    diagnostics = {
        "source": "archived five-minute GitHub pre-off snapshots only",
        "files": 0,
        "lines": 0,
        "invalid_json": 0,
        "outside_window": 0,
        "missing_predictions": 0,
        "missing_odds": 0,
    }
    snapshot_dir = DATA / "odds-snapshots"
    for path in sorted(snapshot_dir.glob("*.jsonl.gz")):
        diagnostics["files"] += 1
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                diagnostics["lines"] += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    diagnostics["invalid_json"] += 1
                    continue
                key = g4.race_key(record.get("race_key"))
                date = key.split("|")[1] if key else ""
                if date not in PILOT_DATES:
                    continue
                try:
                    minutes = float(record.get("minutes_to_post"))
                except (TypeError, ValueError):
                    continue
                if not 2.0 <= minutes <= 30.0:
                    diagnostics["outside_window"] += 1
                    continue
                predictions = record.get("predictions") or []
                odds = record.get("odds") or []
                if len(predictions) < 3:
                    diagnostics["missing_predictions"] += 1
                    continue
                if not odds:
                    diagnostics["missing_odds"] += 1
                    continue
                rows.append({
                    "race_key": key,
                    "date": date,
                    "captured_at": record.get("captured_at"),
                    "minutes_to_post": minutes,
                    "predictions": predictions,
                    "odds": odds,
                })
    return rows, diagnostics


def rank_divergence(model, market):
    model_pos = {no: i for i, no in enumerate(model, start=1)}
    market_pos = {no: i for i, no in enumerate(market, start=1)}
    shared = set(model_pos) & set(market_pos)
    union_top7 = set(model[:7]) | set(market[:7])
    gaps = {
        no: market_pos[no] - model_pos[no]
        for no in shared
    }
    top7_gaps = [abs(gaps[no]) for no in union_top7 if no in gaps]
    overlap = len(set(model[:7]) & set(market[:7]))
    return {
        "overlap_top7": overlap,
        "mean_abs_rank_gap_top7_union": (
            sum(top7_gaps) / len(top7_gaps) if top7_gaps else None
        ),
        "rank_gaps": gaps,
        "model_positions": model_pos,
        "market_positions": market_pos,
    }


def divergence_band(overlap):
    if overlap >= 6:
        return "agreement_6_7"
    if overlap >= 4:
        return "moderate_4_5"
    return "high_divergence_0_3"


def summarize_divergence(races):
    grouped = defaultdict(list)
    for race in races:
        grouped[divergence_band(race["overlap_top7"])].append(race)

    output = {}
    for band in ("agreement_6_7", "moderate_4_5", "high_divergence_0_3"):
        subset = grouped.get(band, [])
        if not subset:
            output[band] = {"races": 0}
            continue
        output[band] = {
            "races": len(subset),
            "avg_top7_overlap": round(
                sum(r["overlap_top7"] for r in subset) / len(subset), 2
            ),
            "avg_abs_rank_gap_top7_union": round(
                sum(r["mean_abs_rank_gap_top7_union"] for r in subset) / len(subset), 2
            ),
            "strategies": [
                evaluate_strategy(subset, "LAB_19", lambda r: r["model_tickets"]),
                evaluate_strategy(subset, "LAB_10", lambda r: r["model_narrow"]),
                evaluate_strategy(subset, "MARKET_19", lambda r: r["market_tickets"]),
                evaluate_strategy(subset, "MARKET_10", lambda r: r["market_narrow"]),
            ],
        }
    return output


def summarize_horse_signals(races):
    buckets = {
        "lab_upgrade_5plus": [],
        "lab_upgrade_3_4": [],
        "lab_upgrade_1_2": [],
        "near_alignment_minus2_to_plus2": [],
        "market_upgrade_3_4": [],
        "market_upgrade_5plus": [],
    }
    top3 = {"LAB": [0, 0], "market": [0, 0]}
    top7 = {"LAB": [0, 0], "market": [0, 0]}
    for race in races:
        winning_horses = {
            no for combo in race["payouts"] for no in combo
        }
        model_pos = race["model_positions"]
        market_pos = race["market_positions"]
        for label, ranking in (("LAB", race["model_rank"]), ("market", race["market_rank"])):
            for limit, store in ((3, top3[label]), (7, top7[label])):
                selected = ranking[:limit]
                store[0] += sum(no in winning_horses for no in selected)
                store[1] += len(selected)
        for no in set(model_pos) & set(market_pos):
            gap = market_pos[no] - model_pos[no]
            hit = int(no in winning_horses)
            if no in race["model_rank"][:7]:
                if gap >= 5:
                    buckets["lab_upgrade_5plus"].append(hit)
                elif gap >= 3:
                    buckets["lab_upgrade_3_4"].append(hit)
                elif gap >= 1:
                    buckets["lab_upgrade_1_2"].append(hit)
                elif gap >= -2:
                    buckets["near_alignment_minus2_to_plus2"].append(hit)
            if no in race["market_rank"][:7]:
                if gap <= -5:
                    buckets["market_upgrade_5plus"].append(hit)
                elif gap <= -3:
                    buckets["market_upgrade_3_4"].append(hit)
    result = {
        "horse_selection_in_winning_trio": {
            "LAB_top3": {
                "hits": top3["LAB"][0], "selected": top3["LAB"][1],
                "rate_pct": round(100 * top3["LAB"][0] / top3["LAB"][1], 2) if top3["LAB"][1] else None,
            },
            "market_top3": {
                "hits": top3["market"][0], "selected": top3["market"][1],
                "rate_pct": round(100 * top3["market"][0] / top3["market"][1], 2) if top3["market"][1] else None,
            },
            "LAB_top7": {
                "hits": top7["LAB"][0], "selected": top7["LAB"][1],
                "rate_pct": round(100 * top7["LAB"][0] / top7["LAB"][1], 2) if top7["LAB"][1] else None,
            },
            "market_top7": {
                "hits": top7["market"][0], "selected": top7["market"][1],
                "rate_pct": round(100 * top7["market"][0] / top7["market"][1], 2) if top7["market"][1] else None,
            },
        },
        "lab_vs_market_rank_gap": {},
    }
    for name, values in buckets.items():
        result["lab_vs_market_rank_gap"][name] = {
            "horses": len(values),
            "horses_in_winning_trio": sum(values),
            "inclusion_rate_pct": round(100 * sum(values) / len(values), 2) if values else None,
        }
    return result


def main():
    outcomes = load_trio_outcomes()
    snapshots, loader_diagnostics = load_github_snapshots()
    selected, selection_diagnostics = select_latest_snapshots(snapshots)

    races = []
    no_outcome = 0
    for snapshot in selected:
        payouts = outcomes.get(snapshot["race_key"])
        if not payouts:
            no_outcome += 1
            continue
        model = snapshot["_model_rank"]
        market = snapshot["_market_rank"]
        model_tickets = build_245(model)
        market_tickets = build_245(market)
        if len(model_tickets) < 10 or len(market_tickets) < 10:
            continue
        divergence = rank_divergence(model, market)
        overlap = divergence["overlap_top7"]
        races.append({
            "race_key": snapshot["race_key"],
            "date": snapshot["race_key"].split("|")[1],
            "model_rank": model,
            "market_rank": market,
            "model_tickets": model_tickets,
            "market_tickets": market_tickets,
            "model_narrow": narrow_by_rank_sum(model_tickets, model),
            "market_narrow": narrow_by_rank_sum(market_tickets, market),
            "overlap_top7": overlap,
            "mean_abs_rank_gap_top7_union": divergence["mean_abs_rank_gap_top7_union"],
            "rank_gaps": divergence["rank_gaps"],
            "model_positions": divergence["model_positions"],
            "market_positions": divergence["market_positions"],
            "payouts": payouts,
        })

    strategies = [
        evaluate_strategy(races, "A_current_19_model_245", lambda r: r["model_tickets"]),
        evaluate_strategy(races, "B_narrow_10_model_rank_sum", lambda r: r["model_narrow"]),
        evaluate_strategy(
            races,
            "C_overlap_adaptive_10_if_overlap_ge_5_else_19",
            lambda r: r["model_narrow"] if r["overlap_top7"] >= OVERLAP_NARROW_THRESHOLD else r["model_tickets"],
        ),
        evaluate_strategy(races, "D_market_19_market_245", lambda r: r["market_tickets"]),
        evaluate_strategy(races, "E_market_narrow_10_market_rank_sum", lambda r: r["market_narrow"]),
    ]

    by_day = {}
    for date in sorted(PILOT_DATES):
        day_races = [r for r in races if r["date"] == date]
        by_day[date] = {
            "eligible_races": len(day_races),
            "model_market_top7_overlap_avg": round(
                sum(r["overlap_top7"] for r in day_races) / len(day_races), 2
            ) if day_races else None,
        }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "feature": "three_day_trio_narrowing_pilot",
        "status": "exploratory_only",
        "period": ["2026-11-03", "2026-11-05"],
        "method": {
            "prediction_source": "timestamped raw model scores saved before race start",
            "snapshot_archive": "GitHub five-minute pre-off archive only; Supabase one-minute rows excluded for reproducibility",
            "market_source": "win odds and explicit popularity from the same pre-off snapshot",
            "snapshot_window_minutes_to_post": [2, 30],
            "snapshot_selection": "one closest-to-post valid snapshot per race; no post-off odds",
            "formation": "deduplicated 2-4-5 3連複, up to 19 unordered combinations",
            "narrowing": "select 10 of the existing 19 combinations with the lowest sum of LAB (or market) ranks",
            "adaptive_rule": f"narrow to 10 only when LAB top-7 and market top-7 overlap is at least {OVERLAP_NARROW_THRESHOLD}; otherwise retain 19",
            "same_budget": f"JPY{BASE_TICKETS * UNIT} per race, redistributed in 100-yen units over selected tickets",
            "outcome_source": "official 3連複 pay_100 in data/daily-payouts.json",
            "thresholds_tuned_on_pilot": False,
            "live_model_changed": False,
        },
        "coverage": {
            "dates_with_official_trio_outcomes": sorted({key.split("|")[1] for key in outcomes}),
            "races_with_official_trio_outcomes": len(outcomes),
            "raw_preoff_snapshots_loaded": len(snapshots),
            "unique_preoff_races_with_model_and_market": len(selected),
            "eligible_races_with_trio_outcome": len(races),
            "races_missing_trio_outcome": no_outcome,
            "by_day": by_day,
            "loader_diagnostics": loader_diagnostics,
            "selection_diagnostics": selection_diagnostics,
        },
        "results": strategies,
        "divergence_analysis": {
            "bands": summarize_divergence(races),
            "horse_level_signals": summarize_horse_signals(races),
            "definition": {
                "overlap_top7": "count of horses shared by LAB top 7 and market top 7",
                "bands": {
                    "agreement_6_7": "top-7 overlap 6 or 7",
                    "moderate_4_5": "top-7 overlap 4 or 5",
                    "high_divergence_0_3": "top-7 overlap 0 to 3",
                },
                "rank_gap": "market rank minus LAB rank; positive means LAB ranks the horse higher",
                "horse_outcome": "horse appears in at least one official winning 3連複 combination",
            },
        },
        "interpretation": (
            "This is a small three-day exploratory sample, not a profitability proof. "
            "Compare both actual 100-yen-per-ticket spend and the equal JPY1900-per-race view. "
            "Do not promote any strategy to live use from this pilot."
        ),
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

# Pilot revision: trigger a fresh Actions run after workflow registration.
