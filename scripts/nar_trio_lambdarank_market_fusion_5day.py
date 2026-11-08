#!/usr/bin/env python3
"""Research-only five-day NAR trio test: LambdaRank fused with timestamped pre-off market odds.

LambdaRank is trained only on races before the five-day window. The fusion weight is
selected on the first three test dates and evaluated on the final two untouched dates.
No final odds, live model files, or production tickets are used or modified.
"""
from __future__ import annotations
import datetime as dt
import itertools
import json
import math
import re
import unicodedata
from pathlib import Path
import nar_backtest as nb
import nar_ml_challenger_backtest as ml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "nar-trio-lambdarank-market-fusion-5day.json"
BLEND_WEIGHTS = (0.00, 0.25, 0.50, 0.75, 1.00)
TICKETS_PER_RACE = 12


def race_key_text(value):
    parts = str(value or "").split("|")
    if len(parts) != 3:
        return None
    date = re.sub(r"\D", "", parts[1])
    race = re.sub(r"\D", "", parts[2])
    if len(date) != 8 or not race:
        return None
    return f"{parts[0]}|{date}|{int(race)}"


def snapshot_date(snapshot):
    key = race_key_text(snapshot.get("race_key"))
    if not key:
        return None
    try:
        return dt.datetime.strptime(key.split("|")[1], "%Y%m%d").date()
    except ValueError:
        return None


def valid_snapshot(snapshot):
    try:
        captured = dt.datetime.fromisoformat(str(snapshot["captured_at"]).replace("Z", "+00:00"))
        scheduled = dt.datetime.fromisoformat(str(snapshot["scheduled_post_time"]).replace("Z", "+00:00"))
        minutes = float(snapshot["minutes_to_post"])
    except (KeyError, TypeError, ValueError):
        return False
    return captured.tzinfo is not None and scheduled.tzinfo is not None and captured < scheduled and 2 <= minutes <= 30


def market_win_probabilities(snapshot):
    """Normalize inverse win odds from the same timestamped pre-off snapshot."""
    inverse = {}
    for row in snapshot.get("odds") or []:
        label = unicodedata.normalize("NFKC", str(row.get("type") or "")).replace(" ", "")
        if "単勝" not in label:
            continue
        digits = re.sub(r"\D", "", str(row.get("n1") or ""))
        try:
            odds = float(row.get("odds"))
        except (TypeError, ValueError):
            continue
        if digits and odds > 0 and math.isfinite(odds):
            inverse[str(int(digits))] = 1.0 / odds
    total = sum(inverse.values())
    return {horse: value / total for horse, value in inverse.items()} if total > 0 else {}


def percentile_scores(values):
    """Convert arbitrary per-horse scores into stable [0,1] within-race ranks."""
    if not values:
        return {}
    ordered = sorted(values.items(), key=lambda item: (-float(item[1]), int(item[0])))
    if len(ordered) == 1:
        return {ordered[0][0]: 0.5}
    result = {}
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average_rank = (index + end - 1) / 2
        score = 1.0 - average_rank / (len(ordered) - 1)
        for pos in range(index, end):
            result[ordered[pos][0]] = score
        index = end
    return result


def blend_rank_scores(model_scores, market_scores, market_weight):
    if set(model_scores) != set(market_scores):
        raise ValueError("Model and market scores must cover the same runners")
    return {
        horse: (1.0 - market_weight) * model_scores[horse] + market_weight * market_scores[horse]
        for horse in model_scores
    }


def rank_horses(group, scores):
    return sorted(group, key=lambda row: (-scores[str(row["horse_no"])], int(row["horse_no"])))


def build_trio_tickets(group, scores, limit=TICKETS_PER_RACE):
    """Rank unordered three-runner sets by the sum of their fused runner scores."""
    candidates = []
    for combo in itertools.combinations((str(row["horse_no"]) for row in group), 3):
        key = tuple(sorted(combo, key=int))
        candidates.append((sum(scores[horse] for horse in key), key))
    candidates.sort(key=lambda item: (-item[0], tuple(map(int, item[1]))))
    return [list(key) for _, key in candidates[:limit]]


def evaluate_races(races, weight):
    totals = {
        "races": 0, "winner_top1_hits": 0, "winner_top3_hits": 0,
        "podium_horses_captured": 0, "podium_horses_total": 0,
        "exact_podium_top3_hits": 0, "trio_hit_races": 0,
        "tickets": 0, "stake_jpy": 0, "return_jpy": 0,
    }
    for item in races:
        scores = blend_rank_scores(item["model_rank_scores"], item["market_rank_scores"], weight)
        ranked = rank_horses(item["group"], scores)
        winner = {str(row["horse_no"]) for row in item["group"] if row["finish"] == 1}
        podium = {str(row["horse_no"]) for row in item["group"] if row["finish"] in (1, 2, 3)}
        if not winner or len(podium) != 3:
            continue
        totals["races"] += 1
        totals["winner_top1_hits"] += int(str(ranked[0]["horse_no"]) in winner)
        totals["winner_top3_hits"] += int(any(str(row["horse_no"]) in winner for row in ranked[:3]))
        predicted_top3 = {str(row["horse_no"]) for row in ranked[:3]}
        totals["podium_horses_captured"] += len(predicted_top3 & podium)
        totals["podium_horses_total"] += 3
        totals["exact_podium_top3_hits"] += int(predicted_top3 == podium)
        tickets = build_trio_tickets(item["group"], scores)
        totals["tickets"] += len(tickets)
        totals["stake_jpy"] += len(tickets) * 100
        returned = sum(nb.payout_for("trio", ticket, item["payout"]) or 0 for ticket in tickets)
        totals["return_jpy"] += returned
        totals["trio_hit_races"] += int(returned > 0)
    count = totals["races"]
    stake = totals["stake_jpy"]
    totals.update({
        "winner_top1_rate_pct": round(100 * totals["winner_top1_hits"] / count, 2) if count else None,
        "winner_top3_capture_rate_pct": round(100 * totals["winner_top3_hits"] / count, 2) if count else None,
        "podium_capture_rate_pct": round(100 * totals["podium_horses_captured"] / totals["podium_horses_total"], 2) if totals["podium_horses_total"] else None,
        "exact_podium_top3_rate_pct": round(100 * totals["exact_podium_top3_hits"] / count, 2) if count else None,
        "trio_hit_rate_pct": round(100 * totals["trio_hit_races"] / count, 2) if count else None,
        "roi_pct": round(100 * totals["return_jpy"] / stake, 2) if stake else None,
        "profit_jpy": totals["return_jpy"] - stake,
        "tickets_per_race": round(totals["tickets"] / count, 2) if count else None,
    })
    return totals


def select_market_weight(dev_races):
    """Tune only on the first three dates; never inspect final two-day holdout results."""
    results = {f"{weight:.2f}": evaluate_races(dev_races, weight) for weight in BLEND_WEIGHTS}
    selected = max(BLEND_WEIGHTS, key=lambda weight: (
        results[f"{weight:.2f}"]["trio_hit_rate_pct"] if results[f"{weight:.2f}"]["trio_hit_rate_pct"] is not None else -1,
        results[f"{weight:.2f}"]["podium_capture_rate_pct"] if results[f"{weight:.2f}"]["podium_capture_rate_pct"] is not None else -1,
        results[f"{weight:.2f}"]["winner_top3_capture_rate_pct"] if results[f"{weight:.2f}"]["winner_top3_capture_rate_pct"] is not None else -1,
        -weight,
    ))
    return selected, results


def feature_matrix(group, category_maps):
    return [[row[key] if key not in ml.CATEGORICAL else category_maps[key].get(row[key], -1)
             for key in ml.RAW_FEATURES] for row in group]


def main():
    snapshots = nb.load_odds_snapshots()
    snapshot_by_key = {
        "|".join(key): record for key, record in snapshots.items()
        if valid_snapshot(record) and market_win_probabilities(record)
    }
    if not snapshot_by_key:
        raise RuntimeError("No valid 2-30 minute pre-off market snapshots with win odds were found")
    snapshot_end = max(snapshot_date(record) for record in snapshot_by_key.values())
    nb.END = snapshot_end
    nb.START = snapshot_end - dt.timedelta(days=1095)
    print(f"TRIO_FUSION_ARCHIVE_WINDOW start={nb.START} end={nb.END}", flush=True)
    races, horses, payouts, downloads = nb.load_archives()
    rows, all_dates = ml.build_rows(races, horses)

    eligible_dates = set()
    for key, snapshot in snapshot_by_key.items():
        parts = key.split("|")
        race_key = nb.race_key(parts[0], parts[1], parts[2])
        payout = payouts.get(race_key) or {}
        if race_key in races and payout.get("trio"):
            eligible_dates.add(snapshot_date(snapshot))
    test_dates = sorted(eligible_dates)[-5:]
    if len(test_dates) < 5:
        raise RuntimeError(f"Need five completed snapshot dates with trio payouts; found {len(test_dates)}: {test_dates}")
    test_date_set = set(test_dates)
    test_start = test_dates[0]
    training_dates = [day for day in all_dates if day < test_start]
    if len(training_dates) < 100:
        raise RuntimeError(f"Insufficient prior training dates: {len(training_dates)}")

    training_date_set = set(training_dates)
    training_rows = [row for row in rows if dt.date.fromisoformat(row["date"]) in training_date_set]
    train_cut = max(1, int(len(training_dates) * 0.85))
    fit_dates = set(training_dates[:train_cut])
    dev_dates = set(training_dates[train_cut:])
    fit_groups = ml.group_rows([row for row in training_rows if dt.date.fromisoformat(row["date"]) in fit_dates])
    dev_groups = ml.group_rows([row for row in training_rows if dt.date.fromisoformat(row["date"]) in dev_dates])
    if not fit_groups or not dev_groups:
        raise RuntimeError("Chronological LambdaRank fit/development split is empty")

    fit_rows = [row for row in training_rows if dt.date.fromisoformat(row["date"]) in fit_dates]
    category_maps = {
        key: {value: index for index, value in enumerate(sorted({row[key] for row in fit_rows}))}
        for key in ml.CATEGORICAL
    }
    print(f"TRIO_FUSION_LAMBDARANK fit_races={len(fit_groups)} dev_races={len(dev_groups)}", flush=True)
    ranker = ml.fit_lambdarank(fit_groups, dev_groups, ml.RAW_FEATURES, category_maps)

    test_rows = [row for row in rows if dt.date.fromisoformat(row["date"]) in test_date_set]
    test_groups = ml.group_rows(test_rows)
    matched = []
    for group in test_groups:
        key = race_key_text(group[0]["race_key"])
        snapshot = snapshot_by_key.get(key)
        parts = key.split("|") if key else []
        race_key = nb.race_key(parts[0], parts[1], parts[2]) if len(parts) == 3 else None
        payout = payouts.get(race_key) if race_key else None
        if not snapshot or not payout or not payout.get("trio"):
            continue
        market_prob = market_win_probabilities(snapshot)
        horse_numbers = {str(row["horse_no"]) for row in group}
        if not horse_numbers.issubset(market_prob):
            continue
        market_ranks = percentile_scores(market_prob)
        model_raw = {
            str(row["horse_no"]): float(score)
            for row, score in zip(group, ranker.predict(feature_matrix(group, category_maps)))
        }
        model_ranks = percentile_scores(model_raw)
        matched.append({
            "date": group[0]["date"], "race_key": key, "group": group,
            "payout": payout, "model_rank_scores": model_ranks,
            "market_rank_scores": {horse: market_ranks[horse] for horse in horse_numbers},
            "snapshot_minutes_to_post": snapshot.get("minutes_to_post"),
            "snapshot_model_version": snapshot.get("model_version"),
        })
    matched.sort(key=lambda item: (item["date"], item["race_key"]))
    if len(matched) < 20:
        raise RuntimeError(f"Insufficient same-race pre-off market coverage for a useful five-day pilot: {len(matched)} races")

    dev_dates_for_fusion = {day.isoformat() for day in test_dates[:3]}
    holdout_dates = {day.isoformat() for day in test_dates[3:]}
    dev_races = [item for item in matched if item["date"] in dev_dates_for_fusion]
    holdout_races = [item for item in matched if item["date"] in holdout_dates]
    if len(dev_races) < 10 or len(holdout_races) < 10:
        raise RuntimeError(f"Insufficient fusion split: development={len(dev_races)} holdout={len(holdout_races)}")

    selected_weight, dev_results = select_market_weight(dev_races)
    holdout_results = {f"{weight:.2f}": evaluate_races(holdout_races, weight) for weight in BLEND_WEIGHTS}
    holdout_results["selected_from_development"] = evaluate_races(holdout_races, selected_weight)
    report = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "evaluated", "research_only": True,
        "test_window": {
            "dates": [day.isoformat() for day in test_dates],
            "development_dates_for_fusion": [day.isoformat() for day in test_dates[:3]],
            "untouched_holdout_dates": [day.isoformat() for day in test_dates[3:]],
            "matched_preoff_races": len(matched), "development_races": len(dev_races),
            "holdout_races": len(holdout_races),
            "snapshot_policy": "closest valid 2-30 minute pre-off snapshot; same races and runners for every blend",
        },
        "training": {
            "start": min(training_dates).isoformat(), "end": max(training_dates).isoformat(),
            "dates": len(training_dates), "races": len(ml.group_rows(training_rows)),
            "fit_races": len(fit_groups), "development_races": len(dev_groups),
            "model": "LightGBM LambdaRank", "features": list(ml.RAW_FEATURES),
            "relevance": "winner=3, second=2, third=1, other=0",
            "market_features_used_inside_ranker": False,
            "fusion_learning": "Select market percentile weight from {0,.25,.50,.75,1.00} using only the first three test dates; maximize trio 12-ticket race-hit rate, then podium capture; ties prefer lower market weight.",
        },
        "development_weight_selection": {"selected_market_weight": round(selected_weight, 2), "selection_results": dev_results},
        "untouched_holdout": holdout_results,
        "all_five_days_descriptive": {f"{weight:.2f}": evaluate_races(matched, weight) for weight in BLEND_WEIGHTS},
        "interpretation": {
            "market_weight": "0.00 = LambdaRank only; 1.00 = pre-off market only; 0.50 = equal blend.",
            "trio_method": "Rank all unordered three-runner sets by the sum of their three fused within-race runner scores; evaluate the top 12 at 100 JPY per ticket.",
            "limitations": [
                "Only the final two of five dates are untouched holdout; this is a pilot, not proof of profitability.",
                "The fusion weight is tuned on three dates and needs validation on a longer later holdout.",
                "Only timestamped pre-off win odds are used; confirmed final odds are excluded.",
                "No live-model.json, live-history.json, or production ticket logic is changed.",
                "The trio set score is an initial additive score; pairwise interaction and pace compatibility remain future work.",
            ],
        },
        "data_source": {"archive_months": downloads, "snapshot_records_considered": len(snapshot_by_key)},
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "dates": report["test_window"]["dates"],
        "matched_races": len(matched), "development_races": len(dev_races),
        "holdout_races": len(holdout_races), "selected_market_weight": selected_weight,
        "holdout_selected": holdout_results["selected_from_development"],
        "report": str(OUT),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
