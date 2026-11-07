#!/usr/bin/env python3
"""Evaluate the fixed D4 challenger on the latest five completed NAR race dates.

All earlier official race history is used only to train/build prior form features.
The five-day evaluation window remains untouched until predictions are scored.
No production model weights or live predictions are modified.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import nar_backtest as nb


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "nar-recent-five-day-backtest.json"
CANDIDATE = "D4_jockey_venue_distance"
BASELINE = "C_performance_pace"


def compact_model(model: dict) -> dict:
    trio = (
        ((model.get("trio_diagnostics") or {}).get("formation_2_4_5_narrowed") or {})
        .get("12") or {}
    )
    bet = (model.get("bet_types") or {}).get("trio") or {}
    return {
        "eligible_races": model.get("races"),
        "winner_top1": model.get("winner_top1"),
        "winner_top1_rate_pct": model.get("winner_top1_rate"),
        "winner_top3": model.get("winner_top3"),
        "winner_top3_rate_pct": model.get("winner_top3_rate"),
        "trio_top5_fixed": {
            "tickets": bet.get("tickets"),
            "hits": bet.get("hits"),
            "hit_rate_pct": bet.get("hit_rate"),
            "stake_jpy": bet.get("stake"),
            "return_jpy": bet.get("return"),
            "roi_pct": bet.get("roi"),
            "profit_jpy": bet.get("profit"),
        },
        "trio_12_ticket_formation_2_4_5": {
            key: trio.get(key)
            for key in (
                "races", "tickets", "funded_tickets", "hits", "hit_races",
                "ticket_hit_rate", "race_hit_rate", "stake", "return",
                "roi", "profit", "budget_per_race_jpy",
            )
        },
    }


def compact_market(row: dict | None) -> dict | None:
    if not row:
        return None
    return {
        key: row.get(key)
        for key in (
            "races", "top1_hits", "top1_rate", "hits", "tickets",
            "stake", "return", "roi", "profit",
        )
    }


def main() -> None:
    end = nb.END
    start = end - dt.timedelta(days=4)
    print(f"RECENT_WINDOW start={start} end={end} (five completed calendar days)", flush=True)

    races, horses, payouts, months = nb.load_archives()
    all_dates = sorted({race["date"] for race in races.values()})
    test_dates = [d for d in all_dates if start <= d <= end]
    train_dates = [d for d in all_dates if d < start]
    if not test_dates:
        raise RuntimeError(f"No NAR races found in requested window {start} through {end}")
    if not train_dates:
        raise RuntimeError("No prior race history is available to train the model")

    test_date_set = set(test_dates)
    test_races = {key: race for key, race in races.items() if race["date"] in test_date_set}
    test_keys = set(test_races)
    test_fields = sum(1 for key in test_keys if len(horses.get(key, [])) >= 3)
    test_with_win_result = sum(
        1 for key in test_keys if (payouts.get(key) or {}).get("win")
    )
    print(
        f"RECENT_WINDOW_DATA dates={len(test_dates)} races={len(test_races)} "
        f"fields_ge3={test_fields} win_results={test_with_win_result}",
        flush=True,
    )

    # Only one monthly odds archive is needed for the November five-day window.
    # Final odds are used strictly as a retrospective market benchmark.
    final_odds, odds_downloads = nb.load_final_odds_archives(start, end)
    print(f"RECENT_WINDOW_FINAL_ODDS races={len(final_odds)}", flush=True)

    fold = nb.evaluate_holdout_fold(
        races, horses, payouts, train_dates, test_dates, final_odds
    )
    models = fold.get("models") or {}
    baseline = models.get(BASELINE) or {}
    d4 = models.get(CANDIDATE) or {}
    if not baseline or not d4:
        raise RuntimeError("The five-day evaluation did not produce both baseline and D4 model results")

    market = fold.get("final_odds_market_comparison") or {}
    baseline_top1 = baseline.get("winner_top1_rate")
    d4_top1 = d4.get("winner_top1_rate")
    baseline_top3 = baseline.get("winner_top3_rate")
    d4_top3 = d4.get("winner_top3_rate")
    baseline_trio = (
        ((baseline.get("trio_diagnostics") or {}).get("formation_2_4_5_narrowed") or {}).get("12") or {}
    )
    d4_trio = (
        ((d4.get("trio_diagnostics") or {}).get("formation_2_4_5_narrowed") or {}).get("12") or {}
    )

    report = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "evaluated",
        "evaluation_window": {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "calendar_days": 5,
            "race_dates_with_data": [d.isoformat() for d in test_dates],
            "races_in_archive": len(test_races),
            "races_with_three_or_more_runners": test_fields,
            "races_with_official_win_result": test_with_win_result,
        },
        "training_window": {
            "start": min(train_dates).isoformat(),
            "end": max(train_dates).isoformat(),
            "race_dates": len(train_dates),
            "races": len([r for r in races.values() if r["date"] in set(train_dates)]),
            "policy": "All training races strictly precede the five-day test window. Same-day results are not used to predict another race on that date.",
        },
        "data_source": {
            "official_archive_months": months,
            "final_odds_downloads": odds_downloads,
            "final_odds_races_available": len(final_odds),
            "final_odds_policy": "Official confirmed final odds are retrospective market benchmark only; they are not model input.",
        },
        "model": {
            "baseline": BASELINE,
            "candidate": CANDIDATE,
            "candidate_weight_adjustments": [
                {"feature": "jockey", "multiplier": 1.5},
                {"feature": "venue", "multiplier": 0.7},
                {"feature": "distance", "multiplier": 0.7},
            ],
            "trained_base_weights": fold.get("trained_weights"),
            "training_updates": fold.get("training_updates"),
            "temperature": fold.get("temperature"),
        },
        "results": {
            "baseline": compact_model(baseline),
            "D4": compact_model(d4),
            "delta_D4_minus_baseline_percentage_points": {
                "winner_top1_rate": round(d4_top1 - baseline_top1, 2)
                    if d4_top1 is not None and baseline_top1 is not None else None,
                "winner_top3_rate": round(d4_top3 - baseline_top3, 2)
                    if d4_top3 is not None and baseline_top3 is not None else None,
                "trio_12_ticket_roi": round(d4_trio["roi"] - baseline_trio["roi"], 2)
                    if d4_trio.get("roi") is not None and baseline_trio.get("roi") is not None else None,
                "trio_12_ticket_race_hit_rate": round(d4_trio["race_hit_rate"] - baseline_trio["race_hit_rate"], 2)
                    if d4_trio.get("race_hit_rate") is not None and baseline_trio.get("race_hit_rate") is not None else None,
            },
        },
        "market_benchmark_same_final_odds_coverage": {
            "baseline_model_top_pick": compact_market(market.get("C_model_top_pick")),
            "D4_model_top_pick": compact_market(market.get(CANDIDATE)),
            "market_favorite": compact_market(market.get("final_odds_favorite")),
        },
        "data_quality": {
            "missing_race_field_count": len(test_races) - test_fields,
            "missing_official_win_result_count": len(test_races) - test_with_win_result,
            "final_odds_coverage_races": len(final_odds),
            "test_dates_are_strictly_after_training": max(train_dates) < min(test_dates),
        },
        "limitations": [
            "Five calendar days are a short sample; results are diagnostic and cannot establish stable hit rate or profitability.",
            "The comparison uses official final odds as a retrospective market benchmark, not as a pre-off input.",
            "D4 is research-only; no live model weights, saved predictions, or production tickets are changed.",
        ],
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "report": str(OUT.relative_to(ROOT)),
        "window": report["evaluation_window"],
        "baseline": report["results"]["baseline"],
        "D4": report["results"]["D4"],
        "delta": report["results"]["delta_D4_minus_baseline_percentage_points"],
        "market": report["market_benchmark_same_final_odds_coverage"],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
