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
from zoneinfo import ZoneInfo

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
        "trio_pool_structures": {
            str(pool): {
                key: value.get(key)
                for key in ("races", "tickets", "hits", "hit_races", "stake", "return", "roi", "profit")
            }
            for pool, value in ((model.get("trio_diagnostics") or {}).get("structures") or {}).items()
        },
    }


def daily_model_metrics(model: dict, market_model: dict | None, market_favorite: dict | None) -> dict:
    trio = (
        ((model.get("trio_diagnostics") or {}).get("formation_2_4_5_narrowed") or {})
        .get("12") or {}
    )
    win = (model.get("bet_types") or {}).get("win") or {}
    races = int(model.get("races") or 0)
    market_races = int((market_model or {}).get("races") or 0)
    favorite_races = int((market_favorite or {}).get("races") or 0)
    model_hits = int((market_model or {}).get("top1_hits") or 0)
    favorite_hits = int((market_favorite or {}).get("top1_hits") or 0)
    return {
        "races": races,
        "winner_top1_hits": int(model.get("winner_top1") or 0),
        "winner_top1_rate_pct": model.get("winner_top1_rate"),
        "winner_top3_hits": int(model.get("winner_top3") or 0),
        "winner_top3_rate_pct": model.get("winner_top3_rate"),
        "win": {
            "stake_jpy": int(win.get("stake") or 0),
            "return_jpy": float(win.get("return") or 0),
            "roi_pct": win.get("roi"),
        },
        "trio_12": {
            "races": int(trio.get("races") or 0),
            "hit_races": int(trio.get("hit_races") or 0),
            "stake_jpy": int(trio.get("stake") or 0),
            "return_jpy": float(trio.get("return") or 0),
            "profit_jpy": float(trio.get("profit") or 0),
            "roi_pct": trio.get("roi"),
            "race_hit_rate_pct": trio.get("race_hit_rate"),
        },
        "market_comparison": {
            "same_races": min(market_races, favorite_races),
            "model_top1_hits": model_hits,
            "model_top1_rate_pct": round(100 * model_hits / market_races, 2) if market_races else None,
            "market_favorite_top1_hits": favorite_hits,
            "market_favorite_top1_rate_pct": round(100 * favorite_hits / favorite_races, 2) if favorite_races else None,
            "top1_gap_percentage_points": (
                round(100 * model_hits / market_races - 100 * favorite_hits / favorite_races, 2)
                if market_races and favorite_races and market_races == favorite_races else None
            ),
            "model_top1_roi_pct": (market_model or {}).get("roi"),
            "market_favorite_roi_pct": (market_favorite or {}).get("roi"),
        },
    }


def build_learning_curve(daily_rows: list[dict], model_names: tuple[str, ...]) -> list[dict]:
    totals = {
        name: {"races": 0, "winner_top1_hits": 0, "winner_top3_hits": 0,
               "trio_races": 0, "trio_hit_races": 0, "trio_stake_jpy": 0,
               "trio_return_jpy": 0.0, "market_top1_hits": 0, "market_races": 0}
        for name in model_names
    }
    curve = []
    for day in daily_rows:
        cumulative = {}
        for name in model_names:
            row = day["models"].get(name)
            if row is None:
                continue
            total = totals[name]
            total["races"] += row["races"]
            total["winner_top1_hits"] += row["winner_top1_hits"]
            total["winner_top3_hits"] += row["winner_top3_hits"]
            trio = row["trio_12"]
            total["trio_races"] += trio["races"]
            total["trio_hit_races"] += trio["hit_races"]
            total["trio_stake_jpy"] += trio["stake_jpy"]
            total["trio_return_jpy"] += trio["return_jpy"]
            market = row["market_comparison"]
            total["market_top1_hits"] += market["market_favorite_top1_hits"]
            total["market_races"] += market["same_races"]
            cumulative[name] = {
                "races": total["races"],
                "winner_top1_hits": total["winner_top1_hits"],
                "winner_top1_rate_pct": round(100 * total["winner_top1_hits"] / total["races"], 2) if total["races"] else None,
                "winner_top3_hits": total["winner_top3_hits"],
                "winner_top3_rate_pct": round(100 * total["winner_top3_hits"] / total["races"], 2) if total["races"] else None,
                "trio_12": {
                    "races": total["trio_races"],
                    "hit_races": total["trio_hit_races"],
                    "race_hit_rate_pct": round(100 * total["trio_hit_races"] / total["trio_races"], 2) if total["trio_races"] else None,
                    "stake_jpy": total["trio_stake_jpy"],
                    "return_jpy": round(total["trio_return_jpy"], 2),
                    "profit_jpy": round(total["trio_return_jpy"] - total["trio_stake_jpy"], 2),
                    "roi_pct": round(100 * total["trio_return_jpy"] / total["trio_stake_jpy"], 2) if total["trio_stake_jpy"] else None,
                },
                "market_favorite_top1_rate_pct": round(100 * total["market_top1_hits"] / total["market_races"], 2) if total["market_races"] else None,
                "market_top1_gap_percentage_points": (
                    round(100 * total["winner_top1_hits"] / total["races"] - 100 * total["market_top1_hits"] / total["market_races"], 2)
                    if total["races"] and total["market_races"] == total["races"] else None
                ),
            }
        curve.append({"date": day["date"], "daily": day["models"], "cumulative": cumulative})
    return curve


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
    # GitHub runners use UTC; NAR race dates and the requested window are Japan-local.
    # Use yesterday in Asia/Tokyo, not nb.END (which is based on the runner's UTC date).
    end = dt.datetime.now(ZoneInfo("Asia/Tokyo")).date() - dt.timedelta(days=1)
    start = end - dt.timedelta(days=4)
    # nar_backtest.py defaults START/END from the runner's UTC date. Override
    # both bounds with the Japan-local window before archive parsing; otherwise
    # the final JST race date can be silently excluded while the runner is still
    # on the previous UTC date.
    nb.END = end
    nb.START = end - dt.timedelta(days=1460)
    print(f"RECENT_WINDOW start={start} end={end} archive_start={nb.START} archive_end={nb.END} (Japan-local)", flush=True)

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

    # Walk-forward learning: refit once per Japan-local race date. Only results
    # strictly before the date being predicted are available to that day's model.
    learning_model_names = (BASELINE, CANDIDATE, "ablate_form")
    learning_days = []
    for day_index, race_date in enumerate(test_dates, 1):
        day_train_dates = [d for d in all_dates if d < race_date]
        day_fold = nb.evaluate_holdout_fold(
            races, horses, payouts, day_train_dates, [race_date], final_odds,
            model_names=learning_model_names,
        )
        day_models = day_fold.get("models") or {}
        day_market = day_fold.get("final_odds_market_comparison") or {}
        market_favorite = day_market.get("final_odds_favorite")
        day_summary = {
            name: daily_model_metrics(
                day_models[name],
                day_market.get("C_model_top_pick" if name == BASELINE else name),
                market_favorite,
            )
            for name in learning_model_names if name in day_models
        }
        learning_days.append({
            "date": race_date.isoformat(),
            "races": day_fold.get("test_races", 0),
            "training_races": day_fold.get("train_races", 0),
            "training_updates": day_fold.get("training_updates", 0),
            "trained_weights": day_fold.get("trained_weights"),
            "temperature": day_fold.get("temperature"),
            "models": day_summary,
        })
        print(
            f"LEARNING_CURVE day={day_index}/{len(test_dates)} date={race_date} "
            f"train_races={day_fold.get('train_races')} test_races={day_fold.get('test_races')} "
            + ", ".join(
                f"{name}:top1={day_summary[name].get('winner_top1_rate_pct')}%"
                for name in learning_model_names if name in day_summary
            ),
            flush=True,
        )
    learning_curve = build_learning_curve(learning_days, learning_model_names)
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

    research_names = (
        "D1_jockey_venue", "D2_jockey_distance", "D3_venue_distance", CANDIDATE,
        "weight_form_down_0_8", "weight_form_up_1_2",
        "weight_jockey_down_0_5", "weight_jockey_up_1_5",
        "weight_distance_down_0_7", "weight_distance_up_1_3",
        "weight_venue_down_0_7", "weight_venue_up_1_3",
        "ablate_form", "ablate_venue", "ablate_distance",
        "ablate_sire", "ablate_jockey", "ablate_age",
    )
    research_candidates = {
        name: compact_model(models[name])
        for name in research_names if name in models
    }

    # Independent adjacent five-day holdout: compare the immediately preceding
    # five calendar days using only history strictly earlier than that window.
    prior_end = start - dt.timedelta(days=1)
    prior_start = prior_end - dt.timedelta(days=4)
    prior_dates = [d for d in all_dates if prior_start <= d <= prior_end]
    prior_train_dates = [d for d in all_dates if d < prior_start]
    prior_odds, prior_odds_downloads = nb.load_final_odds_archives(prior_start, prior_end)
    prior_fold = nb.evaluate_holdout_fold(
        races, horses, payouts, prior_train_dates, prior_dates, prior_odds
    ) if prior_dates and prior_train_dates else {}
    prior_models = prior_fold.get("models") or {}
    prior_candidate_names = (
        "C_performance_pace", "D2_jockey_distance", "D4_jockey_venue_distance",
        "weight_jockey_up_1_5", "ablate_jockey", "ablate_distance", "ablate_form",
    )
    prior_comparison = {
        name: compact_model(prior_models[name])
        for name in prior_candidate_names if name in prior_models
    }

    report = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "evaluated",
        "diagnostic_version": 2,
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
        "research_candidate_comparison": research_candidates,
        "sequential_learning_evaluation": {
            "status": "evaluated",
            "policy": "Walk-forward by Japan-local race date. Each day's model is refit using only official race results strictly before that date. Prior test-day results are added to the next day's training set; same-day results are not used until the next date.",
            "model_names": list(learning_model_names),
            "daily_and_cumulative": learning_curve,
            "training_races_by_day": [
                {"date": row["date"], "training_races": row["training_races"],
                 "training_updates": row["training_updates"]}
                for row in learning_days
            ],
        },
        "independent_adjacent_window_comparison": {
            "evaluation_window": {
                "start": prior_start.isoformat(),
                "end": prior_end.isoformat(),
                "race_dates_with_data": [d.isoformat() for d in prior_dates],
                "races_in_archive": sum(1 for r in races.values() if r["date"] in set(prior_dates)),
                "races_with_official_win_result": sum(
                    1 for key, race in races.items()
                    if race["date"] in set(prior_dates) and (payouts.get(key) or {}).get("win")
                ),
            },
            "training_window": {
                "start": min(prior_train_dates).isoformat() if prior_train_dates else None,
                "end": max(prior_train_dates).isoformat() if prior_train_dates else None,
                "strictly_before_test": bool(prior_train_dates and prior_dates and max(prior_train_dates) < min(prior_dates)),
            },
            "final_odds_races": len(prior_odds),
            "final_odds_downloads": prior_odds_downloads,
            "candidate_comparison": prior_comparison,
            "market_benchmark_same_final_odds_coverage": {
                "baseline_model_top_pick": compact_market(
                    (prior_fold.get("final_odds_market_comparison") or {}).get("C_model_top_pick")
                ),
                "D4_model_top_pick": compact_market(
                    (prior_fold.get("final_odds_market_comparison") or {}).get(CANDIDATE)
                ),
                "market_favorite": compact_market(
                    (prior_fold.get("final_odds_market_comparison") or {}).get("final_odds_favorite")
                ),
            },
            "market_rank_diagnostics": prior_fold.get("market_rank_diagnostics"),
        },
        "diagnostics": {
            "market_rank_diagnostics": fold.get("market_rank_diagnostics"),
            "condition_model_segments": fold.get("condition_model_segments"),
            "condition_model_comparisons": fold.get("condition_model_comparisons"),
            "condition_finish_position_segments": fold.get("condition_finish_position_segments"),
            "integrated_strategy_E": fold.get("integrated_strategy_E"),
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
        "sequential_learning_days": len(report["sequential_learning_evaluation"]["daily_and_cumulative"]),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
