#!/usr/bin/env python3
"""Flat-only audit of the original six-feature NAR learner.

Research only: no production model, predictions, tickets, or database are changed.
The audit filters Banei before training, fits on the first three years, and evaluates
feature ablations plus same-learner daily online updates on the final year.
"""
from __future__ import annotations

import datetime as dt
import json
from collections import defaultdict
from pathlib import Path

import nar_backtest as nb

OUT = Path(__file__).resolve().parents[1] / "data" / "research-flat-model-feature-audit.json"
FEATURES = ("form", "venue", "distance", "sire", "jockey", "age")
MODEL_NAMES = (
    "C_performance_pace",
    "ablate_form",
    "ablate_venue",
    "ablate_distance",
    "ablate_sire",
    "ablate_jockey",
    "ablate_age",
    "D_sequential_daily",
)


def is_banei(race: dict) -> bool:
    venue = str(race.get("venue") or "")
    return any(token in venue for token in ("帯広", "ばんえい", "帯広ば"))


def summarize_weights(trace: list[dict]) -> dict:
    if not trace:
        return {"days": 0, "features": {}}
    summary = {"days": len(trace), "first_date": trace[0]["date"],
               "last_date": trace[-1]["date"], "features": {}}
    for feature in FEATURES:
        values = [float(row["weights"][feature]) for row in trace]
        summary["features"][feature] = {
            "start": round(values[0], 6),
            "end": round(values[-1], 6),
            "mean": round(sum(values) / len(values), 6),
            "min": round(min(values), 6),
            "max": round(max(values), 6),
        }
    yearly = {}
    for year in sorted({row["date"][:4] for row in trace}):
        rows = [row for row in trace if row["date"].startswith(year)]
        yearly[year] = {
            "days": len(rows),
            "start": rows[0]["weights"],
            "end": rows[-1]["weights"],
            "mean": {
                feature: round(sum(float(row["weights"][feature]) for row in rows) / len(rows), 6)
                for feature in FEATURES
            },
        }
    summary["by_year"] = yearly
    return summary


def main() -> None:
    print("FLAT_FEATURE_AUDIT start: loading official NAR archives", flush=True)
    races, horses, payouts, months = nb.load_archives()
    all_count = len(races)
    flat_races = {key: race for key, race in races.items() if not is_banei(race)}
    excluded_count = all_count - len(flat_races)
    if not flat_races:
        raise RuntimeError("平地レースが0件です。ばんえい除外条件を確認してください")

    flat_keys = set(flat_races)
    flat_horses = {key: value for key, value in horses.items() if key in flat_keys}
    flat_payouts = {key: value for key, value in payouts.items() if key in flat_keys}
    dates = sorted({race["date"] for race in flat_races.values()})
    cutoff = nb.END - dt.timedelta(days=365)
    train_dates = [date for date in dates if date <= cutoff]
    test_dates = [date for date in dates if date > cutoff]
    if not train_dates or not test_dates:
        raise RuntimeError("時系列の学習期間または評価期間が空です")

    venues = sorted({str(race.get("venue") or "") for race in flat_races.values()})
    print(
        "FLAT_FEATURE_AUDIT split "
        + json.dumps({
            "all_races": all_count,
            "excluded_banei_races": excluded_count,
            "flat_races": len(flat_races),
            "venues": venues,
            "train_start": train_dates[0].isoformat(),
            "train_end": train_dates[-1].isoformat(),
            "train_races": sum(race["date"] in set(train_dates) for race in flat_races.values()),
            "test_start": test_dates[0].isoformat(),
            "test_end": test_dates[-1].isoformat(),
            "test_races": sum(race["date"] in set(test_dates) for race in flat_races.values()),
        }, ensure_ascii=False),
        flush=True,
    )

    # Capture the original learner's daily training trajectory inside the
    # same training pass used by evaluate_holdout_fold (no duplicate fit).
    training_trace: list[dict] = []
    original_run_backtest = nb.run_backtest

    def traced_run_backtest(*args, **kwargs):
        if kwargs.get("training_only"):
            kwargs["learning_trace"] = training_trace
        return original_run_backtest(*args, **kwargs)

    nb.run_backtest = traced_run_backtest
    try:
        fold = nb.evaluate_holdout_fold(
            flat_races, flat_horses, flat_payouts,
            train_dates, test_dates, model_names=list(MODEL_NAMES),
        )
    finally:
        nb.run_backtest = original_run_backtest

    model_rows = {}
    baseline = (fold.get("models") or {}).get("C_performance_pace") or {}
    for name in MODEL_NAMES:
        metrics = (fold.get("models") or {}).get(name) or {}
        trio = (metrics.get("bet_types") or {}).get("trio") or {}
        row = {
            "races": metrics.get("races"),
            "winner_top1_rate": metrics.get("winner_top1_rate"),
            "winner_top3_rate": metrics.get("winner_top3_rate"),
            "winner_topk_rate": metrics.get("winner_topk_rate"),
            "trio_tickets": trio.get("tickets"),
            "trio_race_count": trio.get("race_count"),
            "trio_hit_rate": trio.get("hit_rate"),
            "trio_roi": trio.get("roi"),
            "trio_profit_yen": trio.get("profit"),
            "trio_stake_yen": trio.get("stake"),
            "trio_return_yen": trio.get("return"),
        }
        if name != "C_performance_pace":
            for metric in ("winner_top1_rate", "winner_top3_rate"):
                if row[metric] is not None and baseline.get(metric) is not None:
                    row[metric + "_delta_pp"] = round(row[metric] - baseline[metric], 4)
            if row["trio_roi"] is not None:
                baseline_trio = (baseline.get("bet_types") or {}).get("trio") or {}
                if baseline_trio.get("roi") is not None:
                    row["trio_roi_delta_pp"] = round(row["trio_roi"] - baseline_trio["roi"], 4)
        model_rows[name] = row

    trace_summary = summarize_weights(training_trace)
    sequential = fold.get("sequential_daily_learning") or {}
    report = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "research_only",
        "scope": "flat_racing_only",
        "source_months": months,
        "data_scope": {
            "source_races_before_filter": all_count,
            "excluded_banei_races": excluded_count,
            "flat_races": len(flat_races),
            "included_venues": venues,
            "banei_venue_present_after_filter": any(is_banei(race) for race in flat_races.values()),
            "train_period": {"start": train_dates[0].isoformat(), "end": train_dates[-1].isoformat()},
            "test_period": {"start": test_dates[0].isoformat(), "end": test_dates[-1].isoformat()},
            "train_races": fold.get("train_races"),
            "test_races": fold.get("test_races"),
            "payout_races_in_test": sum(
                1 for key, race in flat_races.items()
                if race["date"] in set(test_dates) and (flat_payouts.get(key) or {}).get("win")
            ),
        },
        "feature_schema": list(FEATURES),
        "trained_weights": fold.get("trained_weights"),
        "training_weight_trajectory": trace_summary,
        "holdout_sequential_learning": {
            "status": sequential.get("status"),
            "dates": sequential.get("dates"),
            "updates": sequential.get("updates"),
            "first_date": sequential.get("first_date"),
            "last_date": sequential.get("last_date"),
            "final_weights": sequential.get("final_weights"),
            "checkpoints": sequential.get("checkpoints"),
        },
        "models": model_rows,
        "method": {
            "feature_ablation": "Remove one of the six original base features at a time and renormalize the remaining learned weights. The same race set, scoring overlays, and ticket construction are retained.",
            "sequential_comparison": "D_sequential_daily starts from the same fold-trained weights, predicts every race on a Japan-local date before updating on that date's results, and is compared with fixed C_performance_pace.",
            "leakage_controls": "Banei is removed before any training or evaluation. Each prediction uses strictly earlier race history; all races on a date are scored before that date's outcomes are committed. Current-race results and payouts are evaluation targets only.",
            "roi_interpretation": "ROI is a retrospective fixed-ticket research metric, not expected future profit. The six-feature ablations are sensitivity tests, not causal feature importance or automatic deletion decisions.",
            "production_impact": "No production weights, saved predictions, ticket rules, or database records are changed.",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("FLAT_FEATURE_AUDIT_RESULT " + json.dumps({
        "report": str(OUT),
        "data_scope": report["data_scope"],
        "trained_weights": report["trained_weights"],
        "weight_summary": trace_summary,
        "models": model_rows,
        "sequential_summary": {key: sequential.get(key) for key in
                               ("status", "dates", "updates", "first_date", "last_date", "final_weights")},
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
