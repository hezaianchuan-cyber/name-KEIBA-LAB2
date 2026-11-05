#!/usr/bin/env python3
"""Build a daily KEIBA LAB research review from published, reproducible reports.

This report intentionally does not read private per-user prediction rows. It
summarizes the public prospective pre-off evaluation (G4), official payout
archive freshness, and the latest historical/calibration research reports.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
JST = ZoneInfo("Asia/Tokyo")


def read_report(name: str):
    path = DATA / name
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"_read_error": str(exc), "_file": name}


def age_hours(value, now):
    if not value:
        return None
    try:
        stamp = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=dt.timezone.utc)
        return round(max(0.0, (now - stamp.astimezone(dt.timezone.utc)).total_seconds() / 3600), 2)
    except (TypeError, ValueError):
        return None


def compact_metric(metric):
    if not isinstance(metric, dict):
        return None
    return {
        "races": metric.get("races_bet", metric.get("races")),
        "hits": metric.get("hits"),
        "hit_rate_pct": metric.get("hit_rate_pct"),
        "roi_pct": metric.get("roi_pct"),
    }


def build_review(now=None):
    now = now or dt.datetime.now(dt.timezone.utc)
    local_now = now.astimezone(JST)
    payouts = read_report("daily-payouts.json")
    g1 = read_report("model-g1-calibration-backtest.json")
    g2 = read_report("model-g2-odds-divergence-backtest.json")
    g3 = read_report("model-g3-risk-gated-backtest.json")
    g4 = read_report("model-g4-preoff-backtest.json")
    historical = read_report("backtest-summary.json")
    weights = read_report("backtest-5-feature-ablation.json")

    daily_archive = (payouts or {}).get("daily_archive") or {}
    closest = (g4 or {}).get("closest_to_post") or {}
    holdout = closest.get("holdout") or {}
    diagnostics = (g4 or {}).get("diagnostics") or {}
    calibrated = ((g1 or {}).get("holdout_metrics") or {}).get("G1_calibrated") or {}
    weight_exp = (weights or {}).get("feature_weight_experiment") or {}
    weight_holdout = weight_exp.get("holdout_comparison") or {}

    warnings = []
    if not payouts or payouts.get("_read_error"):
        warnings.append("official_payout_report_missing_or_invalid")
    elif daily_archive.get("available") is not True:
        warnings.append("official_daily_payout_archive_unavailable")
    if not g4 or g4.get("_read_error"):
        warnings.append("G4_report_missing_or_invalid")
    elif (g4.get("status") or "").startswith("insufficient"):
        warnings.append("G4_prospective_sample_still_small")
    if not g1 or g1.get("_read_error"):
        warnings.append("G1_calibration_report_missing_or_invalid")
    if not historical or historical.get("_read_error"):
        warnings.append("historical_backtest_report_missing_or_invalid")

    review = {
        "version": "nar-daily-review-v1",
        "generated_at": local_now.isoformat(),
        "as_of_date_jst": local_now.date().isoformat(),
        "status": "needs_attention" if any(
            item in warnings for item in (
                "official_payout_report_missing_or_invalid",
                "official_daily_payout_archive_unavailable",
                "G4_report_missing_or_invalid",
                "G1_calibration_report_missing_or_invalid",
                "historical_backtest_report_missing_or_invalid",
            )
        ) else ("collecting_evidence" if warnings else "ok"),
        "daily_official_results": {
            "report_generated_at": (payouts or {}).get("generated_at"),
            "report_age_hours": age_hours((payouts or {}).get("generated_at"), now),
            "daily_archive_available": daily_archive.get("available"),
            "today_payout_races": daily_archive.get("today_payout_races"),
            "monthly_archive_available": ((payouts or {}).get("monthly_archive") or {}).get("available"),
        },
        "prospective_preoff_evaluation_G4": {
            "generated_at": (g4 or {}).get("generated_at"),
            "age_hours": age_hours((g4 or {}).get("generated_at"), now),
            "status": (g4 or {}).get("status"),
            "evaluation_period": (g4 or {}).get("evaluation_period"),
            "valid_snapshots": diagnostics.get("valid_snapshots"),
            "snapshots_with_official_result": diagnostics.get("snapshots_with_official_result"),
            "snapshots_missing_official_result": diagnostics.get("snapshots_missing_official_result"),
            "supabase_snapshots_joined": diagnostics.get("supabase_snapshots_joined"),
            "closest_to_post": {
                "unique_races": closest.get("unique_races"),
                "development_races": closest.get("development_races"),
                "holdout_races": closest.get("holdout_races"),
                "holdout_model_top": compact_metric(holdout.get("model_top")),
                "holdout_market_favorite": compact_metric(holdout.get("market_favorite")),
                "holdout_value_rule": compact_metric(holdout.get("value_rule")),
            },
            "live_model_changed": (g4 or {}).get("live_model_changed"),
            "profitable_claim": (g4 or {}).get("profitable_claim"),
        },
        "calibration_G1": {
            "generated_at": (g1 or {}).get("generated_at"),
            "status": (g1 or {}).get("status"),
            "selected_temperature": (g1 or {}).get("selected_temperature"),
            "holdout_races": (g1 or {}).get("holdout_races"),
            "baseline": ((g1 or {}).get("holdout_metrics") or {}).get("baseline"),
            "calibrated": calibrated,
            "e_model_changed": (g1 or {}).get("e_model_changed"),
            "profitable_claim": (g1 or {}).get("profitable_claim"),
        },
        "strategy_research": {
            "G2_status": (g2 or {}).get("status"),
            "G2_holdout_races": (g2 or {}).get("holdout_races"),
            "G2_selected_value_tickets": (g2 or {}).get("races_with_selected_value_ticket"),
            "G3_status": (g3 or {}).get("status"),
            "G3_holdout_races": (g3 or {}).get("holdout_races"),
            "G3_selected_value_tickets": (g3 or {}).get("races_with_selected_value_ticket"),
        },
        "weekly_historical_backtest": {
            "generated_at": (historical or {}).get("generated_at"),
            "age_hours": age_hours((historical or {}).get("generated_at"), now),
            "period": (historical or {}).get("period"),
            "race_count": (historical or {}).get("race_count"),
        },
        "weight_research": {
            "generated_at": (weights or {}).get("generated_at"),
            "selected_on_prior_folds": weight_exp.get("selected_on_prior_folds"),
            "promotion_ready": weight_holdout.get("promotion_ready"),
            "holdout_comparison": weight_holdout,
            "live_weights_changed": False,
        },
        "warnings": warnings,
        "interpretation": [
            "G4 is a prospective captured-pre-off-odds evaluation; its sample is rolling and may be too small for a daily conclusion.",
            "G1/G2/G3 and the historical feature-weight research are scheduled weekly; daily review reports their latest published versions.",
            "No live model or production feature weights are changed by this daily review.",
            "Private per-user predictions in Supabase are excluded because their row-level security must not be bypassed by a public GitHub workflow.",
            "No profitability claim is made from small or incomplete samples.",
        ],
    }
    return review


def main():
    review = build_review()
    DATA.mkdir(parents=True, exist_ok=True)
    output = DATA / "daily-review.json"
    output.write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": review["status"],
        "as_of_date_jst": review["as_of_date_jst"],
        "warnings": review["warnings"],
        "g4_status": review["prospective_preoff_evaluation_G4"]["status"],
        "today_payout_races": review["daily_official_results"]["today_payout_races"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
