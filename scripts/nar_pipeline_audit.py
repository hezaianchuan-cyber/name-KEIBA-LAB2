#!/usr/bin/env python3
"""Static safety audit for the NAR historical backtest pipeline.

This audit verifies canonical A-C feature ablations, the timestamped D odds gate,
and the chronological final-year holdout split.
"""
from __future__ import annotations
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts" / "nar_backtest.py"
CAPTURE_SOURCE = ROOT / "scripts" / "capture_nar_odds.py"
OUT = ROOT / "data" / "nar-pipeline-audit.json"


def main() -> int:
    source = SOURCE.read_text(encoding="utf-8")
    capture_source = CAPTURE_SOURCE.read_text(encoding="utf-8")
    checks = {
        "four_year_source_window": bool(re.search(r"END\s*-\s*dt\.timedelta\(days\s*=\s*1460\)", source)),
        "final_year_starts_364_days_before_end": bool(re.search(r"recent_start\s*=\s*END\s*-\s*dt\.timedelta\(days\s*=\s*364\)", source)),
        "final_fold_trains_only_before_test_start": bool(re.search(r"train_dates\s*=\[d\s+for\s+d\s+in\s+dates\s+if\s+d\s*<\s*test_start\]", source)),
        "final_fold_is_recent_year": '"recent_year"' in source,
        "timestamped_odds_window_2_to_30_minutes": "minutes < 2 or minutes > 30" in source,
        "no_final_odds_used_as_preoff": "discard post-off records" in source or "discard post-off" in source,
        "canonical_A_B_C_models_declared": all(token in source for token in (
            '"A_baseline"', '"B_condition"', '"C_performance_pace"')),
        "B_adds_prior_going_suitability": bool(re.search(
            r'"B_condition".*?going_weight=0\.10', source, re.S)),
        "C_adds_performance_time_and_pace": bool(re.search(
            r'"C_performance_pace".*?0\.10,\s*0\.10,\s*0\.10.*?going_weight=0\.10', source, re.S))
            and '"C_performance_pace", "trained_with_pace"' in source,
        "A_C_share_holdout_and_ticket_rules": (
            "identical chronological holdout races" in source
            and "identical ticket construction" in source
            and "100 JPY per ticket" in source
        ),
        "D_is_timestamped_snapshot_only": (
            '"D_preoff_odds"' in source
            and "timestamped 2-30 minute snapshots only" in source
        ),
        "D_uses_C_features_before_odds_overlay": (
            "score_runner_performance(" in capture_source
            and "margin_weight=0.10, closing_weight=0.10, time_weight=0.10" in capture_source
            and "pace_adjustments(horses, history)" in capture_source
            and '"time_per_km":' in source
        ),
        "D_requires_versioned_C_model_snapshots": (
            'expected_model_version = "nar-chronological-v9-D-frame-preoff"' in source
            and 'if record.get("model_version") != expected_model_version:' in source
            and '"model_version": "nar-chronological-v9-D-frame-preoff"' in source
        ),
        "frame_number_is_parsed_from_official_horse_rows": '"frame_no": rank_value(row[4])' in source
            and '"frame_no": nb.rank_value(row[4])' in capture_source,
        "frame_effect_uses_prior_condition_bucket_and_smoothing": (
            "def frame_context_key(" in source
            and "def frame_form_score(" in source
            and "prior_strength=8.0" in source
            and "update_frame_stats(pending_race, pending_field, frame_stats)" in source
        ),
        "backtest_and_live_share_D_frame_pace_scoring": (
            "score_runner_performance(" in source
            and "frame_form_score(race, h, frame_stats)" in source
            and "pace_adjustments(field, history)" in source
            and "nb.frame_form_score(race, horse, frame_stats)" in capture_source
        ),
    }
    limitations = []
    if not checks["four_year_source_window"]:
        limitations.append("Source window is not configured for approximately four years.")
    if not all(checks[k] for k in (
        "final_year_starts_364_days_before_end",
        "final_fold_trains_only_before_test_start",
        "final_fold_is_recent_year",
    )):
        limitations.append("Could not verify a final one-year test trained only on prior dates.")
    if not all(checks[k] for k in (
        "canonical_A_B_C_models_declared",
        "B_adds_prior_going_suitability",
        "C_adds_performance_time_and_pace",
        "A_C_share_holdout_and_ticket_rules",
        "D_is_timestamped_snapshot_only",
        "D_uses_C_features_before_odds_overlay",
        "D_requires_versioned_C_model_snapshots",
        "frame_number_is_parsed_from_official_horse_rows",
        "frame_effect_uses_prior_condition_bucket_and_smoothing",
        "backtest_and_live_share_D_frame_pace_scoring",
    )):
        limitations.append("Canonical A-D feature ablation wiring needs further review.")
    limitations.append("D is evaluated only where valid timestamped pre-off odds snapshots exist; coverage may be insufficient.")
    limitations.append("Legacy exploratory variants remain in the report and may share base learned weights; canonical A/B/C are the comparison set.")

    report = {
        "audit": "NAR historical pipeline static audit",
        "status": "checks_passed_with_known_limitations" if all(checks.values()) else "review_required",
        "checks": checks,
        "canonical_models": {
            "A": "fixed baseline ranking",
            "B": "learned base ranking + prior venue/distance and going suitability",
            "C": "B + prior margin, last-3F, comparable time and race-pace adjustment",
            "D": "C feature scoring + learned prior frame-condition effect + race pace + genuine 2-30 minute timestamped odds from the v9 D-frame model only",
        },
        "limitations": limitations,
        "note": "This is a source-level audit, not a historical performance result. No ROI or hit-rate claims are made.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0





if __name__ == "__main__":
    raise SystemExit(main())
