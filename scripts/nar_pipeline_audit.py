#!/usr/bin/env python3
"""Static safety audit for the NAR historical backtest pipeline.

This audit does not claim that A-D are implemented. It verifies the current
chronological split and reports any remaining comparison limitations.
"""
from __future__ import annotations
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts" / "nar_backtest.py"
OUT = ROOT / "data" / "nar-pipeline-audit.json"


def main() -> int:
    source = SOURCE.read_text(encoding="utf-8")
    checks = {
        "four_year_source_window": bool(re.search(r"END\s*-\s*dt\.timedelta\(days\s*=\s*1460\)", source)),
        "final_year_starts_364_days_before_end": bool(re.search(r"recent_start\s*=\s*END\s*-\s*dt\.timedelta\(days\s*=\s*364\)", source)),
        "final_fold_trains_only_before_test_start": bool(re.search(r"train_dates\s*=\[d\s+for\s+d\s+in\s+dates\s+if\s+d\s*<\s*test_start\]", source)),
        "final_fold_is_recent_year": '"recent_year"' in source,
        "timestamped_odds_window_2_to_30_minutes": "minutes < 2 or minutes > 30" in source,
        "no_final_odds_used_as_preoff": "discard post-off records" in source or "discard post-off" in source,
    }
    model_alias_warning = bool(re.search(
        r'"trained_with_pace"\s*:\s*learned.*?"trained_with_performance"\s*:\s*learned',
        source, re.S))
    checks["model_variants_need_genuine_ablation_review"] = not model_alias_warning

    limitations = []
    if not checks["four_year_source_window"]:
        limitations.append("Source window is not configured for approximately four years.")
    if not all(checks[k] for k in (
        "final_year_starts_364_days_before_end",
        "final_fold_trains_only_before_test_start",
        "final_fold_is_recent_year",
    )):
        limitations.append("Could not verify a final one-year test trained only on prior dates.")
    if not checks["model_variants_need_genuine_ablation_review"]:
        limitations.append("Several named variants share the same learned weights; they are not independent A-D models.")
    limitations.append("A-D ablation comparison is not yet implemented as four clearly isolated feature sets.")
    limitations.append("D can only be evaluated on races with genuine timestamped pre-off odds snapshots.")

    report = {
        "audit": "NAR historical pipeline static audit",
        "status": "checks_passed_with_known_limitations" if all(checks.values()) else "review_required",
        "checks": checks,
        "limitations": limitations,
        "note": "This is a source-level audit, not a historical performance result. No ROI or hit-rate claims are made.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0  # Known model-design limitations are reported, not hidden by a failed CI job.


if __name__ == "__main__":
    raise SystemExit(main())
