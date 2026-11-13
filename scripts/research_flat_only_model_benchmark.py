#!/usr/bin/env python3
"""Research-only, flat-racing benchmark wrapper for the existing NAR ML suite.

This wrapper:
- excludes Banei/Obihiro races before any feature construction or model fitting;
- removes current-race body-weight fields from ML inputs because the archive
  contains result-time values and pre-off availability is not established;
- preserves historical body-weight summaries built only from prior starts;
- runs every model on one common chronological split and writes a separate report.

No production model, ticket generator, or canonical report is modified.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import nar_ml_challenger_backtest as suite  # noqa: E402

OUT = ROOT / "data" / "flat-only-model-benchmark.json"
EXCLUDED_BODY_WEIGHT_FEATURES = {
    "body_weight",
    "body_weight_change",
    "body_weight_delta_pct",
    "race_rel_body_weight_delta_pct",
}
BANEI_PATTERN = re.compile(r"帯広|ばんえい", re.IGNORECASE)


def is_banei_race(race: dict) -> bool:
    venue = str(race.get("venue") or "")
    name = str(race.get("name") or "")
    return bool(BANEI_PATTERN.search(venue) or BANEI_PATTERN.search(name))


def main() -> None:
    original_loader = suite.nb.load_archives

    def load_flat_archives():
        races, horses, payouts, downloads = original_loader()
        excluded = {
            key for key, race in races.items()
            if is_banei_race(race)
        }
        flat_races = {key: race for key, race in races.items() if key not in excluded}
        flat_horses = {key: rows for key, rows in horses.items() if key in flat_races}
        flat_payouts = {key: value for key, value in payouts.items() if key in flat_races}
        venues = sorted({str(r.get("venue") or "unknown") for r in races.values()})
        banei_venues = sorted({
            str(r.get("venue") or "unknown") for r in races.values()
            if is_banei_race(r)
        })
        audit = {
            "source_races": len(races),
            "excluded_banei_races": len(excluded),
            "flat_races": len(flat_races),
            "source_venues": venues,
            "excluded_banei_venues": banei_venues,
            "flat_dates": len({r["date"] for r in flat_races.values()}),
            "flat_horse_race_groups": len(flat_horses),
            "flat_payout_records": len(flat_payouts),
            "filter": "Exclude venue or race name containing 帯広 or ばんえい before feature construction.",
        }
        if not flat_races:
            raise RuntimeError("Flat-only filter produced zero races.")
        if any(is_banei_race(r) for r in flat_races.values()):
            raise RuntimeError("Banei leakage detected after flat-only filtering.")
        print("FLAT_SCOPE_AUDIT " + json.dumps(audit, ensure_ascii=False), flush=True)
        return flat_races, flat_horses, flat_payouts, downloads

    # The underlying ranking suite uses module-level feature lists. Drop any
    # current-race weight fields whose pre-off capture cannot be established.
    suite.RAW_NUMERIC = [
        feature for feature in suite.RAW_NUMERIC
        if feature not in EXCLUDED_BODY_WEIGHT_FEATURES
    ]
    suite.RAW_FEATURES = suite.RAW_NUMERIC + suite.CATEGORICAL
    suite.nb.load_archives = load_flat_archives
    suite.OUT = OUT

    try:
        suite.main()
    finally:
        suite.nb.load_archives = original_loader

    report = json.loads(OUT.read_text(encoding="utf-8"))
    report["benchmark_scope"] = {
        "scope": "NAR flat racing only; Banei/Obihiro excluded before feature generation.",
        "excluded_current_race_features": sorted(EXCLUDED_BODY_WEIGHT_FEATURES),
        "retained_historical_feature": "prior_body_weight_avg (computed only from prior starts)",
        "raw_feature_set": suite.RAW_FEATURES,
        "lab_feature_set": suite.LAB_FEATURES,
        "market_odds_policy": "Official confirmed final odds are retrospective-only and never model inputs.",
        "production_changed": False,
        "canonical_ml_report_changed": False,
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report.get("status"),
        "output": str(OUT),
        "data": report.get("data"),
        "split": report.get("split"),
        "benchmark_scope": report["benchmark_scope"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
