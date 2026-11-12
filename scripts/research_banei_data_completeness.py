#!/usr/bin/env python3
"""Research-only completeness audit for Banei (Obihiro) monthly NAR archives.

This script does not train or modify any model. It measures whether the
historical archive contains the pre-race carried weight and the recorded
horse body-weight fields needed for a separate Banei experiment.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import nar_backtest as nb


OUT = nb.ROOT / "data" / "banei-data-completeness.json"


def is_banei(venue):
    value = nb.norm(venue)
    return "帯広" in value or "ばんえい" in value.lower()


def stats(values):
    clean = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not clean:
        return {"n": 0, "min": None, "p25": None, "median": None, "p75": None, "max": None, "mean": None}
    def percentile(p):
        pos = (len(clean) - 1) * p
        lo, hi = math.floor(pos), math.ceil(pos)
        if lo == hi:
            return round(clean[lo], 3)
        return round(clean[lo] + (clean[hi] - clean[lo]) * (pos - lo), 3)
    return {
        "n": len(clean), "min": round(clean[0], 3), "p25": percentile(0.25),
        "median": percentile(0.5), "p75": percentile(0.75),
        "max": round(clean[-1], 3), "mean": round(sum(clean) / len(clean), 3),
        "zero_count": sum(v == 0 for v in clean),
        "negative_count": sum(v < 0 for v in clean),
    }


def field_summary(rows, key):
    values = [nb.number(row.get(key)) for row in rows]
    valid = [v for v in values if v is not None]
    return {
        "present": len(valid),
        "missing": len(values) - len(valid),
        "coverage_pct": round(100 * len(valid) / len(values), 3) if values else None,
        "distribution": stats(valid),
    }


def main():
    all_races = 0
    all_runners = []
    race_rows = []
    month_rows = []
    downloaded_months = []
    for year, month in nb.month_iter(nb.START, nb.END):
        races, horses, _payouts, metadata = nb.load_parsed_month((year, month))
        banei_keys = {key for key, race in races.items() if is_banei(race.get("venue"))}
        month_runners = [runner for key in banei_keys for runner in horses.get(key, [])]
        month_race_data = [races[key] for key in banei_keys]
        if banei_keys:
            all_races += len(banei_keys)
            all_runners.extend(month_runners)
            race_rows.extend(month_race_data)
        month_rows.append({
            "month": f"{year}-{month:02d}",
            "banei_races": len(banei_keys),
            "banei_runners": len(month_runners),
            "runner_rows_with_finish": sum(nb.number(x.get("finish")) is not None for x in month_runners),
        })
        downloaded_months.append(metadata)
        print(
            f"BANEI_AUDIT_MONTH {year}-{month:02d} "
            f"races={len(banei_keys)} runners={len(month_runners)} "
            f"cache={metadata.get('parsed_cache')}",
            flush=True,
        )

    field_size = len(all_runners)
    field_audit = {
        "carried_weight": field_summary(all_runners, "carried"),
        "body_weight": field_summary(all_runners, "body_weight"),
        "body_weight_change": field_summary(all_runners, "body_weight_change"),
    }
    moisture_raw = Counter(str(r.get("going") or "").strip() for r in race_rows)
    moisture_numeric = [nb.number(r.get("going")) for r in race_rows]
    moisture_valid = [x for x in moisture_numeric if x is not None]
    outcome_counts = Counter(
        "top3" if (nb.number(r.get("finish")) or 999) <= 3
        else "4_6" if (nb.number(r.get("finish")) or 999) <= 6
        else "7_plus" if nb.number(r.get("finish")) is not None
        else "unclassified"
        for r in all_runners
    )
    ratio_values = []
    for runner in all_runners:
        carried = nb.number(runner.get("carried"))
        body = nb.number(runner.get("body_weight"))
        if carried is not None and body is not None and body > 0:
            ratio_values.append(100 * carried / body)

    yearly = defaultdict(lambda: {"races": 0, "runners": 0, "carried_present": 0,
                                  "body_weight_present": 0, "body_weight_change_present": 0})
    for row in month_rows:
        year = row["month"][:4]
        yearly[year]["races"] += row["banei_races"]
        yearly[year]["runners"] += row["banei_runners"]
    for runner in all_runners:
        # Runner rows are not directly date-stamped; yearly coverage is computed
        # from the month-grouped records below instead.
        pass
    for year_month in month_rows:
        year = year_month["month"][:4]
        year_runners = []
        # The exact monthly runner set is re-read from the parsed cache so
        # missingness can be reported by year without storing raw archives.
        y, m = int(year_month["month"][:4]), int(year_month["month"][5:7])
        _r, month_horses, _p, _meta = nb.load_parsed_month((y, m))
        for key, race in _r.items():
            if is_banei(race.get("venue")):
                year_runners.extend(month_horses.get(key, []))
        yearly[year]["carried_present"] += sum(nb.number(x.get("carried")) is not None for x in year_runners)
        yearly[year]["body_weight_present"] += sum(nb.number(x.get("body_weight")) is not None for x in year_runners)
        yearly[year]["body_weight_change_present"] += sum(nb.number(x.get("body_weight_change")) is not None for x in year_runners)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "research_only_data_completeness_audit",
        "period": {"start": nb.START.isoformat(), "end": nb.END.isoformat()},
        "source": "NAR official monthly race archives, parsed by existing scripts/nar_backtest.py",
        "scope": "Only races whose venue name contains 帯広 or ばんえい; no flat-racing rows included.",
        "counts": {
            "banei_races": all_races,
            "banei_runner_rows": field_size,
            "runner_rows_with_finish": sum(nb.number(x.get("finish")) is not None for x in all_runners),
        },
        "runner_field_completeness": field_audit,
        "carried_weight_to_body_weight_pct": stats(ratio_values),
        "race_going_or_moisture": {
            "nonempty_races": sum(bool(str(r.get("going") or "").strip()) for r in race_rows),
            "numeric_races": len(moisture_valid),
            "coverage_pct": round(100 * len(moisture_valid) / len(race_rows), 3) if race_rows else None,
            "numeric_distribution": stats(moisture_valid),
            "raw_value_counts": dict(moisture_raw.most_common(30)),
        },
        "finish_group_counts": dict(outcome_counts),
        "yearly_coverage": {
            year: {
                **values,
                "carried_weight_coverage_pct": round(100 * values["carried_present"] / values["runners"], 3) if values["runners"] else None,
                "body_weight_coverage_pct": round(100 * values["body_weight_present"] / values["runners"], 3) if values["runners"] else None,
                "body_weight_change_coverage_pct": round(100 * values["body_weight_change_present"] / values["runners"], 3) if values["runners"] else None,
            } for year, values in sorted(yearly.items())
        },
        "monthly_counts": month_rows,
        "downloaded_month_metadata": downloaded_months,
        "interpretation_guardrails": [
            "This is a source-data completeness audit, not evidence that any factor improves prediction.",
            "Final finish is used only to describe outcome availability; it is not used as a predictor.",
            "Carried weight, body weight, and body-weight change must be confirmed as available before the race before model use.",
            "Do not combine Banei rows with flat-racing training or evaluation.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("BANEI_AUDIT_RESULT " + json.dumps({
        "races": all_races, "runners": field_size,
        "carried_coverage_pct": field_audit["carried_weight"]["coverage_pct"],
        "body_weight_coverage_pct": field_audit["body_weight"]["coverage_pct"],
        "body_weight_change_coverage_pct": field_audit["body_weight_change"]["coverage_pct"],
        "going_numeric_coverage_pct": report["race_going_or_moisture"]["coverage_pct"],
        "report": str(OUT),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
