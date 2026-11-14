#!/usr/bin/env python3
"""Research-only recent-form subgroup ablation, selected on development data only."""
from __future__ import annotations
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import research_flat_lambdarank_feature_ablation as base
import nar_ml_challenger_backtest as suite

OUT = ROOT / "data" / "flat-lambdarank-recent-form-subgroups.json"

# These seven groups are mutually exclusive and cover all 17 existing recent-form fields.
SUBGROUPS = {
    "career_experience": [
        "prior_runs", "career_win_rate", "career_top3_rate",
    ],
    "recent_finish_consistency": [
        "recent_finish_pct", "recent_top3_rate",
    ],
    "margin_quality": [
        "recent_margin_quality",
    ],
    "closing_kick": [
        "recent_last3f_rank", "race_rel_recent_last3f_rank",
    ],
    "in_race_position": [
        "recent_first_corner_pct", "recent_last_corner_pct",
        "recent_position_gain", "race_rel_recent_last_corner_pct",
    ],
    "time_and_trend": [
        "recent_time_per_km", "recent_time_trend",
        "race_rel_recent_time_per_km",
    ],
    "relative_finish": [
        "race_rel_recent_finish_pct", "race_rel_recent_top3_rate",
    ],
}

def main():
    suite.RAW_NUMERIC = [f for f in suite.RAW_NUMERIC if f not in base.EXCLUDED_BODY_WEIGHT]
    suite.RAW_FEATURES = suite.RAW_NUMERIC + suite.CATEGORICAL
    races, horses, scope = base.load_flat()
    rows, dates = suite.build_rows(races, horses)
    if len(dates) < 100:
        raise RuntimeError("Insufficient chronological dates")

    cut1, cut2 = int(len(dates) * .70), int(len(dates) * .80)
    train_dates, dev_dates = set(dates[:cut1]), set(dates[cut1:cut2])
    def rows_for(date_set):
        return [r for r in rows if dt.date.fromisoformat(r["date"]) in date_set]

    train_rows = rows_for(train_dates)
    dev_rows = rows_for(dev_dates)
    # Split the training period chronologically for early stopping. The development
    # period is used only to compare predeclared subgroup variants, never for fitting.
    train_cut = int(len(sorted(train_dates)) * .90)
    train_date_list = sorted(train_dates)
    fit_dates, early_dates = set(train_date_list[:train_cut]), set(train_date_list[train_cut:])
    fit_groups = suite.group_rows(rows_for(fit_dates))
    early_groups = suite.group_rows(rows_for(early_dates))
    dev_groups = suite.group_rows(dev_rows)
    if not fit_groups or not early_groups or not dev_groups:
        raise RuntimeError("Empty chronological fit/early-stop/development split")

    category_maps = {
        k: {v: i for i, v in enumerate(sorted({r[k] for r in rows_for(fit_dates)}))}
        for k in suite.CATEGORICAL
    }
    features = list(suite.RAW_FEATURES)
    original_recent = set(base.GROUPS["recent_form"])
    declared = [f for group in SUBGROUPS.values() for f in group]
    if len(declared) != len(set(declared)) or set(declared) != original_recent:
        raise RuntimeError(
            "Subgroups must be disjoint and cover all 17 recent-form features exactly; "
            f"missing={sorted(original_recent-set(declared))}, "
            f"extra={sorted(set(declared)-original_recent)}"
        )
    for group_name, members in SUBGROUPS.items():
        unknown = set(members) - set(features)
        if unknown:
            raise RuntimeError(f"Unknown features in {group_name}: {sorted(unknown)}")

    variants = {"baseline_all_45_features": features}
    for group_name, members in SUBGROUPS.items():
        variants[f"without_{group_name}"] = [f for f in features if f not in set(members)]

    baseline_scored = None
    results = {}
    for name, selected_features in variants.items():
        model = base.fit_ranker(fit_groups, early_groups, selected_features, category_maps)
        scored = base.predict_groups(model, dev_groups, selected_features, category_maps)
        metrics = suite.evaluate_scores(name, scored)
        if name == "baseline_all_45_features":
            baseline_scored = scored
        results[name] = {
            "feature_count": len(selected_features),
            "removed_features": sorted(set(features) - set(selected_features)),
            "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
            "development_metrics": metrics,
            "paired_vs_baseline": None if name == "baseline_all_45_features"
                else base.paired_metrics(baseline_scored, scored),
        }
        print("DEV_SUBGROUP_RESULT " + json.dumps(
            {"variant": name, **results[name]}, ensure_ascii=False), flush=True)

    report = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": scope,
        "data": {
            "runner_rows": len(rows), "dates": len(dates),
            "fit_races": len(fit_groups), "early_stopping_races": len(early_groups),
            "development_races": len(dev_groups),
            "holdout_period_reserved_not_loaded_for_evaluation": True,
        },
        "split": {
            "fit_start": train_date_list[0].isoformat(),
            "fit_end": train_date_list[train_cut-1].isoformat(),
            "early_stopping_start": train_date_list[train_cut].isoformat(),
            "early_stopping_end": train_date_list[-1].isoformat(),
            "development_start": dates[cut1].isoformat(),
            "development_end": dates[cut2-1].isoformat(),
            "holdout_start_reserved": dates[cut2].isoformat(),
            "holdout_end_reserved": dates[-1].isoformat(),
            "policy": "Chronological 70/10/20; train period internally split 90/10; variants compared on development period only.",
        },
        "features": {
            "baseline_count": len(features),
            "recent_form_count": len(original_recent),
            "subgroups": SUBGROUPS,
            "subgroups_are_disjoint_and_exhaustive": True,
            "current_race_body_weight_excluded": True,
            "prior_body_weight_avg_retained": "prior_body_weight_avg" in features,
        },
        "training": {
            "fixed_parameters": base.PARAMS,
            "only_varied_dimension": "removal of one predeclared recent-form subgroup",
            "holdout_used": False,
            "selection_policy": "Development metrics are diagnostic for choosing candidate signals; no holdout results are computed or inspected.",
        },
        "results": results,
        "limitations": [
            "Research only; no production model, ticket generator, or canonical report changed.",
            "Ablation indicates dependence on existing signals, not causal effect or profitability.",
            "Development results are for subgroup screening; final assessment requires a fresh future holdout or a pre-registered untouched period.",
            "No final odds, payout, or current-race bodyweight values are used as features.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("RECENT_FORM_SUBGROUPS_COMPLETE " + json.dumps(
        {"data": report["data"], "results": results}, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
