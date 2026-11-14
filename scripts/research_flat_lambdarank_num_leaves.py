#!/usr/bin/env python3
"""Research-only LambdaRank num_leaves sweep.

Only num_leaves changes. The 45 leakage-audited features, relevance labels,
linear label_gain, chronological split, and all other LightGBM parameters
remain fixed. Candidate selection uses development data only; holdout is used
once after selection. No production files are changed.
"""
from __future__ import annotations
import datetime as dt
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import nar_backtest as nb
import nar_ml_challenger_backtest as suite

OUT = ROOT / "data" / "flat-lambdarank-num-leaves.json"
EXCLUDED_BODY_WEIGHT = {
    "body_weight", "body_weight_change", "body_weight_delta_pct",
    "race_rel_body_weight_delta_pct",
}
BANEI_VENUE = re.compile(r"帯広|ばんえい", re.IGNORECASE)
BASE_PARAMS = {
    "objective": "lambdarank", "metric": "ndcg", "n_estimators": 900,
    "learning_rate": 0.035, "num_leaves": 15, "min_child_samples": 80,
    "reg_lambda": 2.0, "random_state": 42, "n_jobs": 4, "verbosity": -1,
    "label_gain": [0, 1, 2, 3, 4],
}
CANDIDATE_LEAVES = [7, 15, 31]


def load_flat():
    races, horses, payouts, downloads = nb.load_archives()
    banei = {k: r for k, r in races.items()
             if BANEI_VENUE.search(str(r.get("venue") or ""))}
    flat = {k: r for k, r in races.items() if k not in banei}
    flat_horses = {k: v for k, v in horses.items() if k in flat}
    if not flat or any(BANEI_VENUE.search(str(r.get("venue") or "")) for r in flat.values()):
        raise RuntimeError("Venue-only flat filter failed")
    scope = {
        "source_races": len(races), "excluded_banei_venue_races": len(banei),
        "flat_races": len(flat),
        "flat_name_only_banei_word_races_retained": sum(
            1 for r in flat.values() if BANEI_VENUE.search(str(r.get("name") or ""))),
        "excluded_venues": sorted({str(r.get("venue") or "") for r in banei.values()}),
        "filter": "Venue only; title/sponsorship mentions do not remove a flat race.",
        "download_months": len(downloads),
    }
    print("FLAT_SCOPE " + json.dumps(scope, ensure_ascii=False), flush=True)
    return flat, flat_horses, scope


def fit_ranker(train_groups, early_groups, features, category_maps, num_leaves):
    import numpy as np
    from lightgbm import LGBMRanker, early_stopping, log_evaluation

    def flatten(groups):
        data, sizes, labels = [], [], []
        for group in groups:
            sizes.append(len(group))
            data.extend(group)
            labels.extend([max(0, 4 - int(r["finish"])) for r in group])
        x = np.asarray([
            [r[k] if k not in suite.CATEGORICAL else category_maps[k].get(r[k], -1)
             for k in features] for r in data
        ], dtype=float)
        return x, np.asarray(labels, dtype=int), sizes

    xtr, ytr, gtr = flatten(train_groups)
    xdv, ydv, gdv = flatten(early_groups)
    params = dict(BASE_PARAMS)
    params["num_leaves"] = num_leaves
    model = LGBMRanker(**params)
    model.fit(
        xtr, ytr, group=gtr, eval_set=[(xdv, ydv)], eval_group=[gdv],
        eval_at=[1, 3, 7],
        callbacks=[early_stopping(60, verbose=False), log_evaluation(0)],
    )
    return model, params


def predict_groups(model, groups, features, category_maps):
    output = []
    for group in groups:
        x = [[r[k] if k not in suite.CATEGORICAL else category_maps[k].get(r[k], -1)
              for k in features] for r in group]
        output.append((group, model.predict(x)))
    return output


def selection_score(m):
    # Ranking-only development objective, prioritizing top-three capture.
    return round(
        0.45 * float(m["top3_capture_rate"] or 0)
        + 0.35 * float(m["top7_podium_coverage_pct"] or 0)
        + 0.20 * float(m["top1_rate"] or 0), 6
    )


def paired_bootstrap(base, challenger, seed=20261114, iterations=3000):
    if len(base) != len(challenger):
        raise RuntimeError("Paired holdout race count mismatch")
    n = len(base)
    if n < 30:
        return {"races": n, "ci95_pp": None, "reason": "fewer_than_30_races"}
    diffs = [challenger[i] - base[i] for i in range(n)]
    rng = random.Random(seed)
    samples = []
    for _ in range(iterations):
        samples.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    samples.sort()
    return {
        "races": n,
        "mean_delta_pp": round(100 * sum(diffs) / n, 3),
        "ci95_pp": [
            round(100 * samples[int(.025 * (iterations - 1))], 3),
            round(100 * samples[int(.975 * (iterations - 1))], 3),
        ],
        "method": "paired race bootstrap", "iterations": iterations,
    }


def paired_metrics(base_scored, alt_scored):
    base_top1, alt_top1, base_top3, alt_top3 = [], [], [], []
    for (g0, s0), (g1, s1) in zip(base_scored, alt_scored):
        if [r["race_key"] for r in g0] != [r["race_key"] for r in g1]:
            raise RuntimeError("Holdout race alignment mismatch")
        winner = next((i for i, r in enumerate(g0) if r["winner"]), None)
        if winner is None:
            continue
        o0 = sorted(range(len(g0)), key=lambda i: (-float(s0[i]), int(g0[i]["horse_no"])))
        o1 = sorted(range(len(g1)), key=lambda i: (-float(s1[i]), int(g1[i]["horse_no"])))
        base_top1.append(float(o0[0] == winner))
        alt_top1.append(float(o1[0] == winner))
        base_top3.append(float(winner in o0[:3]))
        alt_top3.append(float(winner in o1[:3]))
    return {
        "top1": paired_bootstrap(base_top1, alt_top1, seed=201),
        "winner_in_top3": paired_bootstrap(base_top3, alt_top3, seed=203),
    }


def main():
    suite.RAW_NUMERIC = [f for f in suite.RAW_NUMERIC if f not in EXCLUDED_BODY_WEIGHT]
    suite.RAW_FEATURES = suite.RAW_NUMERIC + suite.CATEGORICAL
    races, horses, scope = load_flat()
    rows, dates = suite.build_rows(races, horses)
    if len(dates) < 100:
        raise RuntimeError("Insufficient chronological dates")
    cut1, cut2 = int(len(dates) * .70), int(len(dates) * .80)
    train_dates, dev_dates, hold_dates = set(dates[:cut1]), set(dates[cut1:cut2]), set(dates[cut2:])

    def rows_for(date_set):
        return [r for r in rows if dt.date.fromisoformat(r["date"]) in date_set]

    train_rows = rows_for(train_dates)
    train_groups = suite.group_rows(train_rows)
    dev_groups = suite.group_rows(rows_for(dev_dates))
    holdout_groups = suite.group_rows(rows_for(hold_dates))
    early_stop_groups = dev_groups[:len(dev_groups) // 2]
    selection_groups = dev_groups[len(dev_groups) // 2:]
    if not train_groups or not early_stop_groups or not selection_groups or not holdout_groups:
        raise RuntimeError("Empty chronological split")
    category_maps = {
        k: {v: i for i, v in enumerate(sorted({r[k] for r in train_rows}))}
        for k in suite.CATEGORICAL
    }
    features = list(suite.RAW_FEATURES)
    candidates = []
    models = {}
    for leaves in CANDIDATE_LEAVES:
        model, params = fit_ranker(
            train_groups, early_stop_groups, features, category_maps, leaves
        )
        scored_dev = predict_groups(model, selection_groups, features, category_maps)
        metrics = suite.evaluate_scores(f"num_leaves_{leaves}", scored_dev)
        item = {
            "num_leaves": leaves, "params": params,
            "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
            "development_selection_metrics": metrics,
            "development_selection_score": selection_score(metrics),
        }
        candidates.append(item)
        models[leaves] = model
        print("LEAVES_CANDIDATE " + json.dumps(item, ensure_ascii=False), flush=True)

    selected = max(candidates, key=lambda x: (
        x["development_selection_score"],
        x["development_selection_metrics"]["top7_podium_coverage_pct"],
        x["development_selection_metrics"]["top3_capture_rate"],
    ))
    baseline_model = models[15]
    selected_model = models[selected["num_leaves"]]
    baseline_holdout = predict_groups(baseline_model, holdout_groups, features, category_maps)
    selected_holdout = predict_groups(selected_model, holdout_groups, features, category_maps)
    baseline_metrics = suite.evaluate_scores("baseline_num_leaves_15", baseline_holdout)
    selected_metrics = suite.evaluate_scores(
        f"selected_num_leaves_{selected['num_leaves']}", selected_holdout
    )
    deltas = paired_metrics(baseline_holdout, selected_holdout)

    report = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": scope,
        "data": {
            "runner_rows": len(rows), "dates": len(dates),
            "train_races": len(train_groups), "development_races": len(dev_groups),
            "early_stopping_races": len(early_stop_groups),
            "development_selection_races": len(selection_groups),
            "holdout_races": len(holdout_groups),
        },
        "split": {
            "train_end": dates[cut1 - 1].isoformat(),
            "development_start": dates[cut1].isoformat(),
            "development_end": dates[cut2 - 1].isoformat(),
            "holdout_start": dates[cut2].isoformat(),
            "holdout_end": dates[-1].isoformat(),
            "policy": "Chronological 70/10/20; dev split chronologically in half for early stopping and candidate selection.",
        },
        "features": {
            "count": len(features), "fields": features,
            "excluded_current_race_body_weight": sorted(EXCLUDED_BODY_WEIGHT),
            "prior_body_weight_avg_retained": "prior_body_weight_avg" in features,
        },
        "training": {
            "fixed_parameters": BASE_PARAMS,
            "only_varied_parameter": "num_leaves",
            "candidate_values": CANDIDATE_LEAVES,
            "relevance": "1着=3、2着=2、3着=1、4着以下=0",
            "holdout_used_for_tuning": False,
            "selection_score": "0.45*winner_in_top3 + 0.35*top7_podium_coverage + 0.20*top1_accuracy; ranking only, no odds/payout.",
        },
        "candidates": candidates,
        "selected_num_leaves": selected["num_leaves"],
        "holdout": {
            "baseline_num_leaves_15": baseline_metrics,
            "selected": selected_metrics,
            "selected_minus_baseline_pp": {
                "top1_rate": round(selected_metrics["top1_rate"] - baseline_metrics["top1_rate"], 3),
                "top3_capture_rate": round(selected_metrics["top3_capture_rate"] - baseline_metrics["top3_capture_rate"], 3),
                "top7_podium_coverage_pct": round(
                    selected_metrics["top7_podium_coverage_pct"] - baseline_metrics["top7_podium_coverage_pct"], 3
                ),
                "top7_full_podium_rate": round(
                    selected_metrics["top7_full_podium_rate"] - baseline_metrics["top7_full_podium_rate"], 3
                ),
            },
            "paired_deltas_vs_baseline": deltas,
            "policy": "Final holdout evaluated once after development-only selection.",
        },
        "limitations": [
            "Research only; no production model, ticket generator, or canonical report changed.",
            "Ranking metrics do not establish betting profitability.",
            "No final odds or payout values are used as features or selection targets.",
            "The holdout is historical and cannot guarantee future performance.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("LAMBDA_NUM_LEAVES_COMPLETE " + json.dumps({
        "data": report["data"], "selected_num_leaves": report["selected_num_leaves"],
        "holdout": report["holdout"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
