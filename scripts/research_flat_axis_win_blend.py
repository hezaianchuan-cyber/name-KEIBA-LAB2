#!/usr/bin/env python3
"""Research-only transfer test of Banei winner-similarity / win-specialist blending to flat NAR racing.

Uses the existing leakage-aware chronological feature builder. Current-race body
weight and body-weight change are excluded from predictors. No production files
or live model/ticket logic are changed.
"""
from __future__ import annotations
import datetime as dt
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import nar_backtest as nb
import nar_ml_challenger_backtest as challenger

OUT = ROOT / "data" / "flat-axis-win-blend-research.json"
SIMILARITY_WEIGHTS = (0.60, 0.55, 0.50, 0.45, 0.40)
EXCLUDED = {"body_weight", "body_weight_change", "body_weight_delta_pct", "race_rel_body_weight_delta_pct"}
FEATURES = list(dict.fromkeys(
    [name for name in challenger.LAB_NUMERIC if name not in EXCLUDED]
    + [name for name in challenger.RAW_NUMERIC if name not in EXCLUDED]
))


def grouped(rows):
    buckets = defaultdict(list)
    for row in rows:
        buckets[row["race_key"]].append(row)
    return [sorted(group, key=lambda r: int(r["horse_no"]))
            for _, group in sorted(buckets.items())
            if len(group) >= 3 and sum(int(r["winner"]) for r in group) == 1]


def matrix(rows, center=None, scale=None):
    x = np.asarray([[float(row.get(name) or 0.0) for name in FEATURES]
                    for row in rows], dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if center is not None:
        x = (x - center) / scale
        x = np.clip(x, -6.0, 6.0)
    return x


def fit_similarity(train_rows):
    x = matrix(train_rows)
    center = np.median(x, axis=0)
    scale = np.std(x, axis=0)
    scale[scale < 1e-6] = 1.0
    z = np.clip((x - center) / scale, -6.0, 6.0)
    y = np.asarray([int(r["winner"]) for r in train_rows], dtype=int)
    if y.sum() < 100 or (y == 0).sum() < 100:
        raise RuntimeError("勝ち馬・非勝ち馬の学習行数が不足しています")
    winner_center = np.mean(z[y == 1], axis=0)
    nonwinner_center = np.mean(z[y == 0], axis=0)
    return center, scale, winner_center, nonwinner_center


def similarity_scores(rows, params):
    center, scale, winner_center, nonwinner_center = params
    z = matrix(rows, center, scale)
    d_win = np.sqrt(np.mean((z - winner_center) ** 2, axis=1))
    d_non = np.sqrt(np.mean((z - nonwinner_center) ** 2, axis=1))
    return d_non - d_win


def fit_win_specialist(train_rows):
    x = matrix(train_rows)
    y = np.asarray([int(r["winner"]) for r in train_rows], dtype=int)
    positives = max(int(y.sum()), 1)
    negatives = max(int((y == 0).sum()), 1)
    sample_weight = np.where(y == 1, negatives / positives, 1.0)
    model = HistGradientBoostingClassifier(
        learning_rate=0.06, max_iter=260, max_leaf_nodes=15,
        min_samples_leaf=60, l2_regularization=2.0,
        random_state=42, early_stopping=False,
    )
    model.fit(x, y, sample_weight=sample_weight)
    return model


def percentile(scores):
    order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
    result = np.zeros(len(scores), dtype=float)
    denom = max(len(scores) - 1, 1)
    for rank, idx in enumerate(order):
        result[idx] = 1.0 - rank / denom
    return result


def score_groups(groups, sim_params, win_model):
    scored = []
    for group in groups:
        sim = similarity_scores(group, sim_params)
        win = win_model.predict_proba(matrix(group))[:, 1]
        sim_rank, win_rank = percentile(sim), percentile(win)
        strategies = {
            "LAB_existing": np.asarray([float(r["lab_score"]) for r in group]),
            "winner_similarity_only": sim,
            "win_specialist_only": win,
        }
        for weight in SIMILARITY_WEIGHTS:
            strategies[f"similarity_{int(weight*100)}_win_{100-int(weight*100)}"] = (
                weight * sim_rank + (1.0 - weight) * win_rank
            )
        scored.append((group, strategies))
    return scored


def race_metrics(group, scores):
    order = sorted(range(len(group)), key=lambda i: (-float(scores[i]), i))
    winner = next(i for i, row in enumerate(group) if int(row["winner"]) == 1)
    podium = {i for i, row in enumerate(group)
              if row.get("finish") is not None and int(row["finish"]) <= 3}
    top7 = set(order[:7])
    return {
        "top1": int(order[0] == winner),
        "top3": int(winner in order[:3]),
        "podium_total": len(podium),
        "podium_captured": len(podium & top7),
        "full_podium": int(len(podium) == 3 and len(podium & top7) == 3),
    }


def summarize(scored, name):
    sums = defaultdict(float)
    n = 0
    for group, strategies in scored:
        m = race_metrics(group, strategies[name])
        n += 1
        for key, value in m.items():
            sums[key] += value
    return {
        "races": n,
        "axis_win_rate_pct": round(100 * sums["top1"] / n, 3) if n else None,
        "axis_top3_rate_pct": round(100 * sums["top3"] / n, 3) if n else None,
        "top7_podium_coverage_pct": round(100 * sums["podium_captured"] / sums["podium_total"], 3)
            if sums["podium_total"] else None,
        "top7_full_podium_rate_pct": round(100 * sums["full_podium"] / n, 3) if n else None,
    }


def paired_bootstrap(scored, a, b, iterations=3000):
    """Race-cluster bootstrap for b minus a; paired races are resampled together."""
    records = []
    for group, strategies in scored:
        ma, mb = race_metrics(group, strategies[a]), race_metrics(group, strategies[b])
        records.append((ma, mb))
    if len(records) < 30:
        return {"races": len(records), "status": "sample_too_small"}
    rng = random.Random(20261113 + len(records))
    deltas = defaultdict(list)
    n = len(records)
    for _ in range(iterations):
        sample = [records[rng.randrange(n)] for _ in range(n)]
        for key in ("top1", "top3", "full_podium"):
            deltas[key].append(100 * (
                sum(x[1][key] - x[0][key] for x in sample) / n
            ))
        denom = sum(x[0]["podium_total"] for x in sample)
        if denom:
            deltas["top7_coverage"].append(100 * (
                sum(x[1]["podium_captured"] - x[0]["podium_captured"] for x in sample) / denom
            ))
    out = {"races": n, "delta_definition": f"{b} minus {a}", "confidence_level": 0.95}
    for key, values in deltas.items():
        values.sort()
        out[key] = {
            "delta_pp": round(float(np.mean(values)), 3),
            "ci95_low": round(float(values[int(0.025 * (len(values)-1))]), 3),
            "ci95_high": round(float(values[int(0.975 * (len(values)-1))]), 3),
        }
    return out



def filter_flat_races(races):
    """Exclude Banei/Obihiro races before building chronological features."""
    flat = {
        key: race for key, race in races.items()
        if not any(token in str(race.get("venue") or "") for token in ("帯広", "ばんえい"))
    }
    excluded = len(races) - len(flat)
    if not flat:
        raise RuntimeError("平地レースが0件です。会場フィルターを確認してください")
    return flat, excluded

def main():
    races, horses, payouts, downloaded_months = nb.load_archives()
    races, excluded_banei = filter_flat_races(races)
    rows, dates = challenger.build_rows(races, horses)
    dates = sorted({str(row["date"]) for row in rows})
    if len(dates) < 100:
        raise RuntimeError(f"平地の開催日数が不足しています: {len(dates)}")
    train_end = dates[int(len(dates) * 0.70) - 1]
    dev_end = dates[int(len(dates) * 0.85) - 1]
    train_rows = [r for r in rows if str(r["date"]) <= train_end]
    dev_rows = [r for r in rows if train_end < str(r["date"]) <= dev_end]
    hold_rows = [r for r in rows if str(r["date"]) > dev_end]
    train_groups, dev_groups, hold_groups = grouped(train_rows), grouped(dev_rows), grouped(hold_rows)
    if min(len(train_groups), len(dev_groups), len(hold_groups)) < 100:
        raise RuntimeError("学習・中間確認・最終期間のいずれかで100レース未満です")

    sim_params = fit_similarity(train_rows)
    win_model = fit_win_specialist(train_rows)
    dev_scored = score_groups(dev_groups, sim_params, win_model)
    hold_scored = score_groups(hold_groups, sim_params, win_model)

    dev_results = {name: summarize(dev_scored, name)
                   for name in dev_scored[0][1]}
    hold_results = {name: summarize(hold_scored, name)
                    for name in hold_scored[0][1]}
    anchor = "similarity_50_win_50"
    comparisons = {}
    for weight in SIMILARITY_WEIGHTS:
        name = f"similarity_{int(weight*100)}_win_{100-int(weight*100)}"
        comparisons[name] = {
            "development_vs_LAB": paired_bootstrap(dev_scored, "LAB_existing", name),
            "development_vs_50_50": paired_bootstrap(dev_scored, anchor, name),
            "holdout_descriptive_vs_LAB": paired_bootstrap(hold_scored, "LAB_existing", name),
            "holdout_descriptive_vs_50_50": paired_bootstrap(hold_scored, anchor, name),
        }

    report = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "research_only_flat_axis_win_blend",
        "scope": "地方競馬の平地のみ。ばんえいを含まない。本番モデル・買い目生成は未変更。",
        "excluded_banei_race_count": excluded_banei,
        "included_venues": sorted({str(r.get("venue") or "不明") for r in races.values()}),
        "data": {
            "downloaded_months": downloaded_months,
            "all_rows": len(rows),
            "all_races": len({r["race_key"] for r in rows}),
            "train_races": len(train_groups),
            "development_races": len(dev_groups),
            "holdout_races": len(hold_groups),
            "train_end": train_end,
            "development_end": dev_end,
            "holdout_start": min(str(r["date"]) for r in hold_rows),
            "holdout_end": max(str(r["date"]) for r in hold_rows),
        },
        "features": {
            "count": len(FEATURES),
            "used": FEATURES,
            "excluded": sorted(EXCLUDED),
            "current_race_body_weight_and_change_used": False,
            "race_relative_current_body_weight_change_used": False,
            "current_carried_weight_used": True,
            "feature_timing": "Existing chronological builder constructs every race before updating history with that date's results; no same-day race result is used as a feature.",
        },
        "models": {
            "development": dev_results,
            "holdout_descriptive_only": hold_results,
        },
        "paired_comparisons": comparisons,
        "method": {
            "winner_similarity": "Training-only standardized mean-centroid distance: distance to non-winner centroid minus distance to winner centroid.",
            "win_specialist": "HistGradientBoostingClassifier trained only on the first 70% chronological dates; class-balanced training weights.",
            "blend": "Within-race percentile ranks mixed at similarity shares 60, 55, 50, 45, 40 percent; win specialist supplies the remainder.",
            "selection_rule": "Use the middle 15% only for choosing a candidate ratio. The final 15% is reported as descriptive temporal confirmation and must not be repeatedly tuned against.",
            "LAB_baseline": "Existing lab_score from the same feature builder, on exactly the same races.",
            "holdout_note": "This is a new model experiment but uses the same historical archive window as prior research; do not call it an untouched future holdout.",
            "payouts": "Not used for fitting or ranking; ROI and ticket returns are not assessed in this experiment.",
        },
        "guardrails": [
            "Research-only branch and artifact. No merge to main, production deployment, database migration, or live prediction/ticket changes.",
            "Banei model and data remain separate.",
            "Current-race body_weight, body_weight_change, body_weight_delta_pct and derived race_rel_body_weight_delta_pct are excluded because timestamped pre-off availability is not established.",
            "No odds or post-off information are used.",
            "Do not claim profitability from ranking metrics.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("FLAT_AXIS_WIN_BLEND_RESULT " + json.dumps({
        "data": report["data"], "development": dev_results,
        "holdout_descriptive_only": hold_results,
        "paired_comparisons": comparisons, "report": str(OUT),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
