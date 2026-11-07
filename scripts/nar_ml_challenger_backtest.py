#!/usr/bin/env python3
"""Research-only NAR challenger suite: CatBoost, LightGBM, LambdaRank and Plackett-Luce.

All features are built from information available before the race date. Results from
all races on a date are added only after every race on that date has been featurized.
The final chronological holdout is never used for tuning or early stopping.
"""
from __future__ import annotations
import datetime as dt
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import nar_backtest as nb

OUT = ROOT / "data" / "ml-challenger-backtest.json"
BASE_WEIGHTS = {"form": 0.55, "venue": 0.12, "distance": 0.12,
                "sire": 0.10, "jockey": 0.07, "age": 0.04}
LAB_NUMERIC = ["lab_score", "form", "venue_score", "distance_score", "sire_score",
               "jockey_score", "age_score", "field_size", "distance_m"]
RAW_NUMERIC = [
    "prior_runs", "career_win_rate", "career_top3_rate", "recent_finish_pct",
    "recent_top3_rate", "recent_margin_quality", "recent_last3f_rank",
    "recent_time_per_km", "venue_runs", "venue_top3_rate", "distance_runs",
    "distance_top3_rate", "going_runs", "going_top3_rate", "jockey_runs",
    "jockey_top3_rate", "sire_runs", "sire_top3_rate", "age", "field_size",
    "distance_m"
]
CATEGORICAL = ["venue", "surface", "going", "jockey", "sire"]
LAB_FEATURES = LAB_NUMERIC + ["venue", "surface", "going"]
RAW_FEATURES = RAW_NUMERIC + CATEGORICAL


def mean_or(values, default=0.0):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(values) / len(values) if values else default


def rate(rows, prior=0.5, strength=8.0):
    if not rows:
        return prior
    hits = sum(1 for x in rows if x.get("finish") is not None and x["finish"] <= 3)
    return (hits + prior * strength) / (len(rows) + strength)


def build_rows(races, horses):
    history = defaultdict(list)
    sire_stats = defaultdict(lambda: {"runs": 0, "top3": 0, "wins": 0})
    jockey_stats = defaultdict(lambda: {"runs": 0, "top3": 0, "wins": 0})
    rows = []
    ordered = sorted(races.values(), key=lambda r: (r["date"], r["venue"], int(r["race"])))
    dates = sorted({r["date"] for r in ordered})
    by_date = defaultdict(list)
    for race in ordered:
        by_date[race["date"]].append(race)

    for day_index, day in enumerate(dates, 1):
        day_pending = []
        day_races = by_date[day]
        for race in day_races:
            field = horses.get(race["key"], [])
            if len(field) < 3:
                continue
            for h in field:
                _, parts = nb.score_runner(h, history, sire_stats, jockey_stats, race, BASE_WEIGHTS)
                score = sum(BASE_WEIGHTS[k] * parts[k] for k in BASE_WEIGHTS)
                finish = nb.rank_value(h.get("finish"))
                if finish is None:
                    continue
                past = history.get(h["name"], [])
                recent = past[-5:]
                venue_past = [x for x in past if x.get("venue") == race.get("venue")]
                distance = nb.number(race.get("distance"))
                distance_past = [
                    x for x in past if distance is not None and nb.number(x.get("distance")) is not None
                    and abs(nb.number(x.get("distance")) - distance) <= 200
                ]
                going_past = [
                    x for x in past if race.get("going") and x.get("going") == race.get("going")
                    and (not race.get("surface") or not x.get("surface") or x.get("surface") == race.get("surface"))
                ]
                finish_pct = [
                    max(0.0, min(1.0, (float(x.get("field_size") or 12) - float(x["finish"])) /
                                      max(float(x.get("field_size") or 12) - 1, 1)))
                    for x in recent if x.get("finish") is not None
                ]
                recent_times = [nb.number(x.get("time_per_km")) for x in recent]
                recent_last3f = [nb.number(x.get("last3f_rank_score")) for x in recent]
                recent_margins = [nb.number(x.get("margin_quality")) for x in recent]
                js = jockey_stats[h.get("jockey")] if h.get("jockey") else {"runs": 0, "top3": 0, "wins": 0}
                ss = sire_stats[h.get("sire")] if h.get("sire") else {"runs": 0, "top3": 0, "wins": 0}
                row = {
                    "race_key": "|".join(map(str, race["key"])),
                    "date": race["date"].isoformat(),
                    "horse_no": str(h.get("no", "")),
                    "finish": finish, "winner": int(finish == 1),
                    "lab_score": float(score),
                    "form": float(parts["form"]),
                    "venue_score": float(parts["venue"]),
                    "distance_score": float(parts["distance"]),
                    "sire_score": float(parts["sire"]),
                    "jockey_score": float(parts["jockey"]),
                    "age_score": float(parts["age"]),
                    "field_size": float(race.get("field_size") or len(field)),
                    "distance_m": float(distance or 0),
                    "venue": str(race.get("venue") or "unknown"),
                    "surface": str(race.get("surface") or "unknown"),
                    "going": str(race.get("going") or "unknown"),
                    "jockey": str(h.get("jockey") or "unknown"),
                    "sire": str(h.get("sire") or "unknown"),
                    "prior_runs": float(len(past)),
                    "career_win_rate": float((sum(1 for x in past if x.get("finish") == 1) + 0.5 * 8) / (len(past) + 8)),
                    "career_top3_rate": float(rate(past)),
                    "recent_finish_pct": float(mean_or(finish_pct, 0.5)),
                    "recent_top3_rate": float(rate(recent)),
                    "recent_margin_quality": float(mean_or(recent_margins, 0.5)),
                    "recent_last3f_rank": float(mean_or(recent_last3f, 0.5)),
                    "recent_time_per_km": float(mean_or(recent_times, 0.0)),
                    "venue_runs": float(len(venue_past)),
                    "venue_top3_rate": float(rate(venue_past)),
                    "distance_runs": float(len(distance_past)),
                    "distance_top3_rate": float(rate(distance_past)),
                    "going_runs": float(len(going_past)),
                    "going_top3_rate": float(rate(going_past)),
                    "jockey_runs": float(js["runs"]),
                    "jockey_top3_rate": float((js["top3"] + 0.5 * 10) / (js["runs"] + 10)),
                    "sire_runs": float(ss["runs"]),
                    "sire_top3_rate": float((ss["top3"] + 0.5 * 10) / (ss["runs"] + 10)),
                    "age": float(h.get("age") or 0),
                }
                rows.append(row)

        # Update each race exactly once, only after all races on this date are scored.
        for race in day_races:
            field = horses.get(race["key"], [])
            if len(field) < 3:
                continue
            performance = nb.field_performance_features(field)
            for runner in field:
                if not runner.get("finish"):
                    continue
                sire = runner.get("sire")
                jockey = runner.get("jockey")
                if sire:
                    sire_stats[sire]["runs"] += 1
                    sire_stats[sire]["top3"] += int(runner["finish"] <= 3)
                    sire_stats[sire]["wins"] += int(runner["finish"] == 1)
                if jockey:
                    jockey_stats[jockey]["runs"] += 1
                    jockey_stats[jockey]["top3"] += int(runner["finish"] <= 3)
                    jockey_stats[jockey]["wins"] += int(runner["finish"] == 1)
                history[runner["name"]].append({
                    **performance.get(runner["no"], {}),
                    "date": race["date"], "venue": race["venue"],
                    "distance": race.get("distance"), "going": race.get("going"),
                    "surface": race.get("surface"), "field_size": race.get("field_size"),
                    "finish": runner["finish"],
                    "margin_quality": performance.get(runner["no"], {}).get("margin_quality"),
                    "last3f_rank_score": performance.get(runner["no"], {}).get("last3f_rank_score"),
                    "going_quality": (nb.rank_quality(runner["finish"], race.get("field_size")) or 50) / 100.0,
                    "time_seconds": runner.get("time_seconds"),
                    "time_per_km": (runner.get("time_seconds") * 1000 / race["distance"]
                                    if runner.get("time_seconds") and race.get("distance") else None),
                })
        if day_index == 1 or day_index % max(1, len(dates)//10) == 0 or day_index == len(dates):
            print(f"ML_DATA_PROGRESS {day_index}/{len(dates)} date={day}", flush=True)
    return rows, dates


def group_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["race_key"]].append(row)
    return [g for g in groups.values() if len(g) >= 3 and any(r["winner"] for r in g)]


def evaluate_scores(name, grouped_scores):
    races = top1 = top3 = 0
    for group, scores in grouped_scores:
        winner = next((i for i, r in enumerate(group) if r["winner"]), None)
        if winner is None:
            continue
        order = sorted(range(len(group)), key=lambda i: (-float(scores[i]), group[i]["horse_no"]))
        races += 1
        top1 += int(order[0] == winner)
        top3 += int(winner in order[:3])
    return {"name": name, "races": races, "top1_hits": top1, "top3_hits": top3,
            "top1_rate": round(100 * top1 / races, 3) if races else None,
            "top3_capture_rate": round(100 * top3 / races, 3) if races else None}


def score_model(name, model, test_groups, feature_names, category_maps=None, categorical=False):
    output = []
    for group in test_groups:
        if categorical:
            x = [[r[k] if k in CATEGORICAL else r[k] for k in feature_names] for r in group]
        else:
            x = [[r[k] if k in feature_names and k not in CATEGORICAL else
                  (category_maps[k].get(r[k], -1) if k in CATEGORICAL else r[k])
                  for k in feature_names] for r in group]
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(x)[:, 1]
        else:
            scores = model.predict(x)
        output.append((group, scores))
    return evaluate_scores(name, output)


def fit_classifier(kind, train, dev, features, category_maps):
    import numpy as np
    y_train = np.asarray([r["winner"] for r in train], dtype=int)
    y_dev = np.asarray([r["winner"] for r in dev], dtype=int)
    cat_features = [i for i, k in enumerate(features) if k in CATEGORICAL]
    def matrix(data, cat=False):
        if cat:
            return [[r[k] if k in CATEGORICAL else r[k] for k in features] for r in data]
        return np.asarray([[r[k] if k not in CATEGORICAL else category_maps[k].get(r[k], -1)
                            for k in features] for r in data], dtype=float)
    if kind == "CatBoost":
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(iterations=700, depth=6, learning_rate=0.04,
            loss_function="Logloss", eval_metric="AUC", verbose=False,
            random_seed=42, allow_writing_files=False, thread_count=4)
        model.fit(matrix(train, True), y_train, cat_features=cat_features,
                  eval_set=(matrix(dev, True), y_dev), early_stopping_rounds=60, verbose=False)
        return model
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation
    model = LGBMClassifier(n_estimators=900, learning_rate=0.035, num_leaves=15,
        max_depth=-1, min_child_samples=100, reg_lambda=2.0,
        subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=4, verbosity=-1)
    model.fit(matrix(train), y_train, eval_set=[(matrix(dev), y_dev)],
              callbacks=[early_stopping(60, verbose=False), log_evaluation(0)])
    return model


def fit_lambdarank(train_groups, dev_groups, features, category_maps):
    import numpy as np
    from lightgbm import LGBMRanker, early_stopping, log_evaluation
    def flatten(groups):
        data, sizes, labels = [], [], []
        for group in groups:
            sizes.append(len(group))
            data.extend(group)
            # Graded relevance: winner > second > third > other finishers.
            labels.extend([max(0, 4 - int(r["finish"])) for r in group])
        x = np.asarray([[r[k] if k not in CATEGORICAL else category_maps[k].get(r[k], -1)
                         for k in features] for r in data], dtype=float)
        return x, np.asarray(labels, dtype=int), sizes
    xtr, ytr, gtr = flatten(train_groups)
    xdv, ydv, gdv = flatten(dev_groups)
    model = LGBMRanker(objective="lambdarank", metric="ndcg", n_estimators=900,
        learning_rate=0.035, num_leaves=15, min_child_samples=80, reg_lambda=2.0,
        random_state=42, n_jobs=4, verbosity=-1)
    model.fit(xtr, ytr, group=gtr, eval_set=[(xdv, ydv)], eval_group=[gdv],
              eval_at=[1, 3], callbacks=[early_stopping(60, verbose=False), log_evaluation(0)])
    return model


def fit_plackett_luce(train_groups, features, category_maps, train_scale):
    import numpy as np
    from scipy.optimize import minimize
    # Conditional-logit/Plackett-Luce winner likelihood with L2 regularization.
    x_groups, winner_idx = [], []
    for group in train_groups:
        x = np.asarray([[r[k] if k not in CATEGORICAL else category_maps[k].get(r[k], -1)
                         for k in features] for r in group], dtype=float)
        x_groups.append((x / train_scale, train_scale))
        winner_idx.append(next(i for i, r in enumerate(group) if r["winner"]))
    dimension = len(features)
    def objective(beta):
        loss = 0.0
        grad = np.zeros(dimension)
        for (x, _), winner in zip(x_groups, winner_idx):
            z = x @ beta
            z = z - np.max(z)
            expz = np.exp(np.clip(z, -40, 40))
            probs = expz / expz.sum()
            loss += -math.log(max(float(probs[winner]), 1e-15))
            grad += x.T @ probs - x[winner]
        loss /= len(x_groups)
        grad /= len(x_groups)
        loss += 0.01 * float(beta @ beta)
        grad += 0.02 * beta
        return loss, grad
    result = minimize(objective, np.zeros(dimension), jac=True, method="L-BFGS-B",
                      options={"maxiter": 300, "ftol": 1e-9})
    return {"coef": result.x.tolist(), "converged": bool(result.success),
            "message": str(result.message), "iterations": int(result.nit)}


def pl_scores(model, group, features, category_maps, train_scale):
    import numpy as np
    x = np.asarray([[r[k] if k not in CATEGORICAL else category_maps[k].get(r[k], -1)
                     for k in features] for r in group], dtype=float)
    return (x / train_scale) @ np.asarray(model["coef"])


def main():
    races, horses, payouts, downloads = nb.load_archives()
    rows, dates = build_rows(races, horses)
    if len(dates) < 100:
        raise RuntimeError(f"Insufficient chronological date coverage: {len(dates)}")
    cut_train = max(1, int(len(dates) * 0.70))
    cut_dev = max(cut_train + 1, int(len(dates) * 0.80))
    train_dates = set(dates[:cut_train])
    dev_dates = set(dates[cut_train:cut_dev])
    test_dates = set(dates[cut_dev:])
    train = [r for r in rows if dt.date.fromisoformat(r["date"]) in train_dates]
    dev = [r for r in rows if dt.date.fromisoformat(r["date"]) in dev_dates]
    test = [r for r in rows if dt.date.fromisoformat(r["date"]) in test_dates]
    train_groups, dev_groups, test_groups = group_rows(train), group_rows(dev), group_rows(test)
    if not train_groups or not dev_groups or not test_groups:
        raise RuntimeError("Chronological train/development/holdout split is empty")
    # Category dictionaries and numeric scaling are fit on training data only.
    category_maps = {key: {value: i for i, value in enumerate(sorted({r[key] for r in train}))}
                     for key in CATEGORICAL}
    models = []
    model_objects = {}
    for feature_set, features in (("LAB_feature_enhanced", LAB_FEATURES),
                                  ("independent_raw", RAW_FEATURES)):
        maps = {k: category_maps[k] for k in CATEGORICAL}
        for kind in ("CatBoost", "LightGBM"):
            model = fit_classifier(kind, train, dev, features, maps)
            model_objects[f"{feature_set}_{kind}"] = model
            models.append(score_model(f"{feature_set}_{kind}", model, test_groups,
                                       features, maps, categorical=(kind == "CatBoost")))
    # LAB frozen baseline, on exactly the same eligible holdout races.
    lab_scores = [(g, [r["lab_score"] for r in g]) for g in test_groups]
    models.insert(0, evaluate_scores("LAB_frozen_baseline", lab_scores))

    # LambdaRank and Plackett-Luce use the genuinely LAB-score-free raw feature set.
    ranker = fit_lambdarank(train_groups, dev_groups, RAW_FEATURES, category_maps)
    rank_output = []
    for group in test_groups:
        x = [[r[k] if k not in CATEGORICAL else category_maps[k].get(r[k], -1)
              for k in RAW_FEATURES] for r in group]
        rank_output.append((group, ranker.predict(x)))
    models.append(evaluate_scores("LambdaRank_independent_raw", rank_output))

    import numpy as np
    scale = np.asarray([[r[k] if k not in CATEGORICAL else category_maps[k].get(r[k], -1)
                         for k in RAW_FEATURES] for r in train], dtype=float).std(axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    pl = fit_plackett_luce(train_groups, RAW_FEATURES, category_maps, scale)
    pl_output = [(g, pl_scores(pl, g, RAW_FEATURES, category_maps, scale)) for g in test_groups]
    models.append(evaluate_scores("PlackettLuce_independent_raw", pl_output))

    # Native model importance is computed without consulting holdout outcomes.
    importance = {}
    for name, model in model_objects.items():
        if name.endswith("CatBoost"):
            values = model.get_feature_importance()
        else:
            values = model.booster_.feature_importance(importance_type="gain")
        features = LAB_FEATURES if "LAB_feature_enhanced" in name else RAW_FEATURES
        importance[name] = sorted(
            [{"feature": f, "importance": round(float(v), 6)} for f, v in zip(features, values)],
            key=lambda x: -x["importance"])[:20]
    importance["LambdaRank_independent_raw"] = sorted(
        [{"feature": f, "importance": round(float(v), 6)}
         for f, v in zip(RAW_FEATURES, ranker.booster_.feature_importance(importance_type="gain"))],
        key=lambda x: -x["importance"])[:20]
    report = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "evaluated",
        "period": {"start": dates[0].isoformat(), "end": dates[-1].isoformat()},
        "split": {"train_end": dates[cut_train-1].isoformat(),
                  "development_start": dates[cut_train].isoformat(),
                  "development_end": dates[cut_dev-1].isoformat(),
                  "holdout_start": dates[cut_dev].isoformat(), "holdout_end": dates[-1].isoformat(),
                  "policy": "Chronological 70/10/20 split; early stopping uses development only; holdout untouched."},
        "data": {"races": len(races), "runner_rows": len(rows), "dates": len(dates),
                 "train_races": len(train_groups), "development_races": len(dev_groups),
                 "holdout_races": len(test_groups), "downloaded_months": downloads},
        "feature_sets": {"LAB_feature_enhanced": LAB_FEATURES, "independent_raw": RAW_FEATURES},
        "models": models,
        "feature_importance": importance,
        "plackett_luce_fit": {"converged": pl["converged"], "message": pl["message"],
                              "iterations": pl["iterations"]},
        "limitations": [
            "Research-only ranking; no production model or ticket logic changed.",
            "LAB_feature_enhanced includes LAB score components; independent_raw excludes all LAB-derived scores.",
            "LambdaRank and Plackett-Luce use independent_raw features only.",
            "Top1/Top3 are ranking metrics, not profitability or betting-return evidence.",
            "Feature importance is model-native gain/importance, not causal attribution.",
            "All same-day race results are withheld until that date's feature generation is complete.",
            "All category mappings and PL scaling are fit on training data only; early stopping uses development only."
        ]
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "data": report["data"],
                      "split": report["split"], "models": report["models"],
                      "plackett_luce_fit": report["plackett_luce_fit"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
