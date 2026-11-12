#!/usr/bin/env python3
"""Chronological Banei ablation: prior-form baseline vs baseline + carried weight."""
from __future__ import annotations
import json, math, random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
import nar_backtest as nb

OUT = nb.ROOT / "data" / "banei-carried-weight-ablation.json"
SEED = 20261113
BOOTSTRAPS = 2000

def is_banei(venue):
    v = nb.norm(venue)
    return "帯広" in v or "ばんえい" in v.lower()

def build_rows(races, horses):
    history = defaultdict(list)
    records = []
    ordered = sorted(races.items(), key=lambda item: (item[1]["date"], item[0]))
    for key, race in ordered:
        if not is_banei(race.get("venue")):
            continue
        field = horses.get(key, [])
        if len(field) < 3:
            continue
        weights = [nb.number(h.get("carried")) for h in field]
        valid_weights = [w for w in weights if w is not None]
        if len(valid_weights) < max(3, math.ceil(len(field) * 0.70)):
            continue
        mean_w = sum(valid_weights) / len(valid_weights)
        sd_w = math.sqrt(sum((w-mean_w)**2 for w in valid_weights) / len(valid_weights))
        for h, carried in zip(field, weights):
            name = h.get("name")
            past = history[name]
            finishes = [nb.number(x.get("finish")) for x in past]
            finishes = [x for x in finishes if x is not None and x > 0]
            prior_top3 = sum(1 for x in finishes if x <= 3)
            recent = finishes[-5:]
            prior_field = [nb.number(x.get("field_size")) for x in past[-5:]]
            prior_field = [x for x in prior_field if x is not None and x > 1]
            last_finish = finishes[-1] if finishes else None
            last_size = nb.number(past[-1].get("field_size")) if past else None
            age = nb.number(h.get("age"))
            base = [
                math.log1p(len(finishes)),
                (prior_top3 + 1.5) / (len(finishes) + 5.0),
                (sum(1 for x in recent if x <= 3) + 1.0) / (len(recent) + 3.0),
                (1.0 - (last_finish - 1.0) / (last_size - 1.0)) if last_finish is not None and last_size and last_size > 1 else 0.5,
                (age if age is not None else 5.0),
            ]
            # Carried weight is represented relative to the current race field.
            # This isolates handicap differences within the same Banei race.
            weight_z = ((carried - mean_w) / sd_w) if carried is not None and sd_w > 1e-9 else 0.0
            finish = nb.number(h.get("finish"))
            records.append({
                "date": race["date"].isoformat(),
                "race_key": "|".join(str(x) for x in key),
                "horse_no": str(h.get("no")),
                "base": base,
                "weight_z": weight_z,
                "has_weight": carried is not None,
                "finish": finish,
                "label": int(finish is not None and 0 < finish <= 3),
                "field_size": len(field),
            })
        # Update historical form only after every runner in this race has features.
        for h in field:
            history[h["name"]].append({
                "finish": h.get("finish"),
                "field_size": len(field),
            })
    return records

def metric_pack(records, probs, model_name):
    by_race = defaultdict(list)
    y = np.asarray([r["label"] for r in records], dtype=float)
    p = np.clip(np.asarray(probs, dtype=float), 1e-8, 1-1e-8)
    brier = float(np.mean((p-y)**2))
    logloss = float(np.mean(-(y*np.log(p)+(1-y)*np.log(1-p))))
    bins = []
    ece = 0.0
    for i in range(10):
        mask = (p >= i/10) & (p < (i+1)/10 if i < 9 else p <= 1)
        if np.any(mask):
            conf, acc = float(np.mean(p[mask])), float(np.mean(y[mask]))
            ece += float(np.mean(mask)) * abs(conf-acc)
            bins.append({"range":f"{i*10}-{(i+1)*10}%","n":int(np.sum(mask)),
                         "mean_probability_pct":round(conf*100,2),"actual_top3_pct":round(acc*100,2)})
    for idx, (r, prob) in enumerate(zip(records, p)):
        by_race[r["race_key"]].append((float(prob), r))
    axis_top3 = top3_covered = full_top3 = 0
    actual_podium_count = 0
    for rows in by_race.values():
        rows.sort(key=lambda x:(-x[0], int(x[1]["horse_no"]) if x[1]["horse_no"].isdigit() else x[1]["horse_no"]))
        actual = {r["horse_no"] for _,r in rows if r["label"] == 1}
        if not actual:
            continue
        actual_podium_count += len(actual)
        axis_top3 += int(rows[0][1]["horse_no"] in actual)
        predicted = {r["horse_no"] for _,r in rows[:3]}
        top3_covered += len(predicted & actual)
        full_top3 += int(actual.issubset(predicted))
    races = len(by_race)
    return {
        "model":model_name,"races":races,"runner_rows":len(records),
        "runner_top3_brier":round(brier,6),"runner_top3_logloss":round(logloss,6),
        "runner_top3_ece":round(ece,6),
        "top_ranked_horse_top3_rate_pct":round(100*axis_top3/races,3) if races else None,
        "predicted_top3_actual_podium_coverage_pct":round(100*top3_covered/actual_podium_count,3) if actual_podium_count else None,
        "all_podium_horses_in_predicted_top3_pct":round(100*full_top3/races,3) if races else None,
        "calibration_bins":bins,
        "_per_race":{k:{"axis":int(next((r["horse_no"] for _,r in sorted(v,key=lambda x:-x[0])[:1]),"") in {r["horse_no"] for _,r in v if r["label"]==1}),
                       "coverage":len({r["horse_no"] for _,r in sorted(v,key=lambda x:-x[0])[:3]} & {r["horse_no"] for _,r in v if r["label"]==1}) / max(1,len({r["horse_no"] for _,r in v if r["label"]==1})),
                       "brier":float(np.mean([(1/(1+math.exp(-0))) if False else 0 for _ in []]))}
                   for k,v in by_race.items()}
    }

def fit_predict(train, test, include_weight):
    x_train = np.asarray([r["base"] + ([r["weight_z"]] if include_weight else []) for r in train], dtype=float)
    y_train = np.asarray([r["label"] for r in train], dtype=int)
    x_test = np.asarray([r["base"] + ([r["weight_z"]] if include_weight else []) for r in test], dtype=float)
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.5, max_iter=1000, class_weight=None, random_state=SEED))
    model.fit(x_train, y_train)
    return model.predict_proba(x_test)[:,1], model[-1].coef_[0].tolist()

def paired_bootstrap(base, plus, n=BOOTSTRAPS):
    keys = sorted(set(base) & set(plus))
    if not keys: return {}
    rng = random.Random(SEED)
    deltas = {"axis":[], "coverage":[], "full":[], "brier":[]}
    for _ in range(n):
        sample = [keys[rng.randrange(len(keys))] for _ in keys]
        for metric in deltas:
            deltas[metric].append(sum(plus[k][metric]-base[k][metric] for k in sample)/len(sample))
    return {k:{"mean_delta":round(sum(v)/len(v),6),
               "ci95_low":round(sorted(v)[int(0.025*len(v))],6),
               "ci95_high":round(sorted(v)[int(0.975*len(v))-1],6)}
            for k,v in deltas.items()}

def main():
    races, horses, payouts, metadata = nb.load_archives()
    all_records = build_rows(races, horses)
    # Both models use the exact same runners/races: rows without recorded carried
    # weight are excluded from both, so missingness cannot create a sample advantage.
    complete = [r for r in all_records if r["has_weight"] and r["finish"] is not None and r["field_size"] >= 5]
    dates = sorted({r["date"] for r in complete})
    if len(dates) < 30:
        raise RuntimeError(f"Insufficient Banei dates for chronological split: {len(dates)}")
    train_cut = dates[int(len(dates)*0.70)-1]
    dev_cut = dates[int(len(dates)*0.85)-1]
    train = [r for r in complete if r["date"] <= train_cut]
    holdout = [r for r in complete if r["date"] > dev_cut]
    train_races = {r["race_key"] for r in train}
    holdout_races = {r["race_key"] for r in holdout}
    # Keep only holdout races with at least three classified podium horses.
    counts = defaultdict(int)
    for r in holdout: counts[r["race_key"]] += r["label"]
    valid_races = {k for k,v in counts.items() if v >= 3}
    holdout = [r for r in holdout if r["race_key"] in valid_races]
    if not train or not holdout:
        raise RuntimeError("Empty chronological train or holdout split")
    base_probs, base_coef = fit_predict(train, holdout, False)
    plus_probs, plus_coef = fit_predict(train, holdout, True)
    base_report = metric_pack(holdout, base_probs, "baseline_no_carried_weight")
    plus_report = metric_pack(holdout, plus_probs, "baseline_plus_carried_weight")
    # Race-level paired bootstrap for top-rank accuracy, podium coverage,
    # full-podium capture, and runner-level Brier (resampled by race).
    base_by_race, plus_by_race = {}, {}
    def make_race_metrics(records, probs):
        grouped=defaultdict(list)
        for r,p in zip(records,probs): grouped[r["race_key"]].append((float(p),r))
        out={}
        for key,rows in grouped.items():
            rows.sort(key=lambda x:-x[0])
            actual={r["horse_no"] for _,r in rows if r["label"]==1}
            pred={r["horse_no"] for _,r in rows[:3]}
            top=rows[0][1]["horse_no"]
            out[key]={"axis":float(top in actual),"coverage":len(pred&actual)/max(1,len(actual)),
                      "full":float(actual.issubset(pred)),
                      "brier":sum((p-r["label"])**2 for p,r in rows)/len(rows)}
        return out
    base_by_race=make_race_metrics(holdout,base_probs)
    plus_by_race=make_race_metrics(holdout,plus_probs)
    delta=paired_bootstrap(base_by_race,plus_by_race)
    # Model coefficients show direction only, not causal effect.
    report={
      "generated_at":datetime.now(timezone.utc).isoformat(),
      "status":"research_only_banei_carried_weight_ablation",
      "scope":"Banei/Obihiro only. No flat-racing records. No production changes.",
      "data_period":{"start":nb.START.isoformat(),"end":nb.END.isoformat()},
      "sample":{"all_banei_runner_rows_with_features":len(all_records),
                "carried_weight_complete_rows":len(complete),
                "carried_weight_complete_pct":round(100*len(complete)/len(all_records),3) if all_records else None,
                "unique_dates":len(dates),"train_dates":len({r["date"] for r in train}),
                "train_date_end":train_cut,"dev_date_end":dev_cut,
                "train_races":len(train_races),"train_runner_rows":len(train),
                "holdout_races":len(holdout_races),"holdout_runner_rows":len(holdout),
                "holdout_start":min(r["date"] for r in holdout),"holdout_end":max(r["date"] for r in holdout)},
      "method":{"baseline_features":["prior_run_count","smoothed_career_top3_rate","smoothed_recent5_top3_rate","last_finish_relative_quality","age"],
                "added_feature":"current_race_relative_carried_weight_zscore",
                "same_sample_for_both_models":True,
                "split":"chronological 70% train / 15% development gap / final 15% holdout",
                "model":"StandardScaler + LogisticRegression(C=0.5); fixed settings, no tuning on holdout",
                "label":"finish position <=3; holdout race requires at least 3 classified podium finishers",
                "outcome_not_used_as_feature":True},
      "baseline":{k:v for k,v in base_report.items() if not k.startswith("_")},
      "with_carried_weight":{k:v for k,v in plus_report.items() if not k.startswith("_")},
      "paired_holdout_delta_plus_minus_baseline":delta,
      "coefficients":{"baseline":base_coef,"with_carried_weight":plus_coef},
      "guardrails":["This compares one pre-race feature only; no body weight, body-weight change, or track moisture features.",
                    "The model is a research benchmark, not the production LAB model.",
                    "Feature association and predictive lift do not prove causal effects.",
                    "No betting profitability claim is made."]
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print("BANEI_WEIGHT_ABLATION_RESULT "+json.dumps({
      "sample":report["sample"],"baseline":{k:v for k,v in report["baseline"].items() if k!="calibration_bins"},
      "with_weight":{k:v for k,v in report["with_carried_weight"].items() if k!="calibration_bins"},
      "delta":delta,"coefficients":report["coefficients"],"report":str(OUT)
    },ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
