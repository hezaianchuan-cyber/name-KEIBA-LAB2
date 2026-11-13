#!/usr/bin/env python3
"""Leakage-safe flat-only age and distance feature diagnostics; research only."""
from __future__ import annotations
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path
import nar_backtest as nb

OUT = Path(__file__).resolve().parents[1] / "data" / "research-flat-distance-age-diagnostics.json"

def is_banei(race):
    venue = str(race.get("venue") or "")
    return any(x in venue for x in ("帯広", "ばんえい", "帯広ば"))

def age_band(age):
    try:
        a = int(age)
    except (TypeError, ValueError):
        return "unknown"
    if a <= 3: return "3以下"
    if a <= 5: return "4-5歳"
    if a <= 7: return "6-7歳"
    return "8歳以上"

def bucket_summary(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    result = {}
    for name, values in sorted(groups.items(), key=lambda x: str(x[0])):
        n = len(values)
        result[str(name)] = {
            "runner_starts": n,
            "wins": sum(x["win"] for x in values),
            "top3": sum(x["top3"] for x in values),
            "win_rate_pct": round(100 * sum(x["win"] for x in values) / n, 3) if n else None,
            "top3_rate_pct": round(100 * sum(x["top3"] for x in values) / n, 3) if n else None,
            "mean_finish_quality_pct": round(100 * sum(x["quality"] for x in values) / n, 3) if n else None,
        }
    return result

def main():
    print("DISTANCE_AGE_DIAGNOSTICS loading official archives", flush=True)
    races, horses, payouts, months = nb.load_archives()
    flat = {k:r for k,r in races.items() if not is_banei(r)}
    excluded = len(races) - len(flat)
    dates = sorted({r["date"] for r in flat.values()})
    cutoff = nb.END - dt.timedelta(days=365)
    train_dates = [d for d in dates if d <= cutoff]
    test_dates = [d for d in dates if d > cutoff]
    test_set = set(test_dates)
    history = defaultdict(list)
    age_rows, distance_rows, interaction_rows = [], [], []
    no_distance_history = 0
    distance_exposure = defaultdict(int)
    ordered_dates = sorted(set(train_dates + test_dates))
    by_date = defaultdict(list)
    for key, race in flat.items():
        by_date[race["date"]].append((key, race))
    for date in ordered_dates:
        todays = sorted(by_date[date], key=lambda x:(x[1]["venue"], int(x[1]["race"])))
        pending = []
        for key, race in todays:
            for h in horses.get(key, []):
                finish = h.get("finish")
                if not finish:
                    continue
                name = str(h.get("name") or "").strip()
                if not name:
                    continue
                prior = history[name]
                age = h.get("age")
                if date in test_set:
                    quality = nb.rank_quality(finish, race.get("field_size"))
                    quality = (quality or 0.0) / 100.0
                    age_key = age_band(age)
                    age_rows.append({"age_band":age_key,"win":int(finish == 1),"top3":int(finish <= 3),"quality":quality})
                    current_distance = race.get("distance")
                    prior_dist = [x for x in prior if x.get("distance") and current_distance]
                    gaps = [abs(float(x["distance"])-float(current_distance)) for x in prior_dist]
                    nearest = min(gaps) if gaps else None
                    matching = [x for x in prior_dist if abs(float(x["distance"])-float(current_distance)) <= 200]
                    if not prior_dist:
                        dist_band = "距離履歴なし"
                        no_distance_history += 1
                    elif nearest <= 100:
                        dist_band = "±100m以内"
                    elif nearest <= 200:
                        dist_band = "101-200m"
                    elif nearest <= 400:
                        dist_band = "201-400m"
                    else:
                        dist_band = "400m超"
                    prior_quality = [nb.rank_quality(x.get("finish"), x.get("field_size")) for x in matching]
                    prior_quality = [q for q in prior_quality if q is not None]
                    prior_top3 = sum(1 for x in matching if x.get("finish") and x["finish"] <= 3)
                    distance_rows.append({
                        "distance_band":dist_band,"win":int(finish == 1),"top3":int(finish <= 3),"quality":quality,
                        "matching_starts":len(matching),
                        "prior_distance_top3_rate":(prior_top3/len(matching)) if matching else None,
                        "prior_distance_quality":(sum(prior_quality)/len(prior_quality)/100.0) if prior_quality else None,
                    })
                    interaction_rows.append({"age_band":age_key,"distance_band":dist_band,
                        "win":int(finish == 1),"top3":int(finish <= 3),"quality":quality})
                    distance_exposure[dist_band] += 1
                pending.append((name, {
                    "date":date.isoformat(),"distance":race.get("distance"),
                    "venue":race.get("venue"),"finish":finish,"field_size":race.get("field_size")
                }))
        # Append only after every race on the date has been scored.
        for name, row in pending:
            history[name].append(row)
    # Summaries of prior matching-start counts, useful for shrinkage design.
    match_counts = [r["matching_starts"] for r in distance_rows]
    interaction = defaultdict(list)
    for row in interaction_rows:
        interaction[(row["age_band"],row["distance_band"])].append(row)
    interaction_summary = {}
    for (age, dist), values in sorted(interaction.items()):
        n=len(values)
        interaction_summary[f"{age} / {dist}"] = {
            "runner_starts":n,"top3_rate_pct":round(100*sum(x["top3"] for x in values)/n,3),
            "win_rate_pct":round(100*sum(x["win"] for x in values)/n,3)
        }
    report = {
        "status":"research_only","scope":"flat_only","excluded_banei_races":excluded,
        "flat_races":len(flat),"venues":sorted({r["venue"] for r in flat.values()}),
        "train_period":{"start":min(train_dates).isoformat(),"end":max(train_dates).isoformat()},
        "test_period":{"start":min(test_dates).isoformat(),"end":max(test_dates).isoformat()},
        "test_runner_starts":len(age_rows),"age_bands":bucket_summary(age_rows,"age_band"),
        "distance_nearest_prior_run":bucket_summary(distance_rows,"distance_band"),
        "distance_prior_match_count":{"mean":round(sum(match_counts)/len(match_counts),3) if match_counts else 0,
            "median":sorted(match_counts)[len(match_counts)//2] if match_counts else 0,
            "no_prior_distance_history_runner_starts":no_distance_history},
        "age_distance_interaction":interaction_summary,
        "method":"For each Japan-local race date, all target runners are scored using only horse history from earlier dates; same-date results are appended after the entire date. Age and nearest prior distance experience are descriptive strata, not causal effects or a production model.",
        "source_months":months
    }
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print("DISTANCE_AGE_DIAGNOSTICS_RESULT "+json.dumps(report,ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
