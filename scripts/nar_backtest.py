#!/usr/bin/env python3
"""Download NAR monthly race archives and run a leakage-safe, fixed-ticket backtest.

Raw NAR CSV/ZIP files are processed in the ephemeral runner only. Only aggregate
metrics and a small set of race-level examples are written to data/backtest-summary.json.
"""
from __future__ import annotations
import csv, datetime as dt, io, json, math, os, re, sys, time, urllib.parse, urllib.request, zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "backtest-summary.json"
END = dt.date.today() - dt.timedelta(days=1)
START = END - dt.timedelta(days=364)
UA = "Mozilla/5.0 (compatible; KEIBA-LAB-backtest/1.0)"

def norm(s):
    return re.sub(r"\s+", "", str(s or "")).replace("　", "")

def number(v):
    try:
        s = str(v or "").replace(",", "").strip()
        return float(s) if s else None
    except ValueError:
        return None

def rank_value(v):
    n = number(v)
    return int(n) if n is not None and n > 0 and n < 99 else None

def decode(raw):
    for enc in ("utf-8-sig", "cp932", "shift_jis"):
        try: return raw.decode(enc)
        except UnicodeDecodeError: pass
    return raw.decode("utf-8", errors="replace")

def rows_from_bytes(raw):
    return list(csv.reader(io.StringIO(decode(raw))))

def race_key(venue, date, race):
    d = re.sub(r"\D", "", str(date or ""))
    if len(d) != 8: return None
    r = re.sub(r"\D", "", str(race or ""))
    if not r: return None
    return (norm(venue), d, str(int(r)))

def month_iter(start, end):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        m += 1
        if m == 13: y, m = y + 1, 1

def download_month(y, m):
    url = "https://www.keiba.go.jp/KeibaWeb/DataDownload/RaceDataDownload?" + urllib.parse.urlencode({
        "type": "monthly", "k_year": y, "k_month": m
    })
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Referer": "https://www.keiba.go.jp/"})
    last = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=90) as res:
                data = res.read()
            if not data.startswith(b"PK"):
                raise RuntimeError(f"{y}-{m:02d}: response is not a ZIP archive")
            return data, url
        except Exception as exc:
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"{y}-{m:02d}: download failed: {last}")

def classify(name):
    n = norm(name).lower()
    if any(x in n for x in ("horselist", "horse_list", "出馬表")): return "horses"
    if any(x in n for x in ("racelist", "race_list", "レース一覧")): return "races"
    if any(x in n for x in ("payback", "payout", "払戻")): return "payouts"
    return None

def load_archives():
    races, horses, payouts = {}, defaultdict(list), {}
    downloaded = []
    for y, m in month_iter(START, END):
        blob, url = download_month(y, m)
        downloaded.append({"month": f"{y}-{m:02d}", "bytes": len(blob)})
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            for info in zf.infolist():
                kind = classify(info.filename)
                if not kind or info.is_dir(): continue
                for row in rows_from_bytes(zf.read(info)):
                    if len(row) < 4: continue
                    key = race_key(row[0], row[1], row[2])
                    if not key: continue
                    date = dt.datetime.strptime(key[1], "%Y%m%d").date()
                    if date < START or date > END: continue
                    if kind == "races":
                        races[key] = {
                            "key": key, "date": date, "venue": row[0], "race": key[2],
                            "name": row[5] if len(row) > 5 else "",
                            "distance": number(row[23]) if len(row) > 23 else None,
                            "surface": row[21] if len(row) > 21 else "",
                            "going": row[25] if len(row) > 25 else "",
                        }
                    elif kind == "horses" and len(row) > 35:
                        horse_no = re.sub(r"\D", "", str(row[5] or ""))
                        name = norm(row[6])
                        if not horse_no or not name: continue
                        horses[key].append({
                            "no": horse_no, "name": name, "age": number(row[8]),
                            "sire": norm(row[11]), "jockey": norm(row[14]),
                            "carried": number(row[16]), "finish": rank_value(row[31]),
                            "margin": row[33] if len(row) > 33 else "",
                            "last3f": number(row[34]) if len(row) > 34 else None,
                        })
                    elif kind == "payouts" and len(row) > 52:
                        payouts[key] = {
                            "win": (re.sub(r"\D", "", row[4]), number(row[5])),
                            "place": [(re.sub(r"\D", "", row[i]), number(row[i+1])) for i in (7,10,13) if len(row)>i+1],
                            "quinella": (row[24], row[25], number(row[26])),
                            "exacta": (row[28], row[29], number(row[30])),
                            "wide": [(row[i], row[i+1], number(row[i+2])) for i in (32,36,40) if len(row)>i+2],
                            "trio": (row[44], row[45], row[46], number(row[47])),
                            "trifecta": (row[49], row[50], row[51], number(row[52])),
                        }
    return races, horses, payouts, downloaded

def margin_score(value):
    s = str(value or "").strip()
    if not s: return None
    if any(x in s for x in ("大差", "失格", "中止")): return 0.0
    if "ハナ" in s: return 1.0
    if "アタマ" in s: return 0.5
    if "クビ" in s: return 0.75
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    return float(m.group(1)) if m else None

def score_runner(h, history, sire_stats, jockey_stats, race, model_weights):
    past = history.get(h["name"], [])[-5:]
    if past:
        ranks = [x["finish"] for x in past if x.get("finish")]
        form = max(5, 100 - (sum(ranks)/len(ranks)-1)*12) if ranks else 45
        recent = past[-3:]
        top3 = sum(1 for x in recent if x.get("finish") and x["finish"] <= 3) / len(recent)
        form = min(100, form + top3*12)
    else:
        form = 45
    venue_runs = [x for x in past if x.get("venue") == race["venue"] and x.get("finish")]
    venue_score = max(10, 100-(sum(x["finish"] for x in venue_runs)/len(venue_runs)-1)*12) if venue_runs else 48
    distance_runs = [x for x in past if race.get("distance") and x.get("distance") and abs(x["distance"]-race["distance"]) <= 200 and x.get("finish")]
    distance_score = max(10, 100-(sum(x["finish"] for x in distance_runs)/len(distance_runs)-1)*12) if distance_runs else 48
    sire_score = sire_stats[h["sire"]]["top3"]/sire_stats[h["sire"]]["runs"]*100 if h["sire"] and sire_stats[h["sire"]]["runs"] >= 5 else 50
    j = jockey_stats[h["jockey"]]
    jockey_score = j["top3"]/j["runs"]*100 if h["jockey"] and j["runs"] >= 5 else 50
    age = h.get("age")
    age_score = 72 if age is not None and 4 <= age <= 6 else 58 if age in (3,7) else 45
    parts = {"form":form/100,"venue":venue_score/100,"distance":distance_score/100,
             "sire":sire_score/100,"jockey":jockey_score/100,"age":age_score/100}
    score = sum(model_weights[k]*v for k,v in parts.items())
    return score, parts

def update_model(field, parts_by_no, model_weights):
    valid = [h for h in field if h.get("finish") and h["no"] in parts_by_no]
    if len(valid) < 3: return False
    scores = [sum(model_weights[k]*parts_by_no[h["no"]][k] for k in model_weights) for h in valid]
    peak = max(scores)
    expv = [math.exp((x-peak)*8) for x in scores]
    denom = sum(expv) or 1
    probs = [x/denom for x in expv]
    winner = next((i for i,h in enumerate(valid) if h["finish"] == 1), None)
    if winner is None: return False
    lr = 0.025
    for k in model_weights:
        grad = sum(((1.0 if i == winner else 0.0)-probs[i])*parts_by_no[h["no"]][k] for i,h in enumerate(valid))
        model_weights[k] = max(0.01, min(0.8, model_weights[k] + lr*grad))
    total = sum(model_weights.values()) or 1
    for k in model_weights: model_weights[k] /= total
    return True

def split_nums(s):
    return [re.sub(r"\D", "", x) for x in re.split(r"[-－ー,／/ ]+", str(s or "")) if re.sub(r"\D", "", x)]

def same_unordered(a, b):
    return sorted(a) == sorted(b)

def payout_for(kind, ticket, p):
    if not p: return 0
    if kind == "win":
        return p["win"][1] or 0 if ticket[0] == p["win"][0] else 0
    if kind == "place":
        return next((pay or 0 for no,pay in p["place"] if no == ticket[0]), 0)
    if kind == "quinella":
        a,b,pay = p["quinella"]
        return (pay or 0) if same_unordered(ticket, split_nums(a)+split_nums(b)) and len(split_nums(a)+split_nums(b)) == 2 else (pay or 0) if same_unordered(ticket, [re.sub(r"\D","",a),re.sub(r"\D","",b)]) else 0
    if kind == "exacta":
        a,b,pay = p["exacta"]
        return (pay or 0) if ticket == [re.sub(r"\D","",a),re.sub(r"\D","",b)] else 0
    if kind == "wide":
        for a,b,pay in p["wide"]:
            if pay and same_unordered(ticket,[re.sub(r"\D","",a),re.sub(r"\D","",b)]): return pay
        return 0
    if kind == "trio":
        a,b,c,pay = p["trio"]
        return (pay or 0) if pay and same_unordered(ticket,[re.sub(r"\D","",a),re.sub(r"\D","",b),re.sub(r"\D","",c)]) else 0
    if kind == "trifecta":
        a,b,c,pay = p["trifecta"]
        return (pay or 0) if ticket == [re.sub(r"\D","",a),re.sub(r"\D","",b),re.sub(r"\D","",c)] else 0
    return 0

def run_backtest(races, horses, payouts):
    history = defaultdict(list)
    sire_stats = defaultdict(lambda: {"runs":0,"top3":0})
    jockey_stats = defaultdict(lambda: {"runs":0,"top3":0})
    metrics = {k: {"races":0,"tickets":0,"hits":0,"stake":0,"return":0} for k in ("win","place","wide","quinella","exacta","trio","trifecta")}
    examples = []
    model_weights = {"form":0.55,"venue":0.12,"distance":0.12,"sire":0.10,"jockey":0.07,"age":0.04}
    learning_updates = 0
    ordered = sorted((r for r in races.values()), key=lambda r:(r["date"],r["venue"],int(r["race"])))
    daily_pending = []
    current_date = None

    def commit_day(items):
        nonlocal learning_updates
        for pending_field, pending_parts, pending_race in items:
            if update_model(pending_field, pending_parts, model_weights):
                learning_updates += 1
            for runner in pending_field:
                if runner.get("finish"):
                    history[runner["name"]].append({
                        "date":pending_race["date"],"venue":pending_race["venue"],"distance":pending_race.get("distance"),
                        "finish":runner["finish"],"margin":margin_score(runner.get("margin")),"last3f":runner.get("last3f")
                    })
                    if runner.get("sire"):
                        sire_stats[runner["sire"]]["runs"] += 1
                        sire_stats[runner["sire"]]["top3"] += int(runner["finish"] <= 3)
                    if runner.get("jockey"):
                        jockey_stats[runner["jockey"]]["runs"] += 1
                        jockey_stats[runner["jockey"]]["top3"] += int(runner["finish"] <= 3)

    for race in ordered:
        if current_date is not None and race["date"] != current_date:
            commit_day(daily_pending)
            daily_pending = []
        current_date = race["date"]
        key = race["key"]
        field = horses.get(key, [])
        if len(field) < 3: continue
        # Predict BEFORE adding any current-race results to history or trainer/jockey/sire aggregates.
        scored = []
        parts_by_no = {}
        for h in field:
            score, parts = score_runner(h,history,sire_stats,jockey_stats,race,model_weights)
            scored.append((score,h))
            parts_by_no[h["no"]] = parts
        ranked = [h for score,h in sorted(scored,key=lambda item:item[0],reverse=True)]
        top = ranked[:6]
        if len(top) < 3: continue
        nos = [h["no"] for h in top]
        t = {
            "win":[[nos[0]]],
            "place":[[x] for x in nos[:3]],
            "wide":[[nos[0],nos[1]],[nos[0],nos[2]],[nos[1],nos[2]]],
            "quinella":[[nos[0],nos[1]]],
            "exacta":[[nos[0],nos[1]]],
            "trio":[nos[:3]],
            "trifecta":[nos[:3]],
        }
        payout = payouts.get(key)
        if payout:
            for kind, tickets in t.items():
                m = metrics[kind]
                m["races"] += 1
                for ticket in tickets:
                    m["tickets"] += 1
                    m["stake"] += 100
                    ret = payout_for(kind,ticket,payout)
                    if ret:
                        m["hits"] += 1
                        m["return"] += ret
            if len(examples) < 25:
                examples.append({
                    "date": race["date"].isoformat(), "venue": race["venue"], "race": race["race"],
                    "name": race.get("name",""), "ranked": [{"no":h["no"],"name":h["name"]} for h in ranked[:6]],
                    "tickets": {k:["-".join(x) for x in v] for k,v in t.items()},
                    "winner": payout["win"][0],
                })
        # Keep today's results buffered: no race on the same date can learn from another race's result.
        daily_pending.append((field, parts_by_no, race))
    if daily_pending:
        commit_day(daily_pending)
    summary = {}
    for kind,m in metrics.items():
        summary[kind] = {**m, "hit_rate": round(100*m["hits"]/m["tickets"],2) if m["tickets"] else None,
                         "roi": round(100*m["return"]/m["stake"],2) if m["stake"] else None,
                         "profit": m["return"]-m["stake"]}
    return summary, examples, len(ordered), model_weights, learning_updates

def build_live_history(races, horses):
    horse_history = defaultdict(list)
    sire_stats = defaultdict(lambda: {"runs": 0, "top3": 0})
    jockey_stats = defaultdict(lambda: {"runs": 0, "top3": 0})
    ordered = sorted(races.values(), key=lambda r:(r["date"],r["venue"],int(r["race"])))
    for race in ordered:
        for h in horses.get(race["key"], []):
            finish = h.get("finish")
            if not finish:
                continue
            item = {"date": race["date"].isoformat(), "venue": race["venue"],
                    "distance": race.get("distance"), "finish": finish,
                    "margin": h.get("margin", ""), "last3f": h.get("last3f")}
            horse_history[norm(h["name"])].append(item)
            if h.get("sire"):
                sire_stats[norm(h["sire"])]["runs"] += 1
                sire_stats[norm(h["sire"])]["top3"] += int(finish <= 3)
            if h.get("jockey"):
                jockey_stats[norm(h["jockey"])]["runs"] += 1
                jockey_stats[norm(h["jockey"])]["top3"] += int(finish <= 3)
    return {
        "training_period": {"start": START.isoformat(), "end": END.isoformat()},
        "horse_history": {name: runs[-5:] for name, runs in horse_history.items()},
        "sire_stats": dict(sire_stats),
        "jockey_stats": dict(jockey_stats),
        "note": "Snapshot through period end; only use for target dates after training_period.end."
    }

def holdout_validation(races, horses, payouts):
    dates = sorted({r["date"] for r in races.values()})
    if len(dates) < 30:
        return {"status":"insufficient_dates"}
    split_index = max(1, min(len(dates)-1, int(len(dates)*0.75)))
    split_date = dates[split_index]
    train_races = {k:r for k,r in races.items() if r["date"] < split_date}
    train_horses = {k:v for k,v in horses.items() if k in train_races}
    train_payouts = {k:v for k,v in payouts.items() if k in train_races}
    _, _, _, learned, updates = run_backtest(train_races, train_horses, train_payouts)
    baseline = {"form":0.55,"venue":0.12,"distance":0.12,"sire":0.10,"jockey":0.07,"age":0.04}
    models = {"baseline":baseline, "trained":learned}
    bet_kinds = ("win","place","wide","quinella","exacta","trio","trifecta")
    results = {name:{"races":0,"winner_top1":0,"winner_top3":0,
                     "bet_types":{kind:{"tickets":0,"hits":0,"stake":0,"return":0}
                                  for kind in bet_kinds}} for name in models}
    history = defaultdict(list)
    sire_stats = defaultdict(lambda:{"runs":0,"top3":0})
    jockey_stats = defaultdict(lambda:{"runs":0,"top3":0})
    train_ordered = sorted(train_races.values(),
                           key=lambda r:(r["date"],r["venue"],int(r["race"])))
    for train_race in train_ordered:
        for h in train_horses.get(train_race["key"], []):
            if not h.get("finish"):
                continue
            history[h["name"]].append({"date":train_race["date"],"venue":train_race["venue"],
                "distance":train_race.get("distance"),"finish":h["finish"],
                "margin":margin_score(h.get("margin")),"last3f":h.get("last3f")})
            if h.get("sire"):
                sire_stats[h["sire"]]["runs"] += 1
                sire_stats[h["sire"]]["top3"] += int(h["finish"] <= 3)
            if h.get("jockey"):
                jockey_stats[h["jockey"]]["runs"] += 1
                jockey_stats[h["jockey"]]["top3"] += int(h["finish"] <= 3)
    ordered = sorted((r for r in races.values() if r["date"] >= split_date),
                     key=lambda r:(r["date"],r["venue"],int(r["race"])))
    pending=[]; current=None
    def commit(items):
        for field,race in items:
            for h in field:
                if not h.get("finish"): continue
                history[h["name"]].append({"date":race["date"],"venue":race["venue"],
                    "distance":race.get("distance"),"finish":h["finish"],
                    "margin":margin_score(h.get("margin")),"last3f":h.get("last3f")})
                if h.get("sire"):
                    sire_stats[h["sire"]]["runs"]+=1;sire_stats[h["sire"]]["top3"]+=int(h["finish"]<=3)
                if h.get("jockey"):
                    jockey_stats[h["jockey"]]["runs"]+=1;jockey_stats[h["jockey"]]["top3"]+=int(h["finish"]<=3)
    for race in ordered:
        if current is not None and race["date"] != current:
            commit(pending);pending=[]
        current=race["date"]
        key=race["key"];field=horses.get(key,[])
        if len(field)<3: continue
        p=payouts.get(key)
        if not p: continue
        for model_name,weights in models.items():
            scored=[(score_runner(h,history,sire_stats,jockey_stats,race,weights)[0],h) for h in field]
            ranked=[h for _,h in sorted(scored,key=lambda x:x[0],reverse=True)]
            if len(ranked)<3: continue
            m=results[model_name];m["races"]+=1
            winner=p["win"][0]
            if ranked[0]["no"]==winner:m["winner_top1"]+=1
            if winner in [h["no"] for h in ranked[:3]]:m["winner_top3"]+=1
            n=[h["no"] for h in ranked[:3]]
            ticket_map={
                "win":[[n[0]]],
                "place":[[x] for x in n],
                "wide":[[n[0],n[1]],[n[0],n[2]],[n[1],n[2]]],
                "quinella":[[n[0],n[1]]],
                "exacta":[[n[0],n[1]]],
                "trio":[n[:3]],
                "trifecta":[n[:3]],
            }
            for kind,tickets in ticket_map.items():
                bm=m["bet_types"][kind]
                for ticket in tickets:
                    bm["tickets"]+=1
                    bm["stake"]+=100
                    ret=payout_for(kind,ticket,p) or 0
                    if ret:
                        bm["hits"]+=1
                        bm["return"]+=ret
        pending.append((field,race))
    if pending:commit(pending)
    out={}
    for name,m in results.items():
        n=m["races"] or 1
        bet_summary={}
        for kind,bm in m["bet_types"].items():
            bet_summary[kind]={**bm,
                "hit_rate":round(100*bm["hits"]/bm["tickets"],2) if bm["tickets"] else None,
                "roi":round(100*bm["return"]/bm["stake"],2) if bm["stake"] else None,
                "profit":bm["return"]-bm["stake"]}
        out[name]={**m,"bet_types":bet_summary,
          "winner_top1_rate":round(100*m["winner_top1"]/n,2),
          "winner_top3_rate":round(100*m["winner_top3"]/n,2),
          "win_stake":bet_summary["win"]["stake"],"win_return":bet_summary["win"]["return"],
          "place_stake":bet_summary["place"]["stake"],"place_return":bet_summary["place"]["return"],
          "wide_stake":bet_summary["wide"]["stake"],"wide_return":bet_summary["wide"]["return"],
          "win_roi":bet_summary["win"]["roi"],"place_roi":bet_summary["place"]["roi"],
          "wide_roi":bet_summary["wide"]["roi"]}
    return {"status":"ok","train_end":dates[split_index-1].isoformat(),
            "test_start":split_date.isoformat(),"test_end":dates[-1].isoformat(),
            "train_races":len(train_races),"test_races":len(ordered),
            "training_updates":updates,"trained_weights":learned,"models":out,
            "selection_rule":"Model weights frozen at the train/test boundary. Test-day results are added only after every race on that date is scored."}

def main():
    print(f"Backtest range: {START} through {END}", flush=True)
    races, horses, payouts, months = load_archives()
    metrics, examples, race_count, learned_weights, learning_updates = run_backtest(races, horses, payouts)
    validation = holdout_validation(races, horses, payouts)
    result = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "period": {"start":START.isoformat(),"end":END.isoformat()},
        "source": "NAR official monthly race data; raw archives are not retained",
        "downloaded_months": months,
        "race_count": race_count,
        "horse_rows": sum(len(v) for v in horses.values()),
        "payout_races": len(payouts),
        "evaluated": sum(v["races"] for v in metrics.values()),
        "method": {
            "prediction_order": "date-batched chronological; every race on a date is predicted before any result from that date updates history or model weights",
            "odds_feature": False,
            "ticket_policy": "fixed top-rank tickets per race; 100 JPY per ticket; not a betting recommendation",
            "ticket_counts_per_race": {"win":1,"place":3,"wide":3,"quinella":1,"exacta":1,"trio":1,"trifecta":1},
            "limitations": ["No historical pre-race odds feature; NAR monthly odds are confirmed odds and are not used.",
                            "Only a fixed, transparent ticket policy is tested; scores are heuristic, not calibrated probabilities.",
                            "Missing or incomplete historical rows are excluded where necessary."]
        },
        "metrics": metrics,
        "out_of_sample_validation": validation,
        "online_learning": {"updates": learning_updates, "final_weights": learned_weights,
                            "note": "Daily batch learning: all races on a date are scored before that date's results update weights for later dates."},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    validated_weights = validation.get("trained_weights", learned_weights) if validation.get("status") == "ok" else learned_weights
    live_model = {
        "model_version": "nar-chronological-v4-holdout-seeded",
        "generated_at": result["generated_at"],
        "training_period": result["period"],
        "updates": validation.get("training_updates", learning_updates),
        "validation": validation,
        "feature_schema": ["form", "venue", "distance", "sire", "jockey", "age"],
        "weights": validated_weights,
        "prediction_policy": "Scores use only races strictly earlier than the target race date; current-race finish/popularity/payout are excluded. All races on a date are predicted before any results from that date update the model."
    }
    (OUT.parent / "live-model.json").write_text(json.dumps(live_model,ensure_ascii=False,indent=2),encoding="utf-8")
    live_history = build_live_history(races, horses)
    (OUT.parent / "live-history.json").write_text(json.dumps(live_history,ensure_ascii=False,separators=(",",":")),encoding="utf-8")
    print(json.dumps({"races":race_count,"horse_rows":result["horse_rows"],"payout_races":len(payouts),"metrics":metrics,"learning_updates":learning_updates,"final_weights":learned_weights},ensure_ascii=False),flush=True)
    if race_count == 0 or not payouts:
        raise RuntimeError("No usable historical race/payout data parsed; refusing to publish an empty backtest.")

if __name__ == "__main__":
    main()
