#!/usr/bin/env python3
"""Download NAR monthly race archives and run a leakage-safe, fixed-ticket backtest.

Raw NAR CSV/ZIP files are processed in the ephemeral runner only. Only aggregate
metrics and a small set of race-level examples are written to data/backtest-summary.json.
"""
from __future__ import annotations
import csv, datetime as dt, gzip, io, json, math, os, re, sys, time, urllib.parse, urllib.request, zipfile
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

def merge_payout_row(payouts, key, row):
    """Merge one official payout CSV row; a dead heat may produce several rows."""
    payout = payouts.setdefault(key, {
        "win": [], "place": [], "quinella": [], "exacta": [],
        "wide": [], "trio": [], "trifecta": [], "source_rows": 0,
    })
    payout["source_rows"] += 1

    def add_pay(target, values, amount):
        numbers = [re.sub(r"\\D", "", str(value or "")) for value in values]
        pay = number(amount)
        if all(numbers) and pay is not None and pay > 0:
            item = (numbers, pay)
            if item not in payout[target]:
                payout[target].append(item)

    add_pay("win", [row[4]], row[5])
    for i in (7, 10, 13):
        add_pay("place", [row[i]], row[i+1])
    add_pay("quinella", [row[24], row[25]], row[26])
    add_pay("exacta", [row[28], row[29]], row[30])
    for i in (32, 36, 40):
        add_pay("wide", [row[i], row[i+1]], row[i+2])
    add_pay("trio", [row[44], row[45], row[46]], row[47])
    add_pay("trifecta", [row[49], row[50], row[51]], row[52])
    return payout


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
                        merge_payout_row(payouts, key, row)
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
    winners = {i for i,h in enumerate(valid) if h["finish"] == 1}
    if not winners: return False
    target = 1.0 / len(winners)
    lr = 0.025
    for k in model_weights:
        grad = sum(((target if i in winners else 0.0)-probs[i])*parts_by_no[h["no"]][k] for i,h in enumerate(valid))
        model_weights[k] = max(0.01, min(0.8, model_weights[k] + lr*grad))
    total = sum(model_weights.values()) or 1
    for k in model_weights: model_weights[k] /= total
    return True

def split_nums(s):
    return [re.sub(r"\D", "", x) for x in re.split(r"[-－ー,／/ ]+", str(s or "")) if re.sub(r"\D", "", x)]

def same_unordered(a, b):
    return sorted(a) == sorted(b)

def payout_for(kind, ticket, p):
    """Return the official 100-yen payout for a ticket, including dead heats."""
    if not p:
        return 0
    for numbers, pay in p.get(kind, []):
        if kind in ("win", "place"):
            matched = len(ticket) == 1 and ticket[0] == numbers[0]
        elif kind in ("quinella", "wide", "trio"):
            matched = same_unordered(ticket, numbers)
        else:  # exacta and trifecta are order-sensitive
            matched = ticket == numbers
        if matched:
            return pay or 0
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
        if payout and payout.get("win"):
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
                    "winner": [numbers[0] for numbers, _ in payout["win"]],
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

def evaluate_holdout_fold(races, horses, payouts, train_dates, test_dates):
    train_set = set(train_dates)
    test_set = set(test_dates)
    train_races = {k:r for k,r in races.items() if r["date"] in train_set}
    test_races = {k:r for k,r in races.items() if r["date"] in test_set}
    train_horses = {k:v for k,v in horses.items() if k in train_races}
    train_payouts = {k:v for k,v in payouts.items() if k in train_races}
    _, _, _, learned, updates = run_backtest(train_races, train_horses, train_payouts)
    baseline = {"form":0.55,"venue":0.12,"distance":0.12,"sire":0.10,"jockey":0.07,"age":0.04}
    models = {"baseline":baseline, "trained":learned}
    kinds = ("win","place","wide","quinella","exacta","trio","trifecta")
    results = {name:{"races":0,"winner_top1":0,"winner_top3":0,
                     "bet_types":{kind:{"tickets":0,"hits":0,"stake":0,"return":0}
                                  for kind in kinds}} for name in models}

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

    ordered = sorted(test_races.values(), key=lambda r:(r["date"],r["venue"],int(r["race"])))
    pending=[]; current=None
    def commit(items):
        for field,race in items:
            for h in field:
                if not h.get("finish"): continue
                history[h["name"]].append({"date":race["date"],"venue":race["venue"],
                    "distance":race.get("distance"),"finish":h["finish"],
                    "margin":margin_score(h.get("margin")),"last3f":h.get("last3f")})
                if h.get("sire"):
                    sire_stats[h["sire"]]["runs"]+=1
                    sire_stats[h["sire"]]["top3"]+=int(h["finish"]<=3)
                if h.get("jockey"):
                    jockey_stats[h["jockey"]]["runs"]+=1
                    jockey_stats[h["jockey"]]["top3"]+=int(h["finish"]<=3)

    for race in ordered:
        if current is not None and race["date"] != current:
            commit(pending); pending=[]
        current=race["date"]
        key=race["key"]; field=horses.get(key,[])
        # Add every race to the daily buffer, regardless of payout availability.
        pending.append((field,race))
        if len(field)<3: continue
        payout=payouts.get(key)
        if not payout: continue
        winners={numbers[0] for numbers, _ in payout.get("win", [])}
        if not winners: continue
        for model_name,weights in models.items():
            scored=[(score_runner(h,history,sire_stats,jockey_stats,race,weights)[0],h) for h in field]
            ranked=[h for _,h in sorted(scored,key=lambda x:x[0],reverse=True)]
            if len(ranked)<3: continue
            m=results[model_name]; m["races"]+=1
            if ranked[0]["no"] in winners: m["winner_top1"]+=1
            if any(h["no"] in winners for h in ranked[:3]): m["winner_top3"]+=1
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
                    bm["tickets"]+=1; bm["stake"]+=100
                    ret=payout_for(kind,ticket,payout) or 0
                    if ret:
                        bm["hits"]+=1; bm["return"]+=ret

    if pending: commit(pending)
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
    return {"train_start":min(train_dates).isoformat(),"train_end":max(train_dates).isoformat(),
            "test_start":min(test_dates).isoformat(),"test_end":max(test_dates).isoformat(),
            "train_races":len(train_races),"test_races":len(ordered),
            "training_updates":updates,"trained_weights":learned,"models":out}


def odds_snapshot_inventory():
    directory = ROOT / "data" / "odds-snapshots"
    files = sorted(directory.glob("*.jsonl.gz")) if directory.exists() else []
    race_count = odds_count = 0
    dates = []
    captured = []
    for path in files:
        try:
            with gzip.open(path, "rt", encoding="utf-8") as source:
                for line in source:
                    try:
                        race = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    race_count += 1
                    odds_count += len(race.get("odds") or [])
                    if race.get("date"):
                        dates.append(str(race["date"]).replace("-", ""))
                    if race.get("captured_at"):
                        captured.append(str(race["captured_at"]))
        except OSError:
            continue
    return {
        "snapshot_files": len(files),
        "races": race_count,
        "odds_rows": odds_count,
        "first_date": min(dates) if dates else None,
        "last_date": max(dates) if dates else None,
        "last_captured_at": max(captured) if captured else None,
        "note": "Only timestamped official pre-off daily odds snapshots are eligible for future value backtests; monthly confirmed odds are excluded."
    }


def holdout_validation(races, horses, payouts):
    dates=sorted({r["date"] for r in races.values()})
    if len(dates)<60:
        return {"status":"insufficient_dates","available_dates":len(dates)}
    # Three non-overlapping forward test windows. Each fold trains only on dates
    # strictly before its test window and keeps weights frozen throughout that window.
    bounds=[(0.50,0.65),(0.65,0.80),(0.80,1.00)]
    folds=[]
    for index,(train_fraction,test_end_fraction) in enumerate(bounds,1):
        train_end=max(1,int(len(dates)*train_fraction))
        test_end=max(train_end+1,int(len(dates)*test_end_fraction))
        test_end=min(len(dates),test_end)
        train_dates=dates[:train_end]
        test_dates=dates[train_end:test_end]
        if not train_dates or not test_dates: continue
        folds.append({"fold":index,**evaluate_holdout_fold(races,horses,payouts,train_dates,test_dates)})
    if len(folds)<2:
        return {"status":"insufficient_folds","folds":folds}

    kinds=("win","place","wide","quinella","exacta","trio","trifecta")
    aggregate={}
    for model_name in ("baseline","trained"):
        total={"races":0,"winner_top1":0,"winner_top3":0,
               "bet_types":{kind:{"tickets":0,"hits":0,"stake":0,"return":0} for kind in kinds}}
        for fold in folds:
            fm=fold["models"][model_name]
            total["races"]+=fm["races"]
            total["winner_top1"]+=fm["winner_top1"]
            total["winner_top3"]+=fm["winner_top3"]
            for kind in kinds:
                for key in ("tickets","hits","stake","return"):
                    total["bet_types"][kind][key]+=fm["bet_types"][kind][key]
        bet_summary={}
        for kind,bm in total["bet_types"].items():
            bet_summary[kind]={**bm,
                "hit_rate":round(100*bm["hits"]/bm["tickets"],2) if bm["tickets"] else None,
                "roi":round(100*bm["return"]/bm["stake"],2) if bm["stake"] else None,
                "profit":bm["return"]-bm["stake"]}
        n=total["races"] or 1
        aggregate[model_name]={**total,"bet_types":bet_summary,
            "winner_top1_rate":round(100*total["winner_top1"]/n,2),
            "winner_top3_rate":round(100*total["winner_top3"]/n,2),
            "win_roi":bet_summary["win"]["roi"],"place_roi":bet_summary["place"]["roi"],
            "wide_roi":bet_summary["wide"]["roi"]}
    return {"status":"ok","method":"three rolling-origin folds with non-overlapping test windows; weights frozen within each test window; prior results added only after all races on a date are scored",
            "folds":folds,"test_races":sum(x["models"]["trained"]["races"] for x in folds),
            "test_start":folds[0]["test_start"],"test_end":folds[-1]["test_end"],
            "train_end":folds[-1]["train_end"],"models":aggregate,
            "trained_weights":folds[-1]["trained_weights"],
            "folds_above_100_roi":{kind:sum(1 for fold in folds if (fold["models"]["trained"]["bet_types"][kind]["roi"] or 0)>=100) for kind in kinds},
            "selection_rule":"Each fold uses only strictly earlier training data; test weights remain fixed and race results are added to history only after the entire date is scored."}

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
        "payout_races": sum(1 for p in payouts.values() if p.get("win")),
        "payout_rows": sum(p.get("source_rows", 0) for p in payouts.values()),
        "multi_row_payout_races": sum(1 for p in payouts.values() if p.get("source_rows", 0) > 1),
        "evaluated": sum(v["races"] for v in metrics.values()),
        "method": {
            "prediction_order": "date-batched chronological; every race on a date is predicted before any result from that date updates history or model weights",
            "odds_feature": False,
            "ticket_policy": "fixed top-rank tickets per race; 100 JPY per ticket; not a betting recommendation",
            "ticket_counts_per_race": {"win":1,"place":3,"wide":3,"quinella":1,"exacta":1,"trio":1,"trifecta":1},
            "limitations": ["No historical pre-race odds feature; NAR monthly odds are confirmed odds and are not used.",
                            "All payout rows for dead-heat races are retained and matched across every winning combination.",
                            "Only a fixed, transparent ticket policy is tested; scores are heuristic, not calibrated probabilities.",
                            "Missing or incomplete historical rows are excluded where necessary."]
        },
        "metrics": metrics,
        "out_of_sample_validation": validation,
        "odds_snapshot_archive": odds_snapshot_inventory(),
        "online_learning": {"updates": learning_updates, "final_weights": learned_weights,
                            "note": "Daily batch learning: all races on a date are scored before that date's results update weights for later dates."},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    validated_weights = learned_weights
    live_model = {
        "model_version": "nar-chronological-v5-walk-forward",
        "generated_at": result["generated_at"],
        "training_period": result["period"],
        "updates": learning_updates,
        "validation": validation,
        "feature_schema": ["form", "venue", "distance", "sire", "jockey", "age"],
        "weights": validated_weights,
        "prediction_policy": "Scores use only races strictly earlier than the target race date; current-race finish/popularity/payout are excluded. All races on a date are predicted before any results from that date update the model."
    }
    (OUT.parent / "live-model.json").write_text(json.dumps(live_model,ensure_ascii=False,indent=2),encoding="utf-8")
    live_history = build_live_history(races, horses)
    (OUT.parent / "live-history.json").write_text(json.dumps(live_history,ensure_ascii=False,separators=(",",":")),encoding="utf-8")
    print(json.dumps({"races":race_count,"horse_rows":result["horse_rows"],"payout_races":sum(1 for p in payouts.values() if p.get("win")),"metrics":metrics,"learning_updates":learning_updates,"final_weights":learned_weights},ensure_ascii=False),flush=True)
    if race_count == 0 or not any(p.get("win") for p in payouts.values()):
        raise RuntimeError("No usable historical race/payout data parsed; refusing to publish an empty backtest.")

if __name__ == "__main__":
    main()
