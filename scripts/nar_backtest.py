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

def score_runner(h, history, sire_stats, jockey_stats, race):
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
    return .55*form + .12*venue_score + .12*distance_score + .10*sire_score + .07*jockey_score + .04*age_score

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
    ordered = sorted((r for r in races.values()), key=lambda r:(r["date"],r["venue"],int(r["race"])))
    for race in ordered:
        key = race["key"]
        field = horses.get(key, [])
        if len(field) < 3: continue
        # Predict BEFORE adding any current-race results to history or trainer/jockey/sire aggregates.
        ranked = sorted(field, key=lambda h:score_runner(h,history,sire_stats,jockey_stats,race), reverse=True)
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
        # After evaluation, append the actual result for use by later races only.
        for h in field:
            if h.get("finish"):
                history[h["name"]].append({
                    "date":race["date"],"venue":race["venue"],"distance":race.get("distance"),
                    "finish":h["finish"],"margin":margin_score(h.get("margin")),"last3f":h.get("last3f")
                })
                if h.get("sire"):
                    sire_stats[h["sire"]]["runs"] += 1
                    sire_stats[h["sire"]]["top3"] += int(h["finish"] <= 3)
                if h.get("jockey"):
                    jockey_stats[h["jockey"]]["runs"] += 1
                    jockey_stats[h["jockey"]]["top3"] += int(h["finish"] <= 3)
    summary = {}
    for kind,m in metrics.items():
        summary[kind] = {**m, "hit_rate": round(100*m["hits"]/m["tickets"],2) if m["tickets"] else None,
                         "roi": round(100*m["return"]/m["stake"],2) if m["stake"] else None,
                         "profit": m["return"]-m["stake"]}
    return summary, examples, len(ordered)

def main():
    print(f"Backtest range: {START} through {END}", flush=True)
    races, horses, payouts, months = load_archives()
    metrics, examples, race_count = run_backtest(races, horses, payouts)
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
            "prediction_order": "chronological; each race is predicted before its results enter horse/jockey/sire history",
            "odds_feature": False,
            "ticket_policy": "fixed top-rank tickets per race; 100 JPY per ticket; not a betting recommendation",
            "ticket_counts_per_race": {"win":1,"place":3,"wide":3,"quinella":1,"exacta":1,"trio":1,"trifecta":1},
            "limitations": ["No historical pre-race odds feature; NAR monthly odds are confirmed odds and are not used.",
                            "Only a fixed, transparent ticket policy is tested; scores are heuristic, not calibrated probabilities.",
                            "Missing or incomplete historical rows are excluded where necessary."]
        },
        "metrics": metrics,
        "examples": examples,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"races":race_count,"horse_rows":result["horse_rows"],"payout_races":len(payouts),"metrics":metrics},ensure_ascii=False),flush=True)
    if race_count == 0 or not payouts:
        raise RuntimeError("No usable historical race/payout data parsed; refusing to publish an empty backtest.")

if __name__ == "__main__":
    main()
