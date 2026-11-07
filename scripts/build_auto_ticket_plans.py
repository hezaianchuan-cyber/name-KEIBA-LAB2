#!/usr/bin/env python3
"""Build all-race shadow picks and strictly gated real ticket plans."""
from __future__ import annotations
import datetime as dt, gzip, json, math
from pathlib import Path
from zoneinfo import ZoneInfo
import capture_nar_odds as nar

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data"
TYPES={"win":"単勝","place":"複勝","wide":"ワイド","quinella":"馬連","exacta":"馬単","trio":"3連複","trifecta":"3連単"}
REQUIRED={"win":1,"place":1,"wide":2,"quinella":2,"exacta":2,"trio":3,"trifecta":3}

def no(v):
    d="".join(c for c in str(v or "") if c.isdigit())
    return str(int(d)) if d else ""

def thresholds():
    try:
        report=json.loads((DATA/"backtest-summary.json").read_text(encoding="utf-8"))
        items=report.get("preoff_ticket_type_backtest",{}).get("bet_types",{})
        trio=items.get("trio") or {}
        if trio.get("selected_threshold") and (trio.get("selected_holdout") or {}).get("candidate_pass") is True:
            return {"trio":float(trio["selected_threshold"])}
        return {}
    except (OSError,ValueError,TypeError,KeyError):
        return {}

def latest_snapshots():
    latest={}
    for path in (DATA/"odds-snapshots").glob("*.jsonl.gz"):
        try:
            with gzip.open(path,"rt",encoding="utf-8") as f:
                for line in f:
                    try:
                        r=json.loads(line); key=r.get("race_key")
                        captured=dt.datetime.fromisoformat(r["captured_at"])
                        post=dt.datetime.fromisoformat(r["scheduled_post_time"])
                        if not key or post-captured < dt.timedelta(minutes=2) or post-captured > dt.timedelta(minutes=30): continue
                        if key not in latest or captured>latest[key][0]: latest[key]=(captured,r)
                    except (ValueError,KeyError,TypeError,json.JSONDecodeError): pass
        except OSError: pass
    return {key:value[1] for key,value in latest.items()}

def shadow_tickets(predictions, odds_rows=None, count=12):
    """Build up to 12 shadow-only trio tickets from the model's top seven.

    When captured trio odds exist, rank by a transparent 50:50 probability/EV
    rank fusion. Without odds, retain probability order rather than dropping
    the race from shadow evaluation.
    """
    import itertools
    ranked=sorted(predictions or [],key=lambda x:(float(x.get("win_probability") or 0),float(x.get("score") or 0)),reverse=True)
    unique=[];seen=set()
    for horse in ranked:
        horse_no=no(horse.get("horse_no"))
        if horse_no and horse_no not in seen:
            unique.append(horse);seen.add(horse_no)
    pool=unique[:7]
    numbers=[no(x.get("horse_no")) for x in pool]
    if len(numbers)<3:return []
    probabilities={no(x.get("horse_no")):max(0.0,float(x.get("win_probability") or 0)) for x in pool}
    odds_by_combo={}
    for item in odds_rows or []:
        label=str(item.get("type") or "").replace(" ","")
        if "3連複" not in label and "三連複" not in label:continue
        nums=[no(item.get(key)) for key in ("n1","n2","n3")]
        if len(nums)!=3 or not all(nums) or len(set(nums))!=3:continue
        try:price=float(item.get("odds") or 0)
        except (TypeError,ValueError):continue
        if math.isfinite(price) and price>0:
            odds_by_combo["-".join(sorted(nums,key=int))]=price
    candidates=[]
    for combo in itertools.combinations(numbers,3):
        key="-".join(sorted(combo,key=int))
        probability=nar.combination_probability("trio",combo,probabilities)
        price=odds_by_combo.get(key)
        candidates.append({"key":key,"probability":probability,"odds":price,
                           "ev":probability*price if price is not None else None})
    probability_order=sorted(candidates,key=lambda x:(-x["probability"],x["key"]))
    probability_rank={row["key"]:rank for rank,row in enumerate(probability_order,1)}
    priced=[row for row in candidates if row["odds"] is not None]
    if priced:
        ev_order=sorted(priced,key=lambda x:(-x["ev"],-x["probability"],x["key"]))
        ev_rank={row["key"]:rank for rank,row in enumerate(ev_order,1)}
        selected=sorted(priced,key=lambda x:(probability_rank[x["key"]]+ev_rank[x["key"]],
                                               probability_rank[x["key"]],ev_rank[x["key"]],x["key"]))[:count]
        selected_keys={row["key"] for row in selected}
        selected.extend(row for row in probability_order if row["key"] not in selected_keys)
        basis="probability_ev_rank_fusion"
    else:
        selected=probability_order
        basis="model_probability_only"
    limit=max(0,min(int(count),12))
    return [{"type":"3連複","comb":row["key"],"stake":100,"basis":basis,
             "probability":round(row["probability"],10),
             "odds":round(row["odds"],4) if row["odds"] is not None else None,
             "ev":round(row["ev"],6) if row["ev"] is not None else None}
            for row in selected[:limit]]

def market_ranking(odds):
    rows=[]
    for item in odds or []:
        label=str(item.get("type") or "").replace(" ","")
        if "単勝" not in label: continue
        n=no(item.get("n1"))
        try: price=float(item.get("odds") or 0)
        except (TypeError,ValueError): continue
        if n and price>0:
            try: popularity=int(item.get("popularity") or 999999)
            except (TypeError,ValueError): popularity=999999
            rows.append((price,popularity,int(n),n))
    rows.sort()
    return [row[3] for row in rows[:3]]

def market_favorite(odds):
    ranking=market_ranking(odds)
    return ranking[0] if ranking else None

def base_plan(date,venue,race_no,race_id,predictions,model_version,created_at,odds=None):
    ranked=sorted(predictions or [],key=lambda x:(float(x.get("win_probability") or 0),float(x.get("score") or 0)),reverse=True)
    unique=[];seen=set()
    for h in ranked:
        n=no(h.get("horse_no"))
        if n and n not in seen: unique.append(h);seen.add(n)
    top=unique[:7]
    return {
        "version":2,"createdAt":created_at,"date":date,"raceId":race_id,"venue":venue,"raceNo":str(race_no),
        "decision":"SKIP","skipReason":"仮想買い目は検証専用。購入推奨は締切前オッズと検証済み基準で別判定",
        "skipReasonCode":"SHADOW_ONLY","totalStake":0,"tickets":[],"shadowTickets":shadow_tickets(unique,odds or []),
        "modelTopHorse":no(top[0].get("horse_no")) if top else None,
        "modelTop3":[no(x.get("horse_no")) for x in top[:3]],
        "modelTop7":[no(x.get("horse_no")) for x in top[:7]],
        "shadowTicketSets":{str(n):[x["comb"] for x in shadow_tickets(unique,odds or [])[:n]] for n in (3,6,9,12)},
        "modelPredictions":[{"horseNo":no(x.get("horse_no")),"name":str(x.get("name") or ""),
                             "score":float(x.get("score") or 0),
                             "winProbability":float(x.get("win_probability") or 0)}
                            for x in unique],
        "runnerDraws":[{"horseNo":no(x.get("horse_no")),"name":str(x.get("name") or ""),
                        "frameNo":x.get("frame_no"),"score":float(x.get("score") or 0),
                        "winProbability":float(x.get("win_probability") or 0)}
                       for x in unique],
        "marketFavoriteHorse":market_favorite(odds or []),
        "marketTop3":market_ranking(odds or []),
        "modelLoaded":bool(model_version and len(unique)>=3),"oddsCaptured":bool(odds),
        "oddsRows":len(odds or []),"autoGenerated":True,"modelVersion":model_version,
        "captureTime":created_at,
        "dataStatus":"READY" if len(unique)>=3 else "INSUFFICIENT_DATA"
    }

def make_plan(r,limits):
    predictions=r.get("predictions") or []
    plan=base_plan(r.get("date"),r.get("venue"),r.get("race"),r.get("race_key"),predictions,
                   r.get("model_version"),r.get("captured_at"),r.get("odds") or [])
    plan.update({"captureTime":r.get("captured_at"),"oddsMatchedCandidates":0,"validatedBetTypes":int("trio" in limits)})
    horses=sorted(predictions,key=lambda x:(float(x.get("win_probability") or 0),float(x.get("score") or 0)),reverse=True)
    top=horses[:7]; nums=[no(x.get("horse_no")) for x in top]
    if len(set(nums))<3:
        plan["skipReason"]="予測データまたは出走頭数が不足";plan["skipReasonCode"]="MODEL_NOT_READY"
        return plan
    top_set=set(nums);first_set=set(nums[:2]);candidates=[]
    for x in r.get("candidate_combinations") or []:
        kind=x.get("bet_type")
        if kind!="trio" or kind not in limits:continue
        ns=[no(v) for v in x.get("numbers") or []]
        if len(ns)!=REQUIRED[kind] or len(set(ns))!=len(ns) or any(v not in top_set for v in ns):continue
        if kind in ("exacta","trifecta") and ns[0] not in first_set:continue
        odds=float(x.get("odds") or 0);prob=float(x.get("probability") or 0)
        if odds>0 and prob>0:candidates.append({"kind":kind,"type":TYPES[kind],"comb":"-".join(ns),"odds":odds,"probability":prob,"ev":odds*prob})
    plan["oddsMatchedCandidates"]=len(candidates)
    eligible=[]
    for kind in ("trio",):
        if kind not in limits:continue
        label=TYPES[kind]
        rows=sorted([x for x in candidates if x["kind"]==kind and x["ev"]>=max(1.05,limits[kind])],key=lambda x:(-x["probability"],-x["ev"]))
        if rows:eligible.append({"kind":kind,"type":label,"rows":rows[:12],"avg":sum(x["ev"] for x in rows[:5])/min(5,len(rows)),"best":rows[0]})
    eligible.sort(key=lambda x:(-x["avg"],-x["best"]["probability"]))
    selected=eligible[0] if eligible else None
    allocation=[];allocated=0
    if selected:
        rows=selected["rows"];target=min(2000,len(rows)*100)
        ws=[max(.0001,(x["ev"]-1)*math.sqrt(max(x["probability"],.000001))) for x in rows];total=sum(ws)
        allocation=[{**x,"stake":math.floor((target*w/total)/100)*100} for x,w in zip(rows,ws)] if total else []
        allocation=[x for x in allocation if x["stake"]>=100];allocated=sum(x["stake"] for x in allocation);i=0
        while allocation and allocated+100<=target:
            allocation[i]["stake"]+=100;allocated+=100;i=(i+1)%len(allocation)
    odds_rows=r.get("odds") or []
    model_ok=bool(r.get("model_version")) and len(predictions)>=3
    bet=bool(model_ok and odds_rows and limits and selected and allocation)
    if not model_ok:reason,code="学習モデルまたは予測データ未準備","MODEL_NOT_READY"
    elif not odds_rows:reason,code="締切前オッズ未取得","ODDS_NOT_CAPTURED"
    elif "trio" not in limits:reason,code="3連複の検証済みEV基準が未成立","NO_VALIDATED_TICKET_THRESHOLD"
    elif not candidates:reason,code="候補買い目のオッズ未照合","TICKET_ODDS_NOT_MATCHED"
    else:reason,code="検証済みEV基準を満たす候補なし","EV_BELOW_VALIDATED_THRESHOLD"
    plan.update({"decision":"BET" if bet else "SKIP","skipReason":"" if bet else reason,
        "skipReasonCode":"BET" if bet else code,"totalStake":allocated if bet else 0,
        "tickets":[{"type":x["type"],"comb":x["comb"],"stake":x["stake"],"odds":x["odds"],"probability":x["probability"],"ev":x["ev"]} for x in allocation] if bet else [],
        "modelLoaded":model_ok,"oddsCaptured":bool(odds_rows),"oddsRows":len(odds_rows),
        "validatedBetTypes":int("trio" in limits),"dataStatus":"READY"})
    return plan

def all_racecard_plans(snapshot_map,limits,now):
    races,horses_by_race,odds=nar.read_archives()
    if not races:return []
    context=nar.load_model_context()
    model=context[0] if context else {}
    model_version=model.get("model_version")
    output={}
    today=now.strftime("%Y%m%d")
    for key,race in races.items():
        venue,date,race_no=key
        if date!=today:continue
        field=horses_by_race.get(key,[])
        if len(field)<3:continue
        race_key="|".join(key)
        snap=snapshot_map.get(race_key)
        if snap:
            plan=make_plan(snap,limits)
        else:
            predictions=nar.predict_horses(race,field,[],context)
            plan=base_plan(date,venue,race_no,race_key,predictions,model_version,now.isoformat(),[])
            plan["skipReason"]="締切前オッズスナップショットなし。仮想候補のみ保存"
            plan["skipReasonCode"]="NO_PREOFF_SNAPSHOT"
        plan["marketFavoriteHorse"]=market_favorite(snap.get("odds",[])) if snap else None
        plan["marketTop3"]=market_ranking(snap.get("odds",[])) if snap else []
        output[race_key]=plan
    return list(output.values())

def main():
    now=dt.datetime.now(ZoneInfo("Asia/Tokyo"));limits=thresholds()
    snapshots=latest_snapshots()
    plans_by_key={}
    for r in snapshots.values():
        p=make_plan(r,limits)
        if p and p.get("raceId"):plans_by_key[p["raceId"]]=p
    for p in all_racecard_plans(snapshots,limits,now):
        key=p["raceId"]
        old=plans_by_key.get(key)
        # Prefer the pre-off snapshot record when present; it contains timestamped odds.
        if not old:plans_by_key[key]=p
        else:
            old["shadowTickets"]=p["shadowTickets"]
            old["shadowTicketSets"]=p.get("shadowTicketSets",old.get("shadowTicketSets",{}))
            old["modelTop7"]=p.get("modelTop7",old.get("modelTop7",[]))
            old["modelTopHorse"]=p["modelTopHorse"]
            old["modelTop3"]=p["modelTop3"]
            old["marketFavoriteHorse"]=p["marketFavoriteHorse"] or old.get("marketFavoriteHorse")
    plans=list(plans_by_key.values())
    plans.sort(key=lambda x:(x["date"],x["venue"],int(x["raceNo"] or 0)))
    out={"schema_version":2,"generated_at":now.isoformat(),"source":"official_nar_racecards_plus_preoff_snapshots",
         "plans":plans,"coverage":{"today_racecards":sum(1 for p in plans if p["date"]==now.strftime("%Y%m%d")),
         "shadow_plans":sum(bool(p.get("shadowTickets")) for p in plans),"real_bets":sum(p["decision"]=="BET" for p in plans)}}
    (DATA/"auto-ticket-plans.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"total_races":len(plans),"today_racecards":out["coverage"]["today_racecards"],
        "shadow_plans":out["coverage"]["shadow_plans"],"bet_plans":out["coverage"]["real_bets"],
        "skip_plans":sum(p["decision"]=="SKIP" for p in plans),"validated_bet_types":len(limits)},ensure_ascii=False))
if __name__=="__main__":main()
