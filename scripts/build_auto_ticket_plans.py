#!/usr/bin/env python3
"""Build saved ticket plans from the latest official NAR pre-off snapshots."""
from __future__ import annotations
import datetime as dt, gzip, json, math
from pathlib import Path
from zoneinfo import ZoneInfo

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
        return {k:float(v["selected_threshold"]) for k,v in items.items()
                if v.get("selected_threshold") and (v.get("selected_holdout") or {}).get("candidate_pass") is True}
    except (OSError,ValueError,TypeError,KeyError):
        return {}

def latest_snapshots(now):
    latest={}
    for path in (DATA/"odds-snapshots").glob("*.jsonl.gz"):
        try:
            with gzip.open(path,"rt",encoding="utf-8") as f:
                for line in f:
                    try:
                        r=json.loads(line); key=r.get("race_key")
                        captured=dt.datetime.fromisoformat(r["captured_at"])
                        post=dt.datetime.fromisoformat(r["scheduled_post_time"])
                        if not key or post<=now or not (dt.timedelta(minutes=2)<=post-captured<=dt.timedelta(minutes=30)): continue
                        if key not in latest or captured>latest[key][0]: latest[key]=(captured,r)
                    except (ValueError,KeyError,TypeError,json.JSONDecodeError): pass
        except OSError: pass
    return [v[1] for v in latest.values()]

def make_plan(r,limits):
    horses=sorted(r.get("predictions") or [],key=lambda x:(float(x.get("win_probability") or 0),float(x.get("score") or 0)),reverse=True)
    top=horses[:5]; nums=[no(x.get("horse_no")) for x in top]
    if len(set(nums))<3:return None
    top_set=set(nums); first_set=set(nums[:2]); candidates=[]
    for x in r.get("candidate_combinations") or []:
        kind=x.get("bet_type")
        if kind not in limits:continue
        ns=[no(v) for v in x.get("numbers") or []]
        if len(ns)!=REQUIRED[kind] or len(set(ns))!=len(ns) or any(v not in top_set for v in ns):continue
        if kind in ("exacta","trifecta") and ns[0] not in first_set:continue
        odds=float(x.get("odds") or 0);prob=float(x.get("probability") or 0)
        if odds<=0 or prob<=0:continue
        candidates.append({"kind":kind,"type":TYPES[kind],"comb":"-".join(ns),"odds":odds,"probability":prob,"ev":odds*prob})
    eligible=[]
    for kind,label in TYPES.items():
        rows=sorted([x for x in candidates if x["kind"]==kind and x["ev"]>=max(1.05,limits[kind])],key=lambda x:(-x["probability"],-x["ev"]))
        if rows:eligible.append({"kind":kind,"type":label,"rows":rows[:20],"avg":sum(x["ev"] for x in rows[:5])/min(5,len(rows)),"best":rows[0]})
    eligible.sort(key=lambda x:(-x["avg"],-x["best"]["probability"]))
    selected=eligible[0] if eligible else None
    allocation=[];allocated=0
    if selected:
        rows=selected["rows"];target=min(2000,len(rows)*100)
        ws=[max(.0001,(x["ev"]-1)*math.sqrt(max(x["probability"],.000001))) for x in rows]; total=sum(ws)
        allocation=[{**x,"stake":math.floor((target*w/total)/100)*100} for x,w in zip(rows,ws)] if total else []
        allocation=[x for x in allocation if x["stake"]>=100];allocated=sum(x["stake"] for x in allocation);i=0
        while allocation and allocated+100<=target:
            allocation[i]["stake"]+=100;allocated+=100;i=(i+1)%len(allocation)
    predictions=r.get("predictions") or [];odds_rows=r.get("odds") or []
    model_ok=bool(r.get("model_version")) and len(predictions)>=3
    matched=len(candidates)
    bet=bool(model_ok and odds_rows and limits and selected and allocation)
    if not model_ok: reason,code="学習モデルまたは予測データ未準備","MODEL_NOT_READY"
    elif not odds_rows: reason,code="締切前オッズ未取得","ODDS_NOT_CAPTURED"
    elif not limits: reason,code="券種別の検証済みEV基準が未成立","NO_VALIDATED_TICKET_THRESHOLD"
    elif not matched: reason,code="候補買い目のオッズ未照合","TICKET_ODDS_NOT_MATCHED"
    else: reason,code="検証済みEV基準を満たす候補なし","EV_BELOW_VALIDATED_THRESHOLD"
    return {"version":1,"createdAt":r.get("captured_at"),"date":r.get("date"),"raceId":r.get("race_key"),"venue":r.get("venue"),"raceNo":r.get("race"),"decision":"BET" if bet else "SKIP","skipReason":"" if bet else reason,"skipReasonCode":"BET" if bet else code,"totalStake":allocated if bet else 0,"tickets":[{"type":x["type"],"comb":x["comb"],"stake":x["stake"],"odds":x["odds"],"probability":x["probability"],"ev":x["ev"]} for x in allocation] if bet else [],"modelLoaded":model_ok,"oddsCaptured":bool(odds_rows),"oddsRows":len(odds_rows),"oddsMatchedCandidates":matched,"validatedBetTypes":len(limits),"autoGenerated":True,"modelVersion":r.get("model_version"),"captureTime":r.get("captured_at")}

def main():
    now=dt.datetime.now(ZoneInfo("Asia/Tokyo"));limits=thresholds()
    plans=[p for r in latest_snapshots(now) if (p:=make_plan(r,limits))]
    plans.sort(key=lambda x:(x["date"],x["venue"],int(x["raceNo"] or 0)))
    out={"schema_version":1,"generated_at":now.isoformat(),"source":"official_nar_preoff_snapshots","plans":plans}
    (DATA/"auto-ticket-plans.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"active_races":len(plans),"bet_plans":sum(p["decision"]=="BET" for p in plans),"skip_plans":sum(p["decision"]=="SKIP" for p in plans),"validated_bet_types":len(limits)},ensure_ascii=False))
if __name__=="__main__":main()
