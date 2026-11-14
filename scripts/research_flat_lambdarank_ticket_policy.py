#!/usr/bin/env python3
"""Research-only flat LambdaRank vs LAB ranking and trio ticket-policy benchmark.

Banei is excluded by venue only. Race-name-only Banei sponsorship/title text is
retained as ordinary flat racing. No production code is modified.
"""
from __future__ import annotations
import datetime as dt
import itertools
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import nar_backtest as nb
import nar_ml_challenger_backtest as suite

OUT = ROOT / "data" / "flat-lambdarank-ticket-policy.json"
EXCLUDED_BODY_WEIGHT = {
    "body_weight", "body_weight_change", "body_weight_delta_pct",
    "race_rel_body_weight_delta_pct",
}
BANEI_VENUE = re.compile(r"帯広|ばんえい", re.IGNORECASE)
POLICIES = {
    "top3_single": "Predicted ranks 1-3 as one trio ticket (100 JPY)",
    "axis_plus4": "Rank 1 axis with every pair from ranks 2-5 (6 tickets, 600 JPY)",
    "axis_plus5": "Rank 1 axis with every pair from ranks 2-6 (10 tickets, 1,000 JPY)",
    "top5_box": "All combinations among ranks 1-5 (10 tickets, 1,000 JPY)",
}

def load_flat():
    races, horses, payouts, downloads = nb.load_archives()
    banei = {k:r for k,r in races.items() if BANEI_VENUE.search(str(r.get("venue") or ""))}
    flat = {k:r for k,r in races.items() if k not in banei}
    flat_horses = {k:v for k,v in horses.items() if k in flat}
    flat_payouts = {k:v for k,v in payouts.items() if k in flat}
    name_only = {k:r for k,r in flat.items()
                 if BANEI_VENUE.search(str(r.get("name") or ""))}
    if not flat or any(BANEI_VENUE.search(str(r.get("venue") or "")) for r in flat.values()):
        raise RuntimeError("Venue-only flat filter failed")
    audit = {
        "source_races": len(races), "excluded_banei_venue_races": len(banei),
        "flat_races": len(flat), "flat_name_only_banei_word_races_retained": len(name_only),
        "excluded_venues": sorted({str(r.get("venue") or "") for r in banei.values()}),
        "filter": "Venue only; title/sponsorship mentions do not remove a flat race.",
        "download_months": len(downloads),
    }
    print("CORRECTED_FLAT_SCOPE " + json.dumps(audit, ensure_ascii=False), flush=True)
    return flat, flat_horses, flat_payouts, audit

def percentile(xs, q):
    ys=sorted(xs)
    if not ys: return None
    pos=(len(ys)-1)*q; lo=int(pos); hi=min(lo+1,len(ys)-1); f=pos-lo
    return ys[lo]*(1-f)+ys[hi]*f

def bootstrap_roi(returns, stakes, seed=42, iterations=2000):
    if len(returns)<20: return {"low":None,"high":None,"reason":"fewer_than_20_races"}
    rng=random.Random(seed); n=len(returns); vals=[]
    for _ in range(iterations):
        ix=[rng.randrange(n) for _ in range(n)]
        st=sum(stakes[i] for i in ix)
        if st: vals.append(100*sum(returns[i] for i in ix)/st)
    return {"low":round(percentile(vals,.025),3),"high":round(percentile(vals,.975),3),
            "method":"race_cluster_bootstrap","iterations":iterations}

def tickets(policy, ranked):
    nos=[str(x["horse_no"]) for x in ranked]
    if policy=="top3_single": return [nos[:3]] if len(nos)>=3 else []
    if policy=="axis_plus4": return [[nos[0],*pair] for pair in itertools.combinations(nos[1:5],2)] if len(nos)>=5 else []
    if policy=="axis_plus5": return [[nos[0],*pair] for pair in itertools.combinations(nos[1:6],2)] if len(nos)>=6 else []
    if policy=="top5_box": return [list(c) for c in itertools.combinations(nos[:5],3)] if len(nos)>=5 else []
    raise ValueError(policy)

def trio_payout_exists(p):
    if not p: return False
    idx=p.get("_index") or {}
    return bool(idx.get("trio") or p.get("trio"))

def evaluate_ticket_policy(scored_groups, payouts):
    output={}
    race_vectors={}
    for model_name in ("LAB","LambdaRank"):
        output[model_name]={}; race_vectors[model_name]={}
        for policy in POLICIES:
            returns=[]; stakes=[]; hits=0; tickets_total=0
            for group, scores in scored_groups:
                if len(group)<5: continue
                parts=str(group[0]["race_key"]).split("|")
                if len(parts)!=3: continue
                payout=payouts.get(nb.race_key(parts[0],parts[1],parts[2]))
                if not trio_payout_exists(payout): continue
                order=sorted(range(len(group)),key=lambda i:(-float(scores[model_name][i]),int(group[i]["horse_no"])))
                ranked=[group[i] for i in order]
                picks=tickets(policy,ranked)
                if not picks: continue
                ret=sum(nb.payout_for("trio",ticket,payout) for ticket in picks)
                stake=100*len(picks)
                returns.append(ret); stakes.append(stake); tickets_total+=len(picks); hits+=int(ret>0)
            st=sum(stakes); rt=sum(returns); n=len(returns)
            output[model_name][policy]={
                "races":n,"hit_races":hits,"hit_rate_pct":round(100*hits/n,3) if n else None,
                "tickets":tickets_total,"stake_jpy":st,"return_jpy":rt,
                "roi_pct":round(100*rt/st,3) if st else None,
                "profit_jpy":rt-st,"stake_per_race_jpy":round(st/n,2) if n else None,
                "roi_ci95":bootstrap_roi(returns,stakes),
            }
            race_vectors[model_name][policy]=(returns,stakes)
    deltas={}
    for policy in POLICIES:
        a,sa=race_vectors["LAB"][policy]; b,sb=race_vectors["LambdaRank"][policy]
        n=min(len(a),len(b))
        if n:
            # Both model vectors use the same payout-complete holdout race ordering.
            per_race=[100*(b[i]/sb[i]-a[i]/sa[i]) for i in range(n)]
            rng=random.Random(20261114); boots=[]
            for _ in range(2000):
                sample=[per_race[rng.randrange(n)] for _ in range(n)]
                boots.append(sum(sample)/n)
            deltas[policy]={"lambda_minus_lab_roi_pp":round(sum(per_race)/n,3),
                "paired_race_bootstrap_ci95_pp":[round(percentile(boots,.025),3),round(percentile(boots,.975),3)],
                "races":n}
    return {"models":output,"paired_roi_differences":deltas,"policies":POLICIES}

def main():
    suite.RAW_NUMERIC=[f for f in suite.RAW_NUMERIC if f not in EXCLUDED_BODY_WEIGHT]
    suite.RAW_FEATURES=suite.RAW_NUMERIC+suite.CATEGORICAL
    races,horses,payouts,scope=load_flat()
    rows,dates=suite.build_rows(races,horses)
    if len(dates)<100: raise RuntimeError("Insufficient dates")
    cut1=int(len(dates)*.70); cut2=int(len(dates)*.80)
    train_dates=set(dates[:cut1]); dev_dates=set(dates[cut1:cut2]); holdout_dates=set(dates[cut2:])
    def in_dates(row, ds): return dt.date.fromisoformat(row["date"]) in ds
    train=suite.group_rows([r for r in rows if in_dates(r,train_dates)])
    dev=suite.group_rows([r for r in rows if in_dates(r,dev_dates)])
    holdout=suite.group_rows([r for r in rows if in_dates(r,holdout_dates)])
    mid=len(dev)//2; early,selection=dev[:mid],dev[mid:]
    train_rows=[r for r in rows if in_dates(r,train_dates)]
    category_maps={k:{v:i for i,v in enumerate(sorted({r[k] for r in train_rows}))}
                   for k in suite.CATEGORICAL}
    features=list(suite.RAW_FEATURES)
    model=suite.fit_lambdarank(train,early,features,category_maps)
    lr_scored=[]
    for group in holdout:
        x=[[r[k] if k not in suite.CATEGORICAL else category_maps[k].get(r[k],-1)
            for k in features] for r in group]
        lr_scored.append((group,model.predict(x)))
    lab_scored=[(g,[float(r["lab_score"]) for r in g]) for g in holdout]
    lr_metrics=suite.evaluate_scores("LambdaRank",lr_scored)
    lab_metrics=suite.evaluate_scores("LAB",lab_scored)
    # Join predictions by stable race key; keep only races that have a valid winner.
    lab_by={g[0]["race_key"]:(g,s) for g,s in lab_scored}
    lr_by={g[0]["race_key"]:(g,s) for g,s in lr_scored}
    common=sorted(set(lab_by)&set(lr_by))
    paired=[(lab_by[k][0],{"LAB":lab_by[k][1],"LambdaRank":lr_by[k][1]}) for k in common]
    tickets_report=evaluate_ticket_policy(paired,payouts)
    report={
        "generated_at":dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope":scope,
        "data":{"runner_rows":len(rows),"dates":len(dates),"train_races":len(train),
                "development_races":len(dev),"development_early_stop_races":len(early),
                "development_selection_races":len(selection),"holdout_races":len(holdout)},
        "split":{"train_end":dates[cut1-1].isoformat(),"development_start":dates[cut1].isoformat(),
                 "development_end":dates[cut2-1].isoformat(),"holdout_start":dates[cut2].isoformat(),
                 "holdout_end":dates[-1].isoformat()},
        "features":{"count":len(features),"fields":features,
                    "excluded_current_race_body_weight":sorted(EXCLUDED_BODY_WEIGHT),
                    "prior_body_weight_avg_retained": "prior_body_weight_avg" in features},
        "ranking":{"LAB":lab_metrics,"LambdaRank":lr_metrics},
        "ticket_comparison":tickets_report,
        "interpretation":"Retrospective fixed-payout research only. Ticket results do not establish pre-off executable value or profitability. No production changes.",
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print("TICKET_POLICY_RESULT "+json.dumps({
        "scope":scope,"data":report["data"],"ranking":report["ranking"],
        "ticket_comparison":tickets_report
    },ensure_ascii=False),flush=True)

if __name__=="__main__": main()
