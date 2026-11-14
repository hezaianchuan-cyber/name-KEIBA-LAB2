#!/usr/bin/env python3
"""Export a timestamp-safe, race-level JSONL dataset from the research SQLite DB."""
from __future__ import annotations
import argparse, datetime as dt, json, sqlite3
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def parse_iso(value):
    try: return dt.datetime.fromisoformat(value) if value else None
    except (TypeError,ValueError): return None

def choose_trio_snapshot(predicted_at,scheduled_post_at,snapshots,model_name,model_version=''):
    predicted=parse_iso(predicted_at); post=parse_iso(scheduled_post_at)
    is_a45=('A45' in model_name.upper()) or ('LAMBDARANK-A45' in model_version.upper())
    if predicted is None or post is None: return None
    candidates=[]
    for row in snapshots:
        captured=parse_iso(row["captured_at"])
        if captured is None or captured>=post: continue
        if is_a45 and captured<=predicted: continue
        if not is_a45 and captured>predicted: continue
        candidates.append((captured,row))
    return max(candidates,key=lambda x:x[0])[1] if candidates else None

def build_records(db,race_type="flat"):
    db.row_factory=sqlite3.Row
    groups=db.execute("""SELECT DISTINCT mp.race_id,mp.model_name,mp.model_version,mp.predicted_at,mp.scheduled_post_at,
      r.race_date,r.venue,r.race_type,r.race_number FROM model_predictions mp JOIN races r ON r.race_id=mp.race_id
      WHERE (?='all' OR r.race_type=?) ORDER BY r.race_date,mp.race_id,mp.model_name,mp.predicted_at""",(race_type,race_type)).fetchall()
    output=[]; diag={"candidate_groups":len(groups),"exported":0,"missing_results":0,"missing_trio_odds":0,"missing_trio_payout":0,"partial_result_groups":0}
    for g in groups:
        rid=g["race_id"]
        preds=db.execute("""SELECT horse_no,rank_position,score,features_json,model_name,model_version FROM model_predictions
          WHERE race_id=? AND model_name=? AND model_version=? AND predicted_at=? ORDER BY rank_position,horse_no""",(rid,g["model_name"],g["model_version"],g["predicted_at"])).fetchall()
        results=db.execute("""SELECT horse_no,finish_position,finish_time_seconds,last3f_seconds FROM official_results
          WHERE race_id=? ORDER BY finish_position,horse_no""",(rid,)).fetchall()
        payouts=db.execute("""SELECT combination_key,payout_yen,stake_unit_yen FROM payouts
          WHERE race_id=? AND bet_type='trio' ORDER BY combination_key""",(rid,)).fetchall()
        snapshots=db.execute("SELECT DISTINCT captured_at FROM odds_snapshots WHERE race_id=? AND bet_type='trio' ORDER BY captured_at",(rid,)).fetchall()
        snap=choose_trio_snapshot(g["predicted_at"],g["scheduled_post_at"],snapshots,g["model_name"],g["model_version"])
        if not results: diag["missing_results"]+=1
        if len(results)<len(preds): diag["partial_result_groups"]+=1
        if not snap: diag["missing_trio_odds"]+=1; continue
        quotes=db.execute("""SELECT combination_key,odds FROM odds_snapshots WHERE race_id=? AND bet_type='trio' AND captured_at=?
          ORDER BY combination_key""",(rid,snap["captured_at"])).fetchall()
        if not payouts: diag["missing_trio_payout"]+=1
        output.append({
          "race_id":rid,"race_date":g["race_date"],"venue":g["venue"],"race_type":g["race_type"],"race_number":g["race_number"],
          "model_name":g["model_name"],"model_version":g["model_version"],"predicted_at":g["predicted_at"],
          "scheduled_post_at":g["scheduled_post_at"],"odds_captured_at":snap["captured_at"],
          "odds_timing_policy":"strictly_after_prediction" if ("A45" in g["model_name"].upper() or "LAMBDARANK-A45" in g["model_version"].upper()) else "at_or_before_prediction",
          "predictions":[{**dict(x),"features":json.loads(x["features_json"])} for x in preds],
          "official_results":[dict(x) for x in results],"trio_odds":[dict(x) for x in quotes],
          "trio_payouts":[dict(x) for x in payouts],
          "evaluation_status":"complete" if results and len(results)>=len(preds) and quotes and payouts else "partial"})
    diag["exported"]=len(output); return output,diag

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--db",type=Path,default=ROOT/"data"/"keibalab-research.sqlite")
    p.add_argument("--out",type=Path,default=ROOT/"data"/"research-backtest-dataset.jsonl")
    p.add_argument("--race-type",choices=("flat","banei","all"),default="flat"); a=p.parse_args()
    db=sqlite3.connect(a.db)
    try: records,report=build_records(db,a.race_type)
    finally: db.close()
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text("".join(json.dumps(x,ensure_ascii=False,allow_nan=False)+"\n" for x in records),encoding="utf-8")
    report.update({"race_type":a.race_type,"output":str(a.out),"created_at":dt.datetime.now(dt.timezone.utc).isoformat()})
    a.out.with_suffix(".report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
