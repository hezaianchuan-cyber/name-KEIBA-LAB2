#!/usr/bin/env python3
"""Sync committed NAR pre-off snapshots into a durable local SQLite database.

Requires only Python's standard library and Git. Run from a clone of the
research/keibalab-db-foundation branch. It reads origin/main odds snapshots
and the A45 research branch; it never writes to GitHub or production systems.
"""
from __future__ import annotations
import argparse, datetime as dt, gzip, hashlib, io, json, re, sqlite3, subprocess, unicodedata
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_DB=ROOT/"data"/"keibalab-research.sqlite"
MAIN_REF="refs/remotes/origin/main"
A45_REF="refs/remotes/origin/research/lambdarank-a45-preoff-capture"
PREDICTION_LOCK_FIELDS=("race_key","venue","date","race","scheduled_post_time","captured_at","minutes_to_post","capture_policy","model_version","model_temperature","predictions","combination_predictions","candidate_combinations","model_g_version","model_g_predictions","model_g_candidate_combinations","model_g_strategy","candidate_universe_policy","race_readiness","odds")
BET_TYPES={"単勝":("win",1,False),"複勝":("place",1,False),"馬連":("quinella",2,True),"ワイド":("wide",2,True),"馬単":("exacta",2,False),"3連複":("trio",3,True),"三連複":("trio",3,True),"3連単":("trifecta",3,False),"三連単":("trifecta",3,False)}

def run_git(*args,check=True):
    r=subprocess.run(["git",*args],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=False)
    if check and r.returncode: raise RuntimeError(r.stderr.decode("utf-8","replace").strip())
    return r

def sha_id(*parts): return hashlib.sha256("|".join(str(x) for x in parts).encode()).hexdigest()
def positive(v):
    try: n=float(v)
    except (TypeError,ValueError): return None
    return n if n>0 else None

def verify_prediction_lock(record):
    lock=record.get("prediction_lock") or {}
    if lock.get("version")!="nar-prediction-lock-v1" or lock.get("algorithm")!="sha256": return False
    payload={k:record.get(k) for k in PREDICTION_LOCK_FIELDS}
    raw=json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
    return lock.get("digest")==hashlib.sha256(raw).hexdigest()

def race_type(venue): return "banei" if ("帯広" in str(venue) or "ばんえい" in str(venue)) else "flat"
def normalize_bet_type(value):
    label=unicodedata.normalize("NFKC",str(value or "")).replace(" ","")
    for token,spec in BET_TYPES.items():
        if token in label: return spec
    return None

def normalize_numbers(item,count,unordered):
    values=[]
    for key in ("n1","n2","n3"):
        raw=re.sub(r"\D","",str(item.get(key) or ""))
        if raw: values.append(str(int(raw)))
    if len(values)!=count or len(set(values))!=count: return None
    return sorted(values,key=int) if unordered else values

def archive_paths(ref):
    r=run_git("ls-tree","-r","--full-tree",ref,"--","data/odds-snapshots")
    out=[]
    for line in r.stdout.decode().splitlines():
        parts=line.split("\t",1)
        if len(parts)!=2: continue
        meta,path=parts; fields=meta.split()
        if len(fields)==3 and fields[1]=="blob" and path.endswith(".jsonl.gz"): out.append((path,fields[2]))
    return out

def read_blob(ref,path): return run_git("show",f"{ref}:{path}").stdout
def jsonl_lines(raw):
    for no,line in enumerate(raw.decode("utf-8").splitlines(),1):
        if line.strip():
            try: yield json.loads(line)
            except json.JSONDecodeError as exc: raise ValueError(f"invalid JSON line {no}: {exc}") from exc

def insert_race(db,race_id,venue,date_raw,race_no,scheduled,field_size,source,seen_at):
    date_iso=dt.datetime.strptime(date_raw,"%Y%m%d").date().isoformat()
    db.execute("""INSERT INTO races(race_id,race_date,venue,race_type,race_number,scheduled_post_at,field_size,source,first_seen_at,last_seen_at)
      VALUES(?,?,?,?,?,?,?,?,?,?) ON CONFLICT(race_id) DO UPDATE SET
      scheduled_post_at=COALESCE(excluded.scheduled_post_at,races.scheduled_post_at),
      field_size=MAX(COALESCE(races.field_size,0),COALESCE(excluded.field_size,0)),last_seen_at=excluded.last_seen_at""",
      (race_id,date_iso,venue,race_type(venue),int(race_no),scheduled,field_size,source,seen_at,seen_at))

def insert_prediction_set(db,race_id,record,predictions,model_name,model_version,batch_id,source):
    if not predictions: return 0
    ordered=sorted(predictions,key=lambda p:(float(p.get("score") or p.get("win_probability") or 0),int(p.get("horse_no") or 0)),reverse=True)
    rank_map={str(p.get("horse_no")):i+1 for i,p in enumerate(ordered)}
    inserted=0
    for p in predictions:
        no=int(p["horse_no"]); features=p.get("features") or {}; name=str(p.get("name") or f"horse-{no}")
        db.execute("""INSERT INTO runners(race_id,horse_no,horse_name,frame_no,age,jockey,carried_weight,sire,source,first_seen_at,last_seen_at)
          VALUES(?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(race_id,horse_no) DO UPDATE SET horse_name=excluded.horse_name,last_seen_at=excluded.last_seen_at""",
          (race_id,no,name,p.get("frame_no",features.get("frame_no")),features.get("age"),features.get("jockey"),positive(features.get("carried_weight")),features.get("sire"),source,record["captured_at"],record["captured_at"]))
        version=str(model_version or "unknown"); pred_id=sha_id(race_id,no,model_name,version,record["captured_at"])
        score=p.get("score")
        if score is None: score=p.get("win_probability")
        if score is None: score=0.0
        cur=db.execute("""INSERT OR IGNORE INTO model_predictions
          (prediction_id,race_id,horse_no,model_name,model_version,feature_schema_version,rank_position,score,features_json,predicted_at,scheduled_post_at,minutes_to_post,source,batch_id)
          VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
          (pred_id,race_id,no,model_name,version,"live-capture-v1",int(p.get("rank") or p.get("rank_position") or rank_map[str(no)]),float(score),json.dumps(dict(p),ensure_ascii=False,sort_keys=True),record["captured_at"],record.get("scheduled_post_time"),record.get("minutes_to_post"),source,batch_id))
        inserted+=cur.rowcount
    return inserted

def import_live_snapshot(db,record,batch_id):
    if not verify_prediction_lock(record): return {"rejected":1,"predictions":0,"odds":0}
    race_id=str(record.get("race_key") or ""); parts=race_id.split("|")
    if len(parts)!=3: return {"rejected":1,"predictions":0,"odds":0}
    venue,date_raw,race_no=parts; captured=str(record["captured_at"]); predictions=record.get("predictions") or []
    insert_race(db,race_id,venue,date_raw,race_no,record.get("scheduled_post_time"),len(predictions),"nar_preoff_capture",captured)
    pred_count=insert_prediction_set(db,race_id,record,predictions,"LiveModel",record.get("model_version"),batch_id,"nar_preoff_capture")
    pred_count+=insert_prediction_set(db,race_id,record,record.get("model_g_predictions") or [],"Model-G",record.get("model_g_version"),batch_id,"nar_preoff_capture")
    odds_count=0
    for quote in record.get("odds") or []:
        spec=normalize_bet_type(quote.get("type"))
        if not spec: continue
        kind,count,unordered=spec; numbers=normalize_numbers(quote,count,unordered); odds=positive(quote.get("odds"))
        if not numbers or odds is None: continue
        combo="-".join(numbers); snap_id=sha_id(race_id,kind,combo,captured,"nar_live")
        cur=db.execute("""INSERT OR IGNORE INTO odds_snapshots(snapshot_id,race_id,bet_type,combination_key,odds,captured_at,source,batch_id)
          VALUES(?,?,?,?,?,?,?,?)""",(snap_id,race_id,kind,combo,odds,captured,"nar_live_odds",batch_id))
        odds_count+=cur.rowcount
    return {"rejected":0,"predictions":pred_count,"odds":odds_count}

def import_a45_file(db,records,batch_id):
    inserted=0
    for record in records:
        race_id=str(record.get("race_id") or ""); parts=race_id.split("|")
        if len(parts)!=3: continue
        venue,date_raw,race_no=parts; predictions=record.get("predictions") or []
        if not predictions: continue
        captured=str(record["captured_at"])
        insert_race(db,race_id,venue,date_raw,race_no,record.get("scheduled_post_time"),len(predictions),"research_a45_prediction",captured)
        inserted+=insert_prediction_set(db,race_id,record,predictions,"LambdaRank-A45",record.get("model_version"),batch_id,"research_a45_frozen_prediction")
    return inserted

def start_batch(db,source,path,blob_sha):
    batch_id=sha_id(source,path,blob_sha)
    row=db.execute("SELECT status FROM ingestion_batches WHERE batch_id=?",(batch_id,)).fetchone()
    if row and row[0]=="completed": return batch_id,False
    db.execute("""INSERT OR IGNORE INTO ingestion_batches(batch_id,source,started_at,status,checksum,notes) VALUES(?,?,?,?,?,?)""",
      (batch_id,source,dt.datetime.now(dt.timezone.utc).isoformat(),"running",blob_sha,path))
    return batch_id,True

def finish_batch(db,batch_id,read,inserted,rejected):
    db.execute("""UPDATE ingestion_batches SET completed_at=?,status='completed',rows_read=?,rows_inserted=?,rows_rejected=? WHERE batch_id=?""",
      (dt.datetime.now(dt.timezone.utc).isoformat(),read,inserted,rejected,batch_id))

def sync(db):
    if not (ROOT/".git").exists(): raise RuntimeError("Run this script inside a Git clone of the research branch.")
    run_git("fetch","--no-tags","origin","+refs/heads/main:refs/remotes/origin/main")
    a45=run_git("fetch","--no-tags","origin",f"+refs/heads/research/lambdarank-a45-preoff-capture:{A45_REF}",check=False)
    summary={"odds_archives_seen":0,"odds_archives_imported":0,"live_snapshots_read":0,"live_predictions_inserted":0,"odds_quotes_inserted":0,"a45_records_read":0,"a45_predictions_inserted":0,"rejected_snapshots":0}
    for path,blob_sha in archive_paths(MAIN_REF):
        summary["odds_archives_seen"]+=1; batch_id,do_import=start_batch(db,"main_odds_archive",path,blob_sha)
        if not do_import: continue
        raw=read_blob(MAIN_REF,path); read_count=pred_count=odds_count=rejected=0
        with gzip.GzipFile(fileobj=io.BytesIO(raw),mode="rb") as stream:
            for record in jsonl_lines(stream.read()):
                read_count+=1; result=import_live_snapshot(db,record,batch_id)
                rejected+=result["rejected"]; pred_count+=result["predictions"]; odds_count+=result["odds"]
        finish_batch(db,batch_id,read_count,pred_count+odds_count,rejected); db.commit()
        summary["odds_archives_imported"]+=1; summary["live_snapshots_read"]+=read_count
        summary["live_predictions_inserted"]+=pred_count; summary["odds_quotes_inserted"]+=odds_count
        summary["rejected_snapshots"]+=rejected
    if a45.returncode==0:
        path="data/research-a45-predictions.jsonl"; blob=run_git("rev-parse",f"{A45_REF}:{path}",check=False)
        if blob.returncode==0:
            blob_sha=blob.stdout.decode().strip(); batch_id,do_import=start_batch(db,"a45_prediction_archive",path,blob_sha)
            if do_import:
                records=list(jsonl_lines(read_blob(A45_REF,path))); inserted=import_a45_file(db,records,batch_id)
                finish_batch(db,batch_id,len(records),inserted,0); db.commit()
                summary["a45_records_read"]=len(records); summary["a45_predictions_inserted"]=inserted
    integrity=db.execute("PRAGMA integrity_check").fetchone()[0]; fk=db.execute("PRAGMA foreign_key_check").fetchall()
    summary["integrity_check"]=integrity; summary["foreign_key_violations"]=len(fk)
    summary["table_counts"]={t:db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("races","runners","model_predictions","odds_snapshots","official_results","payouts")}
    if integrity!="ok" or fk: raise RuntimeError(f"SQLite integrity failure: {integrity}; FK={fk[:5]}")
    return summary

def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--db",type=Path,default=DEFAULT_DB); args=parser.parse_args()
    args.db.parent.mkdir(parents=True,exist_ok=True); db=sqlite3.connect(args.db)
    try:
        db.execute("PRAGMA foreign_keys=ON"); db.execute("PRAGMA journal_mode=WAL")
        db.executescript((ROOT/"research_db"/"sqlite_schema_v1.sql").read_text(encoding="utf-8"))
        result=sync(db); db.commit(); print(json.dumps(result,ensure_ascii=False,indent=2))
    finally: db.close()
if __name__=="__main__": main()
