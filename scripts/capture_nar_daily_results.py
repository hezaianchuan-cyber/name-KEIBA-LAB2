#!/usr/bin/env python3
"""Archive official daily NAR finishing positions before the daily ZIP rolls over."""
from __future__ import annotations
import argparse, csv, datetime as dt, hashlib, io, json, re, unicodedata, zipfile
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
JST=ZoneInfo("Asia/Tokyo")
VENUES=("帯広","門別","盛岡","水沢","浦和","船橋","大井","川崎","金沢","笠松","名古屋","園田","姫路","高知","佐賀")

def kind_for(filename):
    base=unicodedata.normalize("NFKC",filename.replace("\\","/").split("/")[-1]).lower()
    if re.search(r"racelist|race.?list|レース一覧",base): return "races"
    if re.search(r"horselist|horse.?list|出馬表",base): return "horses"
    return None

def decode(raw):
    candidates=[]
    for enc in ("utf-8-sig","cp932"):
        try: candidates.append(raw.decode(enc))
        except UnicodeDecodeError: pass
    if not candidates: return raw.decode("cp932",errors="replace")
    return max(candidates,key=lambda s:sum(s.count(v) for v in VENUES))

def digits(value): return re.sub(r"\D","",str(value or ""))
def number(value):
    raw=re.sub(r"[^0-9.]","",str(value or ""))
    try: return float(raw) if raw else None
    except ValueError: return None
def finish(value):
    raw=str(value or "").strip()
    return int(raw) if raw.isdigit() and int(raw)>0 else None

def extract(path):
    blob=path.read_bytes()
    races={}; horses={}
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for member in z.namelist():
            kind=kind_for(member)
            if kind not in ("races","horses") or member.endswith("/"): continue
            for row in csv.reader(io.StringIO(decode(z.read(member)))):
                if kind=="races":
                    if len(row)<27: continue
                    venue=str(row[0]).strip(); date=digits(row[1]); race=digits(row[2])
                    if not venue or len(date)!=8 or not race: continue
                    key=(venue,date,str(int(race)))
                    races[key]={"race_name":str(row[5] or ""),"surface":str(row[21] or ""),
                      "distance_m":int(number(row[23]) or 0) or None,"going":str(row[25] or ""),
                      "field_size":int(number(row[26]) or 0),"post_raw":str(row[3] or "")}
                else:
                    if len(row)<=31: continue
                    venue=str(row[0]).strip(); date=digits(row[1]); race=digits(row[2]); no=digits(row[5])
                    name=str(row[6] or "").strip()
                    if not venue or len(date)!=8 or not race or not no or not name: continue
                    key=(venue,date,str(int(race)))
                    horses.setdefault(key,[]).append({
                      "horse_no":int(no),"horse_name":name,
                      "frame_no":int(number(row[3]) or 0) or None,
                      "sex":str(row[7] or ""),"age":int(number(row[8]) or 0) or None,
                      "sire":str(row[11] or ""),"jockey":str(row[14] or ""),
                      "carried_weight":number(row[16]),"finish_position":finish(row[31])})
    checksum=hashlib.sha256(blob).hexdigest()
    captured=dt.datetime.now(JST).isoformat()
    out={}
    for key,field in horses.items():
        venue,date,race=key
        finished=sum(1 for h in field if h["finish_position"] is not None)
        if not finished: continue
        race_meta=races.get(key,{})
        post=None
        raw=digits(race_meta.get("post_raw"))
        if len(raw) in (3,4):
            raw=raw.zfill(4)
            post=f"{date[:4]}-{date[4:6]}-{date[6:8]}T{raw[:2]}:{raw[2:]}:00+09:00"
        record={"race_id":"|".join(key),"race_date":f"{date[:4]}-{date[4:6]}-{date[6:8]}",
          "venue":venue,"race_number":int(race),"race_name":race_meta.get("race_name"),
          "surface":race_meta.get("surface"),"distance_m":race_meta.get("distance_m"),
          "going":race_meta.get("going"),"scheduled_post_at":post,
          "field_size":race_meta.get("field_size") or len(field),"captured_at":captured,
          "source_checksum":checksum,"runners":field}
        out.setdefault(date,[]).append(record)
    return out,checksum

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive",type=Path,default=ROOT/"data"/"latest-race.zip")
    p.add_argument("--output-dir",type=Path,default=ROOT/"data"/"daily-results")
    a=p.parse_args()
    if not a.archive.is_file(): raise SystemExit(f"missing official daily archive: {a.archive}")
    records,checksum=extract(a.archive)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    written=0
    for date,rows in records.items():
        target=a.output_dir/f"{date}.jsonl"
        payload="".join(json.dumps(x,ensure_ascii=False,allow_nan=False)+"\n" for x in sorted(rows,key=lambda x:(x["venue"],x["race_number"])))
        if not target.exists() or target.read_text(encoding="utf-8")!=payload:
            target.write_text(payload,encoding="utf-8"); written+=1
    print(json.dumps({"archive_sha256":checksum,"dates":len(records),"races_with_results":sum(map(len,records.values())),
      "files_changed":written,"result_rows":sum(sum(1 for h in r["runners"] if h["finish_position"] is not None) for rows in records.values() for r in rows)},ensure_ascii=False,indent=2))

if __name__=="__main__": main()
