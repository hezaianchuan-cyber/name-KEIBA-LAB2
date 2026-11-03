#!/usr/bin/env python3
import csv, io, json, urllib.parse, urllib.request, zipfile
from datetime import datetime, timezone
BASE="https://www.keiba.go.jp/KeibaWeb/DataDownload/OddsDataDownload"
UA="Mozilla/5.0 (compatible; KEIBA-LAB-odds-probe/1.0)"
tests=[]
for year, month in [(2026,m) for m in range(3,12)]:
  for day in (1,11,21):
    variants=[
      {"type":"monthly","k_year":year,"k_month":month,"k_day":day},
      {"type":"monthly","k_year":year,"k_month":month,"k_term":day},
      {"type":"monthly","k_year":year,"k_month":month,"k_from":day},
      {"type":"monthly","k_year":year,"k_month":month,"k_monthly_day":day},
    ]
    for params in variants:
      url=BASE+"?"+urllib.parse.urlencode(params)
      req=urllib.request.Request(url,headers={"User-Agent":UA,"Referer":"https://www.keiba.go.jp/"})
      try:
        with urllib.request.urlopen(req,timeout=25) as res: raw=res.read()
        item={"params":params,"status":"ok","bytes":len(raw),"zip":raw.startswith(b"PK")}
        if raw.startswith(b"PK"):
          with zipfile.ZipFile(io.BytesIO(raw)) as z:
            item["members"]=z.namelist()[:12]
            csvs=[n for n in z.namelist() if n.lower().endswith(".csv")]
            if csvs:
              data=z.read(csvs[0])
              for enc in ("cp932","utf-8-sig","shift_jis"):
                try:
                  rows=list(csv.reader(io.StringIO(data.decode(enc)))); break
                except UnicodeDecodeError: continue
              item["sample_file"]=csvs[0]
              item["sample_rows"]=rows[:3]
        else:
          item["preview"]=raw[:180].decode("utf-8",errors="replace")
        tests.append(item)
        if item["zip"]: break
      except Exception as e:
        tests.append({"params":params,"status":"error","error":str(e)[:240]})
    if any(x.get("zip") for x in tests[-4:]): break
out={"checked_at":datetime.now(timezone.utc).isoformat(),"endpoint":BASE,"tests":tests}
open("odds-monthly-probe.json","w",encoding="utf-8").write(json.dumps(out,ensure_ascii=False,indent=2))
print(json.dumps(out,ensure_ascii=False))
