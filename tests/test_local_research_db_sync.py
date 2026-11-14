import csv, hashlib, io, json, sqlite3, sys, unittest, zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import sync_research_db_local as sync
import export_research_backtest_dataset as exporter

class LocalResearchDBTests(unittest.TestCase):
    def test_odds_normalization(self):
        self.assertEqual(sync.normalize_bet_type("3連複"),("trio",3,True))
        self.assertEqual(sync.normalize_numbers({"n1":"11","n2":"2","n3":"5"},3,True),["2","5","11"])
        self.assertIsNone(sync.normalize_numbers({"n1":"1","n2":"1","n3":"3"},3,True))

    def test_prediction_lock(self):
        record={k:None for k in sync.PREDICTION_LOCK_FIELDS}
        record.update({"race_key":"佐賀|20261114|1","venue":"佐賀","date":"20261114","race":"1",
          "scheduled_post_time":"2026-11-14T15:00:00+09:00","captured_at":"2026-11-14T14:00:00+09:00",
          "minutes_to_post":60,"capture_policy":{},"model_version":"test","model_temperature":0.18,
          "predictions":[],"combination_predictions":[],"candidate_combinations":[],"model_g_version":"g1",
          "model_g_predictions":[],"model_g_candidate_combinations":[],"model_g_strategy":{},
          "candidate_universe_policy":"test","race_readiness":{},"odds":[]})
        payload={k:record.get(k) for k in sync.PREDICTION_LOCK_FIELDS}
        raw=json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
        record["prediction_lock"]={"version":"nar-prediction-lock-v1","algorithm":"sha256","digest":hashlib.sha256(raw).hexdigest()}
        self.assertTrue(sync.verify_prediction_lock(record))
        record["predictions"]=[{"horse_no":"1"}]
        self.assertFalse(sync.verify_prediction_lock(record))


    def test_daily_archive_imports_official_finish_positions(self):
        db=sqlite3.connect(":memory:")
        db.execute("PRAGMA foreign_keys=ON")
        db.executescript((ROOT/"research_db"/"sqlite_schema_v1.sql").read_text())
        race=[""]*27
        race[0]="佐賀"; race[1]="20261114"; race[2]="1"; race[3]="1430"; race[5]="テスト競走"
        race[21]="ダート"; race[23]="1400"; race[25]="良"; race[26]="1"
        horse=[""]*32
        horse[0]="佐賀"; horse[1]="20261114"; horse[2]="1"; horse[3]="1"; horse[5]="1"
        horse[6]="テスト馬"; horse[7]="牡"; horse[8]="4"; horse[11]="父"; horse[14]="騎手"
        horse[16]="56"; horse[31]="1"
        buf=io.BytesIO()
        with zipfile.ZipFile(buf,"w") as z:
            z.writestr("RaceList.csv",",".join(race)+"\n")
            z.writestr("HorseList.csv",",".join(horse)+"\n")
        result=sync.import_daily_race_archive(db,buf.getvalue(),"test-batch")
        self.assertEqual(result["results"],1)
        self.assertEqual(db.execute("SELECT finish_position FROM official_results").fetchone()[0],1)
        db.close()

    def test_daily_payout_report_imports_trio(self):
        db=sqlite3.connect(":memory:")
        db.execute("PRAGMA foreign_keys=ON")
        db.executescript((ROOT/"research_db"/"sqlite_schema_v1.sql").read_text())
        report={"generated_at":"2026-11-14T22:00:00+09:00","days":[{"races":[
          {"race_id":"佐賀|20261114|1","payouts":{"trio":[{"numbers":["11","2","5"],"pay_100":1250}]}}
        ]}]}
        result=sync.import_daily_payout_report(db,json.dumps(report,ensure_ascii=False).encode(),"batch")
        row=db.execute("SELECT combination_key,payout_yen FROM payouts WHERE bet_type='trio'").fetchone()
        self.assertEqual(row,("2-5-11",1250))
        self.assertEqual(result["payouts"],1)
        db.close()

    def test_backtest_timestamp_policy(self):
        snaps=[{"captured_at":"2026-11-14T14:01:00+09:00"},{"captured_at":"2026-11-14T14:02:00+09:00"}]
        pred="2026-11-14T14:00:00+09:00"; post="2026-11-14T15:00:00+09:00"
        self.assertEqual(exporter.choose_trio_snapshot(pred,post,snaps,"LambdaRank-A45")["captured_at"],snaps[1]["captured_at"])
        self.assertIsNone(exporter.choose_trio_snapshot(pred,post,snaps,"LiveModel"))
        same=[{"captured_at":pred}]
        self.assertEqual(exporter.choose_trio_snapshot(pred,post,same,"LiveModel")["captured_at"],pred)

    def test_exporter_separates_data_and_marks_complete(self):
        db=sqlite3.connect(":memory:")
        db.executescript((ROOT/"research_db"/"sqlite_schema_v1.sql").read_text())
        stamp="2026-11-14T14:00:00+09:00"; post="2026-11-14T15:00:00+09:00"; rid="佐賀|20261114|1"
        db.execute("""INSERT INTO races(race_id,race_date,venue,race_number,scheduled_post_at,field_size,source,first_seen_at,last_seen_at)
          VALUES(?,?,?,?,?,?,?,?,?)""",(rid,"2026-11-14","佐賀",1,post,3,"test",stamp,stamp))
        for no in (1,2,3):
            db.execute("""INSERT INTO runners(race_id,horse_no,horse_name,source,first_seen_at,last_seen_at) VALUES(?,?,?,?,?,?)""",(rid,no,f"馬{no}","test",stamp,stamp))
            db.execute("""INSERT INTO model_predictions(prediction_id,race_id,horse_no,model_name,model_version,feature_schema_version,rank_position,score,features_json,predicted_at,scheduled_post_at,source)
              VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",(f"p{no}",rid,no,"LiveModel","v1","v1",no,1/no,"{}",stamp,post,"test"))
            db.execute("INSERT INTO official_results(race_id,horse_no,finish_position,confirmed_at,source) VALUES(?,?,?,?,?)",(rid,no,no,"2026-11-14","test"))
        db.execute("""INSERT INTO odds_snapshots(snapshot_id,race_id,bet_type,combination_key,odds,captured_at,source) VALUES(?,?,?,?,?,?,?)""",("o1",rid,"trio","1-2-3",3.4,stamp,"test"))
        db.execute("INSERT INTO payouts(race_id,bet_type,combination_key,payout_yen,confirmed_at,source) VALUES(?,?,?,?,?,?)",(rid,"trio","1-2-3",340,"2026-11-14","test"))
        records,report=exporter.build_records(db,"flat")
        self.assertEqual(len(records),1)
        self.assertEqual(records[0]["evaluation_status"],"complete")
        self.assertEqual(len(records[0]["trio_odds"]),1)
        self.assertEqual(len(records[0]["trio_payouts"]),1)
        db.close()

if __name__=="__main__": unittest.main()
