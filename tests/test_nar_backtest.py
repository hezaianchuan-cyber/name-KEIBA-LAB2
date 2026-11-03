import sys
import csv
import gzip
import io
import json
import zipfile
import tempfile
import unittest
from collections import defaultdict
from unittest import mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import nar_backtest as nb
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import capture_nar_odds as capture


class BayesianSmoothingTests(unittest.TestCase):
    def test_no_history_returns_neutral_prior(self):
        self.assertAlmostEqual(nb.smoothed_rate(0, 0), 0.50)

    def test_sparse_extreme_rate_is_shrunk_toward_neutral(self):
        self.assertAlmostEqual(nb.smoothed_rate(5, 5), 10 / 15)
        self.assertLess(nb.smoothed_rate(5, 5), 1.0)

    def test_more_observations_reduce_shrinkage(self):
        sparse = nb.smoothed_rate(4, 5)
        larger = nb.smoothed_rate(40, 50)
        self.assertGreater(sparse, 0.50)
        self.assertGreater(larger, sparse)
        self.assertAlmostEqual(larger, 45 / 60)

    def test_same_observed_rate_is_more_conservative_with_fewer_runs(self):
        sparse = nb.smoothed_rate(4, 5)
        large = nb.smoothed_rate(40, 50)
        self.assertLess(abs(sparse - 0.50), abs(0.80 - 0.50))
        self.assertGreater(abs(large - 0.50), abs(sparse - 0.50))


class RaceTimeFeatureTests(unittest.TestCase):
    def test_parse_minutes_seconds(self):
        self.assertAlmostEqual(nb.parse_race_time("1:38.7"), 98.7)

    def test_parse_seconds_without_minutes(self):
        self.assertAlmostEqual(nb.parse_race_time("59.8"), 59.8)

    def test_invalid_or_missing_time_is_neutral(self):
        self.assertIsNone(nb.parse_race_time(""))
        self.assertIsNone(nb.parse_race_time("計不"))

    def test_time_feature_uses_comparable_prior_runs(self):
        horse = {"name": "馬A", "sire": "", "jockey": "", "age": 5}
        history = {"馬A": [
            {"venue": "大井", "distance": 1600, "time_per_km": 65.0},
            {"venue": "大井", "distance": 1600, "time_per_km": 64.0},
            {"venue": "大井", "distance": 1600, "time_per_km": 62.0},
        ]}
        weights = {"form": 0.55, "venue": 0.12, "distance": 0.12,
                   "sire": 0.10, "jockey": 0.07, "age": 0.04}
        _, parts = nb.score_runner_performance(
            horse, history, defaultdict(lambda: {"runs": 0, "top3": 0}),
            defaultdict(lambda: {"runs": 0, "top3": 0}),
            {"venue": "大井", "distance": 1600}, weights,
            margin_weight=0.0, closing_weight=0.0, time_weight=0.10)
        self.assertGreater(parts["time_form"], 0.5)


class RecentHistoryAblationTests(unittest.TestCase):
    def test_recent_history_is_an_explicit_optional_prediction_input(self):
        h={"name":"馬A","sire":"父A","jockey":"騎手A","age":5}
        history={"馬A":[{"finish":1,"field_size":10,"venue":"大井","distance":1600},
                        {"finish":2,"field_size":10,"venue":"大井","distance":1600}]}
        sire=defaultdict(lambda:{"runs":0,"top3":0})
        jockey=defaultdict(lambda:{"runs":0,"top3":0})
        race={"venue":"大井","distance":1600}
        weights={"form":0.55,"venue":0.12,"distance":0.12,"sire":0.10,"jockey":0.07,"age":0.04}
        with_history=nb.score_runner(h,history,sire,jockey,race,weights,include_recent_history=True)[0]
        without_history=nb.score_runner(h,history,sire,jockey,race,weights,include_recent_history=False)[0]
        self.assertNotEqual(with_history,without_history)


class PayoutParsingTests(unittest.TestCase):
    def payout_row(self, winner="1", win_pay="250"):
        row = [""] * 53
        row[4], row[5] = winner, win_pay
        row[7], row[8] = "1", "120"
        row[10], row[11] = "2", "110"
        row[13], row[14] = "3", "130"
        row[24], row[25], row[26] = "1", "2", "400"
        row[28], row[29], row[30] = "1", "2", "600"
        row[32], row[33], row[34] = "1", "2", "200"
        row[36], row[37], row[38] = "1", "3", "250"
        row[40], row[41], row[42] = "2", "3", "300"
        row[44], row[45], row[46], row[47] = "1", "2", "3", "800"
        row[49], row[50], row[51], row[52] = "1", "2", "3", "1500"
        return row

    def test_dead_heat_rows_are_merged_and_all_bet_types_match(self):
        payouts = {}
        key = ("大井", "20261103", "10")
        nb.merge_payout_row(payouts, key, self.payout_row("1", "250"))
        nb.merge_payout_row(payouts, key, self.payout_row("2", "300"))
        payout = payouts[key]

        self.assertEqual(payout["source_rows"], 2)
        self.assertEqual(nb.payout_for("win", ["1"], payout), 250)
        self.assertEqual(nb.payout_for("win", ["2"], payout), 300)
        self.assertEqual(nb.payout_for("place", ["2"], payout), 110)
        self.assertEqual(nb.payout_for("quinella", ["2", "1"], payout), 400)
        self.assertEqual(nb.payout_for("exacta", ["1", "2"], payout), 600)
        self.assertEqual(nb.payout_for("exacta", ["2", "1"], payout), 0)
        self.assertEqual(nb.payout_for("wide", ["2", "1"], payout), 200)
        self.assertEqual(nb.payout_for("trio", ["3", "2", "1"], payout), 800)
        self.assertEqual(nb.payout_for("trifecta", ["1", "2", "3"], payout), 1500)
        self.assertEqual(nb.payout_for("trifecta", ["3", "2", "1"], payout), 0)

    def test_duplicate_payout_combinations_are_not_duplicated(self):
        payouts = {}
        key = ("大井", "20261103", "10")
        row = self.payout_row()
        nb.merge_payout_row(payouts, key, row)
        nb.merge_payout_row(payouts, key, row)
        self.assertEqual(len(payouts[key]["win"]), 1)
        self.assertEqual(payouts[key]["source_rows"], 2)

    def test_indexed_payout_lookup_matches_all_bet_types_and_dead_heats(self):
        payouts = {}
        key = ("大井", "20261103", "10")
        nb.merge_payout_row(payouts, key, self.payout_row("1", "250"))
        nb.merge_payout_row(payouts, key, self.payout_row("2", "300"))
        nb.finalize_payout_indexes(payouts)
        payout = payouts[key]
        self.assertEqual(nb.payout_for("win", ["2"], payout), 300)
        self.assertEqual(nb.payout_for("quinella", ["2", "1"], payout), 400)
        self.assertEqual(nb.payout_for("exacta", ["2", "1"], payout), 0)
        self.assertEqual(nb.payout_for("trio", ["3", "1", "2"], payout), 800)
        self.assertEqual(nb.payout_for("trifecta", ["1", "2", "3"], payout), 1500)


class OddsSnapshotParsingTests(unittest.TestCase):
    def test_race_id_normalizes_race_number(self):
        self.assertEqual(capture.race_id("大井", "20261103", "010"), ("大井", "20261103", "10"))

    def test_post_time_parses_hhmm_and_rejects_invalid_values(self):
        self.assertEqual(capture.post_time("1540").hour, 15)
        self.assertEqual(capture.post_time("940").minute, 40)
        self.assertIsNone(capture.post_time("2561"))

    def test_gzip_snapshot_appends_once_and_preserves_prior_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "2026-11-03.jsonl.gz"
            first = {"race_key":"大井|20261103|10","odds":[{"type":"単勝","odds":2.5}]}
            second = {"race_key":"大井|20261103|11","odds":[{"type":"単勝","odds":3.2}]}
            self.assertTrue(capture.append_snapshot(path, first))
            self.assertFalse(capture.append_snapshot(path, first))
            self.assertTrue(capture.append_snapshot(path, second))
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                rows = [json.loads(line) for line in stream]
            self.assertEqual([row["race_key"] for row in rows], [first["race_key"], second["race_key"]])

    def test_preoff_snapshot_stores_calibrated_win_probability_and_ev(self):
        model = {"model_version":"test-model","temperature":0.18}
        history = defaultdict(list)
        sire_stats = defaultdict(lambda: {"runs":0,"top3":0})
        jockey_stats = defaultdict(lambda: {"runs":0,"top3":0})
        weights = {"form":0.55,"venue":0.12,"distance":0.12,"sire":0.10,"jockey":0.07,"age":0.04}
        context = (model, history, sire_stats, jockey_stats, weights)
        race = {"venue":"大井","distance":1600,"field_size":3}
        horses = [
            {"no":"1","name":"A","age":4,"sire":"S1","jockey":"J1"},
            {"no":"2","name":"B","age":5,"sire":"S2","jockey":"J2"},
            {"no":"3","name":"C","age":7,"sire":"S3","jockey":"J3"},
        ]
        odds = [{"type":"単勝","n1":"1","odds":2.5}]
        predictions = capture.predict_horses(race, horses, odds, context)
        self.assertEqual(len(predictions), 3)
        self.assertAlmostEqual(sum(p["win_probability"] for p in predictions), 1.0, places=6)
        first = next(p for p in predictions if p["horse_no"] == "1")
        self.assertEqual(first["win_odds"], 2.5)
        self.assertAlmostEqual(first["expected_value"], first["win_probability"]*2.5, places=5)

    def test_preoff_D_uses_C_features_and_race_pace_before_odds_overlay(self):
        model = {"model_version":"test-model","temperature":0.18}
        context = (
            model, defaultdict(list),
            defaultdict(lambda: {"runs":0,"top3":0}),
            defaultdict(lambda: {"runs":0,"top3":0}),
            {"form":0.55,"venue":0.12,"distance":0.12,"sire":0.10,"jockey":0.07,"age":0.04},
        )
        race = {"venue":"大井","distance":1600,"field_size":3,"going":"良","surface":"ダート"}
        horses = [
            {"no":"1","name":"A","age":4,"sire":"S1","jockey":"J1"},
            {"no":"2","name":"B","age":5,"sire":"S2","jockey":"J2"},
            {"no":"3","name":"C","age":7,"sire":"S3","jockey":"J3"},
        ]
        with mock.patch.object(nb, "score_runner_performance", side_effect=[(0.20, {}), (0.30, {}), (0.40, {})]) as scorer, \
             mock.patch.object(nb, "pace_adjustments", return_value=(None, None, {"1":0.10,"2":0.05,"3":0.0})) as pace:
            predictions = capture.predict_horses(race, horses, [], context)
        self.assertEqual(scorer.call_count, 3)
        self.assertEqual(scorer.call_args.kwargs["margin_weight"], 0.10)
        self.assertEqual(scorer.call_args.kwargs["closing_weight"], 0.10)
        self.assertEqual(scorer.call_args.kwargs["time_weight"], 0.10)
        self.assertEqual(scorer.call_args.kwargs["going_weight"], 0.10)
        pace.assert_called_once_with(horses, context[1])
        by_no = {p["horse_no"]:p for p in predictions}
        self.assertAlmostEqual(by_no["1"]["score"], 0.30, places=6)
        self.assertAlmostEqual(by_no["2"]["score"], 0.35, places=6)
        self.assertAlmostEqual(by_no["3"]["score"], 0.40, places=6)

    def test_daily_official_zip_parser_reads_races_horses_and_odds(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data"
            data.mkdir()
            race_row = [""] * 27
            race_row[0], race_row[1], race_row[2], race_row[3] = "大井", "20261103", "10", "1540"
            race_row[5], race_row[21], race_row[23], race_row[25], race_row[26] = "サンプル", "ダ", "1600", "良", "3"
            horse_rows = []
            for no, name, age in [("1","馬A","4"),("2","馬B","5"),("3","馬C","6")]:
                row = [""] * 36
                row[0], row[1], row[2], row[5], row[6], row[8], row[11], row[14] = "大井", "20261103", "10", no, name, age, "父", "騎手"
                horse_rows.append(row)
            odds_row = [""] * 10
            odds_row[0], odds_row[1], odds_row[2], odds_row[3], odds_row[4], odds_row[7], odds_row[9] = "大井", "20261103", "10", "単勝", "1", "2.5", "1"
            def csv_text(rows):
                buffer = io.StringIO()
                writer = csv.writer(buffer)
                writer.writerows(rows)
                return buffer.getvalue().encode("utf-8")
            with zipfile.ZipFile(data / "latest-race.zip", "w") as archive:
                archive.writestr("20261103_racelist.csv", csv_text([race_row]))
                archive.writestr("20261103_horselist.csv", csv_text(horse_rows))
            with zipfile.ZipFile(data / "latest-odds.zip", "w") as archive:
                archive.writestr("20261103_odds.csv", csv_text([odds_row]))
            with mock.patch.object(capture, "DATA", data):
                races, horses, odds = capture.read_archives()
            key = ("大井", "20261103", "10")
            self.assertEqual(races[key]["distance"], 1600)
            self.assertEqual(len(horses[key]), 3)
            self.assertEqual(odds[key][0]["type"], "単勝")
            self.assertEqual(odds[key][0]["odds"], 2.5)


class OddsSnapshotBacktestTests(unittest.TestCase):
    def test_loader_keeps_closest_valid_preoff_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "data" / "odds-snapshots"
            folder.mkdir(parents=True)
            path = folder / "2026-11-03.jsonl.gz"
            rows = [
                {"race_key":"大井|20261103|10","captured_at":"2026-11-03T15:00:00+09:00",
                 "scheduled_post_time":"2026-11-03T15:30:00+09:00","minutes_to_post":30,
                 "odds":[{"type":"単勝","n1":"1","odds":2.8}]},
                {"race_key":"大井|20261103|10","captured_at":"2026-11-03T15:10:00+09:00",
                 "scheduled_post_time":"2026-11-03T15:30:00+09:00","minutes_to_post":20,
                 "odds":[{"type":"単勝","n1":"1","odds":2.6}]},
                {"race_key":"大井|20261103|10","captured_at":"2026-11-03T15:20:00+09:00",
                 "scheduled_post_time":"2026-11-03T15:30:00+09:00","minutes_to_post":10,
                 "odds":[{"type":"単勝","n1":"1","odds":2.4}]},
                {"race_key":"大井|20261103|10","captured_at":"2026-11-03T15:25:00+09:00",
                 "scheduled_post_time":"2026-11-03T15:30:00+09:00","minutes_to_post":5,
                 "odds":[{"type":"単勝","n1":"1","odds":2.1}]},
                {"race_key":"大井|20261103|10","captured_at":"2026-11-03T15:31:00+09:00",
                 "scheduled_post_time":"2026-11-03T15:30:00+09:00","minutes_to_post":-1,
                 "odds":[{"type":"単勝","n1":"1","odds":1.8}]},
            ]
            with gzip.open(path, "wt", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            with mock.patch.object(nb, "ROOT", root):
                loaded = nb.load_odds_snapshots()
            key = ("大井", "20261103", "10")
            self.assertIn(key, loaded)
            self.assertEqual(loaded[key]["minutes_to_post"], 5)
            self.assertEqual(loaded[key]["odds"][0]["odds"], 2.1)
            self.assertGreaterEqual(loaded[key]["minutes_to_post"], 2)
            self.assertLessEqual(loaded[key]["minutes_to_post"], 30)


class PreoffValueBacktestTests(unittest.TestCase):
    def test_value_threshold_uses_snapshot_odds_and_settled_payout(self):
        original_root = nb.ROOT
        try:
            with tempfile.TemporaryDirectory() as tmp:
                nb.ROOT = Path(tmp)
                directory = Path(tmp) / "data" / "odds-snapshots"
                directory.mkdir(parents=True)
                record = {
                    "race_key":"大井|20261103|10",
                    "date":"20261103",
                    "captured_at":"2026-11-03T15:25:00+09:00",
                    "scheduled_post_time":"2026-11-03T15:30:00+09:00",
                    "minutes_to_post":5,
                    "model_version":"test-model",
                    "minutes_to_post":5,
                    "predictions":[
                        {"horse_no":"1","win_probability":0.5,"win_odds":2.5,"expected_value":1.25},
                        {"horse_no":"2","win_probability":0.2,"win_odds":3.0,"expected_value":0.6},
                    ],
                    "odds":[{"type":"単勝","n1":"1","odds":2.5}],
                }
                with gzip.open(directory / "2026-11-03.jsonl.gz", "wt", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                payouts = {("大井","20261103","10"): {
                    "win":[(["1"],250)],"place":[],"quinella":[],"exacta":[],
                    "wide":[],"trio":[],"trifecta":[]
                }}
                result = nb.preoff_value_backtest(payouts)
                self.assertEqual(result["status"], "ok")
                self.assertEqual(result["settled_evaluated_races"], 1)
                self.assertEqual(result["thresholds"]["1.10"]["tickets"], 1)
                self.assertEqual(result["thresholds"]["1.10"]["roi"], 250.0)
                self.assertEqual(result["thresholds"]["1.30"]["tickets"], 0)
                self.assertEqual(result["validation_status"], "awaiting_more_dates")
                self.assertFalse(result["thresholds"]["1.10"]["minimum_sample_met"])
                self.assertIsNone(result["thresholds"]["1.10"]["roi_ci95_low"])
        finally:
            nb.ROOT = original_root




class PerBetTypeProbabilityTests(unittest.TestCase):
    def test_plackett_luce_ticket_probabilities_are_consistent(self):
        probabilities = {"1":0.5,"2":0.3,"3":0.2}
        self.assertAlmostEqual(sum(capture.combination_probability("win",[n],probabilities)
                                   for n in probabilities),1.0,places=10)
        self.assertAlmostEqual(sum(capture.combination_probability("place",[n],probabilities)
                                   for n in probabilities),3.0,places=10)
        pairs = [(a,b) for a in probabilities for b in probabilities if a < b]
        self.assertAlmostEqual(sum(capture.combination_probability("quinella",[a,b],probabilities)
                                   for a,b in pairs),1.0,places=10)
        self.assertAlmostEqual(sum(capture.combination_probability("wide",[a,b],probabilities)
                                   for a,b in pairs),3.0,places=10)
        self.assertAlmostEqual(capture.combination_probability("trio",["1","2","3"],probabilities),1.0,places=10)
        exactas = [(a,b) for a in probabilities for b in probabilities if a != b]
        self.assertAlmostEqual(sum(capture.combination_probability("exacta",[a,b],probabilities)
                                   for a,b in exactas),1.0,places=10)
        trifectas = [(a,b,c) for a in probabilities for b in probabilities for c in probabilities
                     if len({a,b,c}) == 3]
        self.assertAlmostEqual(sum(capture.combination_probability("trifecta",[a,b,c],probabilities)
                                   for a,b,c in trifectas),1.0,places=10)

    def test_snapshot_combination_ev_uses_each_market_quote(self):
        horses = [
            {"horse_no":"1","win_probability":0.5},
            {"horse_no":"2","win_probability":0.3},
            {"horse_no":"3","win_probability":0.2},
        ]
        odds = [
            {"type":"単勝","n1":"01","odds":2.5},
            {"type":"複勝","n1":"01","odds":1.3},
            {"type":"ワイド","n1":"01","n2":"02","odds":3.0},
            {"type":"馬連","n1":"01","n2":"02","odds":4.0},
            {"type":"馬単","n1":"01","n2":"02","odds":5.0},
            {"type":"3連複","n1":"01","n2":"02","n3":"03","odds":8.0},
            {"type":"3連単","n1":"01","n2":"02","n3":"03","odds":15.0},
        ]
        result = capture.predict_combinations(odds, horses)
        counts = {kind:sum(1 for x in result if x["bet_type"] == kind)
                  for kind in ("win","place","wide","quinella","exacta","trio","trifecta")}
        self.assertEqual(counts,{"win":3,"place":3,"wide":3,"quinella":3,
                                 "exacta":4,"trio":1,"trifecta":4})
        self.assertTrue(all(x["odds_status"] in ("取得済み","オッズ待ち") for x in result))
        self.assertTrue(any(x["expected_value"] is None for x in result))



class IndependentBetTypeBacktestTests(unittest.TestCase):
    def test_each_bet_type_has_its_own_stake_return_and_roi(self):
        record = {
            "race_key":"大井|20261103|10",
            "model_version":"nar-chronological-v8-abcd-C-preoff",
            "date":"20261103",
            "captured_at":"2026-11-03T15:10:00+09:00",
            "scheduled_post_time":"2026-11-03T15:30:00+09:00",
            "minutes_to_post":20,
            "combination_predictions":[
                {"bet_type":"win","numbers":["1"],"odds":2.5,"probability":0.5,"expected_value":1.25},
                {"bet_type":"place","numbers":["1"],"odds":1.3,"probability":0.9,"expected_value":1.17},
                {"bet_type":"wide","numbers":["1","2"],"odds":4.0,"probability":0.4,"expected_value":1.6},
                {"bet_type":"quinella","numbers":["1","2"],"odds":5.0,"probability":0.3,"expected_value":1.5},
                {"bet_type":"exacta","numbers":["1","2"],"odds":7.0,"probability":0.2,"expected_value":1.4},
                {"bet_type":"trio","numbers":["1","2","3"],"odds":12.0,"probability":0.15,"expected_value":1.8},
                {"bet_type":"trifecta","numbers":["1","2","3"],"odds":25.0,"probability":0.08,"expected_value":2.0},
            ],
        }
        payout = {
            "win":[(["1"],250)], "place":[(["1"],130)],
            "wide":[(["1","2"],400)], "quinella":[(["1","2"],500)],
            "exacta":[(["1","2"],700)], "trio":[(["1","2","3"],1200)],
            "trifecta":[(["1","2","3"],2500)],
        }
        key = ("大井","20261103","10")
        with mock.patch.object(nb,"load_odds_snapshots",return_value={key:record}):
            result = nb.preoff_ticket_type_backtest({key:payout})
        self.assertEqual(result["status"],"ok")
        expected = {
            "win":250.0,"place":130.0,"wide":400.0,"quinella":500.0,
            "exacta":700.0,"trio":1200.0,"trifecta":2500.0,
        }
        for kind, roi in expected.items():
            bucket = result["bet_types"][kind]["overall"]["1.00"]
            self.assertEqual(bucket["stake"],100)
            self.assertEqual(bucket["return"],int(roi))
            self.assertEqual(bucket["roi"],roi)
            self.assertEqual(bucket["tickets"],1)

    def test_unordered_ticket_matching_is_separate_from_ordered_ticket_matching(self):
        payout = {
            "win":[], "place":[], "wide":[(["1","2"],400)],
            "quinella":[(["1","2"],500)], "exacta":[(["1","2"],700)],
            "trio":[(["1","2","3"],1200)], "trifecta":[(["1","2","3"],2500)],
        }
        self.assertEqual(nb.payout_for("wide",["2","1"],payout),400)
        self.assertEqual(nb.payout_for("quinella",["2","1"],payout),500)
        self.assertEqual(nb.payout_for("trio",["3","1","2"],payout),1200)
        self.assertEqual(nb.payout_for("exacta",["2","1"],payout),0)
        self.assertEqual(nb.payout_for("trifecta",["3","2","1"],payout),0)


class FieldSizeNormalizationTests(unittest.TestCase):
    def test_numeric_zero_is_not_treated_as_missing(self):
        self.assertEqual(nb.number(0), 0.0)
        self.assertEqual(nb.number("0"), 0.0)
        self.assertIsNone(nb.number(None))
        self.assertIsNone(nb.number(""))

    def test_finish_rank_is_normalized_by_field_size(self):
        self.assertEqual(nb.rank_quality(1, 16), 100.0)
        self.assertAlmostEqual(nb.rank_quality(8, 16), 53.3333333333, places=6)
        self.assertEqual(nb.rank_quality(1, None), 100.0)


class ProbabilityCalibrationTests(unittest.TestCase):
    def test_softmax_probabilities_sum_to_one(self):
        probs = nb.softmax_probabilities([0.2, 0.4, 0.1], 0.18)
        self.assertAlmostEqual(sum(probs), 1.0, places=10)
        self.assertGreater(probs[1], probs[0])
        self.assertGreater(probs[0], probs[2])

    def test_temperature_fit_uses_only_prediction_scores_and_targets(self):
        records = [{"scores":[0.9,0.2,0.1],"winner_indices":[0]} for _ in range(25)]
        temperature = nb.fit_temperature(records)
        self.assertLessEqual(temperature, 0.10)
        metrics = nb.calibration_metrics(records, temperature)
        self.assertEqual(metrics["races"], 25)
        self.assertEqual(metrics["top1_accuracy"], 100.0)


class ConfidenceFilterBacktestTests(unittest.TestCase):
    def test_confidence_filter_applies_threshold_and_tracks_each_bet_type(self):
        stats = nb.new_confidence_filter_stats()
        ranked = [{"no":"1"}, {"no":"2"}, {"no":"3"}]
        tickets = {"win":[["1"]], "place":[["1"],["2"]]}
        payout = {
            "win":[(["1"],250)], "place":[(["1"],130)],
            "wide":[], "quinella":[], "exacta":[], "trio":[], "trifecta":[]
        }
        nb.update_confidence_filter_stats(stats, 0.30, ranked, {"1"}, tickets, payout)
        result = nb.summarize_confidence_filter_stats(stats)
        self.assertEqual(result["0.20"]["races"], 1)
        self.assertEqual(result["0.30"]["races"], 1)
        self.assertEqual(result["0.35"]["races"], 0)
        self.assertEqual(result["0.30"]["bet_types"]["win"]["stake"], 100)
        self.assertEqual(result["0.30"]["bet_types"]["win"]["return"], 250)
        self.assertEqual(result["0.30"]["bet_types"]["win"]["roi"], 250.0)
        self.assertEqual(result["0.30"]["bet_types"]["place"]["tickets"], 2)

    def test_confidence_filter_empty_bucket_has_no_fabricated_roi(self):
        result = nb.summarize_confidence_filter_stats(nb.new_confidence_filter_stats())
        self.assertIsNone(result["0.50"]["bet_types"]["win"]["roi"])
        self.assertIsNone(result["0.50"]["winner_top1_rate"])


class AutomaticImprovementLoopTests(unittest.TestCase):
    def _fold(self, fold_no, model_rois):
        models = {}
        for model_name, roi in model_rois.items():
            models[model_name] = {"bet_types": {
                "win": {"roi": roi, "tickets": 500, "hits": 50,
                        "stake": 50000, "return": int(50000*roi/100),
                        "profit": int(50000*roi/100)-50000}
            }}
        return {"fold": fold_no, "models": models}

    def test_selects_challenger_only_from_prior_fold_with_required_edge(self):
        prior = self._fold(1, {"baseline": 60.0, "trained": 70.0, "pace": 63.0})
        selected = nb.select_models_from_prior_fold(prior)
        self.assertEqual(selected["win"]["model"], "trained")
        self.assertEqual(selected["win"]["development_roi"], 70.0)

    def test_falls_back_to_baseline_when_sample_or_edge_is_insufficient(self):
        prior = self._fold(1, {"baseline": 60.0, "trained": 64.0})
        prior["models"]["trained"]["bet_types"]["win"]["tickets"] = 299
        selected = nb.select_models_from_prior_fold(prior)
        self.assertEqual(selected["win"]["model"], "baseline")
        prior["models"]["trained"]["bet_types"]["win"]["tickets"] = 500
        prior["models"]["trained"]["bet_types"]["win"]["roi"] = 64.9
        selected = nb.select_models_from_prior_fold(prior)
        self.assertEqual(selected["win"]["model"], "baseline")

    def test_walk_forward_evaluates_selected_model_on_next_fold(self):
        prior = self._fold(1, {"baseline": 60.0, "trained": 70.0})
        following = self._fold(2, {"baseline": 65.0, "trained": 72.0})
        result = nb.evaluate_walk_forward_selection([prior, following])
        win = result["evaluations"][0]["bet_types"]["win"]
        self.assertEqual(win["model"], "trained")
        self.assertEqual(win["validation_fold"], 2)
        self.assertEqual(win["validation_roi"], 72.0)
        self.assertEqual(win["roi_delta_vs_baseline"], 7.0)


if __name__ == "__main__":
    unittest.main()


class CornerPositionAndPaceTests(unittest.TestCase):
    def test_official_corner_order_parser_handles_groups_and_shared_ranks(self):
        positions = nb.parse_corner_order("1,3,(2,7),4")
        self.assertEqual(positions["1"], 1)
        self.assertEqual(positions["3"], 2)
        self.assertEqual(positions["2"], 3.5)
        self.assertEqual(positions["7"], 3.5)
        self.assertEqual(positions["4"], 5)

    def test_corner_history_is_field_normalized_and_uses_first_last_available(self):
        horse = {"no":"2"}
        race = {"field_size":10,"corner_positions":[{"2":3},{"2":5}]}
        features = nb.corner_history_for_runner(horse, race)
        self.assertAlmostEqual(features["first_corner_pct"], 2/9)
        self.assertAlmostEqual(features["last_corner_pct"], 4/9)

    def test_running_style_uses_both_early_and_late_corner_positions(self):
        self.assertEqual(nb.pace_style([])["style"], "unknown")
        self.assertEqual(nb.pace_style([{"first_corner_pct":0.05,"last_corner_pct":0.08}])["style"], "逃げ")
        self.assertEqual(nb.pace_style([{"first_corner_pct":0.25,"last_corner_pct":0.30}])["style"], "先行")
        self.assertEqual(nb.pace_style([{"first_corner_pct":0.75,"last_corner_pct":0.30}])["style"], "差し")
        self.assertEqual(nb.pace_style([{"first_corner_pct":0.85,"last_corner_pct":0.90}])["style"], "追込")
        self.assertGreater(nb.pace_style([{"first_corner_pct":0.75,"last_corner_pct":0.30}])["position_gain"], 0)

    def test_pace_adjustment_only_uses_prior_positions_and_counts_multiple_leaders(self):
        field = [{"no":"1","name":"逃げA"},{"no":"2","name":"逃げB"},
                 {"no":"3","name":"逃げC"},{"no":"4","name":"差しD"}]
        history = {
            "逃げA":[{"first_corner_pct":0.02}],
            "逃げB":[{"first_corner_pct":0.06}],
            "逃げC":[{"first_corner_pct":0.10}],
            "差しD":[{"first_corner_pct":0.80}],
        }
        profiles, leaders, deltas = nb.pace_adjustments(field, history)
        self.assertEqual(leaders, 3)
        self.assertEqual(profiles["4"]["style"], "追込")
        self.assertLess(deltas["1"], 0)
        self.assertGreater(deltas["4"], 0)
        unknown = nb.pace_adjustments(field, {})[2]
        self.assertTrue(all(value == 0 for value in unknown.values()))



class TopKPredictionCoverageTests(unittest.TestCase):
    def test_top_five_six_seven_use_same_ranked_race_and_detect_winner(self):
        ranked = [{"no":str(i)} for i in range(1, 9)]
        self.assertEqual(nb.top_k_winner_coverage(ranked, {"6"}),
                         {"5":0, "6":1, "7":1})
        self.assertEqual(nb.top_k_winner_coverage(ranked, {"8"}),
                         {"5":0, "6":0, "7":0})

    def test_selected_ticket_portfolio_is_unchanged_by_display_top_k(self):
        ranked = [{"no":str(i)} for i in range(1, 9)]
        tickets = nb.build_fixed_tickets(ranked)
        self.assertEqual(len(tickets["wide"]), 3)
        self.assertEqual(len(tickets["exacta"]), 4)
        self.assertEqual(len(tickets["trio"]), 10)
        self.assertEqual(len(tickets["trifecta"]), 24)
        self.assertNotIn("win", tickets)
        self.assertNotIn("place", tickets)
        self.assertNotIn("quinella", tickets)


class ExpandedFixedTicketTests(unittest.TestCase):
    def test_requested_ticket_counts_and_unique_combinations(self):
        ranked = [{"no":str(i)} for i in range(1,7)]
        tickets = nb.build_fixed_tickets(ranked)
        self.assertNotIn("win", tickets)
        self.assertNotIn("place", tickets)
        self.assertNotIn("quinella", tickets)
        self.assertEqual(len(tickets["wide"]), 3)
        self.assertEqual(len(tickets["exacta"]), 4)
        self.assertEqual(len(tickets["trio"]), 10)
        self.assertEqual(len(tickets["trifecta"]), 24)
        self.assertEqual(len({tuple(x) for x in tickets["exacta"]}), 4)
        self.assertEqual(len({tuple(x) for x in tickets["trifecta"]}), 24)
        self.assertTrue(all(len(set(x)) == 3 for x in tickets["trifecta"]))

    def test_three_and_four_runner_fields_do_not_create_invalid_tickets(self):
        three = nb.build_fixed_tickets([{"no":str(i)} for i in range(1,4)])
        self.assertEqual(len(three["trio"]), 1)
        self.assertEqual(len(three["trifecta"]), 4)
        four = nb.build_fixed_tickets([{"no":str(i)} for i in range(1,5)])
        self.assertEqual(len(four["trio"]), 4)
        self.assertEqual(len(four["trifecta"]), 12)

    def test_expanded_exacta_and_trifecta_respect_ranked_formation(self):
        tickets = nb.build_fixed_tickets([{"no":str(i)} for i in range(1,6)])
        self.assertEqual(set(map(tuple,tickets["exacta"])),
                         {("1","2"),("1","3"),("2","1"),("2","3")})
        self.assertIn(["1","2","3"],tickets["trifecta"])
        self.assertIn(["2","1","5"],tickets["trifecta"])
        self.assertNotIn(["3","1","2"],tickets["trifecta"])


class PerformanceFeatureTests(unittest.TestCase):
    def test_field_features_rank_last3f_within_race_and_transform_margin(self):
        field = [
            {"no":"1","last3f":38.0,"margin":"0"},
            {"no":"2","last3f":36.0,"margin":"アタマ"},
            {"no":"3","last3f":40.0,"margin":"2.0"},
            {"no":"4","last3f":None,"margin":"大差"},
        ]
        features = nb.field_performance_features(field)
        self.assertEqual(features["2"]["last3f_rank_score"], 1.0)
        self.assertEqual(features["1"]["last3f_rank_score"], 0.5)
        self.assertEqual(features["3"]["last3f_rank_score"], 0.0)
        self.assertIsNone(features["4"]["last3f_rank_score"])
        self.assertGreater(features["1"]["margin_quality"], features["3"]["margin_quality"])
        self.assertGreater(features["2"]["margin_quality"], features["3"]["margin_quality"])

    def test_d_score_uses_recent_margin_and_closing_features(self):
        horse = {"no":"1","name":"A","sire":"","jockey":"","age":5}
        history = {"A":[
            {"finish":5,"field_size":10,"venue":"X","distance":1400,
             "margin_quality":0.9,"last3f_rank_score":0.9}
        ]}
        weights = {"form":0.55,"venue":0.12,"distance":0.12,"sire":0.10,"jockey":0.07,"age":0.04}
        base, _ = nb.score_runner(horse,history,{}, {"": {"runs":0,"top3":0}}, {"venue":"X","distance":1400},weights)
        d_score, parts = nb.score_runner_d(horse,history,{}, {"": {"runs":0,"top3":0}}, {"venue":"X","distance":1400},weights)
        self.assertGreater(d_score, base)
        self.assertEqual(parts["margin_form"], 0.9)
        self.assertEqual(parts["closing_form"], 0.9)

    def test_performance_ablation_weights_are_independent(self):
        horse = {"no":"1","name":"A","sire":"","jockey":"","age":5}
        history = {"A":[{"finish":5,"field_size":10,"venue":"X","distance":1400,
                         "margin_quality":1.0,"last3f_rank_score":0.0}]}
        weights = {"form":0.55,"venue":0.12,"distance":0.12,"sire":0.10,"jockey":0.07,"age":0.04}
        args = (horse,history,{}, {"": {"runs":0,"top3":0}}, {"venue":"X","distance":1400},weights)
        base = nb.score_runner(*args)[0]
        margin = nb.score_runner_performance(*args,margin_weight=0.10,closing_weight=0.00)[0]
        closing = nb.score_runner_performance(*args,margin_weight=0.00,closing_weight=0.10)[0]
        light = nb.score_runner_performance(*args,margin_weight=0.05,closing_weight=0.05)[0]
        self.assertGreater(margin, base)
        self.assertLess(closing, base)
        self.assertAlmostEqual(light, base + (margin-base + closing-base)/2, places=10)
        # Blend weights must apply to the full normalized B score, not a
        # pre-scaled 0.8 copy of B's weights.
        self.assertAlmostEqual(
            nb.score_runner_performance(*args,margin_weight=0.10,closing_weight=0.10)[0],
            base * 0.80 + 0.10 * 1.0 + 0.10 * 0.0,
            places=10,
        )


class ResearchReadinessAndDataQualityTests(unittest.TestCase):
    def test_data_quality_audit_reports_missing_condition_and_runner_fields(self):
        key = ("大井", "20261103", "1")
        race = {"key": key, "distance": 1600, "surface": "ダート", "going": ""}
        horses = {key: [
            {"no":"1","age":5,"jockey":"騎手A","sire":"父A","finish":1,"last3f":38.0},
            {"no":"2","age":None,"jockey":"","sire":"父B","finish":2,"last3f":None},
            {"no":"3","age":4,"jockey":"騎手B","sire":"","finish":3,"last3f":39.0},
        ]}
        payouts = {key: {"win":[(["1"],250)]}}
        audit = nb.data_quality_audit({key:race}, horses, payouts)
        self.assertEqual(audit["races"], 1)
        self.assertEqual(audit["runners"], 3)
        self.assertEqual(audit["surface_coverage_pct"], 100.0)
        self.assertEqual(audit["going_coverage_pct"], 0.0)
        self.assertAlmostEqual(audit["runner_age_coverage_pct"], 66.67, places=2)
        self.assertAlmostEqual(audit["runner_jockey_coverage_pct"], 66.67, places=2)
        self.assertEqual(audit["races_with_payout"], 1)

    def test_readiness_skips_when_odds_history_or_calibration_evidence_is_insufficient(self):
        model = {"calibration":{"races":100}}
        history = {"A":[{"finish":1}]}
        context = (model, history, {}, {}, {})
        horses = [{"no":str(i),"name":chr(64+i)} for i in range(1,6)]
        predictions = [
            {"horse_no":str(i),"win_probability":0.2,"prior_runs":1 if i == 1 else 0}
            for i in range(1,6)
        ]
        odds = [{"type":"単勝","n1":"1","odds":2.0}]
        result = capture.race_readiness({}, horses, odds, predictions, context)
        self.assertEqual(result["status"], "skip_insufficient_evidence")
        self.assertIn("win_odds_coverage_below_70pct", result["reasons"])
        self.assertIn("historical_form_coverage_below_50pct", result["reasons"])
        self.assertIn("calibration_sample_below_500", result["reasons"])
        self.assertAlmostEqual(result["normalized_entropy"], 1.0, places=6)

    def test_readiness_marks_complete_research_sample_without_calling_it_profitable(self):
        model = {"calibration":{"races":1000}}
        context = (model, {}, {}, {}, {})
        horses = [{"no":str(i),"name":f"H{i}"} for i in range(1,6)]
        predictions = [
            {"horse_no":str(i),"win_probability":p,"prior_runs":5}
            for i,p in enumerate((0.40,0.25,0.15,0.12,0.08),1)
        ]
        odds = [{"type":"単勝","n1":str(i),"odds":float(i+1)} for i in range(1,6)]
        result = capture.race_readiness({}, horses, odds, predictions, context)
        self.assertEqual(result["status"], "eligible_for_research_evaluation")
        self.assertEqual(result["reasons"], [])
        self.assertIn("does not mean profitable", result["policy"])




class ConditionSegmentTests(unittest.TestCase):
    def test_race_segment_labels_cover_venue_surface_going_and_distance_band(self):
        labels = nb.race_segment_labels({
            "venue":"大井", "surface":"ダート", "going":"稍重", "distance":1600
        })
        self.assertEqual(labels["venue"], "大井")
        self.assertEqual(labels["surface"], "ダート")
        self.assertEqual(labels["going"], "稍重")
        self.assertEqual(labels["distance_band"], "mile_1201-1600m")

    def test_missing_conditions_are_not_fabricated(self):
        labels = nb.race_segment_labels({"venue":"", "surface":None, "going":"", "distance":None})
        self.assertEqual(labels, {})



class StrongerSelectionAndOddsGateTests(unittest.TestCase):
    def _fold(self, number, wide_roi, win_roi, trifecta_roi):
        kinds = ("win", "place", "wide", "quinella", "exacta", "trio", "trifecta")
        models = {"baseline": {"bet_types": {}}, "challenger": {"bet_types": {}}}
        for kind in kinds:
            base_roi = 60.0
            challenger_roi = {
                "win": win_roi, "wide": wide_roi, "trifecta": trifecta_roi
            }.get(kind, base_roi)
            for model_name, roi in (("baseline", base_roi), ("challenger", challenger_roi)):
                models[model_name]["bet_types"][kind] = {
                    "roi": roi, "tickets": 12000, "hits": 1000,
                    "stake": 1200000, "return": int(1200000 * roi / 100),
                    "profit": int(1200000 * roi / 100) - 1200000,
                }
        return {"fold": number, "models": models}

    def test_wide_requires_repeated_edge_and_trifecta_has_stricter_hurdle(self):
        first = self._fold(1, wide_roi=70.0, win_roi=70.0, trifecta_roi=71.5)
        second = self._fold(2, wide_roi=71.0, win_roi=55.0, trifecta_roi=71.5)
        selected = nb.select_models_from_prior_folds([first, second])
        self.assertEqual(selected["wide"]["model"], "challenger")
        self.assertEqual(selected["wide"]["edge_history"], [10.0, 11.0])
        self.assertEqual(selected["win"]["model"], "baseline")
        self.assertEqual(selected["trifecta"]["model"], "baseline")
        self.assertEqual(selected["trifecta"]["required_min_tickets"], 5000)
        self.assertEqual(selected["trifecta"]["required_min_roi_gain"], 12.0)

    def test_candidate_that_fails_one_prior_fold_is_not_selected(self):
        first = self._fold(1, wide_roi=70.0, win_roi=70.0, trifecta_roi=60.0)
        second = self._fold(2, wide_roi=62.0, win_roi=70.0, trifecta_roi=60.0)
        selected = nb.select_models_from_prior_folds([first, second])
        self.assertEqual(selected["wide"]["model"], "baseline")

    def test_type_specific_hurdles_keep_trifecta_strict(self):
        selected = nb.select_models_from_prior_folds([
            self._fold(1, wide_roi=63.0, win_roi=60.0, trifecta_roi=71.5)
        ])
        self.assertEqual(selected["wide"]["model"], "challenger")
        self.assertEqual(selected["wide"]["required_min_roi_gain"], 2.5)
        self.assertEqual(selected["trifecta"]["model"], "baseline")
        self.assertEqual(selected["trifecta"]["required_min_roi_gain"], 12.0)

    def test_walk_forward_reports_aggregate_selected_portfolio(self):
        prior = self._fold(1, wide_roi=70.0, win_roi=60.0, trifecta_roi=60.0)
        following = self._fold(2, wide_roi=72.0, win_roi=60.0, trifecta_roi=60.0)
        result = nb.evaluate_walk_forward_selection([prior, following])
        wide = result["selected_portfolio"]["wide"]
        self.assertEqual(wide["roi"], 72.0)
        self.assertEqual(wide["tickets"], 12000)
        self.assertEqual(wide["validation_folds"][0]["model"], "challenger")



class MultiBetConditionDiagnosticsTests(unittest.TestCase):
    def test_exacta_and_trio_condition_comparisons_report_roi_hit_rate_and_sample(self):
        stats = {
            ("baseline", "exacta", "venue", "大井"): {
                "races": 100, "tickets": 400, "hits": 20, "stake": 40000, "return": 32000,
            },
            ("trained_no_recent_history", "exacta", "venue", "大井"): {
                "races": 100, "tickets": 400, "hits": 24, "stake": 40000, "return": 36000,
            },
            ("baseline", "trio", "distance_band", "mile_1201-1600m"): {
                "races": 80, "tickets": 800, "hits": 8, "stake": 80000, "return": 44000,
            },
            ("trained_no_recent_history", "trio", "distance_band", "mile_1201-1600m"): {
                "races": 80, "tickets": 800, "hits": 10, "stake": 80000, "return": 52000,
            },
        }
        segments, comparisons = nb.summarize_condition_model_segments(
            stats, ("exacta", "trio"))
        exacta = comparisons["exacta"][0]
        trio = comparisons["trio"][0]
        self.assertEqual(exacta["roi_delta"], 10.0)
        self.assertEqual(exacta["baseline_hit_rate"], 5.0)
        self.assertEqual(exacta["candidate_hit_rate"], 6.0)
        self.assertEqual(exacta["candidate_tickets"], 400)
        self.assertEqual(trio["roi_delta"], 10.0)
        self.assertEqual(trio["baseline_hit_rate"], 1.0)
        self.assertEqual(trio["candidate_hit_rate"], 1.25)
        self.assertEqual(trio["candidate_profit"], -28000)
        self.assertIn("exacta", segments)
        self.assertIn("trio", segments)



class CandidateRankingResearchTests(unittest.TestCase):
    def test_full_universe_contains_combinations_beyond_fixed_top_five(self):
        horses = [
            {"horse_no":"1","win_probability":0.5,"score":0.9},
            {"horse_no":"2","win_probability":0.3,"score":0.7},
            {"horse_no":"3","win_probability":0.2,"score":0.5},
        ]
        odds = []
        for kind, pairs in [
            ("ワイド", [("1","2"),("1","3"),("2","3")]),
            ("馬単", [("1","2"),("1","3"),("2","1"),("2","3"),("3","1"),("3","2")]),
            ("3連複", [("1","2","3")]),
            ("3連単", [("1","2","3"),("1","3","2"),("2","1","3"),
                       ("2","3","1"),("3","1","2"),("3","2","1")]),
        ]:
            for nums in pairs:
                odds.append({"type":kind,"n1":nums[0],
                             "n2":nums[1] if len(nums)>1 else "",
                             "n3":nums[2] if len(nums)>2 else "",
                             "odds":5.0})
        result = capture.predict_combinations(odds, horses, full_universe=True)
        counts = {kind:sum(1 for row in result if row["bet_type"] == kind)
                  for kind in ("wide","exacta","trio","trifecta")}
        self.assertEqual(counts, {"wide":3,"exacta":6,"trio":1,"trifecta":6})
        unquoted = [row for row in result if row["bet_type"] in {"win","place","quinella"}]
        self.assertTrue(unquoted)
        self.assertTrue(all(row["expected_value"] is None for row in unquoted))

    def test_preoff_ranking_backtest_selects_same_top_n_and_uses_settled_payouts(self):
        key = ("大井","20261103","10")
        candidates = [
            {"bet_type":"exacta","numbers":["1","2"],"probability":0.30,"odds":2.0,"expected_value":0.60},
            {"bet_type":"exacta","numbers":["1","3"],"probability":0.10,"odds":20.0,"expected_value":2.00},
            {"bet_type":"exacta","numbers":["2","1"],"probability":0.20,"odds":3.0,"expected_value":0.60},
            {"bet_type":"exacta","numbers":["2","3"],"probability":0.15,"odds":10.0,"expected_value":1.50},
            {"bet_type":"exacta","numbers":["3","1"],"probability":0.10,"odds":5.0,"expected_value":0.50},
            {"bet_type":"exacta","numbers":["3","2"],"probability":0.15,"odds":4.0,"expected_value":0.60},
        ]
        record = {
            "race_key":"大井|20261103|10","date":"20261103",
            "model_version":"nar-chronological-v8-abcd-C-preoff",
            "captured_at":"2026-11-03T15:10:00+09:00",
            "scheduled_post_time":"2026-11-03T15:30:00+09:00",
            "minutes_to_post":20,"candidate_combinations":candidates,
        }
        payout = {kind:[] for kind in ("win","place","wide","quinella","exacta","trio","trifecta")}
        payout["win"] = [(["9"],100)]
        payout["exacta"] = [(["1","3"],2000)]
        with mock.patch.object(nb,"load_odds_snapshots",return_value={key:record}):
            result = nb.preoff_ranking_strategy_backtest({key:payout})
        self.assertEqual(result["status"],"ok")
        self.assertEqual(result["snapshot_races"],1)
        ev = result["strategies"]["exacta"]["expected_value"]["overall"]
        self.assertEqual(ev["tickets"],4)
        self.assertEqual(ev["hits"],1)
        self.assertEqual(ev["return"],2000)
        self.assertFalse(ev["minimum_sample_met"])
        self.assertEqual(result["selected_by_development"]["exacta"]["status"],
                         "awaiting_development_sample")

    def test_D_backtest_excludes_snapshots_from_older_model_versions(self):
        key = ("大井","20261103","10")
        old_snapshot = {
            "race_key":"大井|20261103|10",
            "model_version":"nar-chronological-v7-fieldsize-calibrated",
        }
        with mock.patch.object(nb, "load_odds_snapshots", return_value={key:old_snapshot}):
            result = nb.preoff_ranking_strategy_backtest({})
        self.assertEqual(result["status"], "awaiting_C_model_preoff_snapshots")
        self.assertEqual(result["snapshot_races"], 0)
        self.assertEqual(result["legacy_snapshot_races_excluded"], 1)
        self.assertEqual(result["required_model_version"], "nar-chronological-v8-abcd-C-preoff")


class TrifectaDiagnosticTests(unittest.TestCase):
    def test_ticket_structures_separate_first_place_pool_from_coverage_pool(self):
        ranked = [{"no": str(i)} for i in range(1, 8)]
        self.assertEqual(len(nb.build_trifecta_tickets(ranked, 1, 5)), 12)
        self.assertEqual(len(nb.build_trifecta_tickets(ranked, 2, 5)), 24)
        self.assertEqual(len(nb.build_trifecta_tickets(ranked, 3, 5)), 36)
        self.assertEqual(len(nb.build_trifecta_tickets(ranked, 2, 6)), 40)
        self.assertEqual(len(nb.build_trifecta_tickets(ranked, 2, 7)), 60)

    def test_narrow_formations_reduce_ticket_count_without_duplicate_horses(self):
        ranked = [{"no": str(i)} for i in range(1, 8)]
        expected = {
            (1, 4, 4): 6, (1, 4, 5): 9, (1, 5, 4): 9,
            (2, 3, 4): 8, (2, 4, 3): 8, (2, 3, 5): 12,
            (2, 5, 3): 12, (2, 4, 4): 12,
        }
        for pools, ticket_count in expected.items():
            tickets = nb.build_trifecta_formation(ranked, *pools)
            self.assertEqual(len(tickets), ticket_count, pools)
            self.assertTrue(all(len(set(ticket)) == 3 for ticket in tickets))
        structures = nb.new_trifecta_diagnostics()["structures"]
        self.assertTrue(all(pools in structures for pools in expected))

    def test_asymmetric_formation_uses_independent_rank_pools(self):
        ranked = [{"no": str(i)} for i in range(1, 8)]
        a = nb.build_trifecta_formation(ranked, 2, 4, 5)
        b = nb.build_trifecta_formation(ranked, 2, 5, 4)
        self.assertEqual(len(a), 18)
        self.assertEqual(len(b), 18)
        self.assertTrue(all(ticket[0] in {"1", "2"} for ticket in a))
        self.assertTrue(all(ticket[1] in {"1", "2", "3", "4"} for ticket in a))
        self.assertTrue(all(ticket[2] in {"1", "2", "3", "4", "5"} for ticket in a))
        self.assertTrue(all(len(set(ticket)) == 3 for ticket in a + b))
        self.assertEqual(nb.build_trifecta_tickets(ranked, 2, 5), nb.build_trifecta_formation(ranked, 2, 5, 5))

    def test_role_coverage_and_ordered_hit_are_reported_separately(self):
        ranked = [{"no": str(i)} for i in range(1, 8)]
        field = [
            {"no": "1", "finish": 2}, {"no": "2", "finish": 1},
            {"no": "3", "finish": 3}, {"no": "4", "finish": 4},
            {"no": "5", "finish": 5}, {"no": "6", "finish": 6},
            {"no": "7", "finish": 7},
        ]
        payout = {kind: [] for kind in
                  ("win", "place", "wide", "quinella", "exacta", "trio", "trifecta")}
        payout["win"] = [(["2"], 100)]
        payout["trifecta"] = [(["2", "1", "3"], 12000)]
        stats = nb.new_trifecta_diagnostics()
        nb.update_trifecta_diagnostics(stats, ranked, field, payout)
        result = nb.summarize_trifecta_diagnostics(stats)
        self.assertEqual(result["rates"]["winner_top2"], 100.0)
        self.assertEqual(result["rates"]["second_top2"], 100.0)
        self.assertEqual(result["rates"]["third_top3"], 100.0)
        self.assertEqual(result["rates"]["all_podium_top5"], 100.0)
        self.assertEqual(result["structures"]["1-5-5"]["hits"], 0)
        self.assertEqual(result["structures"]["2-5-5"]["hits"], 1)
        self.assertEqual(result["structures"]["2-5-5"]["tickets"], 24)
        self.assertEqual(result["structures"]["2-5-5"]["roi"], 500.0)
        self.assertEqual(result["structures"]["2-5-5"]["hit_races"], 1)
        self.assertEqual(result["structures"]["2-5-5"]["race_hit_rate"], 100.0)
        self.assertIsNone(result["structures"]["2-5-5"]["roi_ci95_normal_low"])



class TrioDiagnosticTests(unittest.TestCase):
    def test_top_n_trio_combinations_are_unordered_and_have_expected_counts(self):
        ranked = [{"no": str(i)} for i in range(1, 8)]
        self.assertEqual(len(nb.build_trio_tickets(ranked, 4)), 4)
        self.assertEqual(len(nb.build_trio_tickets(ranked, 5)), 10)
        self.assertEqual(len(nb.build_trio_tickets(ranked, 6)), 20)
        self.assertEqual(len(nb.build_trio_tickets(ranked, 7)), 35)
        self.assertIn(["1", "2", "3"], nb.build_trio_tickets(ranked, 5))
        self.assertNotIn(["1", "3", "2"], nb.build_trio_tickets(ranked, 5))

    def test_trio_diagnostics_separate_podium_coverage_from_ticket_roi(self):
        ranked = [{"no": str(i)} for i in range(1, 8)]
        field = [
            {"no": "1", "finish": 1}, {"no": "2", "finish": 3},
            {"no": "3", "finish": 2}, {"no": "4", "finish": 4},
            {"no": "5", "finish": 5}, {"no": "6", "finish": 6},
            {"no": "7", "finish": 7},
        ]
        payout = {"trio": [(["1", "2", "3"], 1000)]}
        stats = nb.new_trio_diagnostics()
        nb.update_trio_diagnostics(stats, ranked, field, payout)
        result = nb.summarize_trio_diagnostics(stats)
        self.assertEqual(result["rates"]["all_podium_top4"], 100.0)
        self.assertEqual(result["structures"]["top4"]["tickets"], 4)
        self.assertEqual(result["structures"]["top5"]["tickets"], 10)
        self.assertEqual(result["structures"]["top4"]["hits"], 1)
        self.assertEqual(result["structures"]["top4"]["roi"], 250.0)
        self.assertEqual(result["structures"]["top5"]["roi"], 100.0)

    def test_trio_diagnostics_skip_incomplete_podium_results(self):
        ranked = [{"no": str(i)} for i in range(1, 6)]
        field = [{"no": "1", "finish": 1}, {"no": "2", "finish": 2},
                 {"no": "3", "finish": None}]
        stats = nb.new_trio_diagnostics()
        nb.update_trio_diagnostics(stats, ranked, field, {"trio": []})
        self.assertEqual(nb.summarize_trio_diagnostics(stats)["races"], 0)


class IncompleteFinishDiagnosticSampleTests(unittest.TestCase):
    def test_trifecta_diagnostics_report_race_hit_rate_and_roi_interval(self):
        ranked = [{"no": str(i)} for i in range(1, 6)]
        field = [{"no": str(i), "finish": None} for i in range(1, 6)]
        stats = nb.new_trifecta_diagnostics()
        nb.update_trifecta_diagnostics(stats, ranked, field,
            {"trifecta": [(["1", "2", "3"], 5000)]})
        nb.update_trifecta_diagnostics(stats, ranked, field, {"trifecta": []})
        result = nb.summarize_trifecta_diagnostics(stats)["structures"]["2-5-5"]
        self.assertEqual(result["hit_races"], 1)
        self.assertEqual(result["race_hit_rate"], 50.0)
        self.assertIsNotNone(result["roi_ci95_normal_low"])
        self.assertLess(result["roi_ci95_normal_low"], result["roi_ci95_normal_high"])
        self.assertIn("sum_return_sq", result["roi_variance_components"])

    def test_trifecta_roi_keeps_valid_payout_races_without_complete_finish_roles(self):
        ranked = [{"no": str(i)} for i in range(1, 6)]
        field = [{"no": str(i), "finish": None} for i in range(1, 6)]
        payout = {"win": [], "trifecta": [(["1", "2", "3"], 1000)]}
        stats = nb.new_trifecta_diagnostics()
        nb.update_trifecta_diagnostics(stats, ranked, field, payout)
        result = nb.summarize_trifecta_diagnostics(stats)
        self.assertEqual(result["races"], 0)
        self.assertEqual(result["structures"]["2-5-5"]["races"], 1)
        self.assertEqual(result["structures"]["2-5-5"]["tickets"], 24)
        self.assertEqual(result["structures"]["2-5-5"]["roi"], 41.67)

    def test_trio_roi_keeps_valid_payout_races_without_complete_podium_fields(self):
        ranked = [{"no": str(i)} for i in range(1, 6)]
        field = [{"no": str(i), "finish": None} for i in range(1, 6)]
        stats = nb.new_trio_diagnostics()
        nb.update_trio_diagnostics(stats, ranked, field, {"trio": [(["1", "2", "3"], 1000)]})
        result = nb.summarize_trio_diagnostics(stats)
        self.assertEqual(result["races"], 0)
        self.assertEqual(result["structures"]["top5"]["races"], 1)
        self.assertEqual(result["structures"]["top5"]["tickets"], 10)
        self.assertEqual(result["structures"]["top5"]["roi"], 100.0)


class RoleSpecificTrifectaTests(unittest.TestCase):
    def test_role_ranking_uses_only_supplied_prior_history_and_returns_ordered_pool(self):
        scored = [
            (0.70, {"no":"1","name":"逃げ型"}),
            (0.69, {"no":"2","name":"差し型"}),
            (0.68, {"no":"3","name":"安定型"}),
        ]
        history = {
            "逃げ型":[{"finish":1,"field_size":10}] * 5,
            "差し型":[{"finish":2,"field_size":10}] * 5,
            "安定型":[{"finish":3,"field_size":10}] * 5,
        }
        first = nb.role_specific_ranking(scored, history, 1, 0.20)
        second = nb.role_specific_ranking(scored, history, 2, 0.20)
        third = nb.role_specific_ranking(scored, history, 3, 0.20)
        self.assertEqual(first[0]["no"], "1")
        self.assertEqual(second[0]["no"], "2")
        self.assertEqual(third[0]["no"], "3")

    def test_third_place_ranking_rewards_late_gain_and_closing_speed(self):
        scored = [
            (0.70, {"no": "1", "name": "差し脚型"}),
            (0.70, {"no": "2", "name": "先行型"}),
        ]
        history = {
            "差し脚型": [{
                "finish": 3, "field_size": 10,
                "first_corner_pct": 0.90, "last_corner_pct": 0.10,
                "last3f_rank_score": 0.95,
            }],
            "先行型": [{
                "finish": 3, "field_size": 10,
                "first_corner_pct": 0.10, "last_corner_pct": 0.10,
                "last3f_rank_score": 0.10,
            }],
        }
        third = nb.role_specific_ranking(scored, history, 3, 0.40)
        self.assertEqual(third[0]["no"], "1")

    def test_third_place_ranking_handles_missing_corner_gain(self):
        scored = [(0.70, {"no": "1", "name": "closing"})]
        history = {"closing": [{
            "finish": 3, "field_size": 10,
            "last_corner_pct": 0.20, "last3f_rank_score": 0.80,
        }]}
        ranked = nb.role_specific_ranking(scored, history, 3, 0.20)
        self.assertEqual(ranked[0]["no"], "1")

    def test_role_formation_excludes_duplicate_horses(self):
        rankings = {
            1:[{"no":"1"},{"no":"2"}],
            2:[{"no":"1"},{"no":"3"},{"no":"2"}],
            3:[{"no":"2"},{"no":"3"},{"no":"4"}],
        }
        tickets = nb.build_role_trifecta_formation(rankings, 2, 3, 3)
        self.assertTrue(tickets)
        self.assertTrue(all(len(set(ticket)) == 3 for ticket in tickets))
        self.assertTrue(all(ticket[0] in {"1","2"} for ticket in tickets))

    def test_role_diagnostics_cover_multiple_blends_and_formations(self):
        diagnostics = nb.new_role_trifecta_diagnostics()
        self.assertEqual(set(diagnostics), {"0.05","0.10","0.20","0.30","0.40"})
        self.assertEqual(set(diagnostics["0.10"]), {
            "1-3-4","1-4-3","1-4-4","1-4-5","1-5-4",
            "2-3-3","2-3-4","2-4-3","2-4-4","2-5-3",
            "2-3-5","2-5-4","3-4-4","3-5-4"})

    def test_role_diagnostics_record_nonempty_results_for_any_model_bucket(self):
        scored = [
            (0.90, {"no":"1","name":"A"}),
            (0.80, {"no":"2","name":"B"}),
            (0.70, {"no":"3","name":"C"}),
            (0.60, {"no":"4","name":"D"}),
            (0.50, {"no":"5","name":"E"}),
            (0.40, {"no":"6","name":"F"}),
        ]
        history = {
            "A":[{"finish":1,"field_size":6}],
            "B":[{"finish":2,"field_size":6}],
            "C":[{"finish":3,"field_size":6}],
        }
        payout = {"trifecta":[(["1","2","3"],1500)]}
        diagnostics = nb.new_role_trifecta_diagnostics()
        nb.update_role_trifecta_diagnostics(diagnostics, scored, history, payout)
        summary = nb.summarize_role_trifecta_diagnostics(diagnostics)
        for blend in ("0.05","0.10","0.20","0.30","0.40"):
            for formation in ("1-3-4","1-4-3","1-4-4","1-4-5","1-5-4",
                              "2-3-3","2-3-4","2-4-3","2-4-4","2-5-3",
                              "2-3-5","2-5-4","3-4-4","3-5-4"):
                result = summary[blend][formation]
                self.assertEqual(result["races"], 1)
                self.assertGreater(result["tickets"], 0)
                self.assertGreaterEqual(result["stake"], 100)
                self.assertIsNotNone(result["roi"])


class RoleTrifectaUncertaintyAndSelectionTests(unittest.TestCase):
    def _fold(self, fold_no, model_rois):
        models = {}
        for model_name, roi in model_rois.items():
            stake = 50000
            returned = int(stake * roi / 100)
            models[model_name] = {
                "bet_types": {
                    "trifecta": {
                        "roi": roi, "tickets": 500, "hits": 50,
                        "stake": stake, "return": returned,
                        "profit": returned - stake,
                    }
                }
            }
        return {"fold": fold_no, "models": models}

    def _sample_role_inputs(self):
        scored = [
            (0.90, {"no":"1","name":"A"}),
            (0.80, {"no":"2","name":"B"}),
            (0.70, {"no":"3","name":"C"}),
            (0.60, {"no":"4","name":"D"}),
            (0.50, {"no":"5","name":"E"}),
            (0.40, {"no":"6","name":"F"}),
        ]
        history = {
            "A":[{"finish":1,"field_size":6}],
            "B":[{"finish":2,"field_size":6}],
            "C":[{"finish":3,"field_size":6}],
        }
        payout = {"trifecta":[(["1","2","3"],1500)]}
        return scored, history, payout

    def test_role_diagnostics_apply_sample_and_confidence_gate(self):
        scored, history, payout = self._sample_role_inputs()
        diagnostics = nb.new_role_trifecta_diagnostics()
        for _ in range(100):
            nb.update_role_trifecta_diagnostics(diagnostics, scored, history, payout)
        result = nb.summarize_role_trifecta_diagnostics(diagnostics)["0.10"]["1-4-4"]
        self.assertEqual(result["races"], 100)
        self.assertEqual(result["tickets"], 600)
        self.assertEqual(result["stake"], 60000)
        self.assertEqual(result["return"], 150000)
        self.assertEqual(result["roi"], 250.0)
        self.assertEqual(result["roi_ci95_normal_low"], 250.0)
        self.assertTrue(result["minimum_sample_met"])
        self.assertTrue(result["candidate_pass"])

    def _role_bucket_with_roi(self, default_roi, overrides=None):
        overrides = overrides or {}
        bucket = nb.new_role_trifecta_diagnostics()
        for blend, formations in bucket.items():
            for formation, stats in formations.items():
                roi = overrides.get((blend, formation), default_roi)
                races, tickets, stake = 100, 500, 50000
                returned = int(stake * roi / 100)
                race_stake, race_return = stake / races, returned / races
                stats.update({
                    "races": races, "tickets": tickets, "hits": 10,
                    "hit_races": 10, "stake": stake, "return": returned,
                    "sum_return_sq": race_return * race_return * races,
                    "sum_stake_sq": race_stake * race_stake * races,
                    "sum_stake_return": race_stake * race_return * races,
                })
        return bucket

    def test_role_strategy_is_selected_on_prior_fold_and_tested_on_next_fold(self):
        fold1 = self._fold(1, {"baseline":60.0, "trained":70.0})
        fold2 = self._fold(2, {"baseline":55.0, "trained":75.0})
        fold1["models"]["baseline"]["role_trifecta_diagnostics"] = self._role_bucket_with_roi(
            100.0, {("0.10", "2-5-3"): 130.0})
        fold1["models"]["trained"]["role_trifecta_diagnostics"] = self._role_bucket_with_roi(100.0)
        fold2["models"]["baseline"]["role_trifecta_diagnostics"] = self._role_bucket_with_roi(
            100.0, {("0.20", "2-5-5"): 250.0, ("0.10", "2-5-3"): 90.0})
        fold2["models"]["trained"]["role_trifecta_diagnostics"] = self._role_bucket_with_roi(100.0)
        result = nb.evaluate_walk_forward_selection([fold1, fold2])
        selected = result["selected_role_trifecta_strategy_by_validation_fold"][0]
        self.assertEqual(selected["model"], "baseline")
        self.assertEqual(selected["strategy"]["blend"], "0.10")
        self.assertEqual(selected["strategy"]["formation"], "2-5-3")
        self.assertEqual(selected["strategy"]["development_roi"], 130.0)
        self.assertEqual(selected["validation"]["roi"], 90.0)
        self.assertFalse(result["selected_role_trifecta_strategy_portfolio"]["candidate_pass"])

    def test_walk_forward_reports_role_diagnostics_for_selected_model_only(self):
        scored, history, payout = self._sample_role_inputs()
        fold1 = self._fold(1, {"baseline":60.0, "trained":70.0})
        fold2 = self._fold(2, {"baseline":61.0, "trained":75.0})
        for fold in (fold1, fold2):
            for model_name in ("baseline", "trained"):
                bucket = nb.new_role_trifecta_diagnostics()
                nb.update_role_trifecta_diagnostics(bucket, scored, history, payout)
                fold["models"][model_name]["role_trifecta_diagnostics"] = bucket
        result = nb.evaluate_walk_forward_selection([fold1, fold2])
        self.assertEqual(
            result["selected_role_trifecta_models_by_validation_fold"],
            [{"validation_fold":2,"model":"baseline"}],
        )
        selected = result["selected_role_trifecta_diagnostics"]["0.10"]["1-4-4"]
        self.assertEqual(selected["races"], 1)
        self.assertEqual(selected["tickets"], 6)
        self.assertEqual(selected["roi"], 250.0)
        self.assertFalse(selected["minimum_sample_met"])
        self.assertFalse(selected["candidate_pass"])


class BetTypeRaceClusterUncertaintyTests(unittest.TestCase):
    def test_profitable_roi_requires_race_level_sample_and_lower_bound(self):
        stats = {
            "race_count": 100,
            "tickets": 600,
            "stake": 60000,
            "return": 150000,
            "sum_return_sq": 2250000 * 100,
            "sum_stake_sq": 360000 * 100,
            "sum_stake_return": 900000 * 100,
        }
        result = nb.summarize_race_cluster_roi(stats)
        self.assertEqual(result["roi"], 250.0)
        self.assertEqual(result["roi_ci95_normal_low"], 250.0)
        self.assertTrue(result["minimum_sample_met"])
        self.assertTrue(result["candidate_pass"])

    def test_high_point_roi_does_not_pass_with_too_few_races(self):
        stats = {
            "race_count": 99,
            "tickets": 594,
            "stake": 59400,
            "return": 148500,
            "sum_return_sq": 2250000 * 99,
            "sum_stake_sq": 360000 * 99,
            "sum_stake_return": 900000 * 99,
        }
        result = nb.summarize_race_cluster_roi(stats)
        self.assertEqual(result["roi"], 250.0)
        self.assertFalse(result["minimum_sample_met"])
        self.assertFalse(result["candidate_pass"])

    def test_strict_promotion_gate_reports_each_required_condition(self):
        stats = {
            "race_count": 100,
            "tickets": 600,
            "stake": 60000,
            "return": 120000,
            "sum_return_sq": 240000000,
            "sum_stake_sq": 36000000,
            "sum_stake_return": 72000000,
        }
        result = nb.summarize_race_cluster_roi(stats)
        self.assertTrue(result["pass_conditions"]["roi_above_100"])
        self.assertTrue(result["pass_conditions"]["races_at_least_100"])
        self.assertTrue(result["pass_conditions"]["tickets_at_least_300"])
        self.assertTrue(result["pass_conditions"]["roi_ci95_lower_above_100"])
        self.assertEqual(result["roi_ci95_method"], "race_cluster_normal_approximation")
        self.assertTrue(result["candidate_pass"])

    def test_point_roi_above_100_is_not_enough_when_ci_lower_is_not(self):
        stats = {
            "race_count": 100,
            "tickets": 600,
            "stake": 60000,
            "return": 66000,
            "sum_return_sq": 2000000000,
            "sum_stake_sq": 36000000,
            "sum_stake_return": 39600000,
        }
        result = nb.summarize_race_cluster_roi(stats)
        self.assertTrue(result["pass_conditions"]["roi_above_100"])
        self.assertFalse(result["pass_conditions"]["roi_ci95_lower_above_100"])
        self.assertFalse(result["candidate_pass"])

    def test_variable_race_returns_produce_nonzero_uncertainty(self):
        stats = {
            "race_count": 100,
            "tickets": 600,
            "stake": 60000,
            "return": 120000,
            "sum_return_sq": 240000000,
            "sum_stake_sq": 36000000,
            "sum_stake_return": 72000000,
        }
        result = nb.summarize_race_cluster_roi(stats)
        self.assertEqual(result["roi"], 200.0)
        self.assertLess(result["roi_ci95_normal_low"], 200.0)
        self.assertGreater(result["roi_ci95_normal_high"], 200.0)


class RaceRelativeFeatureTests(unittest.TestCase):
    def test_features_are_ranked_within_race_and_weighted(self):
        parts = {
            "1": {"form": 0.9, "jockey": 0.2},
            "2": {"form": 0.5, "jockey": 0.8},
            "3": {"form": 0.1, "jockey": 0.5},
        }
        weights = {"form": 0.5, "jockey": 0.5}
        scores = nb.race_relative_scores(parts, weights)
        self.assertAlmostEqual(scores["1"], 0.50)
        self.assertAlmostEqual(scores["2"], 0.75)
        self.assertAlmostEqual(scores["3"], 0.25)

    def test_tied_feature_values_receive_average_percentile(self):
        parts = {
            "1": {"form": 0.8},
            "2": {"form": 0.8},
            "3": {"form": 0.2},
        }
        scores = nb.race_relative_scores(parts, {"form": 1.0})
        self.assertAlmostEqual(scores["1"], 0.75)
        self.assertAlmostEqual(scores["2"], 0.75)
        self.assertAlmostEqual(scores["3"], 0.0)

    def test_relative_scores_are_invariant_to_monotonic_feature_scaling(self):
        weights = {"form": 0.6, "jockey": 0.4}
        first = {
            "1": {"form": 0.9, "jockey": 0.3},
            "2": {"form": 0.5, "jockey": 0.8},
            "3": {"form": 0.1, "jockey": 0.5},
        }
        scaled = {
            no: {"form": values["form"] * 100 + 17,
                 "jockey": values["jockey"] * 10 - 3}
            for no, values in first.items()
        }
        self.assertEqual(nb.race_relative_scores(first, weights),
                         nb.race_relative_scores(scaled, weights))


class RaceRelativeMissingFeatureTests(unittest.TestCase):
    def test_missing_feature_is_neutral_and_tied_with_half_score(self):
        parts = {
            "1": {"form": 0.9},
            "2": {"form": None},
            "3": {"form": 0.5},
            "4": {"form": 0.1},
        }
        scores = nb.race_relative_scores(parts, {"form": 1.0})
        self.assertAlmostEqual(scores["2"], scores["3"])
        self.assertAlmostEqual(scores["1"], 1.0)
        self.assertAlmostEqual(scores["4"], 0.0)


class TrifectaFormationUniquenessTests(unittest.TestCase):
    def test_formation_never_repeats_a_horse_within_a_ticket(self):
        ranked = [{"no": str(i)} for i in range(1, 9)]
        for counts in ((1, 4, 4), (2, 3, 4), (2, 5, 5), (3, 5, 4)):
            tickets = nb.build_trifecta_formation(ranked, *counts)
            self.assertTrue(tickets)
            self.assertTrue(all(len(ticket) == 3 and len(set(ticket)) == 3
                                for ticket in tickets))
            self.assertEqual(len(tickets), len({tuple(ticket) for ticket in tickets}))

    def test_role_formation_never_repeats_a_horse_within_a_ticket(self):
        role_rankings = {
            1: [{"no": str(i)} for i in (1, 2, 3, 4)],
            2: [{"no": str(i)} for i in (2, 1, 5, 6)],
            3: [{"no": str(i)} for i in (1, 2, 3, 7)],
        }
        tickets = nb.build_role_trifecta_formation(role_rankings, 2, 3, 4)
        self.assertTrue(tickets)
        self.assertTrue(all(len(ticket) == 3 and len(set(ticket)) == 3
                            for ticket in tickets))
        self.assertEqual(len(tickets), len({tuple(ticket) for ticket in tickets}))

class FinishPositionDiagnosticsTests(unittest.TestCase):
    def test_topk_diagnostics_compare_ordinary_and_role_rankings(self):
        stats = nb.new_finish_position_diagnostics()
        field = [
            {"no": "1", "finish": 2},
            {"no": "2", "finish": 1},
            {"no": "3", "finish": 3},
            {"no": "4", "finish": 4},
        ]
        ordinary = [{"no": str(i)} for i in (1, 2, 3, 4)]
        role_rankings = {
            1: [{"no": str(i)} for i in (2, 1, 3, 4)],
            2: [{"no": str(i)} for i in (2, 1, 3, 4)],
            3: [{"no": str(i)} for i in (4, 3, 2, 1)],
        }
        nb.update_finish_position_diagnostics(stats, ordinary, role_rankings, field)
        summary = nb.summarize_finish_position_diagnostics(stats)
        self.assertEqual(summary["ordinary"]["1"]["topk_hits"]["1"], 0)
        self.assertEqual(summary["role_specific"]["1"]["topk_hits"]["1"], 1)
        self.assertEqual(summary["ordinary"]["2"]["topk_hits"]["1"], 1)
        self.assertEqual(summary["role_specific"]["2"]["topk_hits"]["1"], 0)
        self.assertEqual(summary["role_specific"]["3"]["topk_hits"]["1"], 0)
        self.assertEqual(summary["role_specific"]["3"]["topk_hits"]["2"], 1)
        self.assertEqual(summary["role_specific"]["3"]["mean_rank_when_found"], 2.0)
        self.assertEqual(summary["role_specific"]["3"]["mean_reciprocal_rank"], 0.5)



class ConditionSpecificFinishDiagnosticsTests(unittest.TestCase):
    def test_race_segments_include_field_size(self):
        labels = nb.race_segment_labels({
            "venue": "大井", "surface": "ダ", "going": "良",
            "distance": 1600, "field_size": 14,
        })
        self.assertEqual(labels["distance_band"], "mile_1201-1600m")
        self.assertEqual(labels["field_size_band"], "large_>12")

    def test_condition_finish_diagnostics_compare_role_rankings(self):
        stats = nb.new_condition_finish_bucket()
        field = [
            {"no": "1", "finish": 2},
            {"no": "2", "finish": 1},
            {"no": "3", "finish": 3},
            {"no": "4", "finish": 4},
        ]
        ordinary = [{"no": str(i)} for i in (1, 2, 3, 4)]
        role_rankings = {
            1: [{"no": str(i)} for i in (2, 1, 3, 4)],
            2: [{"no": str(i)} for i in (2, 1, 3, 4)],
            3: [{"no": str(i)} for i in (3, 2, 1, 4)],
        }
        nb.update_condition_finish_bucket(stats, ordinary, role_rankings, field)
        summary = nb.summarize_condition_finish_segments({"venue:大井": stats})
        self.assertEqual(summary["venue:大井"]["1"]["ordinary"]["topk_hits"]["1"], 0)
        self.assertEqual(summary["venue:大井"]["1"]["role_specific"]["topk_hits"]["1"], 1)
        self.assertEqual(summary["venue:大井"]["2"]["ordinary"]["topk_hits"]["1"], 1)
        self.assertEqual(summary["venue:大井"]["2"]["role_specific"]["topk_hits"]["1"], 0)
        self.assertEqual(summary["venue:大井"]["3"]["role_specific"]["topk_hits"]["1"], 1)
 
    def test_condition_diagnostics_exclude_missing_finish_roles_from_denominator(self):
        stats = nb.new_condition_finish_bucket()
        field = [
            {"no": "1", "finish": 1},
            {"no": "2", "finish": 4},
            {"no": "3", "finish": 5},
        ]
        ordinary = [{"no": str(i)} for i in (1, 2, 3)]
        role_rankings = {role: ordinary for role in (1, 2, 3)}
        nb.update_condition_finish_bucket(stats, ordinary, role_rankings, field)
        summary = nb.summarize_condition_finish_segments({"field_size:small": stats})
        self.assertEqual(summary["field_size:small"]["1"]["ordinary"]["races"], 1)
        self.assertEqual(summary["field_size:small"]["2"]["ordinary"]["races"], 0)
        self.assertIsNone(summary["field_size:small"]["2"]["ordinary"]["topk_rate"]["1"])
        self.assertEqual(summary["field_size:small"]["3"]["ordinary"]["races"], 0)

    def test_research_fold_export_keeps_condition_diagnostics(self):
        fold = {
            "fold": 2,
            "train_end": "2025-01-01",
            "test_start": "2025-01-02",
            "test_end": "2025-02-01",
            "models": {"trained": {"races": 123, "finish_position_diagnostics": {"ordinary": {}}}},
            "condition_finish_position_segments": {"venue:大井": {"1": {"ordinary": {"races": 20}}}},
            "condition_model_segments": {"trifecta": {"baseline": {"venue:大井": {"roi": 61.0}}}},
            "condition_model_comparisons": {"trifecta": [{"segment": "venue:大井"}]},
            "trained_condition_segments": {"venue:大井": {"races": 20}},
        }
        exported = nb.build_trifecta_research_fold(fold)
        self.assertEqual(exported["races"], 123)
        self.assertEqual(exported["condition_finish_position_segments"],
                         fold["condition_finish_position_segments"])
        self.assertEqual(exported["condition_model_segments"], fold["condition_model_segments"])
        self.assertEqual(exported["condition_model_comparisons"], fold["condition_model_comparisons"])
        self.assertEqual(exported["trained_condition_segments"], fold["trained_condition_segments"])



class GoingSuitabilityPredictionTests(unittest.TestCase):
    def test_going_feature_uses_only_prior_matching_going_and_surface(self):
        horse = {"name":"Runner A", "sire":"", "jockey":"", "age":5}
        weights = {"form":0.55,"venue":0.12,"distance":0.12,"sire":0.10,"jockey":0.07,"age":0.04}
        history = {"Runner A":[
            {"going":"重","surface":"ダート","going_quality":1.0},
            {"going":"重","surface":"ダート","going_quality":1.0},
            {"going":"良","surface":"ダート","going_quality":0.0},
            {"going":"重","surface":"芝","going_quality":0.0},
        ]}
        race = {"venue":"大井","distance":1600,"going":"重","surface":"ダート"}
        _, parts = nb.score_runner_performance(
            horse, history, {}, {}, race, weights,
            margin_weight=0.0, closing_weight=0.0, time_weight=0.0,
            going_weight=0.10
        )
        self.assertAlmostEqual(parts["going_form"], 4.0/6.0)
        self.assertAlmostEqual(parts["performance_weights"]["going"], 0.10)

    def test_going_feature_shrinks_to_neutral_without_matching_history(self):
        horse = {"name":"Runner B", "sire":"", "jockey":"", "age":5}
        weights = {"form":0.55,"venue":0.12,"distance":0.12,"sire":0.10,"jockey":0.07,"age":0.04}
        race = {"venue":"大井","distance":1600,"going":"不良","surface":"ダート"}
        _, parts = nb.score_runner_performance(
            horse, {}, {}, {}, race, weights,
            margin_weight=0.0, closing_weight=0.0, time_weight=0.0,
            going_weight=0.10
        )
        self.assertEqual(parts["going_form"], 0.5)



class RaceBudgetAutoStrategyTests(unittest.TestCase):
    def _snapshot(self, day, tickets):
        return {
            "model_version": "nar-chronological-v8-abcd-C-preoff",
            "race_key": f"大井|{day.replace('-', '')}|1",
            "date": day,
            "captured_at": f"{day}T19:50:00+09:00",
            "scheduled_post_time": f"{day}T20:00:00+09:00",
            "minutes_to_post": 10,
            "combination_predictions": tickets,
        }

    def _ticket(self, kind, number, ev, probability=0.1):
        return {
            "bet_type": kind,
            "numbers": [str(number), str(number + 1)],
            "expected_value": ev,
            "probability": probability,
            "odds": 10.0,
        }

    @mock.patch.object(nb, "load_odds_snapshots")
    def test_selects_best_type_caps_each_race_and_skips_without_candidates(self, load_snapshots):
        records = {}
        for day_num in range(1, 8):
            day = f"2026-10-{day_num:02d}"
            records[str(day_num)] = self._snapshot(day, [self._ticket("wide", 100 + day_num, 0.90)])
        records["8"] = self._snapshot(
            "2026-10-08",
            [self._ticket("wide", i, 1.50) for i in range(1, 26)] +
            [self._ticket("place", i, 1.20) for i in range(30, 55)]
        )
        records["9"] = self._snapshot("2026-10-09", [self._ticket("wide", 60, 0.90)])
        records["10"] = self._snapshot(
            "2026-10-10", [self._ticket("place", 70, 1.30)]
        )
        load_snapshots.return_value = records

        payout = {
            "win": [(["1"], 100)],
            "_index": {"win": {}, "place": {}, "wide": {}, "quinella": {},
                       "exacta": {}, "trio": {}, "trifecta": {}},
        }
        payouts = {
            ("大井", f"202610{day_num:02d}", "1"): payout
            for day_num in range(1, 11)
        }
        validation = {
            "bet_types": {
                kind: {
                    "selected_threshold": "1.05",
                    "development": {"1.05": {"candidate_pass": True}},
                }
                for kind in ("wide", "place")
            }
        }

        result = nb.race_budget_auto_strategy_backtest(payouts, validation)
        self.assertEqual(result["status"], "awaiting_holdout_sample")
        self.assertEqual(result["holdout_races_with_bets"], 2)
        self.assertEqual(result["skipped_races"], 1)
        self.assertEqual(result["selected_bet_type_race_counts"]["wide"], 1)
        self.assertEqual(result["selected_bet_type_race_counts"]["place"], 1)
        self.assertEqual(result["max_race_stake_jpy"], 2000)
        self.assertEqual(result["stake_jpy"], 2100)
        self.assertLessEqual(result["max_race_stake_jpy"], 2000)

    @mock.patch.object(nb, "load_odds_snapshots")
    def test_no_development_validated_type_means_no_bets(self, load_snapshots):
        records = {
            str(i): self._snapshot(
                f"2026-10-{i:02d}", [self._ticket("wide", i, 0.90)]
            )
            for i in range(1, 11)
        }
        records["10"] = self._snapshot(
            "2026-10-10", [self._ticket("wide", 1, 2.0)]
        )
        load_snapshots.return_value = records
        payout = {"win": [(["1"], 100)]}
        payouts = {
            ("大井", f"202610{i:02d}", "1"): payout for i in range(1, 11)
        }

        result = nb.race_budget_auto_strategy_backtest(payouts, {"bet_types": {}})
        self.assertEqual(result["status"], "awaiting_development_validation")
        self.assertEqual(result["stake_jpy"], 0)
        self.assertEqual(result["holdout_races_with_bets"], 0)
