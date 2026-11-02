import sys
import gzip
import json
import tempfile
import unittest
from collections import defaultdict
from unittest import mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import nar_backtest as nb
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import capture_nar_odds as capture


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


class OddsSnapshotBacktestTests(unittest.TestCase):
    def test_loader_keeps_closest_valid_preoff_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "data" / "odds-snapshots"
            folder.mkdir(parents=True)
            path = folder / "2026-11-03.jsonl.gz"
            rows = [
                {"race_key":"大井|20261103|10","captured_at":"2026-11-03T15:20:00+09:00",
                 "scheduled_post_time":"2026-11-03T15:30:00+09:00","minutes_to_post":10,
                 "odds":[{"type":"単勝","n1":"1","odds":2.4}]},
                {"race_key":"大井|20261103|10","captured_at":"2026-11-03T15:27:00+09:00",
                 "scheduled_post_time":"2026-11-03T15:30:00+09:00","minutes_to_post":3,
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
            self.assertEqual(loaded[key]["minutes_to_post"], 3)
            self.assertEqual(loaded[key]["odds"][0]["odds"], 2.1)


class FieldSizeNormalizationTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
