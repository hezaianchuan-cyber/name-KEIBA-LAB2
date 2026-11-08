import datetime as dt
import unittest

from scripts import nar_ml_challenger_backtest as ml


class RichPreRaceFeatureTests(unittest.TestCase):
    def _fixture(self):
        races = {}
        horses = {}
        names = [("A", "20200101", "父A", "母A"),
                 ("B", "20200102", "父B", "母B"),
                 ("C", "20200103", "父C", "母C")]
        for day in range(1, 5):
            date = dt.date(2026, 1, day)
            key = ("大井", date.strftime("%Y%m%d"), "1")
            races[key] = {
                "key": key, "date": date, "venue": "大井", "race": "1",
                "distance": 1600, "field_size": 3, "surface": "ダート",
                "going": "良",
                "corner_positions": [
                    {"1": 1, "2": 2, "3": 3},
                    {"1": 3, "2": 2, "3": 1},
                ],
            }
            field = []
            for index, (name, birth, sire, dam) in enumerate(names, start=1):
                field.append({
                    "no": str(index), "name": name, "birth_date": birth,
                    "sex": "牡", "sire": sire, "dam": dam,
                    "jockey": f"騎手{index}", "age": 5,
                    "frame_no": index, "carried": 56 + index,
                    "body_weight": 470 + index * 5 + (day - 1) * 10,
                    "body_weight_change": -2 + index,
                    "finish": index,
                    "time_seconds": 100.0 - day + index,
                    "last3f": 38.0 + index,
                })
            horses[key] = field
        return races, horses

    def test_new_features_are_pre_race_and_populated_from_prior_history(self):
        races, horses = self._fixture()
        rows, dates = ml.build_rows(races, horses)
        self.assertEqual(len(dates), 4)
        day_four = [row for row in rows if row["date"] == "2026-01-04"]
        self.assertEqual(len(day_four), 3)
        first = next(row for row in day_four if row["horse_no"] == "1")

        self.assertEqual(first["frame_no"], 1.0)
        self.assertEqual(first["carried_weight"], 57.0)
        self.assertEqual(first["body_weight"], 505.0)
        self.assertEqual(first["body_weight_change"], -1.0)
        self.assertGreater(first["prior_body_weight_avg"], 0)
        self.assertGreater(first["body_weight_delta_pct"], 0)
        self.assertEqual(first["predicted_front_count"], 1.0)
        self.assertAlmostEqual(first["front_crowding_ratio"], 1 / 3)
        self.assertAlmostEqual(first["recent_first_corner_pct"], 0.0)
        self.assertAlmostEqual(first["recent_last_corner_pct"], 1.0)
        self.assertAlmostEqual(first["recent_position_gain"], -1.0)
        self.assertGreater(first["recent_time_trend"], 0.5)
        self.assertTrue(0 <= first["recent_opponent_strength"] <= 1)

    def test_walk_forward_splits_are_chronological_and_test_blocks_do_not_overlap(self):
        dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(100)]
        folds = ml.build_walk_forward_splits(dates)
        self.assertEqual(len(folds), 3)
        for fold in folds:
            self.assertLess(fold["train_dates"][-1], fold["development_dates"][0])
            self.assertLess(fold["development_dates"][-1], fold["test_dates"][0])
        tests = [day for fold in folds for day in fold["test_dates"]]
        self.assertEqual(len(tests), len(set(tests)))

    def test_raw_feature_set_contains_all_added_pre_race_signals(self):
        expected = {
            "frame_no", "carried_weight", "body_weight", "body_weight_change",
            "prior_body_weight_avg", "body_weight_delta_pct",
            "recent_time_trend", "recent_opponent_strength",
            "recent_first_corner_pct", "recent_last_corner_pct",
            "recent_position_gain", "predicted_front_count",
            "front_crowding_ratio",
        }
        self.assertTrue(expected.issubset(set(ml.RAW_FEATURES)))


if __name__ == "__main__":
    unittest.main()
