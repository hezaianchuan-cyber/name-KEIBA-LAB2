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

    def test_wide_top3_evaluator_uses_three_pairs_and_fixed_stake(self):
        group = [
            {"race_key": "大井|2026-01-01|1", "horse_no": "1"},
            {"race_key": "大井|2026-01-01|1", "horse_no": "2"},
            {"race_key": "大井|2026-01-01|1", "horse_no": "3"},
            {"race_key": "大井|2026-01-01|1", "horse_no": "4"},
        ]
        key = ml.nb.race_key("大井", "20260101", "1")
        payouts = {key: {"_index": {"wide": {("1", "2"): 500}}}}
        metrics = ml.evaluate_wide_top3([(group, [4.0, 3.0, 2.0, 1.0])], payouts)
        self.assertEqual(metrics["races"], 1)
        self.assertEqual(metrics["hit_races"], 1)
        self.assertEqual(metrics["stake_jpy"], 300)
        self.assertEqual(metrics["return_jpy"], 500)
        self.assertEqual(metrics["roi_pct"], 166.667)
        self.assertEqual(metrics["hit_rate_pct"], 100.0)

    def test_race_relative_features_compare_each_runner_with_today_field(self):
        rows = [
            {"recent_finish_pct": 0.9, "recent_top3_rate": 0.8,
             "recent_last_corner_pct": 0.2, "recent_last3f_rank": 0.9,
             "recent_opponent_strength": 0.8, "recent_time_per_km": 60.0,
             "venue_top3_rate": 0.7, "distance_top3_rate": 0.8,
             "jockey_top3_rate": 0.9, "body_weight_delta_pct": 0.01},
            {"recent_finish_pct": 0.5, "recent_top3_rate": 0.5,
             "recent_last_corner_pct": 0.5, "recent_last3f_rank": 0.5,
             "recent_opponent_strength": 0.5, "recent_time_per_km": 62.0,
             "venue_top3_rate": 0.5, "distance_top3_rate": 0.5,
             "jockey_top3_rate": 0.5, "body_weight_delta_pct": 0.00},
            {"recent_finish_pct": 0.1, "recent_top3_rate": 0.2,
             "recent_last_corner_pct": 0.8, "recent_last3f_rank": 0.1,
             "recent_opponent_strength": 0.2, "recent_time_per_km": 64.0,
             "venue_top3_rate": 0.3, "distance_top3_rate": 0.2,
             "jockey_top3_rate": 0.1, "body_weight_delta_pct": -0.01},
        ]
        result = ml.add_race_relative_features(rows)
        self.assertGreater(result[0]["race_rel_recent_finish_pct"],
                           result[1]["race_rel_recent_finish_pct"])
        self.assertGreater(result[1]["race_rel_recent_finish_pct"],
                           result[2]["race_rel_recent_finish_pct"])
        self.assertGreater(result[0]["race_rel_recent_time_per_km"],
                           result[2]["race_rel_recent_time_per_km"])
        for row in result:
            for key in ("race_rel_recent_finish_pct",
                        "race_rel_recent_top3_rate",
                        "race_rel_recent_last_corner_pct",
                        "race_rel_recent_last3f_rank",
                        "race_rel_recent_opponent_strength",
                        "race_rel_recent_time_per_km",
                        "race_rel_venue_top3_rate",
                        "race_rel_distance_top3_rate",
                        "race_rel_jockey_top3_rate",
                        "race_rel_body_weight_delta_pct"):
                self.assertTrue(-5.0 <= row[key] <= 5.0)

    def test_raw_feature_set_contains_all_added_pre_race_signals(self):
        expected = {
            "frame_no", "carried_weight", "body_weight", "body_weight_change",
            "prior_body_weight_avg", "body_weight_delta_pct",
            "recent_time_trend", "recent_opponent_strength",
            "recent_first_corner_pct", "recent_last_corner_pct",
            "recent_position_gain", "predicted_front_count",
            "front_crowding_ratio", "race_rel_recent_finish_pct",
            "race_rel_recent_top3_rate", "race_rel_recent_last_corner_pct",
            "race_rel_recent_last3f_rank", "race_rel_recent_opponent_strength",
            "race_rel_recent_time_per_km", "race_rel_venue_top3_rate",
            "race_rel_distance_top3_rate", "race_rel_jockey_top3_rate",
            "race_rel_body_weight_delta_pct",
        }
        self.assertTrue(expected.issubset(set(ml.RAW_FEATURES)))


if __name__ == "__main__":
    unittest.main()
