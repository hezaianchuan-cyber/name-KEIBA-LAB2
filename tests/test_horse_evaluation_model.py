import unittest

from scripts.horse_evaluation_model import evaluate_field, MODEL_VERSION


class HorseEvaluationModelTests(unittest.TestCase):
    def setUp(self):
        self.field = [
            {"no": "1", "name": "A", "jockey": "J1"},
            {"no": "2", "name": "B", "jockey": "J2"},
            {"no": "3", "name": "C", "jockey": "J3"},
            {"no": "4", "name": "D", "jockey": "J4"},
        ]
        self.history = {
            "A": [
                {"finish": 5, "field_size": 14, "venue": "大井", "distance": 1600,
                 "going": "良", "margin_quality": 0.9, "last3f_rank_score": 0.95,
                 "first_corner_pct": 0.72, "position_gain": 0.3},
                {"finish": 4, "field_size": 14, "venue": "大井", "distance": 1600,
                 "going": "良", "margin_quality": 0.9, "last3f_rank_score": 0.9,
                 "first_corner_pct": 0.7, "position_gain": 0.25},
            ],
            "B": [
                {"finish": 1, "field_size": 5, "venue": "船橋", "distance": 1200,
                 "going": "稍重", "margin_quality": 0.3, "last3f_rank_score": 0.4},
                {"finish": 4, "field_size": 5, "venue": "船橋", "distance": 1200,
                 "going": "稍重", "margin_quality": 0.3, "last3f_rank_score": 0.4},
            ],
        }

    def test_returns_all_runners_and_separates_win_from_top3(self):
        result = evaluate_field(self.field, self.history,
                                {"venue": "大井", "distance": 1600, "going": "良"})
        self.assertEqual(result["model_version"], MODEL_VERSION)
        self.assertEqual(len(result["predictions"]), 4)
        self.assertEqual(set(result["win_ranking"]), {"1", "2", "3", "4"})
        self.assertEqual(set(result["top3_ranking"]), {"1", "2", "3", "4"})
        self.assertAlmostEqual(result["probability_checks"]["win_sum"], 1.0)
        self.assertAlmostEqual(result["probability_checks"]["top3_sum"], 3.0)

    def test_close_fifth_in_large_field_can_beat_easy_win_in_small_field(self):
        result = evaluate_field(self.field[:2], self.history,
                                {"venue": "大井", "distance": 1600, "going": "良"})
        scores = {row["horse_no"]: row["score"] for row in result["predictions"]}
        self.assertGreater(scores["1"], scores["2"])

    def test_missing_history_is_neutral_and_never_drops_runner(self):
        result = evaluate_field(self.field, {}, {"venue": "大井", "distance": 1600})
        self.assertEqual(len(result["predictions"]), 4)
        for row in result["predictions"]:
            self.assertTrue(all(0 <= value <= 1 for value in row["axes"].values()))

    def test_front_runner_crowding_changes_pace_axis_only(self):
        crowded = evaluate_field(
            self.field, self.history,
            {"venue": "大井", "distance": 1600,
             "predicted_styles": ["front", "front", "front", "closer"]},
        )
        normal = evaluate_field(
            self.field, self.history,
            {"venue": "大井", "distance": 1600,
             "predicted_styles": ["front", "stalker", "closer", "deep_closer"]},
        )
        crowded_a = next(x for x in crowded["predictions"] if x["horse_no"] == "1")
        normal_a = next(x for x in normal["predictions"] if x["horse_no"] == "1")
        self.assertGreater(crowded_a["axes"]["pace"], normal_a["axes"]["pace"])
        for key in ("ability", "condition", "suitability", "race_content", "partnership"):
            self.assertEqual(crowded_a["axes"][key], normal_a["axes"][key])

    def test_invalid_weights_are_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_field(self.field, {}, {"venue": "大井"},
                           axis_weights={"ability": 1.0})


if __name__ == "__main__":
    unittest.main()
