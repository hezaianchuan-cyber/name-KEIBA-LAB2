import unittest

from scripts.nar_model_g import build_g_predictions, select_g_strategy


class ModelGTests(unittest.TestCase):
    def setUp(self):
        self.runners = [
            {"horse_no": str(i), "score": 1.0 - i * 0.1,
             "win_probability": [0.40, 0.25, 0.15, 0.12, 0.08][i-1],
             "win_odds": [2.0, 3.0, 5.0, 8.0, 12.0][i-1],
             "features": {"margin_form": 0.8, "closing_form": 0.8,
                          "time_form": 0.7, "going_form": 0.8}}
            for i in range(1, 6)
        ]
        self.odds = [
            {"type": "単勝", "n1": str(i), "odds": odds}
            for i, odds in enumerate([2.0, 3.0, 5.0, 8.0, 12.0], 1)
        ]

    def test_g_predictions_are_normalized_and_unique(self):
        result = build_g_predictions(self.runners, self.odds, 1000)
        self.assertEqual(len(result), 5)
        self.assertAlmostEqual(sum(x["win_probability"] for x in result), 1.0, places=6)
        self.assertEqual(len({x["horse_no"] for x in result}), 5)

    def test_skip_without_calibration_sample(self):
        result = select_g_strategy({}, self.runners, [], 100)
        self.assertEqual(result["status"], "skip_insufficient_evidence")
        self.assertEqual(result["stake_jpy"], 0)

    def test_skip_without_odds(self):
        runners = [dict(x, win_odds=None) for x in self.runners]
        result = select_g_strategy({}, runners, [], 1000)
        self.assertEqual(result["status"], "skip_insufficient_evidence")

    def test_budget_and_duplicate_horse_protection(self):
        tickets = [
            {"bet_type": "wide", "numbers": ["1", "2"], "probability": 0.5,
             "odds": 3.0, "expected_value": 1.5},
            {"bet_type": "wide", "numbers": ["2", "1"], "probability": 0.5,
             "odds": 3.0, "expected_value": 1.5},
            {"bet_type": "trifecta", "numbers": ["1", "1", "3"], "probability": 0.1,
             "odds": 20.0, "expected_value": 2.0},
        ]
        result = select_g_strategy({}, self.runners, tickets, 1000)
        self.assertEqual(result["selected_bet_type"], "wide")
        self.assertEqual(len(result["tickets"]), 1)
        self.assertLessEqual(result["stake_jpy"], 2000)
        self.assertEqual(result["tickets"][0]["numbers"], ["1", "2"])


if __name__ == "__main__":
    unittest.main()
