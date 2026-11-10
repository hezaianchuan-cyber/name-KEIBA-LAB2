import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import nar_exacta_research as exacta
import nar_backtest as nb


class ExactaResearchTests(unittest.TestCase):
    def setUp(self):
        self.group = [{"horse_no": str(i)} for i in range(1, 9)]
        self.first = {str(i): 1.0 / i for i in range(1, 9)}
        self.second = {str(i): 1.0 / (9 - i) for i in range(1, 9)}

    def test_six_ordered_tickets_and_no_self_pair(self):
        tickets = exacta.build_exacta_tickets(self.group, self.first, self.second)
        self.assertEqual(len(tickets), 6)
        self.assertTrue(all(len(t) == 2 and t[0] != t[1] for t in tickets))
        self.assertEqual(len({tuple(t) for t in tickets}), 6)

    def test_requires_complete_scores(self):
        with self.assertRaises(ValueError):
            exacta.build_exacta_tickets(self.group, {"1": 1.0}, self.second)

    def test_exacta_payout_presence_requires_official_rows(self):
        self.assertFalse(exacta.has_exacta_payout_data(None))
        self.assertFalse(exacta.has_exacta_payout_data({"exacta": []}))
        self.assertTrue(exacta.has_exacta_payout_data({"exacta": [(["1", "2"], 500)]}))
        self.assertTrue(exacta.has_exacta_payout_data({"_index": {"exacta": {("1", "2"): 500}}}))

    def test_missing_exacta_payout_is_excluded_not_counted_as_loss(self):
        race = {
            "group": [
                {"horse_no": str(i), "finish": 1 if i == 1 else 2 if i == 2 else i}
                for i in range(1, 9)
            ],
            "winner_model_probabilities": self.first,
            "podium_model_probabilities": self.second,
            "payout": {"exacta": []},
            "date": "2026-11-07",
            "race_key": "test-race",
        }
        result = exacta.evaluate([race])
        self.assertEqual(result["eligible_result_races"], 1)
        self.assertEqual(result["races"], 0)
        self.assertEqual(result["payout_data_missing_races"], 1)
        self.assertEqual(result["payout_data_coverage_pct"], 0.0)
        self.assertIsNone(result["roi_pct"])
        self.assertEqual(result["stake_jpy"], 0)


if __name__ == "__main__":
    unittest.main()
