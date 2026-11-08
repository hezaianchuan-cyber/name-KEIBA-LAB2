import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import nar_bet_type_comparison as comparison


class BetTypeComparisonTests(unittest.TestCase):
    def setUp(self):
        self.ranked = [{"no": "1"}, {"no": "2"}, {"no": "3"}, {"no": "4"}]

    def test_ticket_counts_and_order(self):
        self.assertEqual(len(comparison.tickets_for_ranked("win", self.ranked)), 1)
        self.assertEqual(len(comparison.tickets_for_ranked("wide_top3", self.ranked)), 3)
        self.assertEqual(comparison.tickets_for_ranked("quinella_top2", self.ranked), [["1", "2"]])
        self.assertEqual(comparison.tickets_for_ranked("exacta_top2", self.ranked), [["1", "2"]])
        self.assertEqual(comparison.tickets_for_ranked("trio_top3", self.ranked), [["1", "2", "3"]])
        trifecta = comparison.tickets_for_ranked("trifecta_top3_box", self.ranked)
        self.assertEqual(len(trifecta), 6)
        self.assertIn(["1", "2", "3"], trifecta)
        self.assertIn(["3", "2", "1"], trifecta)

    def test_bootstrap_requires_enough_races(self):
        result = comparison._cluster_bootstrap_roi([100] * 10, [100] * 10)
        self.assertIsNone(result["low"])
        self.assertEqual(result["reason"], "fewer_than_20_races")


if __name__ == "__main__":
    unittest.main()
