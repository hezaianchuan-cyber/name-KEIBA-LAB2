import importlib.util
import sys
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "nar_trio_combination_probability.py"
SPEC = importlib.util.spec_from_file_location("trio_combo", SCRIPT)
combo = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = combo
SPEC.loader.exec_module(combo)


class TrioCombinationProbabilityTests(unittest.TestCase):
    def test_unordered_trio_probability_sums_over_all_six_orders(self):
        p = {"1": 0.5, "2": 0.3, "3": 0.2}
        probability = combo.plackett_luce_trio_probability(p, ("1", "2", "3"))
        self.assertAlmostEqual(probability, 1.0, places=10)

    def test_trio_probability_is_invariant_to_ticket_order(self):
        p = {"1": 0.4, "2": 0.3, "3": 0.2, "4": 0.1}
        a = combo.plackett_luce_trio_probability(p, ("1", "2", "3"))
        b = combo.plackett_luce_trio_probability(p, ("3", "1", "2"))
        self.assertAlmostEqual(a, b, places=12)

    def test_all_triples_sum_to_one_for_four_horses(self):
        p = {"1": 0.4, "2": 0.3, "3": 0.2, "4": 0.1}
        total = sum(
            combo.plackett_luce_trio_probability(p, trio)
            for trio in __import__("itertools").combinations(p, 3)
        )
        self.assertAlmostEqual(total, 1.0, places=10)

    def test_top_seven_produces_35_unique_unordered_combinations(self):
        p = {str(i): 1.0 / i for i in range(1, 9)}
        ranked = combo.rank_trio_combinations(p)
        self.assertEqual(len(ranked), 35)
        tickets = [tuple(item["combo"]) for item in ranked]
        self.assertEqual(len(set(tickets)), 35)
        self.assertTrue(all(len(set(ticket)) == 3 for ticket in tickets))

    def test_model_probabilities_use_softmax_and_normalize(self):
        rows = [
            {"horse_no": "1", "score": 0.0},
            {"horse_no": "2", "score": -0.1},
            {"horse_no": "3", "score": -0.2},
        ]
        p = combo.model_probabilities(rows, 0.2)
        self.assertAlmostEqual(sum(p.values()), 1.0, places=12)
        self.assertGreater(p["1"], p["2"])
        self.assertGreater(p["2"], p["3"])

    def test_market_probabilities_ignore_place_odds_and_normalize(self):
        odds = [
            {"type": "単勝", "n1": "1", "odds": 2.0},
            {"type": "単勝", "n1": "2", "odds": 4.0},
            {"type": "単勝", "n1": "3", "odds": 8.0},
            {"type": "複勝", "n1": "4", "odds": 1.2},
        ]
        p = combo.market_probabilities(odds)
        self.assertAlmostEqual(sum(p.values()), 1.0, places=12)
        self.assertNotIn("4", p)
        self.assertGreater(p["1"], p["2"])

    def test_blend_uses_identical_runner_intersection(self):
        model = {"1": 0.5, "2": 0.3, "3": 0.2, "4": 0.1}
        market = {"1": 0.4, "2": 0.3, "3": 0.2}
        p = combo.blend_probabilities(model, market, 0.5)
        self.assertEqual(set(p), {"1", "2", "3"})
        self.assertAlmostEqual(sum(p.values()), 1.0, places=12)

    def test_ticket_budget_is_equal_for_10_12_and_19(self):
        for count in (10, 12, 19):
            units = combo.allocate_budget_units(count)
            self.assertEqual(sum(units), 19)
            self.assertTrue(all(unit >= 1 for unit in units))

    def test_existing_245_rank_sum_baseline_is_preserved(self):
        tickets = [
            ("1", "4", "5"),
            ("1", "2", "6"),
            ("2", "3", "7"),
        ]
        ranked = combo.rank_existing_tickets(tickets, ["1", "2", "3", "4", "5", "6", "7"])
        self.assertEqual(ranked[0]["combo"], ["1", "2", "6"])
        self.assertIsNone(ranked[0]["probability"])

    def test_candidate_coverage_is_distinct_from_exact_ticket_hit(self):
        race = {
            "payouts": {("1", "2", "3"): 500},
            "candidate_horses": {"model": ["1", "2", "3", "4", "5", "6", "7"]},
        }
        result = combo.evaluate_tickets(
            [race], lambda _race: [("1", "2", "4")], 1, candidate_strategy="model"
        )
        self.assertEqual(result["hit_races"], 0)
        self.assertEqual(result["candidate_top3_coverage_hits"], 1)
        self.assertEqual(result["candidate_top3_coverage_pct"], 100.0)
        self.assertEqual(result["stake_jpy"], 1900)

    def test_invalid_trio_is_rejected(self):
        with self.assertRaises(ValueError):
            combo.plackett_luce_trio_probability({"1": 0.5, "2": 0.3, "3": 0.2}, ("1", "1", "2"))


if __name__ == "__main__":
    unittest.main()
