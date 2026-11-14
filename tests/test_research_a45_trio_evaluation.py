import unittest

from scripts import research_flat_lambdarank_a45_trio_evaluation as evaluator


class A45TrioEvaluationTests(unittest.TestCase):
    def test_ticket_payout_is_scored_per_100_yen_ticket(self):
        strategy = {
            "ticket_count": 2,
            "all_tickets_quoted": True,
            "tickets": [
                {"numbers": ["1", "2", "3"], "odds": 12.5},
                {"numbers": ["1", "2", "4"], "odds": 25.0},
            ],
        }
        scored = evaluator.score_strategy(
            strategy,
            {("1", "2", "3"): 1250.0},
        )
        self.assertEqual(scored["stake_yen"], 200)
        self.assertEqual(scored["hit_tickets"], 1)
        self.assertEqual(scored["return_yen"], 1250.0)
        self.assertEqual(scored["profit_yen"], 1050.0)
        self.assertEqual(scored["roi_pct"], 625.0)

    def test_dead_heat_multiple_trio_winners_are_summed(self):
        strategy = {
            "ticket_count": 3,
            "all_tickets_quoted": True,
            "tickets": [
                {"numbers": ["1", "2", "3"], "odds": 10.0},
                {"numbers": ["1", "2", "4"], "odds": 20.0},
                {"numbers": ["1", "3", "4"], "odds": 30.0},
            ],
        }
        scored = evaluator.score_strategy(
            strategy,
            {("1", "2", "3"): 900.0, ("1", "2", "4"): 1400.0},
        )
        self.assertEqual(scored["hit_tickets"], 2)
        self.assertEqual(scored["return_yen"], 2300.0)
        self.assertEqual(scored["stake_yen"], 300)


if __name__ == "__main__":
    unittest.main()
