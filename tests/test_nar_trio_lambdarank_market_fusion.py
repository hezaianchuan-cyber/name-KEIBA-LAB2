import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import nar_trio_lambdarank_market_fusion_5day as fusion


class TrioLambdaRankMarketFusionTests(unittest.TestCase):
    def test_market_probabilities_use_only_win_odds_and_normalize(self):
        snapshot = {"odds": [
            {"type": "単勝", "n1": "1", "odds": 2.0},
            {"type": "単勝", "n1": "2", "odds": 4.0},
            {"type": "３連複", "n1": "1", "n2": "2", "n3": "3", "odds": 10.0},
        ]}
        probs = fusion.market_win_probabilities(snapshot)
        self.assertAlmostEqual(probs["1"], 2 / 3)
        self.assertAlmostEqual(probs["2"], 1 / 3)
        self.assertAlmostEqual(sum(probs.values()), 1.0)

    def test_percentile_scores_are_invariant_to_raw_scale(self):
        self.assertEqual(
            fusion.percentile_scores({"1": 100.0, "2": 10.0, "3": -5.0}),
            {"1": 1.0, "2": 0.5, "3": 0.0},
        )

    def test_blend_endpoints_and_equal_weight(self):
        model = {"1": 1.0, "2": 0.5, "3": 0.0}
        market = {"1": 0.0, "2": 0.5, "3": 1.0}
        self.assertEqual(fusion.blend_rank_scores(model, market, 0.0), model)
        self.assertEqual(fusion.blend_rank_scores(model, market, 1.0), market)
        self.assertEqual(fusion.blend_rank_scores(model, market, 0.5),
                         {"1": 0.5, "2": 0.5, "3": 0.5})

    def test_trio_ticket_builder_returns_unique_unordered_combinations(self):
        group = [{"horse_no": str(i)} for i in range(1, 7)]
        scores = {str(i): 1.0 / i for i in range(1, 7)}
        tickets = fusion.build_trio_tickets(group, scores, limit=12)
        self.assertEqual(len(tickets), 12)
        self.assertEqual(len({tuple(ticket) for ticket in tickets}), 12)
        self.assertTrue(all(len(ticket) == 3 for ticket in tickets))
        self.assertTrue(all(ticket == sorted(ticket, key=int) for ticket in tickets))

    def test_weight_selection_uses_development_races_only(self):
        group = [
            {"horse_no": "1", "finish": 1},
            {"horse_no": "2", "finish": 2},
            {"horse_no": "3", "finish": 3},
            {"horse_no": "4", "finish": 4},
        ]
        payout = {"trio": [(["1", "2", "3"], 500)]}
        item = {
            "group": group, "payout": payout,
            "model_rank_scores": {"1": 1.0, "2": 0.8, "3": 0.6, "4": 0.0},
            "market_rank_scores": {"1": 0.0, "2": 0.2, "3": 0.4, "4": 1.0},
        }
        selected, results = fusion.select_market_weight([item])
        self.assertEqual(selected, 0.0)
        self.assertEqual(results["0.00"]["trio_hit_races"], 1)


if __name__ == "__main__":
    unittest.main()
