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

    def test_learned_compatibility_ticket_builder_returns_unique_trios(self):
        group = [{"horse_no": str(i)} for i in range(1, 5)]
        scores = {"1": 1.0, "2": 0.8, "3": 0.6, "4": 0.4}
        pair_scores = {(str(a), str(b)): (a + b) / 10 for a in range(1, 5) for b in range(a + 1, 5)}
        tickets = fusion.build_trio_tickets_learned(group, scores, pair_scores, compatibility_weight=0.5)
        self.assertEqual(len(tickets), 4)
        self.assertEqual(len({tuple(x) for x in tickets}), 4)

    def test_axis_strategy_builds_two_wide_and_six_ev_proxy_trios(self):
        group = [{"horse_no": str(i), "finish": i} for i in range(1, 8)]
        scores = {str(i): 1.0 - (i - 1) * 0.1 for i in range(1, 8)}
        pair_scores = {
            (str(a), str(b)): 0.2 + 0.05 * (8 - a - b)
            for a in range(1, 8) for b in range(a + 1, 8)
        }
        market_probs = {str(i): (8 - i) / 28 for i in range(1, 8)}
        strategy = fusion.build_axis_wide_trio_strategy(
            group, scores, pair_scores, market_probs, compatibility_weight=0.5
        )
        self.assertEqual(strategy["axis"], "1")
        self.assertEqual(strategy["wide_tickets"], [["1", "2"], ["1", "3"]])
        self.assertEqual(len(strategy["trio_opponents"]), 4)
        self.assertNotIn("1", strategy["trio_opponents"])
        self.assertEqual(len(strategy["trio_tickets"]), 6)
        self.assertTrue(all(len(ticket) == 3 for ticket in strategy["trio_tickets"]))
        self.assertEqual(len({tuple(ticket) for ticket in strategy["trio_tickets"]}), 6)

    def test_axis_strategy_evaluates_800_yen_per_race(self):
        group = [{"horse_no": str(i), "finish": i} for i in range(1, 7)]
        pair_scores = {
            (str(a), str(b)): 0.5
            for a in range(1, 7) for b in range(a + 1, 7)
        }
        model_scores = {str(i): 1.0 - (i - 1) * 0.1 for i in range(1, 7)}
        market_scores = dict(model_scores)
        market_probs = {str(i): (7 - i) / 21 for i in range(1, 7)}
        strategy = fusion.build_axis_wide_trio_strategy(
            group, model_scores, pair_scores, market_probs, compatibility_weight=0.5
        )
        item = {
            "group": group,
            "payout": {
                "wide": [(strategy["wide_tickets"][0], 300)],
                "trio": [(strategy["trio_tickets"][0], 500)],
            },
            "model_rank_scores": model_scores,
            "market_rank_scores": market_scores,
            "market_win_probabilities": market_probs,
            "learned_pair_scores": pair_scores,
        }
        result = fusion.evaluate_axis_wide_trio_strategy([item], 0.5, 0.5)
        self.assertEqual(result["stake_per_race_jpy"], 800)
        self.assertEqual(result["wide_stake_jpy"], 200)
        self.assertEqual(result["trio_stake_jpy"], 600)
        self.assertEqual(result["combined_stake_jpy"], 800)
        self.assertEqual(result["wide_return_jpy"], 300)
        self.assertEqual(result["trio_return_jpy"], 500)
        self.assertEqual(result["combined_return_jpy"], 800)

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
