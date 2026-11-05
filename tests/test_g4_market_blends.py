import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import nar_model_g4_preoff_backtest as g4


class PreoffMarketBlendComparisonTests(unittest.TestCase):
    def test_reports_all_predeclared_weights_on_the_same_races(self):
        rows = [
            {
                "blend_candidates": {
                    "0.00": {"top": {"stake": 100, "return": 200, "hit": 1}, "top3_hit": 1},
                    "0.25": {"top": {"stake": 100, "return": 0, "hit": 0}, "top3_hit": 1},
                    "0.50": {"top": {"stake": 100, "return": 0, "hit": 0}, "top3_hit": 0},
                    "0.75": {"top": {"stake": 100, "return": 0, "hit": 0}, "top3_hit": 0},
                    "1.00": {"top": {"stake": 100, "return": 0, "hit": 0}, "top3_hit": 0},
                }
            },
            {
                "blend_candidates": {
                    "0.00": {"top": {"stake": 100, "return": 0, "hit": 0}, "top3_hit": 0},
                    "0.25": {"top": {"stake": 100, "return": 300, "hit": 1}, "top3_hit": 1},
                    "0.50": {"top": {"stake": 100, "return": 300, "hit": 1}, "top3_hit": 1},
                    "0.75": {"top": {"stake": 100, "return": 300, "hit": 1}, "top3_hit": 1},
                    "1.00": {"top": {"stake": 100, "return": 300, "hit": 1}, "top3_hit": 1},
                }
            },
        ]

        report = g4.summarize_blend_candidates(rows)

        self.assertEqual(list(report), ["0.00", "0.25", "0.50", "0.75", "1.00"])
        self.assertEqual(report["0.00"]["races"], 2)
        self.assertEqual(report["0.00"]["top1"]["hits"], 1)
        self.assertEqual(report["0.25"]["top1"]["hits"], 1)
        self.assertEqual(report["0.25"]["top3_coverage_hits"], 2)
        self.assertEqual(report["0.50"]["top3_coverage_pct"], 50.0)
        self.assertEqual(report["1.00"]["market_probability_weight_pct"], 100.0)

    def test_rank_disagreement_counts_winners_in_both_directions(self):
        rows = [
            {
                "race_key": "A|20261105|1",
                "runner_diagnostics": [
                    {"horse_no": "1", "model_rank": 1, "market_rank": 7, "is_winner": 1},
                    {"horse_no": "2", "model_rank": 7, "market_rank": 1, "is_winner": 0},
                ],
            }
        ]
        report = g4.summarize_rank_disagreement(rows)
        self.assertEqual(report["model_ranks_5plus_above_market"]["horses"], 1)
        self.assertEqual(report["model_ranks_5plus_above_market"]["winners"], 1)
        self.assertEqual(report["market_ranks_5plus_above_model"]["horses"], 1)
        self.assertEqual(report["market_ranks_5plus_above_model"]["winners"], 0)

    def test_t10_to_t2_detects_favorite_change_and_winner_shortening(self):
        rows = [
            {
                "race_key": "A|20261105|1",
                "minutes_to_post": 10.0,
                "captured_at": "2026-11-05T10:00:00+00:00",
                "runner_diagnostics": [
                    {"horse_no": "1", "market_rank": 1, "captured_odds": 2.0, "is_winner": 0},
                    {"horse_no": "2", "market_rank": 2, "captured_odds": 4.0, "is_winner": 1},
                ],
            },
            {
                "race_key": "A|20261105|1",
                "minutes_to_post": 2.5,
                "captured_at": "2026-11-05T10:08:00+00:00",
                "runner_diagnostics": [
                    {"horse_no": "1", "market_rank": 2, "captured_odds": 3.0, "is_winner": 0},
                    {"horse_no": "2", "market_rank": 1, "captured_odds": 1.8, "is_winner": 1},
                ],
            },
        ]
        report = g4.summarize_bucket_movement(rows)["T10_to_T2"]
        self.assertEqual(report["matched_races"], 1)
        self.assertEqual(report["favorite_changed_races"], 1)
        self.assertEqual(report["winner_odds_shortened_pct"], 100.0)
        self.assertEqual(report["winner_horses_compared"], 1)

    def test_empty_input_does_not_invent_performance(self):
        report = g4.summarize_blend_candidates([])
        self.assertEqual(len(report), 5)
        self.assertTrue(all(item["races"] == 0 for item in report.values()))
        self.assertTrue(all(item["top3_coverage_pct"] is None for item in report.values()))


if __name__ == "__main__":
    unittest.main()
