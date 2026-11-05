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

    def test_empty_input_does_not_invent_performance(self):
        report = g4.summarize_blend_candidates([])
        self.assertEqual(len(report), 5)
        self.assertTrue(all(item["races"] == 0 for item in report.values()))
        self.assertTrue(all(item["top3_coverage_pct"] is None for item in report.values()))


if __name__ == "__main__":
    unittest.main()
