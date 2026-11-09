import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import nar_axis_longshot_research as research


class AxisLongshotResearchTests(unittest.TestCase):
    def setUp(self):
        self.group = [
            {"horse_no": str(i), "finish": i}
            for i in range(1, 8)
        ]
        self.model = {"1": .99, "2": .85, "3": .75, "4": .65, "5": .55, "6": .50, "7": .10}
        self.market = {"1": .40, "2": .25, "3": .15, "4": .10, "5": .06, "6": .03, "7": .01}

    def test_selects_exactly_one_model_over_market_longshot(self):
        result = research.build_axis_longshot_tickets(self.group, self.model, self.market)
        self.assertEqual(result["axis"], "1")
        self.assertEqual(result["ordinary_opponents"], ["2", "3", "4", "5"])
        self.assertEqual(result["longshot"], "6")
        self.assertEqual(len(result["baseline_tickets"]), 6)
        self.assertEqual(len(result["ten_ticket_tickets"]), 10)
        self.assertEqual(len({tuple(x) for x in result["ten_ticket_tickets"]}), 10)

    def test_does_not_force_longshot_when_no_positive_gap(self):
        market = dict(self.market)
        model = {h: 1.0 - i * .1 for i, h in enumerate(self.model)}
        result = research.build_axis_longshot_tickets(self.group, model, market)
        self.assertIsNone(result["longshot"])
        self.assertIsNone(result["ten_ticket_tickets"])
        self.assertEqual(result["selection_status"], "no_qualified_longshot")

    def test_requires_preoff_probability_for_all_runners(self):
        market = dict(self.market)
        del market["7"]
        result = research.build_axis_longshot_tickets(self.group, self.model, market)
        self.assertEqual(result["selection_status"], "missing_preoff_market_probability")

    def test_evaluation_reports_axis_and_same_race_increment(self):
        group = [
            {"horse_no": str(i), "finish": i}
            for i in range(1, 8)
        ]
        selected = research.build_axis_longshot_tickets(group, self.model, self.market)
        item = {
            "group": group,
            "model_scores": self.model,
            "market_win_probabilities": self.market,
            "payout": {"trio": [(selected["ten_ticket_tickets"][0], 500)]},
        }
        report = research.evaluate_axis_longshot([item])
        self.assertEqual(report["axis_win_rate_pct"], 100.0)
        self.assertEqual(report["axis_podium_rate_pct"], 100.0)
        self.assertEqual(report["longshot_eligible_races"], 1)
        self.assertEqual(report["six_ticket_baseline_eligible_races"]["stake_jpy"], 600)
        self.assertEqual(report["ten_ticket_longshot_eligible_races"]["stake_jpy"], 1000)


if __name__ == "__main__":
    unittest.main()
