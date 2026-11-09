import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import nar_axis_longshot_research as research


class AxisOpponentSeparationTests(unittest.TestCase):
    def setUp(self):
        self.group = [{"horse_no": str(i), "finish": i} for i in range(1, 8)]
        self.axis_scores = {"1": .1, "2": .2, "3": .99, "4": .4, "5": .3, "6": .2, "7": .1}
        self.opponent_scores = {"1": .9, "2": .8, "4": .7, "5": .6, "6": .5, "7": .4}
        self.market = {"1": .40, "2": .25, "3": .15, "4": .10, "5": .06, "6": .005, "7": .01}

    def test_axis_and_opponents_are_ranked_by_separate_scores(self):
        result = research.build_axis_longshot_tickets(
            self.group, self.axis_scores, self.opponent_scores, self.market)
        self.assertEqual(result["axis"], "3")
        self.assertEqual(result["ordinary_opponents"], ["1", "2", "4", "5"])
        self.assertEqual(len(result["baseline_tickets"]), 6)

    def test_selects_one_longshot_from_remaining_opponents(self):
        result = research.build_axis_longshot_tickets(
            self.group, self.axis_scores, self.opponent_scores, self.market)
        self.assertEqual(result["longshot"], "6")
        self.assertEqual(len(result["ten_ticket_tickets"]), 10)

    def test_no_forced_longshot_without_positive_model_market_gap(self):
        opponent = {"1": .9, "2": .8, "3": 0, "4": .7, "5": .6, "6": .1, "7": .2}
        market = {"1": .40, "2": .25, "3": .15, "4": .10, "5": .06, "6": .01, "7": .03}
        result = research.build_axis_longshot_tickets(self.group, self.axis_scores, opponent, market)
        self.assertIsNone(result["longshot"])
        self.assertEqual(result["selection_status"], "no_qualified_longshot")

    def test_requires_market_probability_for_every_runner(self):
        market = dict(self.market)
        del market["7"]
        result = research.build_axis_longshot_tickets(
            self.group, self.axis_scores, self.opponent_scores, market)
        self.assertEqual(result["selection_status"], "missing_preoff_market_probability")

    def test_five_runner_race_retains_six_ticket_baseline(self):
        group = self.group[:5]
        axis = {h: self.axis_scores[h] for h in ("1", "2", "3", "4", "5")}
        opponent = {h: self.opponent_scores[h] for h in axis if h != "3"}
        result = research.build_axis_longshot_tickets(group, axis, opponent, {})
        self.assertEqual(len(result["baseline_tickets"]), 6)
        self.assertEqual(result["selection_status"], "insufficient_runners_for_longshot")

    def test_axis_conditioned_opponent_scores_use_pair_compatibility(self):
        pair_scores = {
            ("1", "2"): .9,
            ("1", "3"): .1,
            ("2", "3"): .5,
        }
        group = [{"horse_no": h} for h in ("1", "2", "3")]
        scores = research.build_axis_conditioned_opponent_scores(
            group, "1", {"1": .9, "2": .2, "3": .8}, pair_scores, .5)
        self.assertGreater(scores["2"], scores["3"])

    def test_evaluation_reports_axis_metrics_and_same_race_increment(self):
        selected = research.build_axis_longshot_tickets(
            self.group, self.axis_scores, self.opponent_scores, self.market)
        item = {
            "group": self.group, "axis_scores": self.axis_scores,
            "opponent_scores": self.opponent_scores,
            "market_win_probabilities": self.market,
            "payout": {"trio": [(selected["ten_ticket_tickets"][0], 500)]},
        }
        report = research.evaluate_axis_longshot([item])
        self.assertEqual(report["axis_win_rate_pct"], 0.0)
        self.assertEqual(report["axis_podium_rate_pct"], 100.0)
        self.assertEqual(report["longshot_eligible_races"], 1)
        self.assertEqual(report["six_ticket_baseline_eligible_races"]["stake_jpy"], 600)
        self.assertEqual(report["ten_ticket_longshot_eligible_races"]["stake_jpy"], 1000)


if __name__ == "__main__":
    unittest.main()
