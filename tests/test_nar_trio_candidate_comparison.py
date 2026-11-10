import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import nar_trio_candidate_comparison as comparison


class TrioCandidateComparisonTests(unittest.TestCase):
    def setUp(self):
        self.group = [{"horse_no": str(i)} for i in range(1, 8)]
        self.ability = {str(i): 1.0 - i / 10 for i in range(1, 8)}
        self.market = {str(i): 1.0 - i / 10 for i in range(1, 8)}
        self.combo = {}
        import itertools
        for a, b in itertools.combinations([str(i) for i in range(1, 8) if i != "1"], 2):
            self.combo[(a, b)] = 0.9 if a in ("2", "3", "4", "5") and b in ("2", "3", "4", "5") else 0.1

    def test_ability_only_selects_top_four(self):
        result = comparison._subset(self.group, "1", self.ability, self.market,
                                    self.combo, "ability")
        self.assertEqual(set(result), {"2", "3", "4", "5"})

    def test_combo_only_selects_high_combo_subset(self):
        result = comparison._subset(self.group, "1", self.ability, self.market,
                                    self.combo, "combo")
        self.assertEqual(set(result), {"2", "3", "4", "5"})

    def test_mixed_scores_preserve_fixed_axis_and_four_opponents(self):
        for mode in ("ability_combo", "ability_market_combo"):
            result = comparison._subset(self.group, "1", self.ability, self.market,
                                        self.combo, mode, combo_weight=0.5,
                                        market_weight=0.25)
            self.assertEqual(len(result), 4)
            self.assertNotIn("1", result)

    def test_axis_conditioned_three_horse_score_keys_are_supported(self):
        combo = {}
        import itertools
        for a, b in itertools.combinations(("2", "3", "4", "5", "6", "7"), 2):
            combo[tuple(sorted(("1", a, b), key=int))] = 0.9 if a in ("2", "3", "4") and b in ("2", "3", "4") else 0.1
        result = comparison._subset(self.group, "1", self.ability, self.market,
                                    combo, "combo")
        self.assertEqual(set(result), {"2", "3", "4", "5"})

    def test_ticket_count_is_six(self):
        self.assertEqual(len(comparison._tickets("1", ("2", "3", "4", "5"))), 6)


    def test_candidate_pool_keeps_axis_fixed_and_emits_six_tickets(self):
        import itertools
        trio_scores = {}
        for a, b in itertools.combinations(("2", "3", "4", "5", "6", "7"), 2):
            trio_scores[tuple(sorted(("1", a, b), key=int))] = (
                0.9 if a in ("2", "3", "4", "5") and b in ("2", "3", "4", "5") else 0.1
            )
        test_group = [dict(row, finish={1: 1, 2: 2, 3: 3}.get(int(row["horse_no"]), 4))
                      for row in self.group]
        race = {
            "group": test_group,
            "winner_model_scores": {str(i): 1.0 if i == 1 else 0.1 for i in range(1, 8)},
            "model_rank_scores": self.ability,
            "winner_axis_trio_scores": trio_scores,
            "payout": {"trio": []},
            "date": "2026-11-08",
            "race_key": "test|20261108|1",
        }
        # Seven runners are valid for direct pool-6 evaluation.
        result = comparison.evaluate_candidate_pool([race], 6, include_diagnostics=True)
        self.assertEqual(result["races"], 1)
        self.assertEqual(result["tickets_per_race"], 6)
        self.assertEqual(result["stake_per_race_jpy"], 600)
        diagnostic = result["race_diagnostics"][0]
        self.assertEqual(diagnostic["axis"], "1")
        self.assertEqual(len(diagnostic["selected_opponents"]), 4)
        self.assertEqual(len(diagnostic["selected_tickets"]), 6)

    def test_candidate_pool_rejects_unsupported_sizes(self):
        with self.assertRaises(ValueError):
            comparison.evaluate_candidate_pool([], 5)


if __name__ == "__main__":
    unittest.main()
