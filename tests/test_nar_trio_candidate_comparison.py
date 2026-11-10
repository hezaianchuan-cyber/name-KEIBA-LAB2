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


if __name__ == "__main__":
    unittest.main()
