import importlib.util
import sys
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "nar_trio_narrowing_pilot.py"
SPEC = importlib.util.spec_from_file_location("trio_pilot", SCRIPT)
pilot = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pilot
SPEC.loader.exec_module(pilot)


class TrioNarrowingPilotTests(unittest.TestCase):
    def test_build_245_has_19_unique_unordered_tickets(self):
        tickets = pilot.build_245(["1", "2", "3", "4", "5", "6", "7"])
        self.assertEqual(len(tickets), 19)
        self.assertEqual(len(set(tickets)), 19)
        self.assertTrue(all(len(set(ticket)) == 3 for ticket in tickets))

    def test_build_245_deduplicates_repeated_horses(self):
        tickets = pilot.build_245(["1", "1", "2", "3", "4", "5", "6", "7"])
        self.assertEqual(len(tickets), 19)
        self.assertEqual(len(set(tickets)), 19)

    def test_narrowing_selects_ten_existing_tickets(self):
        ranked = ["1", "2", "3", "4", "5", "6", "7"]
        tickets = pilot.build_245(ranked)
        narrow = pilot.narrow_by_rank_sum(tickets, ranked)
        self.assertEqual(len(narrow), 10)
        self.assertTrue(set(narrow).issubset(set(tickets)))

    def test_equal_budget_is_exactly_1900_yen(self):
        for count in (10, 19):
            units = pilot.allocate_units(count)
            self.assertEqual(sum(units), 19)
            self.assertTrue(all(unit >= 1 for unit in units))

    def test_rank_divergence_sign_and_bands(self):
        result = pilot.rank_divergence(
            ["1", "2", "3", "4", "5", "6", "7"],
            ["3", "2", "1", "4", "5", "6", "7"],
        )
        self.assertEqual(result["overlap_top7"], 7)
        self.assertEqual(result["rank_gaps"]["1"], 2)
        self.assertEqual(result["rank_gaps"]["3"], -2)
        self.assertEqual(pilot.divergence_band(7), "agreement_6_7")
        self.assertEqual(pilot.divergence_band(4), "moderate_4_5")
        self.assertEqual(pilot.divergence_band(2), "high_divergence_0_3")

    def test_market_rank_uses_explicit_popularity(self):
        odds = [
            {"type": "単勝", "n1": "3", "odds": 2.0, "popularity": "2"},
            {"type": "単勝", "n1": "1", "odds": 4.0, "popularity": "1"},
            {"type": "複勝", "n1": "2", "odds": 1.2, "popularity": "1"},
        ]
        self.assertEqual(pilot.market_rank(odds), ["1", "3"])


if __name__ == "__main__":
    unittest.main()
