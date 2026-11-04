import csv
import io
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_auto_ticket_plans as plans


class ShadowTicketPlanTests(unittest.TestCase):
    def setUp(self):
        self.predictions = [
            {"horse_no": "2", "win_probability": 0.40, "score": 0.9},
            {"horse_no": "1", "win_probability": 0.25, "score": 0.8},
            {"horse_no": "3", "win_probability": 0.20, "score": 0.7},
            {"horse_no": "2", "win_probability": 0.10, "score": 0.6},
        ]

    def test_shadow_portfolio_uses_unique_model_ranks(self):
        tickets = plans.shadow_tickets(self.predictions)
        self.assertEqual(tickets[0]["type"], "単勝")
        self.assertEqual(tickets[0]["comb"], "2")
        self.assertEqual(tickets[1]["comb"], "2")
        self.assertEqual(tickets[2]["comb"], "1")
        self.assertEqual(tickets[3]["comb"], "1-2")
        self.assertEqual(tickets[4]["comb"], "1-2-3")

    def test_market_favorite_uses_lowest_positive_win_odds(self):
        odds = [
            {"type": "単勝", "n1": "1", "odds": 3.2},
            {"type": "単勝", "n1": "2", "odds": 1.8},
            {"type": "複勝", "n1": "3", "odds": 1.1},
            {"type": "単勝", "n1": "4", "odds": 0},
        ]
        self.assertEqual(plans.market_favorite(odds), "2")
        self.assertIsNone(plans.market_favorite([]))

    def test_skip_without_validated_threshold_still_keeps_shadow_picks(self):
        race = {
            "date": "20261105", "venue": "大井", "race": "10",
            "race_key": "大井|20261105|10", "model_version": "test",
            "captured_at": "2026-11-05T12:00:00+09:00",
            "predictions": self.predictions, "odds": [{"type": "単勝", "n1": "2", "odds": 2.1}],
            "candidate_combinations": [],
        }
        plan = plans.make_plan(race, {})
        self.assertEqual(plan["decision"], "SKIP")
        self.assertEqual(plan["skipReasonCode"], "NO_VALIDATED_TICKET_THRESHOLD")
        self.assertEqual(plan["tickets"], [])
        self.assertGreater(len(plan["shadowTickets"]), 0)
        self.assertEqual(plan["modelTopHorse"], "2")
        self.assertEqual(plan["modelTop3"], ["2", "1", "3"])

    def test_missing_odds_does_not_remove_virtual_candidates(self):
        race = {
            "date": "20261105", "venue": "大井", "race": "10",
            "race_key": "大井|20261105|10", "model_version": "test",
            "captured_at": "2026-11-05T12:00:00+09:00",
            "predictions": self.predictions, "odds": [],
            "candidate_combinations": [],
        }
        plan = plans.make_plan(race, {"win": 1.1})
        self.assertEqual(plan["decision"], "SKIP")
        self.assertEqual(plan["skipReasonCode"], "ODDS_NOT_CAPTURED")
        self.assertEqual(len(plan["shadowTickets"]), 5)
        self.assertEqual(plan["totalStake"], 0)


    def test_racecards_still_load_when_odds_archive_is_missing(self):
        key = ("大井", "20261105", "1")
        race_row = ["大井", "20261105", "1", "1230"] + [""] * 23
        horse_row = ["大井", "20261105", "1", "", "", "1", "テスト馬", "", "3", "", "", "テスト父", "", "", "テスト騎手"]
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            with zipfile.ZipFile(data_dir / "latest-race.zip", "w") as archive:
                for filename, row in (("20261105_racelist.csv", race_row), ("20261105_horselist.csv", horse_row)):
                    buffer = io.StringIO()
                    csv.writer(buffer).writerow(row)
                    archive.writestr(filename, buffer.getvalue())
            with patch.object(plans.nar, "DATA", data_dir):
                races, horses, odds = plans.nar.read_archives()
        self.assertIn(key, races)
        self.assertEqual(len(horses[key]), 1)
        self.assertEqual(odds, {})


if __name__ == "__main__":
    unittest.main()
