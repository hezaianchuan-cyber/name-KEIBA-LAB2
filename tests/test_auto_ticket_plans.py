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

    def test_shadow_portfolio_contains_only_unique_trio(self):
        tickets = plans.shadow_tickets(self.predictions)
        self.assertEqual(len(tickets), 1)
        self.assertEqual(tickets[0]["type"], "3連複")
        self.assertEqual(tickets[0]["comb"], "1-2-3")

    def test_real_auto_plan_never_selects_non_trio_even_if_other_thresholds_pass(self):
        race = {
            "date": "20261105", "venue": "大井", "race": "10",
            "race_key": "大井|20261105|10", "model_version": "test",
            "captured_at": "2026-11-05T12:00:00+09:00",
            "predictions": self.predictions,
            "odds": [{"type": "単勝", "n1": "2", "odds": 2.1}],
            "candidate_combinations": [
                {"bet_type": "win", "numbers": ["2"], "odds": 8.0, "probability": 0.8},
                {"bet_type": "trio", "numbers": ["2", "1", "3"], "odds": 3.0, "probability": 0.7},
                {"bet_type": "trifecta", "numbers": ["2", "1", "3"], "odds": 100.0, "probability": 0.5},
            ],
        }
        limits = {"win": 1.1, "trio": 1.1, "trifecta": 1.1}
        plan = plans.make_plan(race, limits)
        self.assertEqual(plan["decision"], "BET")
        self.assertTrue(plan["tickets"])
        self.assertTrue(all(ticket["type"] == "3連複" for ticket in plan["tickets"]))
        self.assertEqual(plan["validatedBetTypes"], 1)

    def test_thresholds_expose_only_validated_trio(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            (data_dir / "backtest-summary.json").write_text(json.dumps({
                "preoff_ticket_type_backtest": {"bet_types": {
                    "win": {"selected_threshold": 1.2, "selected_holdout": {"candidate_pass": True}},
                    "trio": {"selected_threshold": 1.35, "selected_holdout": {"candidate_pass": True}},
                    "trifecta": {"selected_threshold": 1.5, "selected_holdout": {"candidate_pass": True}},
                }}
            }), encoding="utf-8")
            with patch.object(plans, "DATA", data_dir):
                self.assertEqual(plans.thresholds(), {"trio": 1.35})

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
        self.assertEqual(plan["shadowTickets"], [{"type": "3連複", "comb": "1-2-3", "stake": 100, "basis": "model_top_3"}])
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
        self.assertEqual(len(plan["shadowTickets"]), 1)
        self.assertEqual(plan["shadowTickets"][0]["type"], "3連複")
        self.assertEqual(plan["totalStake"], 0)


    def test_racecards_still_load_when_odds_archive_is_missing(self):
        key = ("大井", "20261105", "1")
        race_row = ["大井", "20261105", "1", "1230"] + [""] * 23
        horse_row = ["大井", "20261105", "1", "", "", "1", "テスト馬", "", "3", "", "", "テスト父", "", "", "テスト騎手"]
        duplicate_horse_row = ["大井", "20261105", "1", "", "", "1", "重複馬行", "", "3", "", "", "テスト父", "", "", "テスト騎手"]
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            with zipfile.ZipFile(data_dir / "latest-race.zip", "w") as archive:
                for filename, rows in (("20261105_racelist.csv", [race_row]), ("20261105_horselist.csv", [horse_row, duplicate_horse_row])):
                    buffer = io.StringIO()
                    writer = csv.writer(buffer)
                    writer.writerows(rows)
                    archive.writestr(filename, buffer.getvalue())
            with patch.object(plans.nar, "DATA", data_dir):
                races, horses, odds = plans.nar.read_archives()
        self.assertIn(key, races)
        self.assertEqual(len(horses[key]), 1)
        self.assertEqual(odds, {})


    def test_trio_fusion_builds_12_ranked_preview_independent_of_ev_threshold(self):
        race = {
            "odds": [
                {"type":"単勝","n1":str(i),"odds":float(i+1)} for i in range(1,16)
            ] + [
                {"type":"3連複","n1":"1","n2":"2","n3":"3","odds":2.4},
                {"type":"3連複","n1":"1","n2":"2","n3":"4","odds":18.0},
            ],
            "candidate_combinations": [
                {"bet_type":"trio","numbers":["1","2",str(i)],"odds":float(2+i*3),"probability":0.12/(i-1)}
                for i in range(3,15)
            ]
        }
        rankings = plans.trio_fusion_rankings(race)
        self.assertEqual(set(rankings), {"0.25","0.50","0.75"})
        self.assertEqual(len(rankings["0.50"]), 12)
        self.assertEqual([x["rank"] for x in rankings["0.50"]], list(range(1,13)))
        self.assertTrue(all(x["odds"] > 0 and x["fusedProbability"] > 0 for x in rankings["0.50"]))

    def test_fusion_preview_does_not_turn_unvalidated_plan_into_bet(self):
        race = {
            "date":"20261105","venue":"大井","race":"10","race_key":"大井|20261105|10",
            "model_version":"test","captured_at":"2026-11-05T12:00:00+09:00",
            "predictions":[{"horse_no":str(i),"win_probability":0.2-i*0.01,"score":1-i*0.1} for i in range(1,9)],
            "odds":[{"type":"単勝","n1":str(i),"odds":float(i+1)} for i in range(1,9)] +
                   [{"type":"3連複","n1":"1","n2":"2","n3":str(i),"odds":float(i+2)} for i in range(3,9)],
            "candidate_combinations":[{"bet_type":"trio","numbers":["1","2",str(i)],"odds":float(i+2),"probability":0.08} for i in range(3,9)]
        }
        plan=plans.make_plan(race,{})
        self.assertEqual(plan["decision"],"SKIP")
        self.assertEqual(plan["tickets"],[])
        self.assertEqual(len(plan["shadowTickets"]),6)
        self.assertEqual(plan["trioRanking"]["selectedMarketWeight"],0.50)


class CompetitorBenchmarkDataTests(unittest.TestCase):
    def test_competitor_predictions_require_sourced_records_and_start_empty(self):
        import json
        path = Path(__file__).resolve().parents[1] / "data" / "competitor-predictions.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(report["schema_version"], 1)
        self.assertIn("Only sourced competitor predictions", report["policy"])
        self.assertEqual(report["predictions"], [])

    def test_competitor_comparison_schema_has_cutoff_and_source_fields(self):
        required = {"predictor", "date", "venue", "race", "pick",
                    "captured_at", "cutoff_at", "source_url"}
        example = {
            "predictor": "福来エマ", "date": "2026-11-05", "venue": "大井",
            "race": 10, "pick": 4, "captured_at": "2026-11-05T19:00:00+09:00",
            "cutoff_at": "2026-11-05T19:25:00+09:00",
            "source_url": "https://example.invalid/source"
        }
        self.assertTrue(required.issubset(example))
        self.assertLessEqual(
            __import__("datetime").datetime.fromisoformat(example["captured_at"]),
            __import__("datetime").datetime.fromisoformat(example["cutoff_at"])
        )


if __name__ == "__main__":
    unittest.main()
