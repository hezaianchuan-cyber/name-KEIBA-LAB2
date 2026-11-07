import sys
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import capture_nar_odds as capture


JST = ZoneInfo("Asia/Tokyo")


class CaptureAuditTests(unittest.TestCase):
    def test_audit_classifies_each_race_and_keeps_only_eligible_candidates(self):
        now = datetime(2026, 11, 6, 10, 0, tzinfo=JST)
        races = {
            ("大井", "20261106", "1"): {"venue": "大井", "date": "20261106", "race": "1", "time": "1020"},
            ("大井", "20261106", "2"): {"venue": "大井", "date": "20261106", "race": "2", "time": "1130"},
            ("大井", "20261106", "3"): {"venue": "大井", "date": "20261106", "race": "3", "time": "0959"},
            ("大井", "20261106", "4"): {"venue": "大井", "date": "20261106", "race": "4", "time": "1030"},
            ("大井", "20261106", "5"): {"venue": "大井", "date": "20261106", "race": "5", "time": "1040"},
            ("大井", "20261105", "6"): {"venue": "大井", "date": "20261105", "race": "6", "time": "1040"},
        }
        horses = {
            ("大井", "20261106", "1"): [{"horse_no": str(i)} for i in range(1, 9)],
            ("大井", "20261106", "4"): [{"horse_no": str(i)} for i in range(1, 3)],
            ("大井", "20261106", "5"): [{"horse_no": str(i)} for i in range(1, 9)],
        }
        odds = {
            ("大井", "20261106", "1"): [{"type": "単勝", "n1": "1", "odds": 2.1}],
            ("大井", "20261106", "4"): [{"type": "単勝", "n1": "1", "odds": 2.1}],
        }

        audit, candidates, in_window, with_field, with_odds = capture.build_capture_audit(
            now, races, horses, odds
        )
        by_race = {row["race"]: row for row in audit}

        self.assertEqual(in_window, 3)
        self.assertEqual(with_field, 2)
        self.assertEqual(with_odds, 1)
        self.assertEqual([item[0][2] for item in candidates], ["1"])
        self.assertEqual(by_race["1"]["status"], "eligible_for_capture")
        self.assertEqual(by_race["2"]["status"], "outside_capture_window")
        self.assertEqual(by_race["3"]["status"], "past_capture_cutoff")
        self.assertEqual(by_race["4"]["status"], "missing_field")
        self.assertEqual(by_race["5"]["status"], "missing_positive_odds")
        self.assertEqual(by_race["6"]["status"], "not_current_race_date")

    def test_merge_audit_preserves_prior_snapshot_and_supabase_success(self):
        now = datetime(2026, 11, 6, 11, 0, tzinfo=JST)
        previous = {
            "race_audit": [{
                "race_id": "大井|20261106|1",
                "date": "20261106",
                "snapshot_saved": True,
                "supabase_saved": True,
                "first_seen_at": "2026-11-06T09:00:00+09:00",
                "snapshot_saved_at": "2026-11-06T09:20:00+09:00",
                "supabase_saved_at": "2026-11-06T09:21:00+09:00",
            }]
        }
        incoming = [{
            "race_id": "大井|20261106|1",
            "date": "20261106",
            "status": "past_capture_cutoff",
            "snapshot_saved": False,
            "supabase_saved": False,
        }]
        merged = capture.merge_race_audit(now, incoming, previous)
        self.assertEqual(len(merged), 1)
        self.assertTrue(merged[0]["snapshot_saved"])
        self.assertTrue(merged[0]["supabase_saved"])
        self.assertEqual(merged[0]["first_seen_at"], previous["race_audit"][0]["first_seen_at"])
        self.assertEqual(merged[0]["snapshot_saved_at"], previous["race_audit"][0]["snapshot_saved_at"])
        self.assertEqual(merged[0]["status"], "past_capture_cutoff")
        self.assertEqual(merged[0]["last_checked_at"], now.isoformat())

    def test_empty_audit_does_not_erase_same_day_race_history(self):
        now = datetime(2026, 11, 6, 11, 0, tzinfo=JST)
        previous = {
            "race_audit": [{
                "race_id": "大井|20261106|1",
                "date": "20261106",
                "snapshot_saved": True,
                "supabase_saved": True,
            }]
        }
        merged = capture.merge_race_audit(now, [], previous)
        self.assertEqual(len(merged), 1)
        self.assertTrue(merged[0]["snapshot_saved"])
        self.assertTrue(merged[0]["supabase_saved"])

    def test_odds_snapshot_insert_sets_indexed_capture_timestamp(self):
        source = (
            Path(capture.__file__).resolve().parents[1]
            / "supabase" / "functions" / "nar-odds-capture" / "index.ts"
        ).read_text(encoding="utf-8")
        self.assertIn('captured_at: new Date(now).toISOString(),', source)
        self.assertIn('captured_at: new Date(now).toISOString(),\n          scheduled_post_time:', source)


    def test_supabase_capture_retries_transient_archive_and_plan_attach_failures(self):
        source = (
            Path(capture.__file__).resolve().parents[1]
            / "supabase" / "functions" / "nar-odds-capture" / "index.ts"
        ).read_text(encoding="utf-8")
        self.assertIn("for (let attempt = 1; attempt <= 3; attempt++)", source)
        self.assertIn("NAR archive fetch failed after 3 attempts", source)
        self.assertIn("for (let attempt = 1; attempt <= 3 && !attached; attempt++)", source)
        self.assertIn("Could not attach auto ticket plan after 3 attempts", source)

    def test_supabase_shadow_plan_is_capped_at_twelve_and_respects_readiness(self):
        source = (
            Path(capture.__file__).resolve().parents[1]
            / "supabase" / "functions" / "nar-odds-capture" / "index.ts"
        ).read_text(encoding="utf-8")
        self.assertIn("strategy.selectTickets(pricedCandidates,12,0.5,0)", source)
        self.assertIn("selectedShadow.slice(0,12)", source)
        self.assertIn("base.shadowTicketSets", source)
        self.assertIn('"RACE_QUALITY_GATE"', source)

    def test_capture_script_has_exactly_one_main_entrypoint(self):
        source = Path(capture.__file__).read_text(encoding="utf-8")
        self.assertEqual(source.count('if __name__ == "__main__":'), 1)


if __name__ == "__main__":
    unittest.main()
