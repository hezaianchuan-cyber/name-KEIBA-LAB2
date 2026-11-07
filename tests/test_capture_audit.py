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


if __name__ == "__main__":
    unittest.main()
