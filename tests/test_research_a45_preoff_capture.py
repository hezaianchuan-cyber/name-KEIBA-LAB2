import datetime as dt
import unittest
from zoneinfo import ZoneInfo

from scripts import research_flat_lambdarank_a45_preoff_capture as capture


class A45PreoffCaptureTests(unittest.TestCase):
    def test_parse_post_time_accepts_three_and_four_digits(self):
        self.assertEqual(capture.parse_post_time("930"), dt.time(9, 30))
        self.assertEqual(capture.parse_post_time("1545"), dt.time(15, 45))
        self.assertIsNone(capture.parse_post_time("2560"))

    def test_capture_window_is_preoff_and_excludes_banei(self):
        jst = ZoneInfo("Asia/Tokyo")
        now = dt.datetime(2026, 11, 14, 10, 0, tzinfo=jst)
        races = {
            ("大井", "20261114", "1"): {
                "venue": "大井", "date": dt.date(2026, 11, 14),
                "scheduled_time_raw": "1100", "field_size": 3,
            },
            ("大井", "20261114", "2"): {
                "venue": "大井", "date": dt.date(2026, 11, 14),
                "scheduled_time_raw": "1001", "field_size": 3,
            },
            ("帯広", "20261114", "1"): {
                "venue": "帯広", "date": dt.date(2026, 11, 14),
                "scheduled_time_raw": "1100", "field_size": 3,
            },
        }
        horses = {key: [{"no": "1"}, {"no": "2"}, {"no": "3"}] for key in races}
        eligible = capture.eligible_races(races, horses, now)
        self.assertEqual(set(eligible), {("大井", "20261114", "1")})
        scheduled, minutes = eligible[("大井", "20261114", "1")]
        self.assertEqual(scheduled.hour, 11)
        self.assertEqual(minutes, 60.0)


if __name__ == "__main__":
    unittest.main()
