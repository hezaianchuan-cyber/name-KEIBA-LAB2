import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from nar_trio_compatibility import (
    pair_compatibility, trio_compatibility, trio_score, running_style,
)

class TrioCompatibilityTests(unittest.TestCase):
    def test_running_style_is_bounded(self):
        self.assertEqual(running_style({"recent_first_corner_pct": 0.0}), 1.0)
        self.assertEqual(running_style({"recent_first_corner_pct": 1.0}), 0.0)

    def test_two_front_runners_are_penalized_under_pressure(self):
        a = {"recent_first_corner_pct": 0.05, "front_crowding_ratio": 0.5, "predicted_front_count": 4, "field_size": 12}
        b = dict(a)
        self.assertLess(pair_compatibility(a, b), 0.0)

    def test_pair_score_is_symmetric_and_bounded(self):
        a = {"recent_first_corner_pct": 0.1, "front_crowding_ratio": 0.2, "predicted_front_count": 2, "field_size": 12}
        b = {"recent_first_corner_pct": 0.8, "front_crowding_ratio": 0.2, "predicted_front_count": 2, "field_size": 12}
        self.assertAlmostEqual(pair_compatibility(a, b), pair_compatibility(b, a))
        self.assertGreaterEqual(pair_compatibility(a, b), -0.30)
        self.assertLessEqual(pair_compatibility(a, b), 0.06)

    def test_trio_score_separates_merit_and_compatibility(self):
        group = [{"horse_no": "1", "recent_first_corner_pct": 0.1, "front_crowding_ratio": 0.5, "predicted_front_count": 4, "field_size": 12},
                 {"horse_no": "2", "recent_first_corner_pct": 0.12, "front_crowding_ratio": 0.5, "predicted_front_count": 4, "field_size": 12},
                 {"horse_no": "3", "recent_first_corner_pct": 0.8, "front_crowding_ratio": 0.5, "predicted_front_count": 4, "field_size": 12}]
        scores = {"1": 0.9, "2": 0.8, "3": 0.7}
        self.assertAlmostEqual(trio_score(group, ("1","2","3"), scores, 0.0), 0.8)
        self.assertLess(trio_score(group, ("1","2","3"), scores, 0.5), 0.8)

    def test_invalid_trio_rejected(self):
        with self.assertRaises(ValueError):
            trio_compatibility([{"horse_no":"1"}], ("1","1","1"))

if __name__ == "__main__":
    unittest.main()
