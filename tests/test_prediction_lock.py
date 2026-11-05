import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import capture_nar_odds as capture


class PredictionLockTests(unittest.TestCase):
    def sample(self):
        return {
            "race_key": "大井|20261105|1",
            "venue": "大井",
            "date": "20261105",
            "race": "1",
            "scheduled_post_time": "2026-11-05T18:00:00+09:00",
            "captured_at": "2026-11-05T17:45:00+09:00",
            "minutes_to_post": 15.0,
            "model_version": "nar-test-v1",
            "predictions": [{"horse_no": "1", "win_probability": 0.42}],
            "combination_predictions": {"trio": [["1", "2", "3"]]},
            "odds": [{"type": "単勝", "n1": "1", "odds": 2.4}],
        }

    def test_lock_is_stable_and_verifies_unchanged_record(self):
        record = self.sample()
        capture.attach_prediction_lock(record)
        self.assertTrue(capture.verify_prediction_lock(record))
        self.assertEqual(record["prediction_lock"]["digest"], capture.prediction_lock_digest(record))

    def test_lock_detects_prediction_or_odds_edit(self):
        record = self.sample()
        capture.attach_prediction_lock(record)
        record["predictions"][0]["horse_no"] = "2"
        self.assertFalse(capture.verify_prediction_lock(record))

    def test_lock_detects_timestamp_or_model_change(self):
        record = self.sample()
        capture.attach_prediction_lock(record)
        record["captured_at"] = "2026-11-05T17:46:00+09:00"
        self.assertFalse(capture.verify_prediction_lock(record))

    def test_key_order_does_not_change_digest(self):
        first = self.sample()
        second = dict(reversed(list(self.sample().items())))
        self.assertEqual(capture.prediction_lock_digest(first), capture.prediction_lock_digest(second))

    def test_lock_rejects_post_off_capture_even_with_valid_digest(self):
        record = self.sample()
        record["captured_at"] = "2026-11-05T18:01:00+09:00"
        capture.attach_prediction_lock(record)
        self.assertFalse(capture.verify_prediction_lock(record))

    def test_lock_rejects_capture_outside_preoff_window(self):
        record = self.sample()
        record["captured_at"] = "2026-11-05T17:20:00+09:00"
        capture.attach_prediction_lock(record)
        self.assertFalse(capture.verify_prediction_lock(record))

    def test_lock_rejects_missing_model_or_prediction_payload(self):
        record = self.sample()
        record.pop("model_version")
        capture.attach_prediction_lock(record)
        self.assertFalse(capture.verify_prediction_lock(record))


if __name__ == "__main__":
    unittest.main()
