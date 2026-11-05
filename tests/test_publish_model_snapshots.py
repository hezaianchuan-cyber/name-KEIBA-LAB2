import unittest

from scripts.publish_model_snapshots import lock_digest, record_to_row, verify_record


def valid_record():
    record = {
        "race_key": "園田|20261106|1",
        "venue": "園田",
        "date": "20261106",
        "race": "1",
        "scheduled_post_time": "2026-11-06T10:30:00+09:00",
        "captured_at": "2026-11-06T10:00:00+09:00",
        "minutes_to_post": 30.0,
        "capture_policy": {"window_start_minutes": 30, "stop_before_post_minutes": 2},
        "model_version": "nar-chronological-v9",
        "model_temperature": 0.18,
        "predictions": [
            {"horse_no": "1", "score": 42.0, "win_probability": 0.4},
            {"horse_no": "2", "score": 38.0, "win_probability": 0.3},
            {"horse_no": "3", "score": 35.0, "win_probability": 0.2},
        ],
        "combination_predictions": [],
        "candidate_combinations": [{"bet_type": "trio", "numbers": ["1", "2", "3"], "probability": 0.1}],
        "model_g_version": "nar-model-g-test",
        "model_g_predictions": [],
        "model_g_candidate_combinations": [],
        "model_g_strategy": {},
        "candidate_universe_policy": "all valid combinations",
        "race_readiness": {"ready": True},
        "odds": [{"type": "単勝", "n1": "1", "odds": 2.1}],
    }
    record["prediction_lock"] = {
        "version": "nar-prediction-lock-v1",
        "algorithm": "sha256",
        "digest": lock_digest(record),
        "locked_at": record["captured_at"],
    }
    return record


class PublishModelSnapshotsTests(unittest.TestCase):
    def test_valid_locked_preoff_snapshot_is_mapped(self):
        record = valid_record()
        self.assertTrue(verify_record(record))
        row = record_to_row(record)
        self.assertEqual(row["race_id"], "園田|20261106|1")
        self.assertEqual(row["race_date"], "2026-11-06")
        self.assertEqual(row["model_version"], "nar-chronological-v9")
        self.assertEqual(len(row["predictions"]), 3)
        self.assertEqual(row["prediction_lock"]["digest"], record["prediction_lock"]["digest"])

    def test_modified_prediction_is_rejected_by_lock(self):
        record = valid_record()
        record["predictions"][0]["score"] = 99.0
        self.assertFalse(verify_record(record))
        self.assertIsNone(record_to_row(record))

    def test_snapshot_after_capture_window_is_rejected(self):
        record = valid_record()
        record["scheduled_post_time"] = "2026-11-06T10:31:00+09:00"
        record["prediction_lock"]["digest"] = lock_digest(record)
        self.assertFalse(verify_record(record))

    def test_missing_lock_is_rejected(self):
        record = valid_record()
        record.pop("prediction_lock")
        self.assertFalse(verify_record(record))


if __name__ == "__main__":
    unittest.main()
