import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from publish_model_snapshots import (
    fetch_existing_keys, filter_unpublished_rows, lock_digest, publish, record_to_row, verify_record,
)


def valid_record():
    record = {
        "race_key": "園田|20261106|1",
        "venue": "園田",
        "date": "20261106",
        "race": "1",
        "scheduled_post_time": "2026-11-06T10:30:00+09:00",
        "captured_at": "2026-11-06T09:30:00+09:00",
        "minutes_to_post": 60.0,
        "capture_policy": {"window_start_minutes": 60, "stop_before_post_minutes": 2},
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
        self.assertEqual(row["locked_payload"]["odds"], record["odds"])
        self.assertEqual(row["locked_payload"]["predictions"], record["predictions"])

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

    def test_existing_supabase_keys_are_filtered_from_republish(self):
        rows = [
            {"race_id": "園田|20261106|1", "captured_at": "2026-11-06T09:30:00+09:00"},
            {"race_id": "園田|20261106|1", "captured_at": "2026-11-06T09:35:00+09:00"},
        ]
        existing = {("園田|20261106|1", "2026-11-06T09:30:00+09:00")}
        pending = filter_unpublished_rows(rows, existing)
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["captured_at"], "2026-11-06T09:35:00+09:00")

    def test_publish_uses_small_two_row_batches(self):
        import json
        from unittest.mock import patch

        batches = []

        class Response:
            status = 200
            def __init__(self, payload):
                self.payload = payload
            def __enter__(self):
                return self
            def __exit__(self, *_args):
                return False
            def read(self):
                return json.dumps(self.payload).encode("utf-8")

        def fake_urlopen(request, timeout=0):
            body = json.loads(request.data.decode("utf-8"))
            count = len(body["snapshots"])
            batches.append(count)
            return Response({"ok": True, "received": count, "upserted": count})

        rows = [
            {"race_id": f"園田|20261106|{i}", "captured_at": f"2026-11-06T09:{i:02d}:00+09:00"}
            for i in range(1, 6)
        ]
        with patch("publish_model_snapshots.urllib.request.urlopen", side_effect=fake_urlopen):
            published = publish(rows, "https://example.supabase.co/functions/v1/archive", "token", "anon")

        self.assertEqual(batches, [2, 2, 1])
        self.assertEqual(published, 5)

    def test_existing_key_lookup_reads_only_race_and_capture_columns(self):
        import json
        import urllib.parse
        from unittest.mock import patch

        class Response:
            status = 200
            def __enter__(self):
                return self
            def __exit__(self, *_args):
                return False
            def read(self):
                return json.dumps([{
                    "race_id": "園田|20261106|1",
                    "captured_at": "2026-11-06T09:30:00+09:00",
                }]).encode("utf-8")

        requested = []
        def fake_urlopen(request, timeout=0):
            parsed = urllib.parse.urlparse(request.full_url)
            requested.append((parsed.path, urllib.parse.parse_qs(parsed.query), timeout))
            return Response()

        rows = [{"race_date": "2026-11-06"}]
        with patch("publish_model_snapshots.urllib.request.urlopen", side_effect=fake_urlopen):
            keys = fetch_existing_keys(rows, "https://example.supabase.co", "test-key")

        self.assertEqual(keys, {("園田|20261106|1", "2026-11-06T09:30:00+09:00")})
        self.assertEqual(requested[0][0], "/rest/v1/model_prediction_snapshots")
        self.assertEqual(requested[0][1]["select"], ["race_id,captured_at"])
        self.assertEqual(requested[0][1]["race_date"], ["gte.2026-11-06", "lte.2026-11-06"])
        self.assertEqual(requested[0][2], 30)

    def test_missing_lock_is_rejected(self):
        record = valid_record()
        record.pop("prediction_lock")
        self.assertFalse(verify_record(record))


if __name__ == "__main__":
    unittest.main()
