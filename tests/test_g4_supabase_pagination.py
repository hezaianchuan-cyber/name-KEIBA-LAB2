import gzip
import json
import sys
import tempfile
import urllib.error
import urllib.parse
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import nar_model_g4_preoff_backtest as g4


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return b"[]"


class G4SupabasePaginationTests(unittest.TestCase):
    def test_official_results_skip_supabase_when_archive_covers_snapshot_races(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payouts_file = root / "daily-payouts.json"
            payouts_file.write_text(json.dumps({"days": [{
                "date": "2026-11-07",
                "races": [{
                    "venue": "佐賀",
                    "race": 1,
                    "payouts": {
                        "win": [{"numbers": ["1"], "pay_100": 180}],
                        "trio": [{"numbers": ["1", "2", "3"], "pay_100": 920}],
                    },
                }],
            }]}), encoding="utf-8")
            snapshot_dir = root / "odds-snapshots"
            snapshot_dir.mkdir()
            with gzip.open(snapshot_dir / "2026-11-07.jsonl.gz", "wt", encoding="utf-8") as stream:
                stream.write(json.dumps({"race_key": "佐賀|20261107|1"}) + "\n")

            with (
                patch.object(g4, "PAYOUTS_FILE", payouts_file),
                patch.object(g4, "SNAPSHOT_DIR", snapshot_dir),
                patch.object(g4, "SUPABASE_URL", "https://example.supabase.co"),
                patch.object(g4, "SUPABASE_KEY", "test-key"),
                patch.object(g4.urllib.request, "urlopen", side_effect=AssertionError("unexpected Supabase request")),
            ):
                outcomes = g4.load_outcomes()

        self.assertIn("佐賀|20261107|1", outcomes)
        self.assertEqual(outcomes["佐賀|20261107|1"]["winners"], {"1"})
        self.assertEqual(outcomes["佐賀|20261107|1"]["trio_payouts"], {"1-2-3": 920})
        self.assertEqual(g4.OUTCOME_DIAGNOSTICS["supabase_result_rows"], 0)
        self.assertEqual(g4.OUTCOME_DIAGNOSTICS["supabase_result_errors"], [])

    def test_official_results_fill_current_day_gap_from_supabase(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payouts_file = root / "daily-payouts.json"
            payouts_file.write_text(json.dumps({"days": []}), encoding="utf-8")
            snapshot_dir = root / "odds-snapshots"
            snapshot_dir.mkdir()
            (snapshot_dir / "2026-11-07.jsonl.gz").write_bytes(b"")

            requested = []

            class ResultResponse:
                def __enter__(self):
                    return self

                def __exit__(self, *_args):
                    return False

                def read(self):
                    return json.dumps([{
                        "race_id": "佐賀|20261107|1",
                        "race_date": "2026-11-07",
                        "venue": "佐賀",
                        "race_number": 1,
                        "result_data": {"payouts": {
                            "win": [{"numbers": ["10"], "pay_100": 320}],
                            "trio": [{"numbers": ["1", "6", "10"], "pay_100": 790}],
                        }},
                    }]).encode("utf-8")

            def fake_urlopen(request, timeout=0):
                parsed = urllib.parse.urlparse(request.full_url)
                requested.append((parsed.path, urllib.parse.parse_qs(parsed.query), timeout))
                return ResultResponse()

            with (
                patch.object(g4, "PAYOUTS_FILE", payouts_file),
                patch.object(g4, "SNAPSHOT_DIR", snapshot_dir),
                patch.object(g4, "SUPABASE_URL", "https://example.supabase.co"),
                patch.object(g4, "SUPABASE_KEY", "test-key"),
                patch.object(g4.urllib.request, "urlopen", side_effect=fake_urlopen),
            ):
                outcomes = g4.load_outcomes()

            self.assertIn("佐賀|20261107|1", outcomes)
            self.assertEqual(outcomes["佐賀|20261107|1"]["winners"], {"10"})
            self.assertEqual(outcomes["佐賀|20261107|1"]["trio_payouts"], {"1-6-10": 790})
            self.assertEqual(g4.OUTCOME_DIAGNOSTICS["supabase_result_rows"], 1)
            self.assertEqual(requested[0][0], "/rest/v1/nar_race_results")
            self.assertEqual(requested[0][1]["race_date"], ["eq.2026-11-07"])


    def test_transient_supabase_502_is_retried(self):
        failure = urllib.error.HTTPError(
            "https://example.supabase.co/rest/v1/odds_snapshots",
            502, "Bad Gateway", hdrs=None, fp=None,
        )
        responses = [failure, _Response()]
        request = g4.urllib.request.Request("https://example.supabase.co/rest/v1/odds_snapshots")

        with (
            patch.object(g4.urllib.request, "urlopen", side_effect=responses) as urlopen,
            patch.object(g4.time, "sleep", return_value=None),
        ):
            result = g4.fetch_json_with_retry(request, timeout=1, attempts=2)

        self.assertEqual(result, [])
        self.assertEqual(urlopen.call_count, 2)

    def test_github_model_archive_is_primary_without_optional_supabase_read(self):
        model_rows = [{
            "race_key": "大井|20261104|1",
            "date": "20261104",
            "captured_at": "2026-11-04T10:00:00+09:00",
            "predictions": [{"horse_no": "1"}, {"horse_no": "2"}, {"horse_no": "3"}],
        }]
        requests = []

        def fake_urlopen(request, timeout=30):
            requests.append(request)
            return io.BytesIO(b"[]")

        with (
            patch.dict("os.environ", {"KEIBA_G4_READ_MODEL_SNAPSHOT_TABLE": "0"}),
            patch.object(g4, "SUPABASE_URL", "https://example.supabase.co"),
            patch.object(g4, "SUPABASE_KEY", "test-key"),
            patch.object(g4.urllib.request, "urlopen", side_effect=fake_urlopen),
        ):
            rows, diagnostics = g4.load_supabase_snapshots(model_rows)

        self.assertEqual(rows, [])
        self.assertEqual(len(requests), 1)
        self.assertTrue(requests[0].full_url.split("?")[0].endswith("/odds_snapshots"))
        self.assertEqual(diagnostics["model_snapshot_table_status"], "skipped_github_archive_primary")
        self.assertEqual(diagnostics["github_model_snapshot_races"], 1)

    def test_odds_snapshots_are_fetched_per_day_in_small_indexed_pages(self):
        requested = []

        def fake_urlopen(request, timeout=0):
            parsed = urllib.parse.urlparse(request.full_url)
            query = urllib.parse.parse_qs(parsed.query)
            requested.append((parsed.path, query, timeout))
            return _Response()

        model_rows = [
            {
                "race_key": "船橋|20261104|1",
                "date": "20261104",
                "captured_at": "2026-11-04T08:00:00+00:00",
                "predictions": [{"horse_no": "1", "score": 1.0}] * 3,
            },
            {
                "race_key": "船橋|20261105|1",
                "date": "20261105",
                "captured_at": "2026-11-05T08:00:00+00:00",
                "predictions": [{"horse_no": "1", "score": 1.0}] * 3,
            },
        ]

        with patch.object(g4, "SUPABASE_URL", "https://example.supabase.co"), \
             patch.object(g4, "SUPABASE_KEY", "test-key"), \
             patch.object(g4.urllib.request, "urlopen", side_effect=fake_urlopen):
            _rows, diagnostics = g4.load_supabase_snapshots(model_rows)

        odds_requests = [
            (query, timeout)
            for path, query, timeout in requested
            if path.endswith("/odds_snapshots")
        ]
        self.assertEqual(len(odds_requests), 2)
        self.assertEqual(
            {query["race_date"][0] for query, _timeout in odds_requests},
            {"eq.2026-11-04", "eq.2026-11-05"},
        )
        for query, timeout in odds_requests:
            self.assertEqual(query["limit"], ["20"])
            self.assertEqual(query["order"], ["captured_at.desc,id.desc"])
            self.assertEqual(query["offset"], ["0"])
            self.assertGreaterEqual(timeout, 45)
        self.assertEqual(diagnostics["supabase_status"], "loaded")
        self.assertEqual(diagnostics["supabase_query_rows"], 0)
        self.assertEqual(diagnostics["supabase_odds_fetch_parallelism"], 2)
        self.assertEqual(diagnostics["supabase_odds_days_requested"], 2)


if __name__ == "__main__":
    unittest.main()
