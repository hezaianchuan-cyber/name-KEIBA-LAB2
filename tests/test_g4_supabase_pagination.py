import json
import sys
import tempfile
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

            with patch.object(g4, "PAYOUTS_FILE", payouts_file), \\
                 patch.object(g4, "SNAPSHOT_DIR", snapshot_dir), \\
                 patch.object(g4, "SUPABASE_URL", "https://example.supabase.co"), \\
                 patch.object(g4, "SUPABASE_KEY", "test-key"), \\
                 patch.object(g4.urllib.request, "urlopen", side_effect=fake_urlopen):
                outcomes = g4.load_outcomes()

            self.assertIn("佐賀|20261107|1", outcomes)
            self.assertEqual(outcomes["佐賀|20261107|1"]["winners"], {"10"})
            self.assertEqual(outcomes["佐賀|20261107|1"]["trio_payouts"], {"1-6-10": 790})
            self.assertEqual(g4.OUTCOME_DIAGNOSTICS["supabase_result_rows"], 1)
            self.assertEqual(requested[0][0], "/rest/v1/nar_race_results")
            self.assertEqual(requested[0][1]["race_date"], ["eq.2026-11-07"])


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


if __name__ == "__main__":
    unittest.main()
