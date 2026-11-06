import json
import sys
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
