import sqlite3
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import import_nar_archive_to_research_db as importer


class OfficialArchiveImporterTests(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(":memory:")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript((ROOT / "research_db" / "sqlite_schema_v1.sql").read_text())

    def tearDown(self):
        self.db.close()

    def test_trio_payout_key_is_sorted_and_stable(self):
        self.assertEqual(importer.canonical_payout_key("trio", ["11", "2", "5"]), "2-5-11")

    def test_import_month_is_repeatable_and_keeps_results_separate(self):
        key = ("帯広ば", "20261001", "1")
        race = {
            "key": key, "date": datetime(2026, 10, 1).date(),
            "venue": "帯広ば", "race": "1", "name": "テスト競走",
            "distance": 1400, "field_size": 3, "surface": "ダート",
            "going": "良", "corner_positions": [],
        }
        horses = [
            {"no": "1", "name": "馬A", "frame_no": 1, "sex": "牡", "age": 4,
             "sire": "種牡馬A", "dam": "母A", "jockey": "騎手A",
             "carried": 56, "finish": 1, "time_seconds": 90.1, "last3f": 38.2},
            {"no": "2", "name": "馬B", "frame_no": 2, "sex": "牝", "age": 5,
             "sire": "種牡馬B", "dam": "母B", "jockey": "騎手B",
             "carried": 54, "finish": 2, "time_seconds": 90.4, "last3f": 38.4},
            {"no": "3", "name": "馬C", "frame_no": 3, "sex": "牡", "age": 3,
             "sire": "種牡馬C", "dam": "母C", "jockey": "騎手C",
             "carried": 55, "finish": 3, "time_seconds": 90.8, "last3f": 38.7},
        ]
        payout = {"trio": [(["1", "2", "3"], 1250)]}
        imported_at = "2026-11-14T00:00:00+00:00"
        with patch.object(importer.nar, "download_month", return_value=(b"PK-test", "https://example.invalid/archive")), \
             patch.object(importer.nar, "parse_month_archive",
                          return_value=({key: race}, {key: horses}, {key: payout})):
            first = importer.import_month(self.db, 2026, 10, imported_at)
            second = importer.import_month(self.db, 2026, 10, imported_at)

        self.assertEqual(first["parsed_races"], 1)
        self.assertEqual(second["parsed_runner_rows"], 3)
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM races").fetchone()[0], 1)
        self.assertEqual(self.db.execute("SELECT race_type FROM races").fetchone()[0], "banei")
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM runners").fetchone()[0], 3)
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM official_results").fetchone()[0], 3)
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM payouts").fetchone()[0], 1)
        row = self.db.execute(
            "SELECT bet_type,combination_key,payout_yen FROM payouts"
        ).fetchone()
        self.assertEqual(row, ("trio", "1-2-3", 1250))
        self.assertEqual(self.db.execute("PRAGMA foreign_key_check").fetchall(), [])
        self.assertEqual(self.db.execute("PRAGMA integrity_check").fetchone()[0], "ok")


if __name__ == "__main__":
    unittest.main()
