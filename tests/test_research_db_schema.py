"""Smoke tests for the KEIBALAB SQLite research schema."""
from pathlib import Path
import sqlite3
import unittest


SCHEMA = Path(__file__).resolve().parents[1] / "research_db" / "sqlite_schema_v1.sql"


class ResearchDatabaseSchemaTests(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(":memory:")
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript(SCHEMA.read_text(encoding="utf-8"))

    def tearDown(self):
        self.db.close()

    def test_expected_tables_exist(self):
        names = {
            row[0] for row in self.db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        self.assertTrue({
            "races", "runners", "odds_snapshots", "model_predictions",
            "official_results", "payouts", "ingestion_batches", "schema_meta",
        }.issubset(names))

    def test_race_type_defaults_to_flat(self):
        stamp = "2026-11-14T16:00:00+09:00"
        self.db.execute(
            """INSERT INTO races
            (race_id,race_date,venue,race_number,source,first_seen_at,last_seen_at)
            VALUES (?,?,?,?,?,?,?)""",
            ("saga|20261114|4", "2026-11-14", "佐賀", 4, "test", stamp, stamp),
        )
        self.assertEqual(self.db.execute("SELECT race_type FROM races").fetchone()[0], "flat")

    def test_race_identity_is_unique(self):
        race = ("saga|20261114|4", "2026-11-14", "佐賀", 4, "test", "2026-11-14T16:00:00+09:00")
        self.db.execute(
            """INSERT INTO races
            (race_id,race_date,venue,race_number,source,first_seen_at,last_seen_at)
            VALUES (?,?,?,?,?,?,?)""",
            (race[0], race[1], race[2], race[3], race[4], race[5], race[5]),
        )
        with self.assertRaises(sqlite3.IntegrityError):
            self.db.execute(
                """INSERT INTO races
                (race_id,race_date,venue,race_number,source,first_seen_at,last_seen_at)
                VALUES (?,?,?,?,?,?,?)""",
                ("different-id", race[1], race[2], race[3], race[4], race[5], race[5]),
            )

    def test_odds_are_append_only_by_capture_time(self):
        stamp = "2026-11-14T16:00:00+09:00"
        self.db.execute(
            """INSERT INTO races
            (race_id,race_date,venue,race_number,source,first_seen_at,last_seen_at)
            VALUES (?,?,?,?,?,?,?)""",
            ("saga|20261114|4", "2026-11-14", "佐賀", 4, "test", stamp, stamp),
        )
        row = ("o1", "saga|20261114|4", "trio", "1-2-3", 12.5, stamp, "test")
        self.db.execute(
            """INSERT INTO odds_snapshots
            (snapshot_id,race_id,bet_type,combination_key,odds,captured_at,source)
            VALUES (?,?,?,?,?,?,?)""", row,
        )
        self.db.execute(
            """INSERT INTO odds_snapshots
            (snapshot_id,race_id,bet_type,combination_key,odds,captured_at,source)
            VALUES (?,?,?,?,?,?,?)""",
            ("o2", row[1], row[2], row[3], 10.2, "2026-11-14T16:01:00+09:00", "test"),
        )
        self.assertEqual(
            self.db.execute("SELECT COUNT(*) FROM odds_snapshots").fetchone()[0], 2
        )

    def test_foreign_keys_are_enforced(self):
        with self.assertRaises(sqlite3.IntegrityError):
            self.db.execute(
                """INSERT INTO runners
                (race_id,horse_no,horse_name,source,first_seen_at,last_seen_at)
                VALUES ('missing',1,'horse','test','now','now')"""
            )
        self.assertEqual(self.db.execute("PRAGMA foreign_key_check").fetchall(), [])


if __name__ == "__main__":
    unittest.main()
