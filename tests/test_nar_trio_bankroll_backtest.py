import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from nar_trio_bankroll_backtest import (
    allocate_equal_units,
    allocate_weighted_units,
    build_candidates,
    build_matched_races,
    evaluate_races,
    load_payouts,
    load_snapshots,
    race_key,
    run_backtest,
)


class TrioBankrollAllocationTests(unittest.TestCase):
    def sample_record(self):
        return {
            "race_key": "園田|20261106|1",
            "date": "20261106",
            "captured_at": "2026-11-06T10:00:00+09:00",
            "scheduled_post_time": "2026-11-06T10:20:00+09:00",
            "minutes_to_post": 20.0,
            "model_version": "test-v1",
            "candidate_combinations": [
                {"bet_type": "trio", "numbers": ["1", "2", "3"], "probability": 0.20, "odds": 8.0},
                {"bet_type": "trio", "numbers": ["1", "2", "4"], "probability": 0.10, "odds": 15.0},
                {"bet_type": "trio", "numbers": ["1", "3", "4"], "probability": 0.05, "odds": 30.0},
            ],
            "odds": [
                {"type": "3連複", "n1": "1", "n2": "2", "n3": "3", "odds": 8.0},
                {"type": "3連複", "n1": "1", "n2": "2", "n3": "4", "odds": 15.0},
                {"type": "3連複", "n1": "1", "n2": "3", "n3": "4", "odds": 30.0},
            ],
            "prediction_lock": {
                "version": "nar-prediction-lock-v1",
                "algorithm": "sha256",
                "locked_at": "2026-11-06T10:00:00+09:00",
                "digest": "not-verified-in-this-unit-fixture",
            },
        }

    def test_equal_allocation_respects_budget(self):
        units = allocate_equal_units(12)
        self.assertEqual(sum(units), 19)
        self.assertEqual(max(units), 2)
        self.assertEqual(min(units), 1)

    def test_weighted_allocation_respects_cap_and_leaves_unspent(self):
        units = allocate_weighted_units([100.0, 1.0, 0.0], budget_units=19, max_share=0.30)
        self.assertEqual(sum(units), 10)
        self.assertLessEqual(max(units), 5)
        self.assertEqual(units[2], 0)

    def test_market_fusion_uses_same_snapshot_trio_odds(self):
        rows = build_candidates(self.sample_record())
        self.assertEqual(len(rows), 3)
        self.assertGreater(rows[0]["fused_probability"], rows[2]["fused_probability"])
        self.assertAlmostEqual(rows[0]["odds"], 8.0)
        self.assertAlmostEqual(sum(row["market_probability"] for row in rows), 1.0)

    def test_official_payout_join_and_strategy_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "payouts.json"
            path.write_text(json.dumps({
                "days": [{
                    "date": "2026-11-06",
                    "races": [{
                        "venue": "園田", "race": 1,
                        "payouts": {"trio": [{"numbers": ["1", "2", "3"], "pay_100": 520}]}
                    }]
                }]
            }, ensure_ascii=False), encoding="utf-8")
            payouts = load_payouts(path)
        race = {
            "race_key": "園田|20261106|1",
            "candidates": build_candidates(self.sample_record()),
            "payouts": payouts["園田|20261106|1"],
        }
        equal = evaluate_races([race], "equal")
        balanced = evaluate_races([race], "balanced_30")
        self.assertEqual(equal["races"], 1)
        self.assertEqual(equal["hit_races"], 1)
        self.assertEqual(balanced["hit_races"], 1)
        self.assertLessEqual(balanced["stake_jpy"], 1900)
        self.assertEqual(race_key("園田|20261106|01"), "園田|20261106|1")
        self.assertEqual(race_key("髢\\u0080蛻・|20261106|01"), "門別|20261106|1")

    def test_snapshot_loader_rejects_unverified_lock(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "sample.jsonl.gz"
            with gzip.open(path, "wt", encoding="utf-8") as stream:
                stream.write(json.dumps(self.sample_record(), ensure_ascii=False) + "\n")
            rows, diagnostics = load_snapshots(Path(temp))
        self.assertEqual(rows, [])
        self.assertEqual(diagnostics["invalid_lock_or_shape"], 1)

    def test_empty_inputs_report_not_ready_without_fake_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            snapshot_dir = root / "snapshots"
            snapshot_dir.mkdir()
            payouts = root / "payouts.json"
            payouts.write_text(json.dumps({"days": []}), encoding="utf-8")
            output = root / "report.json"
            report = run_backtest(snapshot_dir, payouts, output)
            self.assertTrue(output.is_file())
        self.assertEqual(report["status"], "not_ready")
        self.assertEqual(report["matched_races"], 0)
        self.assertEqual(report["strategies"], [])


if __name__ == "__main__":
    unittest.main()
