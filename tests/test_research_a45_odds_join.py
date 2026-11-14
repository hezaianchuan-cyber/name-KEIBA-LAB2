import datetime as dt
import unittest

from scripts import research_flat_lambdarank_a45_odds_join as joiner


class A45OddsJoinTests(unittest.TestCase):
    def test_select_snapshot_requires_after_prediction_and_before_post(self):
        prediction = {
            "race_id": "佐賀|20261114|4",
            "captured_at": "2026-11-14T16:38:38+09:00",
            "scheduled_post_time": "2026-11-14T17:00:00+09:00",
        }
        snapshots = [
            {"race_key": prediction["race_id"], "captured_at": "2026-11-14T16:35:00+09:00"},
            {"race_key": prediction["race_id"], "captured_at": "2026-11-14T16:42:37+09:00", "odds": []},
            {"race_key": prediction["race_id"], "captured_at": "2026-11-14T16:59:00+09:00"},
            {"race_key": "別レース", "captured_at": "2026-11-14T16:43:00+09:00"},
        ]
        selected = joiner.select_snapshot(prediction, snapshots)
        self.assertIsNotNone(selected)
        self.assertEqual(selected[0].isoformat(), "2026-11-14T16:42:37+09:00")
        self.assertAlmostEqual(selected[1], 17.383333, places=4)

    def test_market_ranking_uses_same_snapshot_win_odds(self):
        snapshot = {"odds": [
            {"type": "単勝", "n1": "1", "odds": 5.0},
            {"type": "単勝", "n1": "2", "odds": 2.0},
            {"type": "単勝", "n1": "3", "odds": 9.0},
        ]}
        ranked, coverage, win_odds = joiner.market_ranked_numbers(snapshot, ["1", "2", "3", "4"])
        self.assertEqual(ranked, ["2", "1", "3", "4"])
        self.assertEqual(coverage, 3)
        self.assertEqual(win_odds["2"], 2.0)

    def test_trio_odds_normalization_and_ticket_policy_sizes(self):
        odds = [
            {"type": "３連複", "n1": "1", "n2": "2", "n3": "3", "odds": 12.5},
            {"type": "３連複", "n1": "1", "n2": "2", "n3": "4", "odds": 20.0},
            {"type": "単勝", "n1": "1", "n2": "", "n3": "", "odds": 2.0},
        ]
        lookup = joiner.trio_odds_lookup({"odds": odds})
        self.assertEqual(lookup[("1", "2", "3")], 12.5)
        strategies = joiner.build_strategies(["1", "2", "3", "4", "5", "6"], lookup)
        self.assertEqual(strategies["top3_single"]["ticket_count"], 1)
        self.assertEqual(strategies["axis_plus4"]["ticket_count"], 6)
        self.assertEqual(strategies["axis_plus5"]["ticket_count"], 10)
        self.assertEqual(strategies["top5_box"]["ticket_count"], 10)
        self.assertEqual(strategies["top3_single"]["quoted_ticket_count"], 1)


if __name__ == "__main__":
    unittest.main()
