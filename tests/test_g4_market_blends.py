import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import nar_model_g4_preoff_backtest as g4


class PreoffMarketBlendComparisonTests(unittest.TestCase):
    def test_reports_all_predeclared_weights_on_the_same_races(self):
        rows = [
            {
                "blend_candidates": {
                    "0.00": {"top": {"stake": 100, "return": 200, "hit": 1}, "top3_hit": 1},
                    "0.25": {"top": {"stake": 100, "return": 0, "hit": 0}, "top3_hit": 1},
                    "0.50": {"top": {"stake": 100, "return": 0, "hit": 0}, "top3_hit": 0},
                    "0.75": {"top": {"stake": 100, "return": 0, "hit": 0}, "top3_hit": 0},
                    "1.00": {"top": {"stake": 100, "return": 0, "hit": 0}, "top3_hit": 0},
                }
            },
            {
                "blend_candidates": {
                    "0.00": {"top": {"stake": 100, "return": 0, "hit": 0}, "top3_hit": 0},
                    "0.25": {"top": {"stake": 100, "return": 300, "hit": 1}, "top3_hit": 1},
                    "0.50": {"top": {"stake": 100, "return": 300, "hit": 1}, "top3_hit": 1},
                    "0.75": {"top": {"stake": 100, "return": 300, "hit": 1}, "top3_hit": 1},
                    "1.00": {"top": {"stake": 100, "return": 300, "hit": 1}, "top3_hit": 1},
                }
            },
        ]

        report = g4.summarize_blend_candidates(rows)

        self.assertEqual(list(report), ["0.00", "0.25", "0.50", "0.75", "1.00"])
        self.assertEqual(report["0.00"]["races"], 2)
        self.assertEqual(report["0.00"]["top1"]["hits"], 1)
        self.assertEqual(report["0.25"]["top1"]["hits"], 1)
        self.assertEqual(report["0.25"]["top3_coverage_hits"], 2)
        self.assertEqual(report["0.50"]["top3_coverage_pct"], 50.0)
        self.assertEqual(report["1.00"]["market_probability_weight_pct"], 100.0)

    def test_rank_disagreement_counts_winners_in_both_directions(self):
        rows = [
            {
                "race_key": "A|20261105|1",
                "runner_diagnostics": [
                    {"horse_no": "1", "model_rank": 1, "market_rank": 7, "is_winner": 1},
                    {"horse_no": "2", "model_rank": 7, "market_rank": 1, "is_winner": 0},
                ],
            }
        ]
        report = g4.summarize_rank_disagreement(rows)
        self.assertEqual(report["model_ranks_5plus_above_market"]["horses"], 1)
        self.assertEqual(report["model_ranks_5plus_above_market"]["winners"], 1)
        self.assertEqual(report["market_ranks_5plus_above_model"]["horses"], 1)
        self.assertEqual(report["market_ranks_5plus_above_model"]["winners"], 0)

    def test_t10_to_t2_detects_favorite_change_and_winner_shortening(self):
        rows = [
            {
                "race_key": "A|20261105|1",
                "minutes_to_post": 10.0,
                "captured_at": "2026-11-05T10:00:00+00:00",
                "runner_diagnostics": [
                    {"horse_no": "1", "market_rank": 1, "captured_odds": 2.0, "is_winner": 0},
                    {"horse_no": "2", "market_rank": 2, "captured_odds": 4.0, "is_winner": 1},
                ],
            },
            {
                "race_key": "A|20261105|1",
                "minutes_to_post": 2.5,
                "captured_at": "2026-11-05T10:08:00+00:00",
                "runner_diagnostics": [
                    {"horse_no": "1", "market_rank": 2, "captured_odds": 3.0, "is_winner": 0},
                    {"horse_no": "2", "market_rank": 1, "captured_odds": 1.8, "is_winner": 1},
                ],
            },
        ]
        report = g4.summarize_bucket_movement(rows)["T10_to_T2"]
        self.assertEqual(report["matched_races"], 1)
        self.assertEqual(report["favorite_changed_races"], 1)
        self.assertEqual(report["winner_odds_shortened_pct"], 100.0)
        self.assertEqual(report["winner_horses_compared"], 1)
        self.assertEqual(report["nonwinner_horses_compared"], 1)
        self.assertEqual(report["nonwinner_odds_shortened_pct"], 0.0)

    def test_market_edge_conditions_separate_agreement_from_disagreement(self):
        def row(model_hit, market_hit, agree, model_market_rank, market_model_rank):
            return {
                "model_top": {"stake": 100, "return": 200 if model_hit else 0, "hit": model_hit},
                "market_top": {"stake": 100, "return": 200 if market_hit else 0, "hit": market_hit},
                "model_market_top_agree": agree,
                "model_top_market_rank": model_market_rank,
                "market_favorite_model_rank": market_model_rank,
                "model_top3_hit": model_hit,
                "market_top3_hit": market_hit,
                "blend_candidates": {
                    key: {"top": {"stake": 100, "return": 0, "hit": 0}}
                    for key in ("0.25", "0.50", "0.75")
                },
            }

        report = g4.summarize_market_edge_conditions([
            row(1, 1, 1, 1, 1),
            row(0, 1, 0, 6, 7),
        ])
        self.assertEqual(report["model_agrees_with_market_favorite"]["races"], 1)
        self.assertEqual(report["model_disagrees_with_market_favorite"]["races"], 1)
        self.assertEqual(report["model_top_is_market_rank_4_to_6"]["races"], 1)
        self.assertEqual(report["market_favorite_is_model_rank_7_plus"]["races"], 1)
        self.assertEqual(report["model_disagrees_with_market_favorite"]["market_favorite_hit_rate_pct"], 100.0)

    def test_empty_input_does_not_invent_performance(self):
        report = g4.summarize_blend_candidates([])
        self.assertEqual(len(report), 5)
        self.assertTrue(all(item["races"] == 0 for item in report.values()))
        self.assertTrue(all(item["top3_coverage_pct"] is None for item in report.values()))

    def test_trio_candidate_probabilities_use_exact_captured_preoff_odds(self):
        candidates = [
            {"bet_type": "trio", "numbers": ["1", "2", "3"], "probability": 0.10, "odds": 999, "expected_value": 99.9},
            {"bet_type": "trio", "numbers": ["1", "2", "4"], "probability": 0.08, "odds": 999, "expected_value": 79.9},
            {"bet_type": "win", "numbers": ["1"], "probability": 0.5},
        ]
        captured_odds = [
            {"type": "３連複", "n1": "1", "n2": "2", "n3": "3", "odds": 10.0},
            {"type": "3連複", "n1": "4", "n2": "2", "n3": "1", "odds": 30.0},
        ]
        joined = g4.build_trio_combo_candidates(candidates, captured_odds)
        by_key = {row["key"]: row for row in joined}
        self.assertEqual(by_key["1-2-3"]["expected_value"], 1.0)
        self.assertEqual(by_key["1-2-4"]["expected_value"], 2.4)
        self.assertEqual(len(joined), 2)

    def test_trio_candidate_rankings_compare_probability_ev_and_balanced(self):
        candidates = [
            {"key": "1-2-3", "probability": 0.10, "expected_value": 1.0},
            {"key": "1-2-4", "probability": 0.08, "expected_value": 2.4},
            {"key": "1-2-5", "probability": 0.05, "expected_value": 1.8},
        ]
        rankings = g4.rank_trio_combo_candidates(candidates)
        self.assertEqual(rankings["probability"][0], "1-2-3")
        self.assertEqual(rankings["expected_value"][0], "1-2-4")
        self.assertEqual(rankings["balanced_50_50"][0], "1-2-4")
        ticket_sets = g4.build_trio_candidate_ticket_sets(candidates)
        self.assertEqual(len(ticket_sets["balanced_50_50"]["3"]), 3)

    def test_245_ticket_builder_and_rank_sum_narrowing_are_bounded(self):
        tickets = g4.build_245_tickets(["1", "2", "3", "4", "5", "6", "7", "8"])
        narrowed = g4.narrow_trio_tickets(tickets, ["1", "2", "3", "4", "5", "6", "7", "8"], 12)
        self.assertGreaterEqual(len(tickets), 12)
        self.assertEqual(len(narrowed), 12)
        self.assertEqual(len(set(narrowed)), 12)
        self.assertTrue(all(len(combo) == 3 for combo in narrowed))

    def test_top7_full_podium_coverage_is_distinct_from_top3_winner_coverage(self):
        rows = [{
            "model_top7_podium_hit": 0,
            "model_top7_podium_captured": 2,
            "market_top7_podium_hit": 1,
            "market_top7_podium_captured": 3,
            "blend_candidates": {
                "0.00": {"top7_podium_hit": 0, "top7_podium_captured": 2},
                "0.25": {"top7_podium_hit": 1, "top7_podium_captured": 3},
                "0.50": {"top7_podium_hit": 1, "top7_podium_captured": 3},
                "0.75": {"top7_podium_hit": 0, "top7_podium_captured": 2},
                "1.00": {"top7_podium_hit": 1, "top7_podium_captured": 3},
            },
        }]
        report = g4.summarize_podium_coverage(rows)
        self.assertEqual(report["model"]["full_podium_hits"], 0)
        self.assertEqual(report["market"]["full_podium_coverage_pct"], 100.0)
        self.assertEqual(report["blend_candidates"]["0.50"]["mean_podium_horses_captured"], 3.0)

    def test_partial_trio_ticket_summary_compares_3_6_9_12_choices(self):
        row = {
            "trio_ticket_candidates": {
                "3": [["1", "2", "3"], ["1", "2", "4"], ["1", "3", "4"]],
                "6": [["1", "2", "3"], ["1", "2", "4"], ["1", "3", "4"], ["2", "3", "4"], ["1", "2", "5"], ["1", "3", "5"]],
                "9": [["1", "2", "3"], ["1", "2", "4"], ["1", "3", "4"], ["2", "3", "4"], ["1", "2", "5"], ["1", "3", "5"], ["2", "3", "5"], ["1", "4", "5"], ["2", "4", "5"]],
                "12": [["1", "2", "3"], ["1", "2", "4"], ["1", "3", "4"], ["2", "3", "4"], ["1", "2", "5"], ["1", "3", "5"], ["2", "3", "5"], ["1", "4", "5"], ["2", "4", "5"], ["1", "2", "6"], ["1", "3", "6"], ["2", "3", "6"]],
            },
            "trio_payouts": {"1-2-3": 1500},
        }
        report = g4.summarize_trio_ticket_strategies([row])
        self.assertEqual(report["3"]["hit_races"], 1)
        self.assertEqual(report["6"]["tickets"], 6)
        self.assertEqual(report["12"]["tickets"], 12)
        self.assertEqual(report["12"]["return_jpy"], 1500)




class SupabaseOddsTimestampTests(unittest.TestCase):
    def test_payload_capture_time_overrides_database_insert_time(self):
        item = {
            "captured_at": "2026-11-06T08:22:01.621745+00:00",
            "odds_data": {
                "captured_at": "2026-11-06T08:21:01.233Z",
                "scheduled_post_time": "2026-11-06T08:45:00.000Z",
                "minutes_to_post": 22.98,
            },
        }
        captured = g4.supabase_odds_capture_time(item)
        self.assertEqual(captured, g4.parse_time("2026-11-06T08:21:01.233Z"))
        self.assertAlmostEqual(g4.supabase_odds_minutes_to_post(item, captured), 23.97945, places=4)

    def test_database_timestamp_is_fallback_when_payload_timestamp_is_missing(self):
        item = {"captured_at": "2026-11-06T08:22:01.621745+00:00", "odds_data": {}}
        self.assertEqual(
            g4.supabase_odds_capture_time(item),
            g4.parse_time("2026-11-06T08:22:01.621745+00:00"),
        )


class ModelSnapshotSourceMergeTests(unittest.TestCase):
    def test_partial_supabase_table_does_not_drop_github_only_races(self):
        github_rows = [
            {
                "race_key": "園田|20261106|1",
                "date": "20261106",
                "captured_at": "2026-11-06T10:00:00+09:00",
                "model_version": "github-archive",
                "predictions": [{"horse_no": "1", "score": 50}],
            },
            {
                "race_key": "船橋|20261106|2",
                "date": "20261106",
                "captured_at": "2026-11-06T10:01:00+09:00",
                "model_version": "github-archive",
                "predictions": [{"horse_no": "2", "score": 45}],
            },
        ]
        supabase_rows = [
            {
                "race_id": "園田|20261106|1",
                "race_date": "2026-11-06",
                "captured_at": "2026-11-06T10:00:00+09:00",
                "model_version": "supabase-authoritative",
                "predictions": [{"horse_no": "1", "score": 55}],
            }
        ]

        merged = g4.merge_model_snapshot_sources(github_rows, supabase_rows)

        self.assertEqual(set(merged), {"園田|20261106|1", "船橋|20261106|2"})
        self.assertEqual(len(merged["園田|20261106|1"]), 1)
        self.assertEqual(
            merged["園田|20261106|1"][0][1]["model_version"],
            "supabase-authoritative",
        )
        self.assertEqual(
            merged["船橋|20261106|2"][0][1]["model_version"],
            "github-archive",
        )

    def test_same_race_different_capture_times_are_preserved(self):
        github_rows = [
            {"race_key": "笠松|20261106|3", "captured_at": "2026-11-06T10:00:00+09:00", "date": "20261106"},
        ]
        supabase_rows = [
            {"race_id": "笠松|20261106|3", "captured_at": "2026-11-06T10:01:00+09:00", "race_date": "2026-11-06"},
        ]
        merged = g4.merge_model_snapshot_sources(github_rows, supabase_rows)
        self.assertEqual(len(merged["笠松|20261106|3"]), 2)


if __name__ == "__main__":
    unittest.main()
