import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import nar_exacta_conditional_model as conditional


class ConditionalExactaTests(unittest.TestCase):
    def test_ordered_features_are_directional_and_ignore_finish(self):
        first = {"horse_no": "1", "finish": 1, "recent_finish_pct": 0.8, "venue": "A"}
        second = {"horse_no": "2", "finish": 2, "recent_finish_pct": 0.4, "venue": "A"}
        forward = conditional.ordered_pair_features(first, second)
        reverse = conditional.ordered_pair_features(second, first)
        self.assertNotEqual(forward, reverse)
        changed_result = {**second, "finish": 8}
        self.assertEqual(forward, conditional.ordered_pair_features(first, changed_result))

    def test_training_data_uses_winner_condition_and_second_place_label(self):
        groups = []
        for race in range(12):
            group = []
            for horse in range(1, 6):
                group.append({
                    "horse_no": str(horse),
                    "finish": 1 if horse == 1 else 2 if horse == 2 else horse,
                    "recent_finish_pct": 0.1 * horse + race * 0.001,
                    "venue": "A",
                })
            groups.append(group)
        x, y, counts, races = conditional.build_training_data(groups, 2)
        self.assertEqual(races, 12)
        self.assertEqual(counts[1], 12)
        self.assertEqual(counts[0], 24)
        self.assertEqual(x.shape[0], 36)
        self.assertEqual(int(y.sum()), 12)

    def test_conditional_ticket_count_and_order(self):
        group = [{"horse_no": str(i)} for i in range(1, 7)]
        first = {str(i): 1 / i for i in range(1, 7)}
        conditional_scores = {
            str(first_no): {
                str(second_no): 1 / (1 + second_no)
                for second_no in range(1, 7) if second_no != first_no
            }
            for first_no in range(1, 7)
        }
        tickets = conditional.build_conditional_exacta_tickets(
            group, first, conditional_scores, limit=6
        )
        self.assertEqual(len(tickets), 6)
        self.assertTrue(all(ticket[0] != ticket[1] for ticket in tickets))
        self.assertEqual(len({tuple(ticket) for ticket in tickets}), 6)
        self.assertTrue(all(ticket[0] in {"1", "2"} for ticket in tickets))


if __name__ == "__main__":
    unittest.main()
