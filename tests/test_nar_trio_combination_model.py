import sys
import unittest
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import nar_trio_combination_model as model


class TrioCombinationModelTests(unittest.TestCase):
    def setUp(self):
        self.group = [
            {"horse_no": str(i), "finish": finish,
             "recent_first_corner_pct": i / 10,
             "recent_last_corner_pct": (8 - i) / 10,
             "recent_last3f_rank": float(i),
             "recent_time_trend": i / 20,
             "recent_opponent_strength": i / 15,
             "predicted_front_count": 2.0,
             "front_crowding_ratio": 0.3,
             "carried_weight": 55 + i / 10,
             "body_weight_delta_pct": i / 100}
            for i, finish in enumerate((1, 2, 3, 4, 5, 6, 7), start=1)
        ]

    def test_training_data_has_one_exact_top_three_positive(self):
        x, y, meta = model.build_trio_training_data([self.group], 4)
        self.assertEqual(sum(y), 1)
        self.assertEqual(len(y), 5)
        self.assertEqual(meta["positive_triplets"], 1)
        self.assertEqual(meta["negative_triplets"], 4)
        self.assertEqual(len(x[0]), 75)

    def test_trio_features_are_order_invariant(self):
        a, b, c = self.group[:3]
        self.assertEqual(model.trio_features(a, b, c), model.trio_features(c, a, b))

    def test_choose_axis_trio_tickets_uses_specialist_axis_and_combo_scores(self):
        axis_scores = {str(i): (1.0 if i == 3 else 0.5 - i / 100) for i in range(1, 8)}
        individual = {str(i): 1.0 - i / 10 for i in range(1, 8)}
        opponents = [str(i) for i in range(1, 8) if i != 3]
        scores = {}
        for a, b in combinations(opponents, 2):
            scores[tuple(sorted(("3", a, b), key=int))] = (
                0.9 if a in ("1", "2", "4", "5") and b in ("1", "2", "4", "5")
                else 0.1
            )
        result = model.choose_axis_trio_tickets(
            self.group, axis_scores, individual, scores, opponent_count=4
        )
        self.assertEqual(result["axis"], "3")
        self.assertEqual(set(result["opponents"]), {"1", "2", "4", "5"})
        self.assertEqual(len(result["trio_tickets"]), 6)
        self.assertEqual(len(result["wide_tickets"]), 2)
        self.assertTrue(all("3" in ticket for ticket in result["trio_tickets"]))


if __name__ == "__main__":
    unittest.main()
