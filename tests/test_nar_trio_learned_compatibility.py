import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import nar_trio_learned_compatibility as learned


def make_group(seed):
    group = []
    for no in range(1, 7):
        group.append({
            "horse_no": str(no),
            "finish": no if no <= 3 else no + (seed % 2),
            "recent_first_corner_pct": min(0.95, 0.08 * no + seed * 0.001),
            "recent_last_corner_pct": min(0.98, 0.10 * no + seed * 0.001),
            "recent_last3f_rank": no / 6,
            "recent_time_trend": 0.4 + no / 20,
            "recent_opponent_strength": 0.3 + no / 10,
            "predicted_front_count": 2 + (seed % 3),
            "front_crowding_ratio": 0.1 + (seed % 4) / 10,
            "carried_weight": 54 + no,
            "body_weight_delta_pct": (no - 3) / 100,
        })
    return group


class LearnedCompatibilityTests(unittest.TestCase):
    def test_pair_feature_vector_has_stable_width(self):
        group = make_group(1)
        values = learned.pair_features(group[0], group[1])
        self.assertEqual(len(values), len(learned.FEATURE_NAMES))

    def test_training_sampling_and_prediction(self):
        groups = [make_group(i) for i in range(30)]
        x, y, counts = learned.build_pair_training_data(groups, max_negative_pairs_per_race=8)
        self.assertEqual(counts[1], 90)
        self.assertEqual(counts[0], 240)
        self.assertEqual(len(x), len(y))
        fitted = learned.fit_pair_model(groups, max_negative_pairs_per_race=8)
        pair_scores = learned.predict_pair_compatibility(groups[0], fitted)
        self.assertEqual(len(pair_scores), 15)
        self.assertTrue(all(0 <= value <= 1 for value in pair_scores.values()))
        individual = {str(i): 1 - i / 10 for i in range(1, 7)}
        trio = learned.learned_trio_score(("1", "2", "3"), individual, pair_scores, 0.2)
        self.assertTrue(0 <= trio <= 1)

    def test_duplicate_trio_rejected(self):
        with self.assertRaises(ValueError):
            learned.learned_trio_score(("1", "1", "2"), {"1": 0.9, "2": 0.8},
                                       {("1", "2"): 0.5}, 0.2)


if __name__ == "__main__":
    unittest.main()
