import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import nar_axis_conditional_pair_ranker as ranker


def make_race(seed):
    group = []
    scores = {}
    for no in range(1, 9):
        group.append({
            "horse_no": str(no),
            "finish": no if no <= 3 else no + seed % 2,
            "recent_first_corner_pct": min(.95, .08 * no + seed * .001),
            "recent_last_corner_pct": min(.98, .10 * no + seed * .001),
            "recent_last3f_rank": no / 8,
            "recent_time_trend": .4 + no / 20,
            "recent_opponent_strength": .3 + no / 10,
            "predicted_front_count": 2 + seed % 3,
            "front_crowding_ratio": .1 + seed % 4 / 10,
            "carried_weight": 54 + no,
            "body_weight_delta_pct": (no - 3) / 100,
        })
        scores[str(no)] = 1 - no / 10
    return {
        "group": group,
        "axis_horse_no": "1",
        "candidate_horse_nos": ["2", "3", "4", "5", "6", "7"],
        "individual_scores": scores,
    }


class AxisConditionalPairTests(unittest.TestCase):
    def test_only_axis_podium_races_create_labels(self):
        races = [make_race(i) for i in range(20)]
        races[0]["group"][0]["finish"] = 5
        x, y, d = ranker.build_axis_pair_training_data(races)
        self.assertEqual(d["axis_podium_races"], 19)
        self.assertEqual(len(x), len(y))
        self.assertGreater(d["positive_pairs"], 0)

    def test_axis_pair_features_are_symmetric(self):
        race = make_race(1)
        group = {r["horse_no"]: r for r in race["group"]}
        ab = ranker.axis_pair_features(group["1"], group["2"], group["3"], race["individual_scores"])
        ba = ranker.axis_pair_features(group["1"], group["3"], group["2"], race["individual_scores"])
        self.assertEqual(ab, ba)
        self.assertEqual(len(ab), len(ranker.FEATURE_NAMES))

    def test_candidate_pool_excludes_axis_and_validates_membership(self):
        race = make_race(1)
        with self.assertRaises(ValueError):
            ranker.build_axis_pair_training_data([{
                **race, "candidate_horse_nos": ["2", "99"]
            }])

    def test_fit_and_rank(self):
        races = [make_race(i) for i in range(40)]
        fitted = ranker.fit_axis_pair_model(races)
        result = ranker.rank_axis_partner_pairs(
            races[0]["group"], "1", races[0]["candidate_horse_nos"],
            races[0]["individual_scores"], fitted,
        )
        self.assertEqual(len(result), 10)
        self.assertTrue(all(0 <= item["score"] <= 1 for item in result))
        self.assertNotIn("1", result[0]["pair"])


if __name__ == "__main__":
    unittest.main()
