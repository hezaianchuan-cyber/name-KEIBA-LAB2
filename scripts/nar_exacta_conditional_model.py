"""Research-only conditional runner-up model for ordered NAR exacta pairs.

Training conditions on the actual historical winner; inference conditions on each
predicted first-place candidate. Only pre-race numeric runner features are used.
The module is intentionally isolated from live prediction and production ticket code.
"""
from __future__ import annotations

import itertools
from array import array
from collections import Counter

FEATURES = (
    "prior_runs", "career_win_rate", "career_top3_rate", "recent_finish_pct",
    "recent_top3_rate", "recent_margin_quality", "recent_last3f_rank",
    "recent_time_per_km", "recent_time_trend", "recent_opponent_strength",
    "recent_first_corner_pct", "recent_last_corner_pct", "recent_position_gain",
    "prior_body_weight_avg", "body_weight_delta_pct", "body_weight",
    "body_weight_change", "frame_no", "carried_weight", "predicted_front_count",
    "front_crowding_ratio", "race_rel_recent_finish_pct",
    "race_rel_recent_top3_rate", "race_rel_recent_last_corner_pct",
    "race_rel_recent_last3f_rank", "race_rel_recent_opponent_strength",
    "race_rel_recent_time_per_km", "race_rel_venue_top3_rate",
    "race_rel_distance_top3_rate", "race_rel_jockey_top3_rate",
    "race_rel_body_weight_delta_pct", "venue_top3_rate", "distance_top3_rate",
    "going_top3_rate", "jockey_top3_rate", "sire_top3_rate", "age",
    "field_size", "distance_m",
)
MATCH_FEATURES = ("venue", "surface", "going", "jockey", "sire")


def _number(row, key):
    try:
        value = float(row.get(key) or 0.0)
        return value if value == value and abs(value) != float("inf") else 0.0
    except (TypeError, ValueError):
        return 0.0


def ordered_pair_features(first, second):
    """Directional features: first is the assumed winner, second the runner-up candidate."""
    a = [_number(first, key) for key in FEATURES]
    b = [_number(second, key) for key in FEATURES]
    return a + b + [y - x for x, y in zip(a, b)] + [x * y for x, y in zip(a, b)] + [
        float(str(first.get(key) or "") == str(second.get(key) or ""))
        for key in MATCH_FEATURES
    ]


def build_training_data(groups, max_negative_candidates_per_race=6):
    """Use actual winner as conditioning horse; sample negatives deterministically."""
    features = array("f")
    labels = bytearray()
    counts = Counter()
    races_used = 0
    width = len(FEATURES) * 4 + len(MATCH_FEATURES)
    for group in groups:
        winners = [row for row in group if row.get("finish") == 1]
        runners_up = [row for row in group if row.get("finish") == 2]
        if len(winners) != 1 or len(runners_up) != 1:
            continue
        first = winners[0]
        candidates = sorted(
            (row for row in group if str(row.get("horse_no")) != str(first.get("horse_no"))),
            key=lambda row: int(row.get("horse_no") or 0),
        )
        positive_no = str(runners_up[0].get("horse_no"))
        positive = [row for row in candidates if str(row.get("horse_no")) == positive_no]
        if len(positive) != 1:
            continue
        negatives = [row for row in candidates if str(row.get("horse_no")) != positive_no]
        if len(negatives) > max_negative_candidates_per_race:
            step = len(negatives) / max_negative_candidates_per_race
            negatives = [negatives[int(i * step)] for i in range(max_negative_candidates_per_race)]
        for candidate, label in [(positive[0], 1)] + [(row, 0) for row in negatives]:
            features.extend(ordered_pair_features(first, candidate))
            labels.append(label)
            counts[label] += 1
        races_used += 1
    if not labels:
        return None, None, dict(counts), races_used
    import numpy as np
    x = np.frombuffer(features, dtype=np.float32).reshape(-1, width).copy()
    y = np.frombuffer(labels, dtype=np.uint8).copy()
    return x, y, dict(counts), races_used


def fit_conditional_second_model(groups, max_negative_candidates_per_race=6):
    """Fit a regularized conditional runner-up classifier on pre-window races only."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x, y, counts, races_used = build_training_data(groups, max_negative_candidates_per_race)
    if x is None or counts.get(1, 0) < 10 or counts.get(0, 0) < 10:
        raise ValueError(f"Insufficient conditional runner-up labels: {counts}")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=300, C=0.25, random_state=42),
    )
    model.fit(x, y)
    return {
        "model": model, "training_races": races_used,
        "training_pairs": int(len(y)), "positive_pairs": counts[1],
        "negative_pairs": counts[0], "features": list(FEATURES),
        "conditioning_policy": "Actual winner from historical pre-window race; no finish/result fields are model inputs.",
    }


def predict_for_first_candidates(group, fitted):
    """Return P(second horse is runner-up | each candidate first horse) for all ordered pairs."""
    model = fitted["model"]
    rows = {str(row["horse_no"]): row for row in group}
    horses = sorted(rows, key=int)
    output = {}
    for first_no in horses:
        candidates = [horse for horse in horses if horse != first_no]
        matrix = [ordered_pair_features(rows[first_no], rows[second_no]) for second_no in candidates]
        probabilities = model.predict_proba(matrix)[:, 1].tolist()
        output[first_no] = {horse: float(probability)
                            for horse, probability in zip(candidates, probabilities)}
    return output


def build_conditional_exacta_tickets(group, first_scores, conditional_second_scores, limit=6):
    """Rank all ordered pairs from the top-2 winner pool by P(first)*P(second|first)."""
    horses = [str(row["horse_no"]) for row in group]
    if len(horses) < 5 or set(horses) != set(first_scores):
        raise ValueError("At least five runners and complete first-place scores are required")
    if set(horses) != set(conditional_second_scores):
        raise ValueError("Conditional second-place scores must cover every possible first horse")
    firsts = sorted(horses, key=lambda h: (-float(first_scores[h]), int(h)))[:2]
    candidates = []
    for first in firsts:
        second_map = conditional_second_scores[first]
        if set(second_map) != set(horses) - {first}:
            raise ValueError(f"Incomplete conditional runner-up scores for first-place horse {first}")
        for second, conditional_probability in second_map.items():
            score = float(first_scores[first]) * float(conditional_probability)
            candidates.append((score, int(first), int(second), [first, second]))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [item[3] for item in candidates[:limit]]
