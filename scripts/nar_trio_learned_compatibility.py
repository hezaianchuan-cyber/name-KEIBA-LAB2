"""Leakage-safe learned pair compatibility for NAR trio research.

The model learns from pre-race runner attributes whether two runners from the
same historical race both finish in the top three. It is research-only and is
not wired into production predictions.
"""
from __future__ import annotations

import itertools
from collections import Counter


FEATURE_NAMES = (
    "front_mean", "front_gap", "both_forward",
    "last_corner_mean", "last_corner_gap",
    "last3f_mean", "last3f_gap",
    "time_trend_mean", "time_trend_gap",
    "opponent_strength_mean", "opponent_strength_gap",
    "front_count", "crowding_ratio",
    "carried_weight_gap", "body_weight_delta_gap",
)


def _num(row, key, default=0.5):
    try:
        value = float(row.get(key))
        return value if value == value and abs(value) != float("inf") else default
    except (TypeError, ValueError):
        return default


def pair_features(a, b):
    front_a, front_b = _num(a, "recent_first_corner_pct"), _num(b, "recent_first_corner_pct")
    last_a, last_b = _num(a, "recent_last_corner_pct"), _num(b, "recent_last_corner_pct")
    last3_a, last3_b = _num(a, "recent_last3f_rank"), _num(b, "recent_last3f_rank")
    trend_a, trend_b = _num(a, "recent_time_trend"), _num(b, "recent_time_trend")
    opp_a, opp_b = _num(a, "recent_opponent_strength"), _num(b, "recent_opponent_strength")
    crowd_a, crowd_b = _num(a, "front_crowding_ratio", 0.0), _num(b, "front_crowding_ratio", 0.0)
    return [
        (front_a + front_b) / 2, abs(front_a - front_b),
        max(0.0, 0.20 - front_a) * max(0.0, 0.20 - front_b),
        (last_a + last_b) / 2, abs(last_a - last_b),
        (last3_a + last3_b) / 2, abs(last3_a - last3_b),
        (trend_a + trend_b) / 2, abs(trend_a - trend_b),
        (opp_a + opp_b) / 2, abs(opp_a - opp_b),
        max(_num(a, "predicted_front_count", 0.0), _num(b, "predicted_front_count", 0.0)),
        max(crowd_a, crowd_b),
        abs(_num(a, "carried_weight", 56.0) - _num(b, "carried_weight", 56.0)),
        abs(_num(a, "body_weight_delta_pct", 0.0) - _num(b, "body_weight_delta_pct", 0.0)),
    ]


def build_pair_training_data(groups, max_negative_pairs_per_race=20):
    """Keep every positive pair and deterministic bounded negative sampling per race."""
    x_rows, labels = [], []
    counts = Counter()
    for group in groups:
        eligible = [row for row in group if row.get("finish") is not None]
        positives, negatives = [], []
        for a, b in itertools.combinations(eligible, 2):
            label = int(int(a["finish"]) <= 3 and int(b["finish"]) <= 3)
            item = (pair_features(a, b), label)
            (positives if label else negatives).append(item)
        if len(negatives) > max_negative_pairs_per_race:
            # Evenly spaced deterministic sample, independent of race outcomes.
            step = len(negatives) / max_negative_pairs_per_race
            negatives = [negatives[int(i * step)] for i in range(max_negative_pairs_per_race)]
        for features, label in positives + negatives:
            x_rows.append(features)
            labels.append(label)
            counts[label] += 1
    return x_rows, labels, dict(counts)


def fit_pair_model(groups, max_negative_pairs_per_race=20):
    """Fit a balanced regularized logistic pair model on pre-holdout historical races."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x_rows, labels, counts = build_pair_training_data(groups, max_negative_pairs_per_race)
    if counts.get(1, 0) < 10 or counts.get(0, 0) < 10:
        raise ValueError(f"Insufficient pair labels for training: {counts}")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=500, C=0.5, random_state=42),
    )
    model.fit(x_rows, labels)
    return {
        "model": model,
        "training_pairs": len(labels),
        "positive_pairs": counts[1],
        "negative_pairs": counts[0],
        "features": list(FEATURE_NAMES),
    }


def predict_pair_compatibility(group, fitted):
    model = fitted["model"]
    rows = {str(row["horse_no"]): row for row in group}
    pairs = list(itertools.combinations(sorted(rows, key=int), 2))
    features = [pair_features(rows[a], rows[b]) for a, b in pairs]
    probabilities = model.predict_proba(features)[:, 1].tolist()
    if len(probabilities) <= 1:
        ranks = [0.5] * len(probabilities)
    else:
        ordered = sorted(range(len(probabilities)), key=lambda i: probabilities[i])
        ranks = [0.0] * len(probabilities)
        index = 0
        while index < len(ordered):
            end = index + 1
            while end < len(ordered) and probabilities[ordered[end]] == probabilities[ordered[index]]:
                end += 1
            rank = (index + end - 1) / 2 / (len(ordered) - 1)
            for pos in range(index, end):
                ranks[ordered[pos]] = rank
            index = end
    return {pair: rank for pair, rank in zip(pairs, ranks)}


def learned_trio_score(horse_numbers, individual_scores, pair_scores, weight):
    """Blend mean individual merit with the learned within-race pair compatibility rank."""
    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must be between 0 and 1")
    numbers = tuple(sorted(map(str, horse_numbers), key=int))
    if len(numbers) != 3 or len(set(numbers)) != 3:
        raise ValueError("A trio must contain three distinct horses")
    individual = sum(float(individual_scores[n]) for n in numbers) / 3
    pairs = [tuple(sorted(pair, key=int)) for pair in itertools.combinations(numbers, 2)]
    compatibility = sum(float(pair_scores[pair]) for pair in pairs) / 3
    return (1 - weight) * individual + weight * compatibility
