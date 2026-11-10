"""Leakage-safe supervised scoring of exact three-runner top-three combinations.

Training examples are built only from historical races before the evaluation window.
One positive (the exact official top-three set) and a deterministic bounded sample of
negative triplets are retained per race. Scores are used for ranking only, not treated
as calibrated probabilities or expected yen returns.
"""
from __future__ import annotations

import itertools

from nar_trio_learned_compatibility import pair_features


def trio_features(a, b, c):
    rows = sorted((a, b, c), key=lambda row: int(row["horse_no"]))
    pairs = [
        pair_features(rows[0], rows[1]),
        pair_features(rows[0], rows[2]),
        pair_features(rows[1], rows[2]),
    ]
    # Preserve pair identity/order and add per-feature mean and range across the trio.
    means = [sum(pair[i] for pair in pairs) / 3.0 for i in range(len(pairs[0]))]
    ranges = [max(pair[i] for pair in pairs) - min(pair[i] for pair in pairs)
              for i in range(len(pairs[0]))]
    return [value for pair in pairs for value in pair] + means + ranges


def build_trio_training_data(groups, max_negative_triplets_per_race=12):
    x_rows, labels = [], []
    counts = {0: 0, 1: 0}
    used_races = 0
    for group in groups:
        eligible = [row for row in group if row.get("finish") is not None]
        if len(eligible) < 3:
            continue
        podium = {str(row["horse_no"]) for row in eligible
                  if int(row["finish"]) in (1, 2, 3)}
        if len(podium) != 3:
            continue
        positives, negatives = [], []
        for combo in itertools.combinations(eligible, 3):
            numbers = {str(row["horse_no"]) for row in combo}
            features = trio_features(*combo)
            if numbers == podium:
                positives.append(features)
            else:
                negatives.append(features)
        if len(positives) != 1 or not negatives:
            continue
        if len(negatives) > max_negative_triplets_per_race:
            step = len(negatives) / max_negative_triplets_per_race
            negatives = [negatives[int(i * step)]
                         for i in range(max_negative_triplets_per_race)]
        x_rows.append(positives[0])
        labels.append(1)
        counts[1] += 1
        for features in negatives:
            x_rows.append(features)
            labels.append(0)
            counts[0] += 1
        used_races += 1
    return x_rows, labels, {
        "training_races": used_races,
        "positive_triplets": counts[1],
        "negative_triplets": counts[0],
        "features": 60,
        "negative_sample_cap_per_race": max_negative_triplets_per_race,
    }


def fit_trio_model(groups, max_negative_triplets_per_race=12):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x_rows, labels, metadata = build_trio_training_data(
        groups, max_negative_triplets_per_race
    )
    if metadata["positive_triplets"] < 20 or metadata["negative_triplets"] < 20:
        raise ValueError(f"Insufficient trio labels for training: {metadata}")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced", max_iter=300, C=0.2,
            solver="lbfgs", random_state=42,
        ),
    )
    model.fit(x_rows, labels)
    return {"model": model, **metadata}


def predict_trio_scores(group, fitted, axis):
    rows = {str(row["horse_no"]): row for row in group}
    if axis not in rows:
        raise ValueError("axis must be present in group")
    opponents = sorted((horse for horse in rows if horse != axis), key=int)
    combos = list(itertools.combinations(opponents, 2))
    features = [trio_features(rows[axis], rows[a], rows[b]) for a, b in combos]
    probabilities = fitted["model"].predict_proba(features)[:, 1].tolist()
    return {
        tuple(sorted((axis, a, b), key=int)): float(score)
        for (a, b), score in zip(combos, probabilities)
    }


def choose_axis_trio_tickets(group, axis_scores, individual_scores, trio_scores,
                             opponent_count=4):
    rows = {str(row["horse_no"]): row for row in group}
    axis = min(rows, key=lambda horse: (-float(axis_scores[horse]), int(horse)))
    opponents = sorted(
        (horse for horse in rows if horse != axis),
        key=lambda horse: (-float(individual_scores[horse]), int(horse)),
    )
    if len(opponents) < opponent_count:
        raise ValueError("Not enough opponents for requested trio ticket structure")
    best_subset, best_score = None, float("-inf")
    for subset in itertools.combinations(opponents, opponent_count):
        score = sum(
            float(trio_scores[tuple(sorted((axis, a, b), key=int))])
            for a, b in itertools.combinations(subset, 2)
        )
        key = tuple(sorted(subset, key=int))
        if score > best_score + 1e-12 or (
            abs(score - best_score) <= 1e-12 and
            (best_subset is None or tuple(map(int, key)) < tuple(map(int, best_subset)))
        ):
            best_subset, best_score = key, score
    tickets = [
        sorted((axis, a, b), key=int)
        for a, b in itertools.combinations(best_subset, 2)
    ]
    wide_opponents = opponents[:2]
    return {
        "axis": axis,
        "opponents": list(best_subset),
        "trio_tickets": tickets,
        "wide_tickets": [[axis, horse] for horse in wide_opponents],
        "trio_score_sum": round(best_score, 8),
    }
