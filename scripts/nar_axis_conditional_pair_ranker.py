"""Axis-conditioned partner-pair ranker for NAR trio research.

Research-only. The axis horse and candidate pool must be generated from
pre-off predictions. Historical finish positions are used only as labels.
A training race contributes labels only when its fixed axis finished top 3.
"""
from __future__ import annotations

import itertools

from nar_trio_learned_compatibility import pair_features

FEATURE_NAMES = (
    "pair_front_mean", "pair_front_gap", "pair_both_forward",
    "pair_last_corner_mean", "pair_last_corner_gap",
    "pair_last3f_mean", "pair_last3f_gap",
    "pair_time_trend_mean", "pair_time_trend_gap",
    "pair_opponent_strength_mean", "pair_opponent_strength_gap",
    "pair_front_count", "pair_crowding_ratio",
    "pair_carried_weight_gap", "pair_body_weight_delta_gap",
    "candidate_score_mean", "candidate_score_gap",
    "axis_score_mean", "axis_score_gap",
)


def _score(scores, horse_no):
    try:
        value = float(scores[str(horse_no)])
        if value == value and abs(value) != float("inf"):
            return value
    except (KeyError, TypeError, ValueError):
        pass
    raise ValueError(f"Missing finite individual score for horse {horse_no}")


def axis_pair_features(axis, a, b, scores):
    """Symmetric features for a partner pair conditional on a fixed axis."""
    sa, sb, sx = _score(scores, a["horse_no"]), _score(scores, b["horse_no"]), _score(scores, axis["horse_no"])
    base = pair_features(a, b)
    def metric(row, key, default=0.0):
        try:
            value = float(row.get(key, default))
            return value if value == value and abs(value) != float("inf") else default
        except (TypeError, ValueError):
            return default

    # Reliability-adjusted distance/venue/going form. The underlying rates and
    # sample counts are computed only from races before the target race.
    def reliable_rate(row, rate_key, runs_key, prior=8.0):
        runs = max(0.0, metric(row, runs_key))
        rate = max(0.0, min(1.0, metric(row, rate_key, 0.5)))
        return (rate * runs + 0.5 * prior) / (runs + prior)

    da, db, dx = (
        reliable_rate(a, "distance_top3_rate", "distance_runs"),
        reliable_rate(b, "distance_top3_rate", "distance_runs"),
        reliable_rate(axis, "distance_top3_rate", "distance_runs"),
    )
    va, vb = (
        reliable_rate(a, "venue_top3_rate", "venue_runs"),
        reliable_rate(b, "venue_top3_rate", "venue_runs"),
    )
    ga, gb = (
        reliable_rate(a, "going_top3_rate", "going_runs"),
        reliable_rate(b, "going_top3_rate", "going_runs"),
    )
    return base + [
        (sa + sb) / 2.0, abs(sa - sb),
        (abs(sa - sx) + abs(sb - sx)) / 2.0,
        abs(abs(sa - sx) - abs(sb - sx)),
        (da + db) / 2.0, abs(da - db), min(da, db),
        (da + db) / 2.0 - dx, dx,
        (va + vb) / 2.0, abs(va - vb),
        (ga + gb) / 2.0, abs(ga - gb),
    ]


def _candidate_rows(group, candidate_horse_nos):
    by_no = {str(row["horse_no"]): row for row in group}
    candidates = [str(no) for no in candidate_horse_nos]
    if len(candidates) < 2 or len(set(candidates)) != len(candidates):
        raise ValueError("Candidate pool must contain at least two unique horses")
    if any(no not in by_no for no in candidates):
        raise ValueError("Every candidate must exist in the race group")
    return by_no, candidates


def build_axis_pair_training_data(races):
    """Create pair labels only for races where the preselected axis made top 3.

    Each race dict contains group, axis_horse_no, candidate_horse_nos and
    individual_scores. Candidate membership/scores must be pre-off values.
    """
    x_rows, labels, diagnostics = [], [], {
        "races_seen": len(races), "axis_podium_races": 0,
        "races_with_positive_candidate_pair": 0,
        "positive_pairs": 0, "negative_pairs": 0,
    }
    for race in races:
        group = race["group"]
        by_no = {str(row["horse_no"]): row for row in group}
        axis_no = str(race["axis_horse_no"])
        if axis_no not in by_no:
            raise ValueError("Axis horse does not exist in race group")
        axis = by_no[axis_no]
        if axis.get("finish") is None or int(axis["finish"]) > 3:
            continue
        diagnostics["axis_podium_races"] += 1
        by_no, candidates = _candidate_rows(group, race["candidate_horse_nos"])
        candidates = [no for no in candidates if no != axis_no]
        if len(candidates) < 2:
            continue
        scores = race["individual_scores"]
        positives_in_race = 0
        for left, right in itertools.combinations(candidates, 2):
            a, b = by_no[left], by_no[right]
            # A starter without a classified finish (e.g. DNF/disqualification)
            # is a negative partner label, not a reason to remove the pair.
            label = int(
                a.get("finish") is not None and b.get("finish") is not None
                and int(a["finish"]) <= 3 and int(b["finish"]) <= 3
            )
            x_rows.append(axis_pair_features(axis, a, b, scores))
            labels.append(label)
            diagnostics["positive_pairs" if label else "negative_pairs"] += 1
            positives_in_race += label
        diagnostics["races_with_positive_candidate_pair"] += int(positives_in_race > 0)
    return x_rows, labels, diagnostics


def fit_axis_pair_model(races):
    """Fit a regularized logistic model on chronological training races only."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x_rows, labels, diagnostics = build_axis_pair_training_data(races)
    if diagnostics["positive_pairs"] < 10 or diagnostics["negative_pairs"] < 10:
        raise ValueError("Insufficient conditional pair labels: " + str(diagnostics))
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=600, C=0.5, random_state=42),
    )
    model.fit(x_rows, labels)
    return {
        "model": model,
        "features": list(FEATURE_NAMES),
        "training_pairs": len(labels),
        "diagnostics": diagnostics,
    }


def rank_axis_partner_pairs(group, axis_horse_no, candidate_horse_nos, individual_scores, fitted):
    """Rank unordered partner pairs for a pre-off axis and candidate pool."""
    by_no, candidates = _candidate_rows(group, candidate_horse_nos)
    axis_no = str(axis_horse_no)
    if axis_no not in by_no:
        raise ValueError("Axis horse does not exist in race group")
    candidates = [no for no in candidates if no != axis_no]
    if len(candidates) < 2:
        raise ValueError("At least two non-axis candidates are required")
    pairs = list(itertools.combinations(candidates, 2))
    features = [axis_pair_features(by_no[axis_no], by_no[a], by_no[b], individual_scores)
                for a, b in pairs]
    probabilities = fitted["model"].predict_proba(features)[:, 1].tolist()
    return sorted(
        [{"pair": tuple(sorted((a, b), key=int)), "score": float(score)}
         for (a, b), score in zip(pairs, probabilities)],
        key=lambda item: (-item["score"], item["pair"]),
    )
