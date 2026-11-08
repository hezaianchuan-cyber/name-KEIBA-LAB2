"""Research-only horse evaluation model (v1).

The model separates latent win strength from top-three reliability. It consumes
strictly pre-race history supplied by the caller; it never reads current-race
finishes, payouts, or odds. Missing evidence is neutral rather than fabricated.
"""
from __future__ import annotations

import math
from statistics import mean, pstdev

AXIS_WEIGHTS = {
    "ability": 0.28,
    "condition": 0.20,
    "suitability": 0.18,
    "pace": 0.12,
    "race_content": 0.14,
    "partnership": 0.08,
}
MODEL_VERSION = "horse-evaluation-v1"


def clamp01(value, default=0.5):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    return max(0.0, min(1.0, value))


def _weighted_recent(values):
    values = [clamp01(v) for v in values if v is not None]
    if not values:
        return 0.5
    weights = list(range(1, len(values) + 1))
    return sum(v * w for v, w in zip(values, weights)) / sum(weights)


def _rank_quality(row):
    finish, size = row.get("finish"), row.get("field_size")
    try:
        finish, size = float(finish), float(size)
    except (TypeError, ValueError):
        return None
    if finish <= 0 or size <= 1 or finish > size:
        return None
    return clamp01((size - finish) / (size - 1))


def _margin_quality(row):
    value = row.get("margin_quality")
    if value is not None:
        return clamp01(value)
    # Historical parser stores textual margin in some data sources.
    try:
        gap = max(0.0, float(str(row.get("margin", "")).replace(",", "")))
    except (TypeError, ValueError):
        return None
    return math.exp(-gap / 2.0)


def _style(history):
    rows = history[-5:]
    positions = [r.get("first_corner_pct") for r in rows]
    positions = [float(x) for x in positions if x is not None]
    if not positions:
        return "unknown"
    avg = mean(positions)
    if avg <= 0.12:
        return "front"
    if avg <= 0.38:
        return "stalker"
    if avg <= 0.65:
        return "closer"
    return "deep_closer"


def _axis_scores(history, race, horse):
    recent = history[-5:]
    quality = [_rank_quality(r) for r in recent]
    quality = [v for v in quality if v is not None]
    margins = [_margin_quality(r) for r in recent]
    margins = [v for v in margins if v is not None]
    closings = [r.get("last3f_rank_score") for r in recent
                if r.get("last3f_rank_score") is not None]

    # Ability emphasizes quality of past performance, not raw finishing position.
    ability = 0.65 * _weighted_recent(quality) + 0.20 * _weighted_recent(margins) + 0.15 * _weighted_recent(closings)

    # Current condition: trend and consistency are distinct from lifetime ability.
    condition = 0.5
    if len(quality) >= 2:
        split = max(1, len(quality) // 2)
        older, newer = mean(quality[:split]), mean(quality[split:])
        trend = clamp01(0.5 + (newer - older) * 1.5)
        consistency = clamp01(1.0 - pstdev(quality) * 1.5)
        condition = 0.65 * trend + 0.35 * consistency
    elif quality:
        condition = 0.5 + (quality[-1] - 0.5) * 0.5

    # Suitability is based only on prior runs under comparable conditions.
    venue = str(race.get("venue") or "")
    distance = race.get("distance")
    going = str(race.get("going") or "")
    comparable = []
    for row in recent:
        same_venue = bool(venue) and str(row.get("venue") or "") == venue
        try:
            same_distance = distance is not None and row.get("distance") is not None and abs(float(row["distance"]) - float(distance)) <= 200
        except (TypeError, ValueError):
            same_distance = False
        same_going = bool(going) and str(row.get("going") or "") == going
        if same_venue or same_distance or same_going:
            q = _rank_quality(row)
            if q is not None:
                comparable.append(q)
    suitability = _weighted_recent(comparable) if comparable else 0.5

    # Race pace: reward a runner's historical style only when today's field
    # projects a genuinely contested lead; otherwise stay neutral.
    style = _style(history)
    profiles = race.get("predicted_styles") or []
    front_count = sum(1 for x in profiles if x in ("front", "逃げ"))
    pace = 0.5
    if front_count >= 3:
        pace = {"front": 0.35, "stalker": 0.48, "closer": 0.68, "deep_closer": 0.72}.get(style, 0.5)
    elif front_count == 1:
        pace = {"front": 0.68, "stalker": 0.58, "closer": 0.43, "deep_closer": 0.38}.get(style, 0.5)

    # Race-content axis values late-race performance and position gain separately
    # from the finish result. Unknown components remain neutral.
    position_gains = [r.get("position_gain") for r in recent if r.get("position_gain") is not None]
    race_content = 0.7 * _weighted_recent(closings) + 0.3 * _weighted_recent(
        [clamp01(0.5 + float(x)) for x in position_gains]
    ) if closings or position_gains else 0.5

    # Partnership requires prior horse/jockey pair evidence; absent data is neutral.
    partnership = 0.5
    pair_stats = (horse.get("jockey_pair_stats") or {})
    if pair_stats:
        runs = max(0, int(pair_stats.get("runs", 0) or 0))
        top3 = max(0, int(pair_stats.get("top3", 0) or 0))
        partnership = (top3 + 2.0) / (runs + 4.0)
    elif jockey_stats and horse.get("jockey"):
        stat = jockey_stats.get(horse["jockey"], {})
        partnership = (float(stat.get("top3", 0)) + 5.0) / (float(stat.get("runs", 0)) + 10.0)

    return {
        "ability": clamp01(ability),
        "condition": clamp01(condition),
        "suitability": clamp01(suitability),
        "pace": clamp01(pace),
        "race_content": clamp01(race_content),
        "partnership": clamp01(partnership),
    }


def _softmax(scores, temperature=0.12):
    temperature = max(0.02, float(temperature))
    peak = max(scores)
    exps = [math.exp((s - peak) / temperature) for s in scores]
    total = sum(exps)
    return [v / total for v in exps]


def _top3_probabilities(scores, temperature=0.10):
    """Independent inclusion probabilities calibrated to sum to min(3, field size)."""
    target = min(3, len(scores))
    if not scores:
        return []
    lo, hi = -30.0, 30.0
    scaled = [s / max(0.02, temperature) for s in scores]
    for _ in range(80):
        mid = (lo + hi) / 2
        total = sum(1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x + mid)))) for x in scaled)
        if total < target:
            lo = mid
        else:
            hi = mid
    shift = (lo + hi) / 2
    return [1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x + shift)))) for x in scaled]


def evaluate_field(field, history_by_name, race, jockey_stats=None,
                   axis_weights=None, win_temperature=0.12, top3_temperature=0.10):
    """Return every runner, separately ranked/probabilized for win and top-three."""
    weights = dict(axis_weights or AXIS_WEIGHTS)
    if set(weights) != set(AXIS_WEIGHTS) or any(float(v) < 0 for v in weights.values()):
        raise ValueError("axis_weights must contain the six known non-negative axes")
    total_weight = sum(float(v) for v in weights.values())
    if total_weight <= 0:
        raise ValueError("axis_weights must have positive total weight")
    weights = {k: float(v) / total_weight for k, v in weights.items()}

    rows = []
    for horse in field:
        name = str(horse.get("name") or "")
        history = history_by_name.get(name, [])
        axes = _axis_scores(history, race, horse)
        score = sum(axes[k] * weights[k] for k in weights)
        rows.append({"horse_no": str(horse.get("no", "")), "name": name,
                     "score": score, "axes": axes})
    win_probs = _softmax([r["score"] for r in rows], win_temperature)
    top3_probs = _top3_probabilities([r["score"] for r in rows], top3_temperature)
    for row, win_p, top3_p in zip(rows, win_probs, top3_probs):
        row["win_probability"] = win_p
        row["top3_probability"] = top3_p
    return {
        "model_version": MODEL_VERSION,
        "weights": weights,
        "predictions": sorted(rows, key=lambda r: r["score"], reverse=True),
        "win_ranking": [r["horse_no"] for r in sorted(rows, key=lambda r: r["win_probability"], reverse=True)],
        "top3_ranking": [r["horse_no"] for r in sorted(rows, key=lambda r: r["top3_probability"], reverse=True)],
        "probability_checks": {
            "win_sum": sum(win_probs),
            "top3_sum": sum(top3_probs),
        },
        "research_only": True,
    }
