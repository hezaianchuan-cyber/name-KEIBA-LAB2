"""Research-only trio compatibility scoring primitives.

The individual LambdaRank score remains the horse's ability/condition estimate.
This module adds a separate, transparent pair-compatibility adjustment for trio
ranking. It does not change production predictions.
"""
from __future__ import annotations

def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))

def running_style(row: dict) -> float:
    """Estimated front-running tendency in [0,1]; larger means more forward."""
    # The historical first-corner percentile is a fraction of field position.
    return _clamp(1.0 - float(row.get("recent_first_corner_pct", 0.5)))

def pair_compatibility(a: dict, b: dict, field_size: int | None = None) -> float:
    """Return a bounded (-1,+1) pair adjustment, without using race outcomes.

    Penalize two strongly forward horses when the field has multiple likely
    leaders; mildly reward a front/closer balance. This is a hypothesis to test,
    not a claim that mixed running styles always win.
    """
    sa, sb = running_style(a), running_style(b)
    both_forward = sa * sb
    style_separation = abs(sa - sb)
    crowding = _clamp(float(a.get("front_crowding_ratio", b.get("front_crowding_ratio", 0.0))) * 3.0)
    front_count = max(float(a.get("predicted_front_count", 0.0)),
                      float(b.get("predicted_front_count", 0.0)))
    field = max(3.0, float(field_size or a.get("field_size") or b.get("field_size") or 12.0))
    multiple_leaders = _clamp((front_count - 1.0) / max(field * 0.25, 1.0))
    clash_penalty = 0.30 * both_forward * max(crowding, multiple_leaders)
    balance_bonus = 0.06 * style_separation * (1.0 - 0.5 * crowding)
    return _clamp(balance_bonus - clash_penalty, -0.30, 0.06)

def trio_compatibility(group: list[dict], horse_numbers: tuple[str, ...] | list[str]) -> float:
    """Mean of the three pair adjustments for an unordered trio."""
    by_no = {str(row["horse_no"]): row for row in group}
    numbers = tuple(map(str, horse_numbers))
    if len(numbers) != 3 or len(set(numbers)) != 3:
        raise ValueError("A trio must contain three distinct horse numbers")
    if any(number not in by_no for number in numbers):
        raise ValueError("Every trio horse must exist in the race group")
    pairs = ((numbers[0], numbers[1]), (numbers[0], numbers[2]), (numbers[1], numbers[2]))
    return sum(pair_compatibility(by_no[x], by_no[y]) for x, y in pairs) / 3.0

def trio_score(group: list[dict], horse_numbers: tuple[str, ...] | list[str],
               individual_scores: dict[str, float], compatibility_weight: float = 0.15) -> float:
    """Separate individual merit from pair compatibility in the final set score."""
    if not 0.0 <= compatibility_weight <= 1.0:
        raise ValueError("compatibility_weight must be between 0 and 1")
    numbers = tuple(map(str, horse_numbers))
    if len(numbers) != 3 or any(number not in individual_scores for number in numbers):
        raise ValueError("Individual scores must exist for exactly three selected horses")
    individual = sum(float(individual_scores[number]) for number in numbers) / 3.0
    compatibility = trio_compatibility(group, numbers)
    return (1.0 - compatibility_weight) * individual + compatibility_weight * (0.5 + compatibility)
