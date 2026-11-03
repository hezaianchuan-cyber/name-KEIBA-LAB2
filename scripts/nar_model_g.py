#!/usr/bin/env python3
"""Experimental Model G challenger for prospective NAR pre-off evaluation.

G is deliberately separate from model E. It consumes only timestamped odds and
features available before the off. It never treats confirmed/final odds as live
inputs and never labels an unvalidated candidate as profitable.
"""
from __future__ import annotations

import math
import unicodedata

MODEL_VERSION = "nar-model-g-experimental-1"
MIN_CALIBRATION_RACES = 500
MIN_EXPECTED_VALUE = 1.08
MAX_TICKETS = 20
TICKET_UNIT_JPY = 100
RACE_CAP_JPY = 2000
UNORDERED = {"wide", "quinella", "trio"}


def _number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _horse_no(value):
    digits = "".join(ch for ch in str(value or "") if ch.isdigit())
    return str(int(digits)) if digits else ""


def _win_market_probabilities(odds_rows):
    raw = {}
    for row in odds_rows or []:
        label = unicodedata.normalize("NFKC", str(row.get("type") or "")).replace(" ", "")
        if "単勝" not in label:
            continue
        no = _horse_no(row.get("n1"))
        odds = _number(row.get("odds"))
        if no and odds and odds > 0:
            raw[no] = 1.0 / odds
    total = sum(raw.values())
    return ({no: value / total for no, value in raw.items()} if total > 0 else {})


def build_g_predictions(horse_predictions, odds_rows, calibration_races, temperature=0.18):
    """Add a small, auditable overlooked-runner adjustment to calibrated C scores.

    The adjustment is intentionally capped and remains experimental until
    prospective snapshots supply an independent calibration/holdout sample.
    """
    source = [dict(row) for row in (horse_predictions or [])]
    if not source:
        return []
    market = _win_market_probabilities(odds_rows)
    model_order = sorted(source, key=lambda x: _number(x.get("win_probability")) or 0, reverse=True)
    market_order = sorted(market, key=market.get, reverse=True)
    model_rank = {_horse_no(row.get("horse_no")): i + 1 for i, row in enumerate(model_order)}
    market_rank = {no: i + 1 for i, no in enumerate(market_order)}
    temp = []
    for row in source:
        no = _horse_no(row.get("horse_no"))
        features = row.get("features") or {}
        usable = [_number(features.get(key)) for key in
                  ("margin_form", "closing_form", "time_form", "going_form")]
        usable = [min(1.0, max(0.0, x)) for x in usable if x is not None]
        signal = sum(usable) / len(usable) if usable else 0.5
        m_rank = market_rank.get(no)
        r_rank = model_rank.get(no, len(source))
        market_is_cold = m_rank is not None and m_rank > max(2, math.ceil(len(source) * 0.50))
        model_respects_form = r_rank <= max(3, math.ceil(len(source) * 0.60))
        overlooked = market_is_cold and model_respects_form and signal >= 0.60
        # A capped score adjustment avoids allowing one sparse historical feature
        # to dominate the existing chronological model.
        bonus = min(0.025, max(0.0, signal - 0.50) * 0.05) if overlooked else 0.0
        base_score = _number(row.get("score")) or 0.0
        row["g_score"] = round(base_score + bonus, 8)
        row["market_win_probability"] = round(market.get(no, 0.0), 8)
        row["market_rank"] = m_rank
        row["model_c_rank"] = r_rank
        row["overlooked_signal"] = round(signal, 6)
        row["overlooked_runner_flag"] = bool(overlooked)
        row["g_adjustment"] = round(bonus, 8)
        temp.append(row)

    # Preserve the existing chronological temperature calibration. This is a
    # calibrated-base challenger, not a claim that G itself is calibrated.
    scores = [row["g_score"] for row in temp]
    temperature = _number(temperature) or 0.18
    if temperature <= 0:
        temperature = 0.18
    max_score = max(scores)
    weights = [math.exp((score - max_score) / temperature) for score in scores]
    total = sum(weights)
    probabilities = [weight / total for weight in weights] if total else [1 / len(temp)] * len(temp)
    # For a zero-adjustment race this returns the existing C distribution exactly
    # up to rounding, avoiding an uncalibrated re-temperature of the base model.
    if not any(row["g_adjustment"] for row in temp):
        probabilities = [_number(row.get("win_probability")) or 0.0 for row in temp]
        ptotal = sum(probabilities)
        probabilities = [p / ptotal for p in probabilities] if ptotal > 0 else [1 / len(temp)] * len(temp)
    for row, probability in zip(temp, probabilities):
        row["win_probability"] = round(probability, 10)
        row["g_win_probability"] = round(probability, 10)
    return temp


def _ticket_key(ticket):
    kind = str(ticket.get("bet_type") or "")
    numbers = [_horse_no(x) for x in ticket.get("numbers") or []]
    if not numbers or any(not x for x in numbers) or len(set(numbers)) != len(numbers):
        return None
    key_numbers = tuple(sorted(numbers)) if kind in UNORDERED else tuple(numbers)
    return kind, key_numbers


def select_g_strategy(race, g_predictions, candidate_combinations, calibration_races):
    """Choose one bet type, skip weak races, and cap spend at JPY 2,000."""
    reasons = []
    runners = list(g_predictions or [])
    if len(runners) < 5:
        reasons.append("出走頭数5頭未満")
    if int(calibration_races or 0) < MIN_CALIBRATION_RACES:
        reasons.append("校正用データ500レース未満")
    odds_coverage = sum(1 for row in runners if (_number(row.get("win_odds")) or 0) > 0) / max(1, len(runners))
    if odds_coverage < 0.70:
        reasons.append("単勝オッズ取得率70%未満")
    if abs(sum(_number(row.get("win_probability")) or 0 for row in runners) - 1.0) > 0.01:
        reasons.append("確率分布が不正")
    if reasons:
        return {
            "model_version": MODEL_VERSION, "status": "skip_insufficient_evidence",
            "reasons": reasons, "selected_bet_type": None, "tickets": [],
            "stake_jpy": 0, "race_cap_jpy": RACE_CAP_JPY,
            "odds_coverage": round(odds_coverage, 4),
            "calibration_races": int(calibration_races or 0),
            "validated_profitable": False,
        }

    by_type = {}
    seen = set()
    for ticket in candidate_combinations or []:
        key = _ticket_key(ticket)
        if key is None or key in seen:
            continue
        seen.add(key)
        odds = _number(ticket.get("odds"))
        probability = _number(ticket.get("probability"))
        ev = _number(ticket.get("expected_value"))
        if odds is None or odds <= 0 or probability is None or probability <= 0 or ev is None:
            continue
        if ev < MIN_EXPECTED_VALUE:
            continue
        item = dict(ticket)
        item["expected_value"] = ev
        item["probability"] = probability
        item["odds"] = odds
        by_type.setdefault(key[0], []).append(item)

    qualified = {
        kind: sorted(items, key=lambda x: (x["expected_value"], x["probability"]), reverse=True)
        for kind, items in by_type.items() if items
    }
    if not qualified:
        return {
            "model_version": MODEL_VERSION, "status": "skip_no_positive_value",
            "reasons": ["期待値1.08以上かつ有効オッズの買い目なし"],
            "selected_bet_type": None, "tickets": [], "stake_jpy": 0,
            "race_cap_jpy": RACE_CAP_JPY, "odds_coverage": round(odds_coverage, 4),
            "calibration_races": int(calibration_races or 0),
            "validated_profitable": False,
        }

    selected_type = max(
        qualified,
        key=lambda kind: (
            sum(x["expected_value"] for x in qualified[kind]) / len(qualified[kind]),
            sum(x["probability"] for x in qualified[kind]) / len(qualified[kind]),
            -len(qualified[kind]),
        ),
    )
    selected = qualified[selected_type][:MAX_TICKETS]
    tickets = []
    for item in selected:
        numbers = [_horse_no(x) for x in item.get("numbers") or []]
        tickets.append({
            "bet_type": selected_type, "numbers": numbers,
            "probability": round(item["probability"], 8),
            "odds": round(item["odds"], 4),
            "expected_value": round(item["expected_value"], 6),
            "stake_jpy": TICKET_UNIT_JPY,
        })
    stake = sum(item["stake_jpy"] for item in tickets)
    if stake > RACE_CAP_JPY:
        raise AssertionError("Model G exceeded the per-race stake cap")
    return {
        "model_version": MODEL_VERSION,
        "status": "experimental_only_not_validated",
        "reasons": ["Gは新規チャレンジャー。将来の直前オッズスナップショットで未使用期間検証が必要"],
        "selected_bet_type": selected_type,
        "tickets": tickets,
        "ticket_count": len(tickets),
        "stake_jpy": stake,
        "race_cap_jpy": RACE_CAP_JPY,
        "odds_coverage": round(odds_coverage, 4),
        "calibration_races": int(calibration_races or 0),
        "minimum_expected_value": MIN_EXPECTED_VALUE,
        "validated_profitable": False,
        "race_context": {
            "venue": race.get("venue"), "distance": race.get("distance"),
            "surface": race.get("surface"), "going": race.get("going"),
        },
    }
