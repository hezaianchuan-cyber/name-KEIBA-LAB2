#!/usr/bin/env python3
"""Walk-forward comparison: fixed win-axis + LightGBM LambdaRank opponent pool.

The canonical chronological backtest supplies the fixed pre-off axis. The
existing ML challenger pipeline supplies strictly pre-race LambdaRank features
and scores. A conditional pair model is fitted only on a later development
window and evaluated once on an untouched chronological holdout.

This writes an aggregate research artifact only. No live model/ticket files are
changed and no real-money betting is performed. Feature ablations share identical
chronological partitions and holdout races.
"""
from __future__ import annotations

import itertools
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import nar_backtest as nb
import nar_ml_challenger_backtest as ml
from nar_axis_conditional_pair_ranker import (
    build_axis_pair_training_data,
    fit_axis_pair_model,
    rank_axis_partner_pairs,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "axis-conditional-pair-backtest.json"
BOOTSTRAP_REPS = 2000
TICKETS_PER_RACE = 6
STAKE_PER_TICKET = 100


def collect_fixed_axes(races, horses, payouts):
    """Replay every race chronologically and capture the fixed pre-off win axis."""
    axis_by_key = {}
    original_evaluator = nb.evaluate_field

    def capture_evaluator(field, history_by_name, race, *args, **kwargs):
        result = original_evaluator(field, history_by_name, race, *args, **kwargs)
        win_order = result.get("win_ranking") or []
        if win_order:
            predictions = result.get("predictions") or []
            prediction_by_no = {
                str(row.get("horse_no")): {
                    "name": row.get("name"),
                    "score": row.get("score"),
                    "win_probability": row.get("win_probability"),
                    "top3_probability": row.get("top3_probability"),
                    "axes": row.get("axes") or {},
                }
                for row in predictions
            }
            axis_by_key["|".join(map(str, race["key"]))] = {
                "axis": str(win_order[0]),
                "top7": [str(no) for no in win_order[:7]],
                "win_ranking": [str(no) for no in win_order],
                "predictions": prediction_by_no,
            }
        return result

    nb.evaluate_field = capture_evaluator
    try:
        # This preserves the canonical date-batched prior-history replay while
        # skipping unrelated ticket/report aggregation.
        nb.run_backtest(races, horses, payouts, training_only=True)
    finally:
        nb.evaluate_field = original_evaluator
    return axis_by_key


def _ranker_matrix(group, category_maps):
    return [[
        row[key] if key not in ml.CATEGORICAL else category_maps[key].get(row[key], -1)
        for key in ml.RAW_FEATURES
    ] for row in group]


def make_ranked_records(rows, groups, axis_by_key, payouts, ranker, category_maps):
    records = []
    for group in groups:
        race_key_text = str(group[0]["race_key"])
        axis_info = axis_by_key.get(race_key_text)
        if not axis_info:
            continue
        axis = axis_info["axis"]
        score_values = ranker.predict(_ranker_matrix(group, category_maps))
        score_by_no = {
            str(row["horse_no"]): float(score)
            for row, score in zip(group, score_values)
        }
        if axis not in score_by_no:
            continue
        ranked_opponents = sorted(
            (no for no in score_by_no if no != axis),
            key=lambda no: (-score_by_no[no], int(no)),
        )
        if len(ranked_opponents) < 4:
            continue
        candidates = [axis] + ranked_opponents[:6]
        parts = race_key_text.split("|")
        if len(parts) != 3:
            continue
        payout_key = nb.race_key(parts[0], parts[1], parts[2])
        records.append({
            "date": str(group[0]["date"]),
            "race_key": race_key_text,
            "group": group,
            "axis_horse_no": axis,
            "axis_top7_horse_nos": axis_info["top7"],
            "candidate_horse_nos": candidates,
            "individual_scores": score_by_no,
            "payout": payouts.get(payout_key) or {},
        })
    records.sort(key=lambda row: (row["date"], row["race_key"]))
    return records


def _baseline_pairs(record):
    axis = str(record["axis_horse_no"])
    scores = record["individual_scores"]
    opponents = [str(no) for no in record["candidate_horse_nos"] if str(no) != axis]
    opponents.sort(key=lambda no: (-float(scores[no]), int(no)))
    return list(itertools.combinations(opponents[:4], 2))


def _tickets(axis, pairs):
    return [tuple(sorted((str(axis), str(a), str(b)), key=int)) for a, b in pairs]


def _bootstrap_ci(race_rows):
    if len(race_rows) < 30:
        return None
    rng = random.Random(20261111)
    values = []
    for _ in range(BOOTSTRAP_REPS):
        sample = [race_rows[rng.randrange(len(race_rows))] for _ in race_rows]
        stake = sum(row[0] for row in sample)
        returned = sum(row[1] for row in sample)
        if stake:
            values.append(100.0 * returned / stake)
    values.sort()
    if not values:
        return None
    return [
        round(values[int(0.025 * (len(values) - 1))], 2),
        round(values[int(0.975 * (len(values) - 1))], 2),
    ]


def evaluate(records, fitted=None, include_bootstrap=True):
    races = ticket_hits = hit_races = stake = returned = 0
    axis_podium = candidate_covered = selected_covered = 0
    top7_podium_horses = total_podium_horses = 0
    full_podium_in_top7_races = races_with_three_classified_podium = 0
    race_rows = []
    for record in records:
        payout = record["payout"]
        if not payout.get("trio") or not payout.get("win"):
            continue
        axis = str(record["axis_horse_no"])
        by_no = {str(row["horse_no"]): row for row in record["group"]}
        if axis not in by_no:
            continue
        opponents = [str(no) for no in record["candidate_horse_nos"] if str(no) != axis]
        if len(opponents) < 4:
            continue
        if fitted is None:
            selected_pairs = _baseline_pairs(record)
        else:
            ranked_pairs = rank_axis_partner_pairs(
                record["group"], axis, record["candidate_horse_nos"],
                record["individual_scores"], fitted,
            )
            selected_pairs = [tuple(item["pair"]) for item in ranked_pairs[:TICKETS_PER_RACE]]
        tickets = _tickets(axis, selected_pairs)
        if len(tickets) != TICKETS_PER_RACE:
            continue
        race_return = sum(nb.payout_for("trio", ticket, payout) or 0 for ticket in tickets)
        race_hits = sum(
            1 for ticket in tickets if (nb.payout_for("trio", ticket, payout) or 0) > 0
        )
        race_stake = TICKETS_PER_RACE * STAKE_PER_TICKET
        races += 1
        ticket_hits += race_hits
        hit_races += int(race_hits > 0)
        stake += race_stake
        returned += race_return
        race_rows.append((race_stake, race_return))

        podium = {
            no for no, row in by_no.items()
            if row.get("finish") is not None and int(row["finish"]) <= 3
        }
        top7 = set(str(no) for no in record.get("axis_top7_horse_nos", []))
        total_podium_horses += len(podium)
        top7_podium_horses += len(podium & top7)
        if len(podium) == 3:
            races_with_three_classified_podium += 1
            full_podium_in_top7_races += int(podium.issubset(top7))
        if axis in podium:
            axis_podium += 1
            actual_partners = podium - {axis}
            if len(actual_partners) == 2 and actual_partners.issubset(set(opponents)):
                candidate_covered += 1
            if len(actual_partners) == 2 and any(set(pair) == actual_partners for pair in selected_pairs):
                selected_covered += 1

    return {
        "races": races,
        "tickets": races * TICKETS_PER_RACE,
        "ticket_hits": ticket_hits,
        "hit_races": hit_races,
        "ticket_hit_rate_pct": round(100 * ticket_hits / (races * TICKETS_PER_RACE), 2) if races else None,
        "race_hit_rate_pct": round(100 * hit_races / races, 2) if races else None,
        "stake_jpy": stake,
        "return_jpy": returned,
        "profit_jpy": returned - stake,
        "roi_pct": round(100 * returned / stake, 2) if stake else None,
        "roi_95ci_bootstrap_pct": _bootstrap_ci(race_rows) if include_bootstrap else None,
        "axis_podium_races": axis_podium,
        "axis_top3_rate_pct": round(100 * axis_podium / races, 2) if races else None,
        "top7_podium_coverage_pct": round(100 * top7_podium_horses / total_podium_horses, 2) if total_podium_horses else None,
        "top7_full_podium_races": full_podium_in_top7_races,
        "top7_full_podium_rate_pct": round(100 * full_podium_in_top7_races / races_with_three_classified_podium, 2) if races_with_three_classified_podium else None,
        "races_with_three_classified_podium": races_with_three_classified_podium,
        "both_partners_in_candidate_pool": candidate_covered,
        "both_partners_selected": selected_covered,
        "partner_pair_capture_pct": round(100 * selected_covered / axis_podium, 2) if axis_podium else None,
        "note": "Offline simulation only; fixed axis plus six unordered trio tickets at 100 JPY virtual stake each.",
    }


def paired_comparison(records, fitted):
    """Paired race-cluster bootstrap of conditional-pair vs ability-top4."""
    paired = []
    for record in records:
        payout = record["payout"]
        if not payout.get("trio") or not payout.get("win"):
            continue
        axis = str(record["axis_horse_no"])
        by_no = {str(row["horse_no"]): row for row in record["group"]}
        if axis not in by_no:
            continue
        opponents = [str(no) for no in record["candidate_horse_nos"] if str(no) != axis]
        if len(opponents) < 4:
            continue

        baseline_pairs = _baseline_pairs(record)
        ranked_pairs = rank_axis_partner_pairs(
            record["group"], axis, record["candidate_horse_nos"],
            record["individual_scores"], fitted,
        )
        model_pairs = [tuple(item["pair"]) for item in ranked_pairs[:TICKETS_PER_RACE]]
        baseline_tickets = _tickets(axis, baseline_pairs)
        model_tickets = _tickets(axis, model_pairs)
        if len(baseline_tickets) != TICKETS_PER_RACE or len(model_tickets) != TICKETS_PER_RACE:
            continue

        baseline_returns = [nb.payout_for("trio", ticket, payout) or 0 for ticket in baseline_tickets]
        model_returns = [nb.payout_for("trio", ticket, payout) or 0 for ticket in model_tickets]
        paired.append((
            sum(baseline_returns), sum(model_returns),
            sum(value > 0 for value in baseline_returns),
            sum(value > 0 for value in model_returns),
            int(any(value > 0 for value in baseline_returns)),
            int(any(value > 0 for value in model_returns)),
        ))

    n = len(paired)
    if not n:
        return {"races": 0, "error": "No paired races"}
    delta_returns = [row[1] - row[0] for row in paired]
    delta_ticket_hits = [row[3] - row[2] for row in paired]
    delta_race_hits = [row[5] - row[4] for row in paired]
    stake = n * TICKETS_PER_RACE * STAKE_PER_TICKET
    roi_delta = 100.0 * sum(delta_returns) / stake
    ticket_hit_delta = 100.0 * sum(delta_ticket_hits) / (n * TICKETS_PER_RACE)
    race_hit_delta = 100.0 * sum(delta_race_hits) / n

    rng = random.Random(20261112)
    boot_roi, boot_ticket, boot_race = [], [], []
    for _ in range(BOOTSTRAP_REPS):
        sample = [paired[rng.randrange(n)] for _ in range(n)]
        boot_roi.append(
            100.0 * sum(row[1] - row[0] for row in sample)
            / (n * TICKETS_PER_RACE * STAKE_PER_TICKET)
        )
        boot_ticket.append(
            100.0 * sum(row[3] - row[2] for row in sample) / (n * TICKETS_PER_RACE)
        )
        boot_race.append(100.0 * sum(row[5] - row[4] for row in sample) / n)

    def interval(values):
        values.sort()
        return [
            round(values[int(0.025 * (len(values) - 1))], 2),
            round(values[int(0.975 * (len(values) - 1))], 2),
        ]

    return {
        "races": n,
        "roi_delta_pp_model_minus_baseline": round(roi_delta, 2),
        "roi_delta_95ci_paired_bootstrap_pp": interval(boot_roi),
        "ticket_hit_rate_delta_pp_model_minus_baseline": round(ticket_hit_delta, 2),
        "ticket_hit_rate_delta_95ci_paired_bootstrap_pp": interval(boot_ticket),
        "race_hit_rate_delta_pp_model_minus_baseline": round(race_hit_delta, 2),
        "race_hit_rate_delta_95ci_paired_bootstrap_pp": interval(boot_race),
        "races_model_return_higher": sum(row[1] > row[0] for row in paired),
        "races_baseline_return_higher": sum(row[0] > row[1] for row in paired),
        "races_equal_return": sum(row[0] == row[1] for row in paired),
        "note": "Race-level paired bootstrap; same races and virtual stake, 2,000 resamples. Interval crossing zero means the observed difference is not clearly separated from zero.",
    }




def paired_model_comparison(records, fitted_a, fitted_b):
    """Direct same-race comparison between two feature-ablation rankers."""
    paired = []
    for record in records:
        payout = record["payout"]
        if not payout.get("trio") or not payout.get("win"):
            continue
        axis = str(record["axis_horse_no"])
        by_no = {str(row["horse_no"]): row for row in record["group"]}
        if axis not in by_no:
            continue
        opponents = [str(no) for no in record["candidate_horse_nos"] if str(no) != axis]
        if len(opponents) < 4:
            continue
        ranked_a = rank_axis_partner_pairs(
            record["group"], axis, record["candidate_horse_nos"],
            record["individual_scores"], fitted_a,
        )
        ranked_b = rank_axis_partner_pairs(
            record["group"], axis, record["candidate_horse_nos"],
            record["individual_scores"], fitted_b,
        )
        tickets_a = _tickets(axis, [tuple(item["pair"]) for item in ranked_a[:TICKETS_PER_RACE]])
        tickets_b = _tickets(axis, [tuple(item["pair"]) for item in ranked_b[:TICKETS_PER_RACE]])
        if len(tickets_a) != TICKETS_PER_RACE or len(tickets_b) != TICKETS_PER_RACE:
            continue
        returns_a = [nb.payout_for("trio", ticket, payout) or 0 for ticket in tickets_a]
        returns_b = [nb.payout_for("trio", ticket, payout) or 0 for ticket in tickets_b]
        paired.append((
            sum(returns_a), sum(returns_b),
            sum(value > 0 for value in returns_a),
            sum(value > 0 for value in returns_b),
            int(any(value > 0 for value in returns_a)),
            int(any(value > 0 for value in returns_b)),
        ))

    n = len(paired)
    if not n:
        return {"races": 0, "error": "No paired races"}
    roi_delta = 100.0 * sum(row[0] - row[1] for row in paired) / (n * TICKETS_PER_RACE * STAKE_PER_TICKET)
    ticket_delta = 100.0 * sum(row[2] - row[3] for row in paired) / (n * TICKETS_PER_RACE)
    race_delta = 100.0 * sum(row[4] - row[5] for row in paired) / n
    rng = random.Random(20261113)
    boot_roi, boot_ticket, boot_race = [], [], []
    for _ in range(BOOTSTRAP_REPS):
        sample = [paired[rng.randrange(n)] for _ in range(n)]
        boot_roi.append(100.0 * sum(row[0] - row[1] for row in sample) / (n * TICKETS_PER_RACE * STAKE_PER_TICKET))
        boot_ticket.append(100.0 * sum(row[2] - row[3] for row in sample) / (n * TICKETS_PER_RACE))
        boot_race.append(100.0 * sum(row[4] - row[5] for row in sample) / n)

    def interval(values):
        values.sort()
        return [
            round(values[int(0.025 * (len(values) - 1))], 2),
            round(values[int(0.975 * (len(values) - 1))], 2),
        ]

    return {
        "races": n,
        "roi_delta_pp_A_minus_B": round(roi_delta, 2),
        "roi_delta_95ci_paired_bootstrap_pp": interval(boot_roi),
        "ticket_hit_rate_delta_pp_A_minus_B": round(ticket_delta, 2),
        "ticket_hit_rate_delta_95ci_paired_bootstrap_pp": interval(boot_ticket),
        "race_hit_rate_delta_pp_A_minus_B": round(race_delta, 2),
        "race_hit_rate_delta_95ci_paired_bootstrap_pp": interval(boot_race),
        "races_A_return_higher": sum(row[0] > row[1] for row in paired),
        "races_B_return_higher": sum(row[1] > row[0] for row in paired),
        "races_equal_return": sum(row[0] == row[1] for row in paired),
        "note": "Direct paired race-cluster bootstrap; A minus B, identical races and virtual stake, 2,000 resamples. ROI interval crossing zero means no clear evidence of superiority.",
    }



def changed_ticket_diagnostics(records, fitted_a, fitted_b, label):
    """Explain changed partner tickets race by race; no feature is tuned on holdout."""
    changed = []
    summary = {"label": label, "races_checked": 0, "races_with_changed_tickets": 0,
               "a_only_ticket_hits": 0, "b_only_ticket_hits": 0,
               "a_only_return_jpy": 0, "b_only_return_jpy": 0}
    for record in records:
        payout = record["payout"]
        if not payout.get("trio") or not payout.get("win"):
            continue
        axis = str(record["axis_horse_no"])
        by_no = {str(row["horse_no"]): row for row in record["group"]}
        if axis not in by_no:
            continue
        ranked_a = rank_axis_partner_pairs(record["group"], axis, record["candidate_horse_nos"], record["individual_scores"], fitted_a)
        ranked_b = rank_axis_partner_pairs(record["group"], axis, record["candidate_horse_nos"], record["individual_scores"], fitted_b)
        pairs_a = [tuple(item["pair"]) for item in ranked_a[:TICKETS_PER_RACE]]
        pairs_b = [tuple(item["pair"]) for item in ranked_b[:TICKETS_PER_RACE]]
        if len(pairs_a) != TICKETS_PER_RACE or len(pairs_b) != TICKETS_PER_RACE:
            continue
        summary["races_checked"] += 1
        if set(pairs_a) == set(pairs_b):
            continue
        summary["races_with_changed_tickets"] += 1
        a_only = sorted(set(pairs_a) - set(pairs_b))
        b_only = sorted(set(pairs_b) - set(pairs_a))
        def ticket_info(pair):
            ticket = _tickets(axis, [pair])[0]
            ret = nb.payout_for("trio", ticket, payout) or 0
            return {"pair": list(pair), "ticket": list(ticket), "return_jpy": ret, "hit": ret > 0}
        a_info, b_info = [ticket_info(p) for p in a_only], [ticket_info(p) for p in b_only]
        summary["a_only_ticket_hits"] += sum(x["hit"] for x in a_info)
        summary["b_only_ticket_hits"] += sum(x["hit"] for x in b_info)
        summary["a_only_return_jpy"] += sum(x["return_jpy"] for x in a_info)
        summary["b_only_return_jpy"] += sum(x["return_jpy"] for x in b_info)
        podium = sorted([no for no, row in by_no.items() if row.get("finish") is not None and int(row["finish"]) <= 3], key=int)
        relevant = sorted({no for pair in a_only + b_only for no in pair}, key=int)
        horse_details = {}
        for no in relevant:
            row = by_no[no]
            horse_details[no] = {
                "name": row.get("name"), "finish": row.get("finish"),
                "distance_top3_rate": row.get("distance_top3_rate"), "distance_runs": row.get("distance_runs"),
                "venue_distance_top3_rate": row.get("venue_distance_top3_rate"), "venue_distance_runs": row.get("venue_distance_runs"),
                "venue_top3_rate": row.get("venue_top3_rate"), "venue_runs": row.get("venue_runs"),
                "going_top3_rate": row.get("going_top3_rate"), "going_runs": row.get("going_runs"),
                "individual_score": record["individual_scores"].get(no),
            }
        changed.append({
            "date": record["date"], "race_key": record["race_key"],
            "venue": (record["group"] or [{}])[0].get("venue"),
            "distance_m": (record["group"] or [{}])[0].get("distance_m"),
            "axis_horse_no": axis, "axis_name": by_no[axis].get("name"),
            "podium_horse_nos": podium, "A_only_pairs": a_info, "B_only_pairs": b_info,
            "changed_horses": horse_details,
        })
    return {"summary": summary, "changed_races": changed,
            "note": "変更された買い目、着順、過去の距離・競馬場距離実績、仮想払戻を記録。着順と払戻は評価用のみ。"}
    

def venue_distance_ablation(records, fitted_a, fitted_b, fitted_c):
    """Descriptive venue-by-distance results for all variants; no subgroup tuning."""
    def band(group):
        distance = float((group or [{}])[0].get("distance_m") or 0)
        if distance <= 0: return "距離不明"
        if distance <= 1200: return "1200m以下"
        if distance <= 1600: return "1300〜1600m"
        return "1700m以上"
    output = {}
    venues = sorted({str((r.get("group") or [{}])[0].get("venue") or "不明") for r in records})
    for venue in venues:
        venue_rows = [r for r in records if str((r.get("group") or [{}])[0].get("venue") or "不明") == venue]
        output[venue] = {}
        for distance_band in ("1200m以下", "1300〜1600m", "1700m以上", "距離不明"):
            subset = [r for r in venue_rows if band(r.get("group")) == distance_band]
            if subset:
                output[venue][distance_band] = {
                    "A通常距離適性": evaluate(subset, fitted_a, include_bootstrap=False),
                    "B距離適性なし": evaluate(subset, fitted_b, include_bootstrap=False),
                    "C競馬場内距離適性": evaluate(subset, fitted_c, include_bootstrap=False),
                    "sample_warning": "100レース未満は参考値。100レース以上でも別期間での再現確認が必要。重み調整には直結させない。",
                }
    return {"by_venue_and_distance": output,
            "policy": "全競馬場・全距離帯の探索用集計。小標本に合わせて重みを最適化しない。"}


def venue_holdout_breakdown(records, fitted=None):
    """Venue-level ticket results and leave-one-venue-out impact on the same holdout."""
    overall = evaluate(records, fitted)
    venues = sorted({str((row.get("group") or [{}])[0].get("venue") or "不明") for row in records})
    by_venue = {}
    for venue in venues:
        selected = [row for row in records if str((row.get("group") or [{}])[0].get("venue") or "不明") == venue]
        venue_result = evaluate(selected, fitted, include_bootstrap=False)
        remaining = [row for row in records if str((row.get("group") or [{}])[0].get("venue") or "不明") != venue]
        without = evaluate(remaining, fitted, include_bootstrap=False)
        by_venue[venue] = {
            "venue_result": venue_result,
            "overall_without_venue": without,
            "leave_one_out": {
                "races_removed": venue_result["races"],
                "overall_roi_change_pp": round(without["roi_pct"] - overall["roi_pct"], 2)
                    if without.get("roi_pct") is not None and overall.get("roi_pct") is not None else None,
                "overall_profit_change_jpy": without["profit_jpy"] - overall["profit_jpy"]
                    if without.get("profit_jpy") is not None and overall.get("profit_jpy") is not None else None,
                "overall_race_hit_rate_change_pp": round(without["race_hit_rate_pct"] - overall["race_hit_rate_pct"], 2)
                    if without.get("race_hit_rate_pct") is not None and overall.get("race_hit_rate_pct") is not None else None,
            },
            "sample_warning": (
                "100レース未満。見送り判断には使わず参考値として扱う"
                if venue_result["races"] < 100 else
                "最低サンプル数は満たすが、別期間での再現確認が必要"
            ),
        }
    return {
        "overall": overall,
        "by_venue": by_venue,
        "policy": "同一ホールドアウト・同一6点・1点100円の仮想3連複。各競馬場を除外した場合の全体ROI変化を算出。見送りルールの採用には別期間での再検証が必要。",
    }



def venue_distance_breakdown(records, fitted=None):
    """Distance-band diagnostics inside the four venues with weakest model ROI."""
    target_venues = {"盛岡", "水沢", "園田", "門別"}

    def band(group):
        distance = float((group or [{}])[0].get("distance_m") or 0)
        if distance <= 0:
            return "距離不明"
        if distance <= 1200:
            return "1200m以下"
        if distance <= 1600:
            return "1300〜1600m"
        return "1700m以上"

    output = {}
    for venue in sorted(target_venues):
        venue_rows = [
            row for row in records
            if str((row.get("group") or [{}])[0].get("venue") or "不明") == venue
        ]
        by_band = {}
        for distance_band in ("1200m以下", "1300〜1600m", "1700m以上", "距離不明"):
            subset = [row for row in venue_rows if band(row.get("group")) == distance_band]
            if not subset:
                continue
            baseline = evaluate(subset, include_bootstrap=False)
            model = evaluate(subset, fitted, include_bootstrap=False)
            by_band[distance_band] = {
                "baseline_ability_top4": baseline,
                "conditional_pair_ranker": model,
                "roi_delta_pp_model_minus_baseline": round(
                    model["roi_pct"] - baseline["roi_pct"], 2
                ) if model.get("roi_pct") is not None and baseline.get("roi_pct") is not None else None,
                "sample_warning": (
                    "100レース未満。傾向確認のみ"
                    if model["races"] < 100 else
                    "100レース以上。別期間での再現確認が必要"
                ),
            }
        output[venue] = by_band
    return {
        "by_venue_and_distance_band": output,
        "policy": "同一ホールドアウト内の距離帯別診断。距離帯の小標本から見送りルールを採用しない。",
    }


def main():
    print(
        f"PAIR_RESEARCH_ARCHIVE_WINDOW start={nb.START.isoformat()} end={nb.END.isoformat()}",
        flush=True,
    )
    races, horses, payouts, months = nb.load_archives()
    print(
        f"PAIR_RESEARCH_ARCHIVES races={len(races)} horse_races={len(horses)} "
        f"payout_races={len(payouts)} months={len(months)}",
        flush=True,
    )

    # The axis is captured from the fixed horse-evaluation win ranking using
    # only each race's prior history. The candidate pool below is LambdaRank.
    axis_by_key = collect_fixed_axes(races, horses, payouts)

    rows, all_dates = ml.build_rows(races, horses, include_unplaced=True)
    groups = ml.group_rows(rows)
    if len(all_dates) < 20:
        raise RuntimeError(f"Insufficient chronological dates: {len(all_dates)}")
    fit_end = max(1, int(len(all_dates) * 0.60))
    rank_dev_end = max(fit_end + 1, int(len(all_dates) * 0.70))
    pair_dev_end = max(rank_dev_end + 1, int(len(all_dates) * 0.85))
    if pair_dev_end >= len(all_dates):
        raise RuntimeError("Chronological fit/development/holdout split is too small")

    fit_dates = {day.isoformat() for day in all_dates[:fit_end]}
    rank_dev_dates = {day.isoformat() for day in all_dates[fit_end:rank_dev_end]}
    pair_dev_dates = {day.isoformat() for day in all_dates[rank_dev_end:pair_dev_end]}
    holdout_dates = {day.isoformat() for day in all_dates[pair_dev_end:]}
    fit_groups = [g for g in groups if g[0]["date"] in fit_dates]
    rank_dev_groups = [g for g in groups if g[0]["date"] in rank_dev_dates]
    fit_rows = [row for row in rows if row["date"] in fit_dates]
    pair_dev_groups = [g for g in groups if g[0]["date"] in pair_dev_dates]
    holdout_groups = [g for g in groups if g[0]["date"] in holdout_dates]
    if not fit_groups or not rank_dev_groups or not pair_dev_groups or not holdout_groups:
        raise RuntimeError("One or more chronological data partitions are empty")

    category_maps = {
        key: {value: index for index, value in enumerate(sorted({row[key] for row in fit_rows}))}
        for key in ml.CATEGORICAL
    }
    print(
        f"PAIR_RESEARCH_SPLIT dates={len(all_dates)} fit={len(fit_groups)} "
        f"rank_dev={len(rank_dev_groups)} pair_dev={len(pair_dev_groups)} "
        f"holdout={len(holdout_groups)}",
        flush=True,
    )
    ranker = ml.fit_lambdarank(
        fit_groups, rank_dev_groups, ml.RAW_FEATURES, category_maps
    )
    ranked_records = make_ranked_records(
        rows, pair_dev_groups + holdout_groups, axis_by_key, payouts, ranker, category_maps
    )
    development = [row for row in ranked_records if row["date"] in pair_dev_dates]
    holdout = [row for row in ranked_records if row["date"] in holdout_dates]
    if not development or not holdout:
        raise RuntimeError(
            f"No matched axis/LambdaRank records: development={len(development)} holdout={len(holdout)}"
        )

    x_rows, labels, diagnostics = build_axis_pair_training_data(development)
    fitted = fit_axis_pair_model(development)
    baseline_dev = evaluate(development)
    baseline_holdout = evaluate(holdout)
    pair_holdout = evaluate(holdout, fitted)
    no_distance_fitted = fit_axis_pair_model(development, "no_distance")
    no_distance_holdout = evaluate(holdout, no_distance_fitted)
    venue_distance_fitted = fit_axis_pair_model(development, "venue_distance")
    venue_distance_holdout = evaluate(holdout, venue_distance_fitted)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "research_only_evaluated",
        "data_source": "NAR official monthly archives; canonical chronological axis replay plus existing pre-race LambdaRank feature pipeline",
        "period": {"start": all_dates[0].isoformat(), "end": all_dates[-1].isoformat()},
        "partitions": {
            "fit_end": all_dates[fit_end - 1].isoformat(),
            "rank_dev_start": all_dates[fit_end].isoformat(),
            "rank_dev_end": all_dates[rank_dev_end - 1].isoformat(),
            "pair_development_start": all_dates[rank_dev_end].isoformat(),
            "pair_development_end": all_dates[pair_dev_end - 1].isoformat(),
            "holdout_start": all_dates[pair_dev_end].isoformat(),
            "holdout_end": all_dates[-1].isoformat(),
            "fit_dates": len(fit_dates),
            "rank_dev_dates": len(rank_dev_dates),
            "pair_development_dates": len(pair_dev_dates),
            "holdout_dates": len(holdout_dates),
        },
        "runner_data_quality": {
            "runner_rows": len(rows),
            "started_without_rank_rows": sum(1 for row in rows if row.get("finish") is None),
            "started_without_rank_status_counts": {
                status: sum(1 for row in rows if row.get("finish") is None and row.get("finish_raw") == status)
                for status in sorted({row.get("finish_raw") for row in rows if row.get("finish") is None})
            },
            "policy": "Only explicit starter statuses containing 失格/中止/落馬 are retained without numeric finish; pre-race 取消/除外 and blank statuses remain excluded.",
        },
        "development": {
            "matched_races": len(development),
            "axis_pair_training": diagnostics,
            "training_pair_rows": len(labels),
            "baseline_ability_top4": baseline_dev,
            "note": "Development period fits the conditional pair model; development performance is descriptive only.",
        },
        "holdout": {
            "matched_races": len(holdout),
            "baseline_ability_top4": baseline_holdout,
            "conditional_pair_ranker": pair_holdout,
            "paired_difference": paired_comparison(holdout, fitted),
            "feature_ablation": {
                "A_full_distance": {"feature_mode": "full", "metrics": pair_holdout, "paired_vs_baseline": paired_comparison(holdout, fitted)},
                "B_no_distance": {"feature_mode": "no_distance", "metrics": no_distance_holdout, "paired_vs_baseline": paired_comparison(holdout, no_distance_fitted)},
                "C_venue_distance": {"feature_mode": "venue_distance", "metrics": venue_distance_holdout, "paired_vs_baseline": paired_comparison(holdout, venue_distance_fitted)},
                "paired_comparisons": {
                    "A_full_distance_vs_B_no_distance": paired_model_comparison(holdout, fitted, no_distance_fitted),
                    "A_full_distance_vs_C_venue_distance": paired_model_comparison(holdout, fitted, venue_distance_fitted),
                    "B_no_distance_vs_C_venue_distance": paired_model_comparison(holdout, no_distance_fitted, venue_distance_fitted),
                },
                "changed_ticket_diagnostics": {
                    "A_vs_B": changed_ticket_diagnostics(holdout, fitted, no_distance_fitted, "A距離適性あり vs B距離適性なし"),
                    "A_vs_C": changed_ticket_diagnostics(holdout, fitted, venue_distance_fitted, "A通常距離適性 vs C競馬場内距離適性"),
                },
                "venue_distance_ablation": venue_distance_ablation(holdout, fitted, no_distance_fitted, venue_distance_fitted),
                "same_development_and_holdout_races": True,
                "note": "All three variants use identical chronological partitions, fixed axes, LambdaRank candidate pools, six virtual trio tickets per race and the same holdout. C replaces general distance suitability with same-venue and within-200m historical suitability. No holdout tuning.",
            },
            "same_races_and_budget": True,
            "venue_breakdown_baseline": venue_holdout_breakdown(holdout),
            "venue_breakdown_conditional_pair_ranker": venue_holdout_breakdown(holdout, fitted),
            "venue_distance_breakdown_conditional_pair_ranker": venue_distance_breakdown(holdout, fitted),
        },
        "model": {
            "axis_source": "fixed pre-off win ranking from horse_evaluation_model via canonical chronological replay",
            "opponent_pool_source": "LightGBM LambdaRank trained on first 60% of dates; up to six highest-ranked non-axis candidates",
            "candidate_pool_size_distribution_holdout": {
                str(size): sum(
                    1 for record in holdout
                    if len([no for no in record["candidate_horse_nos"] if str(no) != str(record["axis_horse_no"])]) == size
                )
                for size in (4, 5, 6)
            },
            "pair_universe_by_candidate_count": {str(size): size * (size - 1) // 2 for size in (4, 5, 6)},
            "selected_pairs": 6,
            "training_rule": "only pair-development races where the fixed axis finished top three; finishes used only as labels",
            "features": fitted["features"],
            "hyperparameters": {
                "pair_model": {"C": 0.5, "max_iter": 600, "class_weight": "balanced", "random_state": 42},
                "lambdarank": "existing nar_ml_challenger_backtest.fit_lambdarank defaults; early stopping on separate rank-development dates",
            },
        },
        "policy": "No holdout tuning, no final-odds features, no live integration, no deployment. Chronological holdout is untouched.",
        "archive_months": months,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "development": diagnostics,
        "holdout_baseline": baseline_holdout,
        "holdout_pair_ranker": pair_holdout,
        "report_path": str(OUT),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
