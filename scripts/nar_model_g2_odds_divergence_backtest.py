#!/usr/bin/env python3
"""Leakage-safe G2 odds-divergence retrospective test.

G2 keeps the validated G1 temperature fixed and compares its win probabilities
with normalized official final win-odds probabilities. The pre-registered
selection rule is model/market probability ratio >= 1.20 and model probability
times final odds >= 1.08. At most one JPY100 win bet is made per race.
Final odds are retrospective only; this is not a live pre-off result.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import random
from pathlib import Path

import nar_backtest as nb

ROOT = Path(__file__).resolve().parents[1]
G1_REPORT = ROOT / "data" / "model-g1-calibration-backtest.json"
OUT = ROOT / "data" / "model-g2-odds-divergence-backtest.json"
START_DATE = dt.date(2026, 3, 1)
MIN_ODDS_COVERAGE = 0.70
MIN_DIVERGENCE = 1.20
MIN_EXPECTED_VALUE = 1.08
DIVERGENCE_GRID = (1.5, 1.75, 2.0, 2.5, 3.0)
EXPECTED_VALUE_GRID = (1.15, 1.25, 1.4, 1.6)
MIN_DEV_BETS = 100
MIN_SELECTION_RATE = 0.05
MAX_SELECTION_RATE = 0.40
UNIT_STAKE = 100
BOOTSTRAP_REPS = 2000


def interval(values):
    values = sorted(values)
    if not values:
        return None
    return [round(values[int(0.025 * (len(values)-1))], 6),
            round(values[int(0.975 * (len(values)-1))], 6)]


def race_bootstrap_roi(rows):
    if len(rows) < 30:
        return {"method": "paired race-level bootstrap", "replicates": 0,
                "roi_delta_95ci": None}
    rng = random.Random(20261104 + len(rows))
    deltas = []
    n = len(rows)
    for _ in range(BOOTSTRAP_REPS):
        sample = [rows[rng.randrange(n)] for _ in range(n)]
        stake = sum(x["stake"] for x in sample)
        if stake:
            deltas.append(100 * sum(x["return"] for x in sample) / stake)
    return {"method": "race-level bootstrap ROI percentile interval",
            "replicates": BOOTSTRAP_REPS, "roi_95ci": interval(deltas)}


def summarize_bets(rows):
    stake = sum(x["stake"] for x in rows)
    returned = sum(x["return"] for x in rows)
    hits = sum(x["hit"] for x in rows)
    return {
        "races_bet": len(rows), "tickets": len(rows), "hits": hits,
        "hit_rate": round(100 * hits / len(rows), 2) if rows else None,
        "stake_jpy": stake, "return_jpy": returned,
        "profit_jpy": returned - stake,
        "roi": round(100 * returned / stake, 2) if stake else None,
        "roi_95ci": race_bootstrap_roi(rows),
    }


def score_records(records, odds_by_race, payouts, temperature, min_divergence=MIN_DIVERGENCE, min_expected_value=MIN_EXPECTED_VALUE, min_odds_coverage=MIN_ODDS_COVERAGE):
    evaluated = []
    for record in records:
        key_parts = str(record.get("race_key") or "").split("|")
        if len(key_parts) != 3:
            continue
        key = nb.race_key(*key_parts)
        if not key:
            continue
        odds = odds_by_race.get(key, {})
        scores = record["scores"]
        if len(scores) < 3:
            continue
        odds_by_no = {str(int(k)): float(v) for k, v in odds.items()
                      if str(k).isdigit() and float(v) > 0}
        if not odds_by_no:
            continue
        # score order is the official horse-list order; the monthly archive odds
        # are keyed by horse number, so the G1 run record must preserve horse order.
        horse_numbers = record.get("horse_numbers") or []
        if len(horse_numbers) != len(scores):
            continue
        field = [str(int(no)) for no in horse_numbers]
        matched = [no for no in field if no in odds_by_no]
        coverage = len(matched) / len(field) if field else 0.0
        if coverage < min_odds_coverage:
            continue
        model = nb.softmax_probabilities(scores, temperature)
        raw_market = {no: 1.0 / odds_by_no[no] for no in matched}
        market_total = sum(raw_market.values())
        if market_total <= 0:
            continue
        market = {no: value / market_total for no, value in raw_market.items()}
        model_by_no = {no: model[i] for i, no in enumerate(field)}
        candidates = []
        for no in matched:
            p = model_by_no[no]
            mp = market[no]
            price = odds_by_no[no]
            divergence = p / mp if mp > 0 else 0.0
            ev = p * price
            if divergence >= min_divergence and ev >= min_expected_value:
                candidates.append((ev, divergence, p, price, no))
        candidates.sort(reverse=True)
        payout = payouts.get(key)
        winners = set()
        for nums, _ in (payout or {}).get("win", []):
            if nums:
                winners.add(str(nums[0]))
        if not winners:
            continue
        best_model = max(field, key=lambda no: model_by_no[no])
        favorite = min(matched, key=lambda no: (odds_by_no[no], int(no)))
        model_winner_mass = sum(model_by_no[no] for no in winners if no in model_by_no)
        market_winner_mass = sum(market.get(no, 0.0) for no in winners)
        selected = candidates[0] if candidates else None
        def win_row(no, label):
            ret = nb.payout_for("win", (no,), payout) or 0
            return {"label": label, "stake": UNIT_STAKE, "return": ret,
                    "hit": int(no in winners)}
        g1_top = win_row(best_model, "G1_top_pick")
        market_top = win_row(favorite, "market_favorite")
        g2_row = None
        if selected:
            ev, divergence, p, price, no = selected
            g2_row = win_row(no, "G2_value_pick")
            g2_row.update({"horse_no": no, "model_probability": round(p, 8),
                           "market_probability": round(market.get(no, 0.0), 8),
                           "final_odds": round(price, 4),
                           "probability_ratio": round(divergence, 4),
                           "expected_value": round(ev, 4)})
        evaluated.append({
            "date": record["date"], "race_key": "|".join(key),
            "coverage": coverage, "winner_mass_g1": model_winner_mass,
            "winner_mass_market": market_winner_mass,
            "g1_top": g1_top, "market_top": market_top, "g2": g2_row,
        })
    return evaluated


def probability_metrics(rows, key):
    valid = [r[key] for r in rows if r.get(key) is not None]
    if not valid:
        return {"races": 0, "log_loss": None, "brier_winner_mass": None}
    # Winner-mass log loss is directly comparable for dead heats. Brier is
    # reported as a scalar squared error on winner mass for market vs model.
    log_loss = -sum(math.log(max(1e-12, x)) for x in valid) / len(valid)
    brier = sum((x - 1.0) ** 2 for x in valid) / len(valid)
    return {"races": len(valid), "log_loss": round(log_loss, 6),
            "brier_winner_mass": round(brier, 6)}


def main():
    if not G1_REPORT.exists():
        raise RuntimeError("G1 report is missing; G2 cannot run without the validated G1 baseline.")
    g1 = json.loads(G1_REPORT.read_text(encoding="utf-8"))
    if not g1.get("g1_pass") or g1.get("status") != "G1_improved_on_untouched_holdout":
        raise RuntimeError("G1 did not pass; refusing to promote its temperature into G2.")
    temperature = float(g1["selected_temperature"])
    # NAR monthly archives for the current month may not be published yet.
    # Restrict retrospective tests to the last fully completed calendar month.
    nb.END = dt.date.today().replace(day=1) - dt.timedelta(days=1)
    races, horses, payouts, archive_months = nb.load_archives()
    _, _, _, _, _, records = nb.run_backtest(races, horses, payouts)
    records = [r for r in records if r.get("date") and
               dt.date.fromisoformat(r["date"]) >= START_DATE]
    # Attach horse numbers in exactly the same order as scores/winner indices.
    race_by_key = { "|".join(k): race for k, race in races.items() }
    aligned = []
    for record in records:
        race = race_by_key.get(record.get("race_key"))
        field = horses.get(race["key"], []) if race else []
        if len(field) != len(record.get("scores", [])):
            continue
        aligned.append({**record, "horse_numbers": [h["no"] for h in field]})
    odds_by_race, odds_months = nb.load_final_odds_archives(START_DATE, nb.END)
    all_dates = sorted({r["date"] for r in aligned})
    if len(all_dates) < 10:
        raise RuntimeError("Insufficient dates for chronological G2 evaluation.")
    split = max(1, min(len(all_dates)-1, int(len(all_dates)*0.70)))
    dev_end = all_dates[split-1]
    holdout_start = all_dates[split]

    # Tune thresholds on development data only; do not inspect the holdout.
    candidates = []
    for divergence in DIVERGENCE_GRID:
        for expected_value in EXPECTED_VALUE_GRID:
            trial = score_records(aligned, odds_by_race, payouts, temperature,
                                  divergence, expected_value, MIN_ODDS_COVERAGE)
            dev_trial = [r["g2"] for r in trial if r["date"] <= dev_end and r.get("g2")]
            dev_races = sum(1 for r in trial if r["date"] <= dev_end)
            selection_rate = len(dev_trial) / dev_races if dev_races else 0.0
            summary = summarize_bets(dev_trial)
            ci = (summary.get("roi_95ci") or {}).get("roi_95ci")
            eligible = (len(dev_trial) >= MIN_DEV_BETS
                        and MIN_SELECTION_RATE <= selection_rate <= MAX_SELECTION_RATE
                        and ci is not None and ci[0] > 100.0)
            candidates.append({
                "min_probability_ratio": divergence,
                "min_model_probability_times_final_odds": expected_value,
                "development_bets": len(dev_trial),
                "development_selection_rate_pct": round(100 * selection_rate, 2),
                "development_roi": summary["roi"],
                "development_roi_95ci": ci,
                "eligible": eligible,
            })
    eligible_candidates = [c for c in candidates if c["eligible"]]
    if eligible_candidates:
        chosen = max(eligible_candidates,
                     key=lambda c: (c["development_roi_95ci"][0],
                                    c["development_roi"] or -1,
                                    -c["development_bets"]))
        selected_divergence = chosen["min_probability_ratio"]
        selected_ev = chosen["min_model_probability_times_final_odds"]
        tuning_status = "selected_on_development_only"
    else:
        chosen = None
        selected_divergence = None
        selected_ev = None
        tuning_status = "no_eligible_development_threshold_skip_all"
    data_ready_dates = []
    alignment = {"aligned_score_records": len(aligned), "exact_odds_race_key_matches": 0,
                 "aligned_races_with_win_payout": 0, "records_with_odds_and_win_payout": 0,
                 "records_with_70pct_odds_coverage": 0}
    for record in aligned:
        parts = str(record.get("race_key") or "").split("|")
        if len(parts) != 3: continue
        key = nb.race_key(*parts)
        if not key: continue
        odds = odds_by_race.get(key, {})
        payout = payouts.get(key) or {}
        if odds: alignment["exact_odds_race_key_matches"] += 1
        if payout.get("win"): alignment["aligned_races_with_win_payout"] += 1
        if odds and payout.get("win"):
            alignment["records_with_odds_and_win_payout"] += 1
            nums = [str(int(x)) for x in (record.get("horse_numbers") or []) if str(x).isdigit()]
            odds_nums = {str(int(x)) for x in odds if str(x).isdigit()}
            coverage = len(set(nums) & odds_nums) / len(nums) if nums else 0.0
            if coverage >= MIN_ODDS_COVERAGE:
                alignment["records_with_70pct_odds_coverage"] += 1
                data_ready_dates.append(record["date"])
    # Always retain aligned G1/market benchmark rows, even when no G2/G3
    # threshold qualifies. Infinite thresholds suppress value selections only.
    evaluated = score_records(
        aligned, odds_by_race, payouts, temperature,
        selected_divergence if chosen else math.inf,
        selected_ev if chosen else math.inf,
        MIN_ODDS_COVERAGE,
    )
    development = [r for r in evaluated if r["date"] <= dev_end]
    holdout = [r for r in evaluated if r["date"] >= holdout_start]
    def flatten(rows, field):
        return [r[field] for r in rows if r.get(field)]
    dev_g2 = flatten(development, "g2")
    hold_g2 = flatten(holdout, "g2")
    dev_g1 = flatten(development, "g1_top")
    hold_g1 = flatten(holdout, "g1_top")
    dev_market = flatten(development, "market_top")
    hold_market = flatten(holdout, "market_top")
    report = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "feature": "G2_improved_skip_decision_development_tuned",
        "status": ("evaluated" if chosen and len(holdout) >= 100 else
                   "no_eligible_development_strategy" if not chosen else "insufficient_holdout_sample"),
        "g1_temperature": temperature, "g1_report_status": g1.get("status"),
        "evaluation_period": [START_DATE.isoformat(), nb.END.isoformat()],
        "races_with_g1_scores": len(aligned),
        "data_alignment_diagnostics": alignment,
        "races_with_final_odds_and_outcomes": alignment["records_with_odds_and_win_payout"],
        "races_with_70pct_odds_coverage_and_outcomes": alignment["records_with_70pct_odds_coverage"],
        "races_with_benchmark_metrics": len(evaluated),
        "races_with_selected_value_ticket": sum(1 for r in evaluated if r.get("g2")),
        "development_period": [all_dates[0], dev_end],
        "holdout_period": [holdout_start, all_dates[-1]],
        "development_races_with_odds": sum(1 for d in data_ready_dates if d <= dev_end),
        "holdout_races_with_odds": sum(1 for d in data_ready_dates if d >= holdout_start),
        "official_final_odds_months": odds_months,
        "rule": {
            "minimum_odds_coverage": MIN_ODDS_COVERAGE,
            "model_to_market_probability_ratio_minimum": selected_divergence,
            "model_probability_times_final_odds_minimum": selected_ev,
            "selection": "At most one highest-EV qualifying horse per race; JPY100 win bet; otherwise no bet.",
            "threshold_selection": "Development-only grid search; untouched holdout excluded from threshold selection.",
            "tuning_status": tuning_status,
            "development_candidate_grid": candidates,
            "selected_candidate": chosen,
        },
        "odds_note": "Official confirmed final win odds are used retrospectively. This does not represent odds available 2-30 minutes before post time and must not be presented as live profitability.",
        "e_model_changed": False, "live_model_changed": False,
        "profitable_claim": False,
        "holdout_probability_quality": {
            "G1_model_winner_mass": probability_metrics(holdout, "winner_mass_g1"),
            "market_winner_mass": probability_metrics(holdout, "winner_mass_market"),
        },
        "development_bets": {
            "G1_top_pick": summarize_bets(dev_g1),
            "market_favorite": summarize_bets(dev_market),
            "G2_value_pick": summarize_bets(dev_g2),
        },
        "holdout_bets": {
            "G1_top_pick": summarize_bets(hold_g1),
            "market_favorite": summarize_bets(hold_market),
            "G2_value_pick": summarize_bets(hold_g2),
        },
        "holdout_G2_selection_rate_pct": round(100 * len(hold_g2) / len(holdout), 2) if holdout else None,
        "g2_pass": bool(len(holdout) >= 100 and len(hold_g2) >= 100
                        and summarize_bets(hold_g2)["roi_95ci"]["roi_95ci"]
                        and summarize_bets(hold_g2)["roi_95ci"]["roi_95ci"][0] > 100),
        "interpretation": "G2 is only considered ROI-confirmed if the untouched holdout has at least 100 eligible races and 100 G2 bets and the race-bootstrap 95% ROI lower bound exceeds 100%. The skip rule is tuned only on development data; the untouched holdout evaluates the selected thresholds once. A pass still requires at least 100 holdout bets and a race-bootstrap 95% ROI lower bound above 100%.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
