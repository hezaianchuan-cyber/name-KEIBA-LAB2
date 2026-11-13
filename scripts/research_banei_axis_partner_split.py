#!/usr/bin/env python3
"""Research-only Banei split-role model: winner-like axis + top-three partners.

Uses stable horse history keys (normalized name + birth date when available).
No current-race body weight/change, post-off data, odds, or flat-racing rows.
"""
from __future__ import annotations
import json, math, random
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import nar_backtest as nb
from research_banei_historical_body_weight_profile import is_banei, safe_mean, safe_sd, paired_bootstrap
from research_banei_winner_profile_similarity import ranking_metrics, fit_winner_similarity

OUT = nb.ROOT / "data" / "banei-axis-partner-split.json"
SEED = 20261113
BOOTSTRAPS = 2000
# Axis uses the full winner-profile vector. Partner model deliberately uses
# performance/age + prior body-weight level only; it excludes noisy body
# weight trend/variation and current carried-weight-relative-to-field.
PARTNER_INDICES = list(range(0, 7))
PARTNER_INDICES_EXTENDED = list(range(0, 7)) + [11, 12]


def identity_key(horse):
    name = nb.norm(horse.get("name"))
    birth = "".join(ch for ch in str(horse.get("birth_date") or "") if ch.isdigit())
    return (name, birth) if len(birth) == 8 else (name, "birthdate_missing")


def build_rows_stable(races, horses):
    history = defaultdict(list)
    records = []
    name_births = defaultdict(set)
    missing_birth_rows = 0
    ordered = sorted(races.items(), key=lambda item: (item[1]["date"], item[0]))
    for key, race in ordered:
        if not is_banei(race.get("venue")):
            continue
        field = horses.get(key, [])
        if len(field) < 5:
            continue
        for h in field:
            name = nb.norm(h.get("name"))
            birth = "".join(ch for ch in str(h.get("birth_date") or "") if ch.isdigit())
            if name and len(birth) == 8:
                name_births[name].add(birth)
            elif name:
                missing_birth_rows += 1

        carried = [nb.number(h.get("carried")) for h in field]
        valid = [x for x in carried if x is not None]
        if len(valid) >= max(3, math.ceil(len(field) * 0.70)):
            mean_c, sd_c = safe_mean(valid, 0.0), safe_sd(valid, 1.0)
            if sd_c < 1e-9:
                sd_c = 1.0
        else:
            mean_c, sd_c = 0.0, 1.0

        # Featurize every runner before updating any history from this race.
        pending = []
        for horse, current_carried in zip(field, carried):
            hkey = identity_key(horse)
            past = history[hkey]
            finishes = [nb.number(x.get("finish")) for x in past]
            finishes = [x for x in finishes if x is not None and x > 0]
            recent = finishes[-5:]
            last_finish = finishes[-1] if finishes else None
            last_field = nb.number(past[-1].get("field_size")) if past else None
            age = nb.number(horse.get("age"))
            past_body = [nb.number(x.get("body_weight")) for x in past]
            past_body = [x for x in past_body if x is not None and x > 0]
            past_changes = [nb.number(x.get("body_weight_change")) for x in past]
            past_changes = [x for x in past_changes if x is not None]
            past_carried = [nb.number(x.get("carried")) for x in past]
            past_carried = [x for x in past_carried if x is not None]
            ratios = [
                100.0 * nb.number(x.get("carried")) / nb.number(x.get("body_weight"))
                for x in past
                if nb.number(x.get("carried")) is not None
                and nb.number(x.get("body_weight")) is not None
                and nb.number(x.get("body_weight")) > 0
            ]
            base = [
                math.log1p(len(finishes)),
                (sum(x <= 3 for x in finishes) + 1.5) / (len(finishes) + 5.0),
                (sum(x <= 3 for x in recent) + 1.0) / (len(recent) + 3.0),
                (1.0 - (last_finish - 1.0) / (last_field - 1.0))
                if last_finish is not None and last_field and last_field > 1 else 0.5,
                age if age is not None else 5.0,
            ]
            body = [
                past_body[-1] if past_body else None,
                safe_mean(past_body[-3:]),
                (past_body[-1] - past_body[-2]) if len(past_body) >= 2 else None,
                safe_sd(past_body[-5:]) if len(past_body) >= 2 else None,
                past_changes[-1] if past_changes else None,
                safe_mean(past_changes[-3:]),
                safe_mean(ratios[-3:]),
                safe_mean(past_carried[-3:]),
                len(past_body),
            ]
            carried_z = ((current_carried - mean_c) / sd_c
                         if current_carried is not None and len(valid) >= 3 else None)
            finish = nb.number(horse.get("finish"))
            records.append({
                "date": race["date"].isoformat(),
                "race_key": "|".join(str(x) for x in key),
                "horse_no": str(horse.get("no")),
                "horse_identity": hkey,
                "base": base, "body_profile": body, "carried_z": carried_z,
                "finish": finish,
                "label_top3": int(finish is not None and 0 < finish <= 3),
                "label_nonwinner_podium": int(finish is not None and finish in (2, 3)),
                "label_win": int(finish is not None and finish == 1),
                "field_size": len(field),
            })
            pending.append((hkey, horse))
        for hkey, horse in pending:
            history[hkey].append({
                "finish": horse.get("finish"), "field_size": len(field),
                "body_weight": horse.get("body_weight"),
                "body_weight_change": horse.get("body_weight_change"),
                "carried": horse.get("carried"),
            })
    collisions = {name: sorted(births) for name, births in name_births.items() if len(births) > 1}
    audit = {
        "normalized_names_with_multiple_birth_dates": len(collisions),
        "examples_first_20": dict(list(sorted(collisions.items()))[:20]),
        "runner_rows_missing_valid_birth_date": missing_birth_rows,
        "history_key": "normalized horse name + 8-digit birth date; name-only fallback when birth date unavailable",
    }
    return records, audit


def partner_probabilities(train, test, target_key="label_top3", model_kind="logistic", feature_indices=None):
    indices = feature_indices or PARTNER_INDICES
    def vec(row):
        all_features = row["base"] + row["body_profile"] + [row["carried_z"]]
        return [all_features[i] for i in indices]
    x_train = np.asarray([vec(r) for r in train], dtype=float)
    x_test = np.asarray([vec(r) for r in test], dtype=float)
    y = np.asarray([r[target_key] for r in train], dtype=int)
    if model_kind == "hist_gradient_boosting":
        model = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            HistGradientBoostingClassifier(
                learning_rate=0.05, max_iter=160, max_leaf_nodes=15,
                min_samples_leaf=30, l2_regularization=2.0,
                early_stopping=True, random_state=SEED,
            ),
        )
    else:
        model = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            LogisticRegression(C=0.5, max_iter=1500, random_state=SEED),
        )
    model.fit(x_train, y)
    return model.predict_proba(x_test)[:, 1]


def blend_partner_scores(records, first, second):
    """Blend partner models after converting scores to within-race percentiles."""
    grouped = defaultdict(list)
    for i, row in enumerate(records):
        grouped[row["race_key"]].append(i)
    blended = np.zeros(len(records), dtype=float)
    for indices in grouped.values():
        for scores in (first, second):
            order = sorted(indices, key=lambda i: (float(scores[i]), -int(records[i]["horse_no"])))
            ranks = ({order[0]: 1.0} if len(order) == 1 else
                     {idx: rank / (len(order) - 1) for rank, idx in enumerate(order)})
            for idx in indices:
                blended[idx] += ranks[idx] / 2.0
    return blended


def split_role_rank(records, axis_scores, partner_scores):
    grouped = defaultdict(list)
    for row, axis, partner in zip(records, axis_scores, partner_scores):
        grouped[row["race_key"]].append((float(axis), float(partner), row))
    per_race = {}
    axis_rows, partner_rows = [], []
    for key, entries in grouped.items():
        podium = {r["horse_no"] for _, _, r in entries if r["label_top3"]}
        if len(podium) < 3:
            continue
        # The axis is chosen only by winner-profile similarity.
        axis_entry = max(entries, key=lambda x: (x[0], -int(x[2]["horse_no"])))
        others = [x for x in entries if x is not axis_entry]
        others.sort(key=lambda x: (-x[1], int(x[2]["horse_no"])))
        selected = [axis_entry] + others[:2]
        winner = next((r["horse_no"] for _, _, r in entries if r["label_win"]), None)
        axis_no = axis_entry[2]["horse_no"]
        selected_nos = {x[2]["horse_no"] for x in selected}
        per_race[key] = {
            "axis_top3": float(axis_no in podium),
            "axis_win": float(axis_no == winner),
            "top3_coverage": len(selected_nos & podium) / 3.0,
            "full_top3": float(podium.issubset(selected_nos)),
            "winner_in_top3": float(winner in selected_nos) if winner else 0.0,
        }
        axis_rows.append((axis_no, podium, winner))
        partner_rows.append((selected[1][2]["horse_no"], selected[2][2]["horse_no"], podium))
    names = ("axis_top3", "axis_win", "top3_coverage", "full_top3", "winner_in_top3")
    summary = {"races": len(per_race)}
    for name in names:
        summary[name + "_pct"] = round(100 * np.mean([x[name] for x in per_race.values()]), 3) if per_race else None
    partner_hits = sum((a in p) + (b in p) for a, b, p in partner_rows)
    summary["partner_slots_top3_pct"] = round(100 * partner_hits / max(1, 2 * len(partner_rows)), 3)
    return per_race, summary


def main():
    races, horses, _payouts, _metadata = nb.load_archives()
    rows, identity_audit = build_rows_stable(races, horses)
    usable = [r for r in rows if r["finish"] is not None and r["field_size"] >= 5]
    dates = sorted({r["date"] for r in usable})
    if len(dates) < 100:
        raise RuntimeError(f"ばんえい開催日数不足: {len(dates)}")
    train_cut = dates[int(len(dates) * 0.70) - 1]
    dev_cut = dates[int(len(dates) * 0.85) - 1]
    train = [r for r in usable if r["date"] <= train_cut]
    holdout_raw = [r for r in usable if r["date"] > dev_cut]
    podium_counts = defaultdict(int)
    for r in holdout_raw:
        podium_counts[r["race_key"]] += r["label_top3"]
    holdout_keys = {k for k, n in podium_counts.items() if n >= 3}
    holdout = [r for r in holdout_raw if r["race_key"] in holdout_keys]
    if not train or not holdout:
        raise RuntimeError("学習または検証データが空です")

    axis_scores, prototype = fit_winner_similarity(train, holdout)
    partner_top3_scores = partner_probabilities(train, holdout, "label_top3")
    split_top3_races, split_top3_summary = split_role_rank(holdout, axis_scores, partner_top3_scores)

    # Specialist partner model learns only second/third-place finishers.
    partner_23_scores = partner_probabilities(train, holdout, "label_nonwinner_podium")
    split_23_races, split_23_summary = split_role_rank(holdout, axis_scores, partner_23_scores)

    # Blend the partner rankings by within-race percentile, not raw score scale.
    partner_blend_scores = blend_partner_scores(holdout, partner_top3_scores, partner_23_scores)
    split_blend_races, split_blend_summary = split_role_rank(holdout, axis_scores, partner_blend_scores)

    # Non-linear partner rankers use the same 2nd/3rd-place target and no
    # current carried-weight-relative feature.
    partner_23_hgb = partner_probabilities(
        train, holdout, "label_nonwinner_podium", model_kind="hist_gradient_boosting"
    )
    split_23_hgb_races, split_23_hgb_summary = split_role_rank(holdout, axis_scores, partner_23_hgb)
    partner_23_hgb_ext = partner_probabilities(
        train, holdout, "label_nonwinner_podium", model_kind="hist_gradient_boosting",
        feature_indices=PARTNER_INDICES_EXTENDED
    )
    split_23_hgb_ext_races, split_23_hgb_ext_summary = split_role_rank(holdout, axis_scores, partner_23_hgb_ext)

    axis_races, axis_summary = ranking_metrics(holdout, axis_scores)
    top3_all_races, top3_all_summary = ranking_metrics(holdout, partner_top3_scores)
    podium23_all_races, podium23_all_summary = ranking_metrics(holdout, partner_23_scores)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "research_only_banei_axis_partner_split",
        "scope": "ばんえい・帯広のみ。平地データ混在なし。本番変更なし。",
        "sample": {
            "train_races": len({r["race_key"] for r in train}),
            "train_runner_rows": len(train),
            "holdout_races": len(holdout_keys),
            "holdout_runner_rows": len(holdout),
            "train_end": train_cut,
            "development_gap_end": dev_cut,
            "holdout_start": min(r["date"] for r in holdout),
            "holdout_end": max(r["date"] for r in holdout),
        },
        "identity_audit": identity_audit,
        "models": {
            "winner_similarity_axis_only_rank": axis_summary,
            "top3_partner_model_all_slots": top3_all_summary,
            "second_third_partner_model_all_slots": podium23_all_summary,
            "split_axis_plus_top3_partners": split_top3_summary,
            "split_axis_plus_second_third_partners": split_23_summary,
            "split_axis_plus_blended_partners": split_blend_summary,
            "split_axis_plus_second_third_hist_gradient_boosting": split_23_hgb_summary,
            "split_axis_plus_second_third_hist_gradient_boosting_extended_features": split_23_hgb_ext_summary,
        },
        "paired_comparisons": {
            "split_top3_vs_winner_similarity": paired_bootstrap(axis_races, split_top3_races),
            "split_23_vs_winner_similarity": paired_bootstrap(axis_races, split_23_races),
            "split_blend_vs_winner_similarity": paired_bootstrap(axis_races, split_blend_races),
            "split_23_vs_split_top3": paired_bootstrap(split_top3_races, split_23_races),
            "split_blend_vs_split_top3": paired_bootstrap(split_top3_races, split_blend_races),
            "split_23_hgb_vs_logistic": paired_bootstrap(split_23_races, split_23_hgb_races),
            "split_23_hgb_extended_vs_logistic": paired_bootstrap(split_23_races, split_23_hgb_ext_races),
            "split_23_hgb_extended_vs_hgb": paired_bootstrap(split_23_hgb_races, split_23_hgb_ext_races),
            "split_23_vs_top3_partner_all_slots": paired_bootstrap(top3_all_races, split_23_races),
        },
        "method": {
            "axis": "勝ち馬特徴照合モデル（過去成績・年齢・過去馬体重・斤量相対値）",
            "partners": "3着内分類モデルと2・3着専用分類モデルを比較し、順位化して混合する方式も検証",
            "partner_feature_indices": PARTNER_INDICES,
            "partner_extended_feature_indices": PARTNER_INDICES_EXTENDED,
            "partner_model_algorithms": ["logistic_regression", "hist_gradient_boosting"],
            "current_race_body_weight_and_change_used": False,
            "current_carried_weight_relative_to_field_used_for_partners": False,
            "same_race_history_update_delayed_until_all_features_built": True,
        },
        "guardrails": [
            "研究用。実運用・画面・買い目生成には反映しない。",
            "勝ち馬軸と相手2頭を別々に選ぶ方式を同一検証レースで評価。",
            "回収率は固定した買い目と実際の発売時オッズで別途検証する。",
            "時系列の最終15%を検証用に保持し、途中15%は学習から除外。",
            "馬体重は過去走の記録のみ。今回の馬体重・増減は使わない。",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("BANEI_AXIS_PARTNER_SPLIT " + json.dumps({
        "status": report["status"], "sample": report["sample"],
        "models": report["models"], "identity_audit": identity_audit,
        "paired_comparisons": report["paired_comparisons"], "report": str(OUT)
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
