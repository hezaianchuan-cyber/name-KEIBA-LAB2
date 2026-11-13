#!/usr/bin/env python3
"""Research-only rank-6+ diagnostics and chronological online-learning comparison.

Flat NAR racing only. Banei/Obihiro excluded. Current-race body weight and
body-weight change are excluded. No production model or ticket logic is changed.
"""
from __future__ import annotations
import datetime as dt
import itertools
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import SGDClassifier

import nar_backtest as nb
import research_flat_axis_win_blend as axis_blend

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "flat-rank6-online-learning-research.json"
FEATURES = axis_blend.FEATURES
EXCLUDED = {"body_weight", "body_weight_change", "body_weight_delta_pct",
            "body_weight_delta_pct", "race_rel_body_weight_delta_pct"}
SIM_SHARE = 0.40
WIN_SHARE = 0.60
TICKET_COUNT = 6
TICKET_COST = 100
BOOTSTRAP = 3000
DIAGNOSTIC_FEATURES = [
    "prior_runs", "career_win_rate", "career_top3_rate", "recent_finish_pct",
    "recent_top3_rate", "recent_margin_quality", "recent_last3f_rank",
    "recent_time_trend", "recent_opponent_strength", "recent_first_corner_pct",
    "recent_last_corner_pct", "recent_position_gain", "venue_top3_rate",
    "distance_top3_rate", "going_top3_rate", "jockey_top3_rate",
    "sire_top3_rate", "carried_weight", "age", "field_size", "distance_m",
    "front_crowding_ratio", "race_rel_recent_finish_pct",
    "race_rel_recent_top3_rate", "race_rel_recent_last_corner_pct",
    "race_rel_recent_opponent_strength",
]

def matrix(rows, center=None, scale=None):
    x = np.asarray([[float(r.get(name) or 0.0) for name in FEATURES] for r in rows], dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if center is not None:
        x = np.clip((x - center) / scale, -6.0, 6.0)
    return x

def grouped(rows):
    buckets = defaultdict(list)
    for row in rows:
        buckets[row["race_key"]].append(row)
    return [sorted(g, key=lambda r: int(r["horse_no"]))
            for _, g in sorted(buckets.items())
            if len(g) >= 3 and sum(int(r["winner"]) for r in g) == 1]

def pct_rank(values):
    order = sorted(range(len(values)), key=lambda i: (-float(values[i]), i))
    out = np.zeros(len(values), dtype=float)
    denom = max(len(order) - 1, 1)
    for rank, idx in enumerate(order):
        out[idx] = 1.0 - rank / denom
    return out

def similarity_from_centers(rows, center, scale, win_center, lose_center):
    z = matrix(rows, center, scale)
    d_win = np.sqrt(np.mean((z - win_center) ** 2, axis=1))
    d_lose = np.sqrt(np.mean((z - lose_center) ** 2, axis=1))
    return d_lose - d_win

def init_online_state(train_rows):
    x = matrix(train_rows)
    center = np.median(x, axis=0)
    scale = np.std(x, axis=0)
    scale[scale < 1e-6] = 1.0
    z = np.clip((x - center) / scale, -6.0, 6.0)
    y = np.asarray([int(r["winner"]) for r in train_rows], dtype=int)
    if y.sum() < 100 or (y == 0).sum() < 100:
        raise RuntimeError("初期学習データが不足しています")
    clf = SGDClassifier(loss="log_loss", penalty="l2", alpha=0.0005,
                        learning_rate="optimal", random_state=42)
    weights = np.where(y == 1, (y == 0).sum() / max(y.sum(), 1), 1.0)
    clf.partial_fit(z, y, classes=np.asarray([0, 1]), sample_weight=weights)
    return {
        "center": center, "scale": scale, "clf": clf,
        "win_sum": z[y == 1].sum(axis=0), "win_n": int(y.sum()),
        "lose_sum": z[y == 0].sum(axis=0), "lose_n": int((y == 0).sum()),
        "updates": 0,
    }

def online_scores(rows, state):
    z = matrix(rows, state["center"], state["scale"])
    win_center = state["win_sum"] / max(state["win_n"], 1)
    lose_center = state["lose_sum"] / max(state["lose_n"], 1)
    sim = similarity_from_centers(rows, state["center"], state["scale"],
                                  win_center, lose_center)
    win_prob = state["clf"].predict_proba(z)[:, 1]
    return SIM_SHARE * pct_rank(sim) + WIN_SHARE * pct_rank(win_prob)

def update_online_state(day_rows, state):
    if not day_rows:
        return
    x = matrix(day_rows, state["center"], state["scale"])
    y = np.asarray([int(r["winner"]) for r in day_rows], dtype=int)
    if len(np.unique(y)) == 2:
        weights = np.where(y == 1, (y == 0).sum() / max((y == 1).sum(), 1), 1.0)
        state["clf"].partial_fit(x, y, sample_weight=weights)
    elif len(y):
        # partial_fit can update one-class batches after classes were initialized.
        state["clf"].partial_fit(x, y, sample_weight=np.ones(len(y)))
    state["win_sum"] += x[y == 1].sum(axis=0)
    state["win_n"] += int(y.sum())
    state["lose_sum"] += x[y == 0].sum(axis=0)
    state["lose_n"] += int((y == 0).sum())
    state["updates"] += 1

def race_payout(group, payouts):
    parts = str(group[0]["race_key"]).split("|")
    if len(parts) != 3:
        return None
    return payouts.get(nb.race_key(parts[0], parts[1], parts[2]))

def ticket_result(group, scores, payout):
    order = sorted(range(len(group)), key=lambda i: (-float(scores[i]), int(group[i]["horse_no"])))
    if len(order) < 7:
        return None
    axis = order[0]
    candidates = []
    for a, b in itertools.combinations(order[1:7], 2):
        pair_score = float(scores[a]) + float(scores[b])
        tie = tuple(sorted((int(group[a]["horse_no"]), int(group[b]["horse_no"]))))
        candidates.append((pair_score, tie, a, b))
    candidates.sort(key=lambda x: (-x[0], x[1]))
    tickets = [tuple(sorted((str(group[axis]["horse_no"]),
                             str(group[a]["horse_no"]),
                             str(group[b]["horse_no"])), key=int))
               for _, _, a, b in candidates[:TICKET_COUNT]]
    returned = sum(nb.payout_for("trio", list(t), payout) or 0 for t in tickets)
    finish = {str(r["horse_no"]): r.get("finish") for r in group}
    podium = {no for no, f in finish.items() if f is not None and int(f) in (1, 2, 3)}
    ranked = {str(group[i]["horse_no"]): rank + 1 for rank, i in enumerate(order)}
    axis_no = str(group[axis]["horse_no"])
    return {
        "race_key": group[0]["race_key"], "axis_win": int(finish.get(axis_no) == 1),
        "axis_top3": int(axis_no in podium), "top7_coverage": len(podium & set(ranked_no for ranked_no, rank in ranked.items() if rank <= 7)),
        "full_podium_top7": int(len(podium) == 3 and all(ranked.get(no, 99) <= 7 for no in podium)),
        "trio_hit": int(returned > 0), "stake": len(tickets) * TICKET_COST,
        "returned": returned, "podium_ranks": sorted(ranked.get(no, 99) for no in podium),
    }

def summarize(records):
    n = len(records)
    stake = sum(x["stake"] for x in records)
    returned = sum(x["returned"] for x in records)
    podium_n = sum(len(x["podium_ranks"]) for x in records)
    return {
        "races": n,
        "axis_win_rate_pct": round(100 * sum(x["axis_win"] for x in records) / n, 3) if n else None,
        "axis_top3_rate_pct": round(100 * sum(x["axis_top3"] for x in records) / n, 3) if n else None,
        "top7_podium_coverage_pct": round(100 * sum(x["top7_coverage"] for x in records) / podium_n, 3) if podium_n else None,
        "top7_full_podium_rate_pct": round(100 * sum(x["full_podium_top7"] for x in records) / n, 3) if n else None,
        "trio_hit_rate_pct": round(100 * sum(x["trio_hit"] for x in records) / n, 3) if n else None,
        "roi_pct": round(100 * returned / stake, 3) if stake else None,
        "stake_jpy": stake, "return_jpy": returned, "profit_jpy": returned - stake,
    }

def paired_bootstrap(base, challenger, seed):
    a = {r["race_key"]: r for r in base}
    b = {r["race_key"]: r for r in challenger}
    keys = sorted(set(a) & set(b))
    if len(keys) < 30:
        return {"races": len(keys), "status": "比較レース不足"}
    rng = random.Random(seed)
    n = len(keys)
    metrics = {k: [] for k in ("axis_win", "axis_top3", "trio_hit", "roi")}
    for _ in range(BOOTSTRAP):
        sample = [keys[rng.randrange(n)] for _ in range(n)]
        aa, bb = [a[k] for k in sample], [b[k] for k in sample]
        for key in ("axis_win", "axis_top3", "trio_hit"):
            metrics[key].append(100 * (sum(x[key] for x in bb) - sum(x[key] for x in aa)) / n)
        sa, sb = sum(x["stake"] for x in aa), sum(x["stake"] for x in bb)
        if sa and sb:
            metrics["roi"].append(100 * sum(x["returned"] for x in bb) / sb
                                  - 100 * sum(x["returned"] for x in aa) / sa)
    def ci(values):
        values = sorted(values)
        return {"delta_pp": round(float(np.mean(values)), 3),
                "ci95_low": round(float(values[int(.025 * (len(values)-1))]), 3),
                "ci95_high": round(float(values[int(.975 * (len(values)-1))]), 3)}
    return {"races": n, "challenger_minus_static": {k: ci(v) for k, v in metrics.items()}}

def diagnostic_report(scored_groups):
    feature_rows = defaultdict(lambda: {"podium": [], "not_podium": []})
    rank_buckets = defaultdict(lambda: {"runners": 0, "podium": 0})
    segments = defaultdict(lambda: {"runners": 0, "podium": 0})
    venue_segments = defaultdict(lambda: {"runners": 0, "podium": 0})
    for group, scores in scored_groups:
        order = sorted(range(len(group)), key=lambda i: (-float(scores[i]), int(group[i]["horse_no"])))
        for rank, idx in enumerate(order, 1):
            row = group[idx]
            if rank < 6:
                continue
            podium = row.get("finish") is not None and int(row["finish"]) in (1, 2, 3)
            bucket = "6-7位" if rank <= 7 else "8-10位" if rank <= 10 else "11位以下"
            rank_buckets[bucket]["runners"] += 1
            rank_buckets[bucket]["podium"] += int(podium)
            dist = float(row.get("distance_m") or 0)
            db = "1200m未満" if dist < 1200 else "1200-1599m" if dist < 1600 else "1600-1999m" if dist < 2000 else "2000m以上"
            segments[db]["runners"] += 1
            segments[db]["podium"] += int(podium)
            venue = str(row.get("venue") or "不明")
            venue_segments[venue]["runners"] += 1
            venue_segments[venue]["podium"] += int(podium)
            for feature in DIAGNOSTIC_FEATURES:
                value = float(row.get(feature) or 0.0)
                feature_rows[feature]["podium" if podium else "not_podium"].append(value)
    feature_summary = {}
    for feature, values in feature_rows.items():
        a, b = values["podium"], values["not_podium"]
        feature_summary[feature] = {
            "rank6plus_podium_mean": round(float(np.mean(a)), 4) if a else None,
            "rank6plus_nonpodium_mean": round(float(np.mean(b)), 4) if b else None,
            "podium_n": len(a), "nonpodium_n": len(b),
        }
    def rates(source):
        return {k: {"runners": v["runners"], "podium": v["podium"],
                    "podium_rate_pct": round(100*v["podium"]/v["runners"], 3) if v["runners"] else None}
                for k, v in sorted(source.items())}
    return {"rank_bucket": rates(rank_buckets), "distance_band": rates(segments),
            "venue": rates(venue_segments), "feature_means": feature_summary,
            "interpretation_note": "結果を使った事後的な特徴比較であり、予測規則として採用するには別期間での再検証が必要。"}

def main():
    races, horses, payouts, months = nb.load_archives()
    races, excluded_banei = axis_blend.filter_flat_races(races)
    rows, _ = axis_blend.challenger.build_rows(races, horses)
    dates = sorted({str(r["date"]) for r in rows})
    if len(dates) < 100:
        raise RuntimeError(f"平地開催日数が不足: {len(dates)}")
    warm_end = dates[int(len(dates) * .70) - 1]
    warm_rows = [r for r in rows if str(r["date"]) <= warm_end]
    future_rows = [r for r in rows if str(r["date"]) > warm_end]
    warm_groups = grouped(warm_rows)
    if len(warm_groups) < 100:
        raise RuntimeError("初期学習レースが不足しています")
    sim_params = axis_blend.fit_similarity(warm_rows)
    fixed_win = axis_blend.fit_win_specialist(warm_rows)
    online = init_online_state(warm_rows)

    # Compare the frozen 40:60 model with a daily-updated online model on identical races.
    by_day = defaultdict(list)
    for row in future_rows:
        by_day[str(row["date"])].append(row)
    static_records, online_records = [], []
    static_diag, online_diag = [], []
    progress = 0
    for day in sorted(by_day):
        day_rows = by_day[day]
        day_groups = grouped(day_rows)
        for group in day_groups:
            payout = race_payout(group, payouts)
            if not payout or len(group) < 7:
                continue
            if sum(int(r.get("finish") == 1) for r in group) != 1:
                continue
            if sum(1 for r in group if r.get("finish") is not None and int(r["finish"]) in (1,2,3)) != 3:
                continue
            fixed_scores = axis_blend.SIMILARITY_WEIGHTS  # marker; model scored below
            static_sim = axis_blend.similarity_scores(group, sim_params)
            static_prob = fixed_win.predict_proba(axis_blend.matrix(group))[:, 1]
            static_score = SIM_SHARE * axis_blend.percentile(static_sim) + WIN_SHARE * axis_blend.percentile(static_prob)
            online_score = online_scores(group, online)
            sr = ticket_result(group, static_score, payout)
            ir = ticket_result(group, online_score, payout)
            if sr and ir:
                static_records.append(sr)
                online_records.append(ir)
                static_diag.append((group, static_score))
                online_diag.append((group, online_score))
        # Strict date boundary: predictions for all races on this date are made
        # before any result from this date updates the online learner.
        update_online_state(day_rows, online)
        progress += 1
        if progress % max(1, len(by_day)//10) == 0 or progress == len(by_day):
            print(f"ONLINE_LEARNING_PROGRESS {progress}/{len(by_day)} date={day}", flush=True)

    # For rank-6+ diagnostics, use the static 40:60 reference over the same
    # evaluation window, then separately report online-model diagnostics.
    report = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "research_only_flat_rank6_online_learning",
        "scope": "地方競馬の平地のみ。ばんえい・帯広除外。",
        "excluded_banei_race_count": excluded_banei,
        "included_venues": sorted({str(r.get("venue") or "不明") for r in races.values()}),
        "data": {"archive_months": months, "initial_training_end": warm_end,
                 "evaluation_start": min(by_day) if by_day else None,
                 "evaluation_end": max(by_day) if by_day else None,
                 "initial_training_races": len(warm_groups),
                 "evaluation_races": len(static_records),
                 "evaluation_dates": len(by_day)},
        "model_comparison": {
            "fixed_40_60": summarize(static_records),
            "daily_online_update": summarize(online_records),
            "paired_bootstrap": paired_bootstrap(static_records, online_records, 20261114),
            "online_update_days": online["updates"],
            "online_algorithm": "初期70%でSGDロジスティック分類器と勝ち馬/非勝ち馬の特徴平均を初期化。以後、各開催日の全レースを予測し終えてから、その日の結果で1回だけpartial_fitと平均値を更新。順位内40:60合成。"
        },
        "rank6plus_diagnostics": {
            "fixed_40_60": diagnostic_report(static_diag),
            "daily_online_update": diagnostic_report(online_diag),
        },
        "features": {"count": len(FEATURES), "excluded": sorted(EXCLUDED),
                     "current_race_body_weight_used": False,
                     "current_race_body_weight_change_used": False},
        "guardrails": [
            "平地のみ。ばんえい・帯広を除外。",
            "初期学習は時系列の先頭70%。残り30%は日ごとに予測後、その日の全結果で更新。",
            "同一開催日の全レースを予測する前に当日の結果を学習へ入れない。",
            "当該レースの馬体重・馬体重増減は特徴量から除外。",
            "勝ち馬・着順・払戻は予測特徴量に使わない。払戻は検証集計だけに使用。",
            "固定モデルと逐次学習モデルは同一レース・同一6点・同一600円で比較。",
            "過去期間の逐次検証であり将来の利益を保証しない。本番モデル・買い目生成・DBは変更しない。",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("FLAT_RANK6_ONLINE_LEARNING_RESULT " + json.dumps(report, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
