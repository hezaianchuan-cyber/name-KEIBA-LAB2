#!/usr/bin/env python3
"""Publish daily official NAR payout results for the KEIBA LAB UI."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import nar_backtest as nb

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "daily-payouts.json"
JST = ZoneInfo("Asia/Tokyo")
KINDS = {
    "win": "単勝", "place": "複勝", "wide": "ワイド",
    "quinella": "馬連", "exacta": "馬単", "trio": "3連複",
    "trifecta": "3連単",
}


def main():
    today = dt.datetime.now(JST).date()
    start = today - dt.timedelta(days=120)
    # Reuse the same official monthly archive parser as the backtest.
    nb.START, nb.END = start, today
    races, _horses, payouts, downloaded = nb.load_archives()

    days = {}
    for key, payout in payouts.items():
        race = races.get(key, {})
        date = key[1]
        day = days.setdefault(date, {
            "date": f"{date[:4]}-{date[4:6]}-{date[6:8]}",
            "races": [],
        })
        entries = {}
        for kind, label in KINDS.items():
            rows = payout.get(kind, [])
            if rows:
                entries[kind] = [
                    {"numbers": [str(n) for n in numbers], "pay_100": int(round(pay))}
                    for numbers, pay in rows
                ]
        if not entries:
            continue
        day["races"].append({
            "venue": str(race.get("venue") or key[0]),
            "race": int(key[2]),
            "name": str(race.get("name") or ""),
            "payouts": entries,
        })

    for day in days.values():
        day["races"].sort(key=lambda x: (x["venue"], x["race"]))
        day["race_count"] = len(day["races"])
        day["payout_entry_count"] = sum(
            len(items) for race in day["races"] for items in race["payouts"].values()
        )

    output = {
        "generated_at": dt.datetime.now(JST).isoformat(),
        "source": "NAR official monthly race and payout archives",
        "period": {"start": start.isoformat(), "end": today.isoformat()},
        "days": [days[k] for k in sorted(days, reverse=True)],
        "downloaded_months": downloaded,
        "note": "All amounts are official payout amounts per 100 yen. They are race-result data, not the user's actual purchase history or personal profit/loss.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({
        "period": output["period"], "days": len(output["days"]),
        "races": sum(day["race_count"] for day in output["days"]),
        "payout_entries": sum(day["payout_entry_count"] for day in output["days"]),
        "output_bytes": OUT.stat().st_size,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
