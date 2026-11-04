#!/usr/bin/env python3
"""Publish daily official NAR payout results for the KEIBA LAB UI."""
from __future__ import annotations

import datetime as dt
import io
import json
import urllib.request
import zipfile
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



def load_today_payouts(today):
    """Read the rapidly refreshed official daily race archive for today's results."""
    url = "https://www.keiba.go.jp/KeibaWeb/DataDownload/RaceDataDownload?type=daily"
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; KEIBA-LAB-payouts/1.0)",
            "Referer": "https://www.keiba.go.jp/",
        },
    )
    last_error = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                blob = response.read()
            if not blob.startswith(b"PK"):
                raise RuntimeError("official daily race response is not a ZIP archive")
            today_key = today.strftime("%Y%m%d")
            payouts = {}
            with zipfile.ZipFile(io.BytesIO(blob)) as archive:
                for info in archive.infolist():
                    if info.is_dir() or nb.classify(info.filename) != "payouts":
                        continue
                    for row in nb.rows_from_bytes(archive.read(info)):
                        if len(row) <= 52:
                            continue
                        key = nb.race_key(row[0], row[1], row[2])
                        if not key or key[1] != today_key:
                            continue
                        nb.merge_payout_row(payouts, key, row)
            return payouts, len(blob), None
        except Exception as exc:
            last_error = str(exc)
            if attempt < 2:
                import time
                time.sleep(2 * (attempt + 1))
    return {}, 0, last_error


def main():
    today = dt.datetime.now(JST).date()
    start = today - dt.timedelta(days=120)
    # Reuse the same official monthly archive parser as the backtest.
    nb.START, nb.END = start, today
    races, _horses, payouts, downloaded = nb.load_archives()

    # Monthly archives are refreshed only around 02:00 JST and can therefore
    # lag behind today's races. The official daily archive updates about every
    # two minutes and is the authoritative source for same-day payouts.
    today_payouts, daily_archive_bytes, daily_archive_error = load_today_payouts(today)
    for key, payout in today_payouts.items():
        target = payouts.setdefault(key, {
            "win": [], "place": [], "quinella": [], "exacta": [],
            "wide": [], "trio": [], "trifecta": [], "source_rows": 0,
        })
        for kind in KINDS:
            for item in payout.get(kind, []):
                if item not in target[kind]:
                    target[kind].append(item)
        target["source_rows"] = target.get("source_rows", 0) + payout.get("source_rows", 0)

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
        "daily_archive": {
            "available": daily_archive_bytes > 0,
            "bytes": daily_archive_bytes,
            "today_payout_races": len(today_payouts),
            "error": daily_archive_error,
        },
        "note": "All amounts are official payout amounts per 100 yen. They are race-result data, not the user's actual purchase history or personal profit/loss.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({
        "period": output["period"], "days": len(output["days"]),
        "races": sum(day["race_count"] for day in output["days"]),
        "payout_entries": sum(day["payout_entry_count"] for day in output["days"]),
        "output_bytes": OUT.stat().st_size,
        "today_payout_races": len(today_payouts),
        "daily_archive_error": daily_archive_error,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
