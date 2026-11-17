#!/usr/bin/env python3
"""One-race, low-volume probe of JRA's publicly accessible result page.

This is deliberately separate from every NAR collector/model. It downloads one
public JRA result page, extracts result fields and the 3-renpuku payout, and
writes a compact diagnostic JSON. It does not submit bets or use post-race
fields as prediction features.
"""
from __future__ import annotations

import json
import re
import sys
import time
import unicodedata
import urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "jra-free-data-probe.json"
SAMPLE_URL = (
    "https://www.jra.go.jp/JRADB/accessS.html?"
    "CNAME=pw01sde1003202603041120261115%2F13"
)
USER_AGENT = "KEIBALAB-JRA-free-data-probe/0.2"
EQUIPMENT_LABELS = (
    "ブリンカー着用",
    "チークピーシズ着用",
    "シャドーロール着用",
    "ホライゾネット着用",
)


def clean(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "")
    return re.sub(r"\s+", " ", value).strip()


def parse_horse_name(value: str) -> tuple[str, str, list[str], bool]:
    raw = clean(value)
    equipment = [label for label in EQUIPMENT_LABELS if label in raw]
    is_foreign_bred = "マルガイ" in raw
    name = raw.replace("マルガイ", "")
    for label in equipment:
        name = name.replace(label, "")
    return raw, clean(name), equipment, is_foreign_bred


def parse_body_weight(value: str) -> tuple[int | None, int | None]:
    match = re.search(r"(\d+)\s*\(([+-]?\d+)\)", clean(value))
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


class PageParser(HTMLParser):
    """Collect visible page text and basic HTML table rows without dependencies."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.text_parts: list[str] = []
        self.rows: list[list[str]] = []
        self.links: list[str] = []
        self.table_depth = 0
        self.row: list[str] | None = None
        self.cell_parts: list[str] | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        attrs = dict(attrs)
        if tag == "a" and attrs.get("href"):
            self.links.append(attrs["href"])
        if tag == "table":
            self.table_depth += 1
        elif tag == "tr" and self.table_depth:
            self.row = []
        elif tag in ("td", "th") and self.row is not None:
            self.cell_parts = []
        elif tag == "img":
            alt = clean(attrs.get("alt", ""))
            if alt:
                self.text_parts.append(alt)
                if self.cell_parts is not None:
                    self.cell_parts.append(alt)

    def handle_data(self, data: str) -> None:
        if data:
            self.text_parts.append(data)
            if self.cell_parts is not None:
                self.cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self.cell_parts is not None:
            self.row.append(clean(" ".join(self.cell_parts)))
            self.cell_parts = None
        elif tag == "tr" and self.row is not None:
            if self.row:
                self.rows.append(self.row)
            self.row = None
        elif tag == "table" and self.table_depth:
            self.table_depth -= 1


def decode_html(raw: bytes, declared_charset: str | None) -> str:
    candidates = []
    if declared_charset:
        candidates.append(declared_charset)
    meta = re.search(rb'<meta[^>]+charset=["\']?([A-Za-z0-9._-]+)', raw[:4096], re.I)
    if meta:
        candidates.append(meta.group(1).decode("ascii", errors="ignore"))
    candidates.extend(["utf-8-sig", "cp932", "euc_jp"])

    decoded = []
    for charset in dict.fromkeys(candidates):
        try:
            value = raw.decode(charset, errors="replace")
        except LookupError:
            continue
        japanese_chars = len(re.findall(r"[ぁ-んァ-ヶ一-龥]", value))
        replacement_chars = value.count("\ufffd")
        decoded.append((japanese_chars - replacement_chars * 1000, value))
    if not decoded:
        return raw.decode("cp932", errors="replace")
    return max(decoded, key=lambda item: item[0])[1]


def fetch_html(url: str) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "ja,en;q=0.8",
        },
    )
    with urllib.request.urlopen(request, timeout=25) as response:
        raw = response.read()
        charset = response.headers.get_content_charset()
    return decode_html(raw, charset)


def parse_result(html: str, source_url: str) -> dict:
    source_url = unquote(source_url)
    parser = PageParser()
    parser.feed(html)
    page_text = clean(" ".join(parser.text_parts))
    related_result_links = sorted({
        unquote(urljoin(source_url, href))
        for href in parser.links
        if "accessS.html" in href and "CNAME=" in href
    })

    # JRA result table columns:
    # finish, frame, horse number, horse name, sex/age, carried weight,
    # jockey, time, margin, corner positions, estimated final 3F,
    # body weight, trainer, win popularity.
    runner_rows = []
    for row in parser.rows:
        if len(row) < 14 or not re.fullmatch(r"\d{1,2}", clean(row[0])):
            continue
        finish = int(row[0])
        if not 1 <= finish <= 30 or not re.search(r"[一-龥ぁ-んァ-ン]", row[3]):
            continue

        horse_raw, horse_name, equipment, is_foreign_bred = parse_horse_name(row[3])
        body_weight_kg, body_weight_change_kg = parse_body_weight(row[11])
        sex_age = clean(row[4])
        age_match = re.search(r"\d+", sex_age)
        frame_match = re.search(r"\d+", clean(row[1]))
        carried_weight_match = re.search(r"\d+(?:\.\d+)?", clean(row[5]))
        last_3f_match = re.search(r"\d+(?:\.\d+)?", clean(row[10]))

        runner_rows.append({
            "finish": finish,
            "frame_number": int(frame_match.group(0)) if frame_match else None,
            "horse_number": clean(row[2]),
            "horse_name": horse_name,
            "horse_name_raw": horse_raw,
            "is_foreign_bred": is_foreign_bred,
            "equipment_flags": equipment,
            "sex_age_raw": sex_age,
            "age": int(age_match.group(0)) if age_match else None,
            "sex": sex_age.replace(age_match.group(0), "") if age_match else sex_age,
            "carried_weight_kg": float(carried_weight_match.group(0)) if carried_weight_match else None,
            "jockey": clean(row[6]),
            "time": clean(row[7]),
            "margin": clean(row[8]),
            "corner_positions": [int(v) for v in re.findall(r"\d+", row[9])],
            "estimated_last_3f_sec": float(last_3f_match.group(0)) if last_3f_match else None,
            "body_weight_kg": body_weight_kg,
            "body_weight_change_kg": body_weight_change_kg,
            "trainer": clean(row[12]),
            # This is popularity rank, not a decimal odds value.
            "win_popularity": int(row[13]) if clean(row[13]).isdigit() else None,
        })

    date_match = re.search(
        r"(20\d{2}年\d{1,2}月\d{1,2}日\([^)]+\))\s*"
        r"(\d+回[^\s]+?\d+日)",
        page_text,
    )
    venue_match = re.search(r"\d+回(札幌|函館|福島|新潟|東京|中山|中京|京都|阪神|小倉)\d+日", page_text)
    race_match = re.search(r"(\d{1,2})レース", page_text)
    course_match = re.search(r"([0-9,]+)\s*メートル\s*\(([^)]+)\)", page_text)
    weather_match = re.search(r"天候\s*([^ ]+)", page_text)
    going_match = re.search(r"(芝|ダート)\s*(良|稍重|重|不良)", page_text)

    # Keep separators between the combination, payout and popularity.
    # Removing all whitespace makes a 9-horse combination + 220-yen payout
    # ambiguous ("2-3-9 220" becomes "2-3-9220").
    trifecta_match = re.search(
        r"3連複\s*(\d{1,2}-\d{1,2}-\d{1,2})\s*([0-9,]+)円\s*(\d+)番人気",
        page_text,
    )

    race = {
        "date_label": date_match.group(1) if date_match else None,
        "meeting_day": date_match.group(2) if date_match else None,
        "venue": venue_match.group(1) if venue_match else None,
        "race_number": int(race_match.group(1)) if race_match else None,
        "distance_m": int(course_match.group(1).replace(",", "")) if course_match else None,
        "course_detail": course_match.group(2) if course_match else None,
        "weather": weather_match.group(1) if weather_match else None,
        "going": f"{going_match.group(1)}{going_match.group(2)}" if going_match else None,
    }
    payout = None
    if trifecta_match:
        payout = {
            "bet_type": "trifecta",
            "combination": trifecta_match.group(1),
            "payout_yen_per_100": int(trifecta_match.group(2).replace(",", "")),
            "popularity": int(trifecta_match.group(3)),
        }

    return {
        "source": "JRA public race result page",
        "source_url": source_url,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "race": race,
        "runner_count": len(runner_rows),
        "runners": runner_rows,
        "trifecta_payout": payout,
        "related_result_links": related_result_links,
        "diagnostics": {
            "html_table_rows": len(parser.rows),
            "runner_rows_detected": len(runner_rows),
            "has_trifecta_payout": payout is not None,
            "page_text_chars": len(page_text),
            "related_result_link_count": len(related_result_links),
        },
    }


def collect_target_day(sample_report: dict, target_ymd: str = "20261115") -> dict:
    reports = {unquote(sample_report["source_url"]): sample_report}
    errors: list[dict] = []

    # A result page exposes the 11R pages for all venues on that date.
    # Those three seed pages expose the remaining 1R-12R links for each venue.
    seed_pattern = re.compile(rf"11{target_ymd}/")
    seed_urls = sorted({
        unquote(url) for url in sample_report.get("related_result_links", [])
        if target_ymd in url and seed_pattern.search(url)
    })

    for url in seed_urls:
        if url in reports:
            continue
        time.sleep(1.0)
        try:
            reports[url] = parse_result(fetch_html(url), url)
        except Exception as exc:
            errors.append({"url": url, "error": f"{type(exc).__name__}: {exc}"})

    target_urls = {sample_report["source_url"]}
    for report in reports.values():
        target_urls.update(
            url for url in report.get("related_result_links", [])
            if target_ymd in url and "accessS.html" in url
        )

    # Fetch each unique official race-result page once, sequentially, with a
    # one-second pause to keep the exploratory request rate modest.
    for url in sorted(target_urls):
        if url in reports:
            continue
        time.sleep(1.0)
        try:
            reports[url] = parse_result(fetch_html(url), url)
        except Exception as exc:
            errors.append({"url": url, "error": f"{type(exc).__name__}: {exc}"})

    valid_reports = [
        report for report in reports.values()
        if (report.get("race") or {}).get("date_label", "").startswith("2026年11月15日")
    ]
    unique_reports = {}
    for report in valid_reports:
        race = report.get("race") or {}
        key = (race.get("date_label"), race.get("venue"), race.get("race_number"))
        unique_reports.setdefault(key, report)
    duplicate_page_count = len(valid_reports) - len(unique_reports)

    summaries = []
    for report in unique_reports.values():
        race = report.get("race") or {}
        summaries.append({
            "venue": race.get("venue"),
            "race_number": race.get("race_number"),
            "distance_m": race.get("distance_m"),
            "course_detail": race.get("course_detail"),
            "runner_count": report.get("runner_count", 0),
            "trifecta_payout": report.get("trifecta_payout"),
            "source_url": report.get("source_url"),
        })
    summaries.sort(key=lambda row: (row.get("venue") or "", row.get("race_number") or 0))

    venues = sorted({row["venue"] for row in summaries if row.get("venue")})
    return {
        "target_date": "2026-11-15",
        "target_url_count": len(target_urls),
        "race_page_count": len(valid_reports),
        "duplicate_page_count": duplicate_page_count,
        "race_count": len(summaries),
        "venue_count": len(venues),
        "venues": venues,
        "runner_count_total": sum(row["runner_count"] for row in summaries),
        "trifecta_payout_count": sum(bool(row["trifecta_payout"]) for row in summaries),
        "races_missing_payout": [
            f'{row.get("venue")} {row.get("race_number")}R'
            for row in summaries if not row["trifecta_payout"]
        ],
        "race_summaries": summaries,
        "errors": errors,
    }


def main() -> int:
    try:
        sample = parse_result(fetch_html(SAMPLE_URL), SAMPLE_URL)
        batch = collect_target_day(sample)
    except Exception as exc:
        print(f"JRA_FREE_DATA_PROBE_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    report = dict(sample)
    report.pop("related_result_links", None)
    report["batch"] = batch
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    passed = (
        batch["race_count"] >= 30
        and batch["venue_count"] == 3
        and batch["trifecta_payout_count"] >= 30
        and batch["runner_count_total"] >= 300
    )
    print(json.dumps({
        "status": "PASS" if passed else "INCOMPLETE",
        "sample_race": report["race"],
        "sample_runner_count": report["runner_count"],
        "batch": {
            key: batch[key]
            for key in (
                "target_date", "target_url_count", "race_page_count", "duplicate_page_count", "race_count", "venue_count",
                "venues", "runner_count_total", "trifecta_payout_count",
                "races_missing_payout", "errors",
            )
        },
        "output": str(OUTPUT.relative_to(ROOT)),
    }, ensure_ascii=False, indent=2))
    return 0 if passed else 2

if __name__ == "__main__":
    raise SystemExit(main())
