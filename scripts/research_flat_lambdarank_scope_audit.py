#!/usr/bin/env python3
"""Research-only audit of NAR flat/Banei archive scope classification."""
from __future__ import annotations
import collections
import datetime as dt
import json
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import nar_backtest as nb

PATTERN = re.compile(r"帯広|ばんえい", re.IGNORECASE)
OUT = ROOT / "data" / "flat-lambdarank-scope-audit.json"

def main():
    races, horses, payouts, downloads = nb.load_archives()
    venue_hits, name_hits, both_hits = [], [], []
    venue_counts, name_counts = collections.Counter(), collections.Counter()
    venue_name_samples = collections.defaultdict(list)
    for key, race in races.items():
        venue = str(race.get("venue") or "")
        name = str(race.get("name") or "")
        vh, nh = bool(PATTERN.search(venue)), bool(PATTERN.search(name))
        if vh:
            venue_hits.append(key)
            venue_counts[venue] += 1
        if nh:
            name_hits.append(key)
            name_counts[(venue, name)] += 1
            if len(venue_name_samples[venue]) < 12:
                venue_name_samples[venue].append({
                    "date": race["date"].isoformat(),
                    "race": race.get("race"),
                    "name": name,
                    "key": "|".join(map(str, key)),
                })
        if vh and nh:
            both_hits.append(key)

    venue_only = set(venue_hits) - set(both_hits)
    name_only = set(name_hits) - set(both_hits)
    # A strict venue-only Banei definition is the primary reliable test.
    report = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_races": len(races),
        "venue_match_races": len(venue_hits),
        "name_match_races": len(name_hits),
        "both_match_races": len(both_hits),
        "venue_only_races": len(venue_only),
        "name_only_races": len(name_only),
        "venue_match_counts": dict(sorted(venue_counts.items())),
        "name_match_counts_by_venue": {
            venue: len(items) for venue, items in sorted(venue_name_samples.items())
        },
        "name_only_samples_by_venue": {
            venue: samples for venue, samples in sorted(venue_name_samples.items())
            if any(x["key"] in {"|".join(map(str, k)) for k in name_only} for x in samples)
        },
        "name_match_examples": [
            {"venue": venue, "race_name": name, "count": count}
            for (venue, name), count in name_counts.most_common(100)
        ],
        "classification": {
            "venue_only_banei_candidate_races": len(venue_hits),
            "name_only_quarantine_races": len(name_only),
            "policy": "Do not silently treat race-name-only matches as Banei. Inspect examples; venue match is the primary scope key."
        },
        "download_months": len(downloads),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SCOPE_AUDIT_RESULT " + json.dumps(report, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
