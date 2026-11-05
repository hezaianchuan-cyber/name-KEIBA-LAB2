#!/usr/bin/env python3
"""Verify SHA-256 locks on persisted pre-off NAR prediction snapshots."""
from __future__ import annotations

import gzip
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import capture_nar_odds as capture


def audit_prediction_locks():
    directory = ROOT / "data" / "odds-snapshots"
    files = sorted(directory.glob("*.jsonl.gz")) if directory.exists() else []
    counts = Counter()
    first_capture = None
    last_capture = None
    models = set()
    invalid_examples = []
    for path in files:
        try:
            with gzip.open(path, "rt", encoding="utf-8") as source:
                for line_number, line in enumerate(source, 1):
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        counts["malformed_rows"] += 1
                        continue
                    counts["records"] += 1
                    captured = record.get("captured_at")
                    if captured:
                        first_capture = min(first_capture, captured) if first_capture else captured
                        last_capture = max(last_capture, captured) if last_capture else captured
                    if record.get("model_version"):
                        models.add(str(record["model_version"]))
                    lock = record.get("prediction_lock")
                    if not lock:
                        counts["legacy_unlocked"] += 1
                    elif capture.verify_prediction_lock(record):
                        counts["verified"] += 1
                    else:
                        counts["invalid"] += 1
                        if len(invalid_examples) < 20:
                            invalid_examples.append({
                                "file": path.name,
                                "line": line_number,
                                "race_key": record.get("race_key"),
                                "captured_at": captured,
                            })
        except (OSError, EOFError) as exc:
            counts["unreadable_files"] += 1
            invalid_examples.append({"file": path.name, "error": str(exc)})

    report = {
        "version": "nar-prediction-lock-audit-v1",
        "files": len(files),
        "records": counts["records"],
        "verified": counts["verified"],
        "invalid": counts["invalid"],
        "legacy_unlocked": counts["legacy_unlocked"],
        "malformed_rows": counts["malformed_rows"],
        "unreadable_files": counts["unreadable_files"],
        "first_capture": first_capture,
        "last_capture": last_capture,
        "model_versions": sorted(models),
        "invalid_examples": invalid_examples,
        "status": "passed" if counts["invalid"] == 0 and counts["unreadable_files"] == 0 else "failed",
        "policy": "Legacy snapshots without a digest are reported separately and never treated as verified. Every newly captured snapshot must carry a valid SHA-256 digest covering pre-off predictions, model version, odds, race identity and timestamps.",
    }
    output = ROOT / "data" / "prediction-lock-audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    result = audit_prediction_locks()
    raise SystemExit(0 if result["status"] == "passed" else 1)
