#!/usr/bin/env python3
"""Download NAR monthly race archives and run a leakage-safe, fixed-ticket backtest.

Raw NAR CSV/ZIP files are processed in the ephemeral runner only. Only aggregate
metrics and a small set of race-level examples are written to data/backtest-summary.json.
"""
from __future__ import annotations
import base64, csv, datetime as dt, gzip, io, json, math, os, pickle, random, re, sys, time, urllib.parse, urllib.request, zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "backtest-summary.json"
END = dt.date.today() - dt.timedelta(days=1)
START = END - dt.timedelta(days=1460)  # four years: one year warm-up + three untouched annual test windows
UA = "Mozilla/5.0 (compatible; KEIBA-LAB-backtest/1.0)"


# Durable stage checkpoints live on a dedicated GitHub branch, separate from
# production data and reports. Completed expensive stages/folds can therefore
# be reused after a runner timeout or a failed workflow.
BACKTEST_CHECKPOINT_VERSION = "nar-four-year-stage-checkpoint-v1"
BACKTEST_CHECKPOINT_BRANCH = "nar-backtest-checkpoints"
BACKTEST_CHECKPOINT_PATH = "nar-backtest-checkpoint.pkl.gz"


def checkpoint_signature():
    return {
        "version": BACKTEST_CHECKPOINT_VERSION,
        "commit_sha": os.environ.get("GITHUB_SHA", "local"),
        "start": START.isoformat(),
        "end": END.isoformat(),
    }


def _github_checkpoint_request(url, method="GET", payload=None, raw=False):
    token = (os.environ.get("NAR_BACKTEST_CHECKPOINT_TOKEN")
             or os.environ.get("GITHUB_TOKEN"))
    if not token:
        raise RuntimeError("GitHub checkpoint token is not configured")
    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": "KEIBA-LAB-backtest-checkpoint",
        "Accept": "application/vnd.github.raw" if raw else "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=45) as response:
        return response.read()


def _checkpoint_api_base():
    repository = os.environ.get("GITHUB_REPOSITORY")
    if not repository:
        raise RuntimeError("GITHUB_REPOSITORY is not configured")
    return f"https://api.github.com/repos/{repository}"


def load_backtest_checkpoint():
    local_path = ROOT / ".cache" / BACKTEST_CHECKPOINT_PATH
    token = (os.environ.get("NAR_BACKTEST_CHECKPOINT_TOKEN")
             or os.environ.get("GITHUB_TOKEN"))
    if not token:
        if local_path.is_file():
            try:
                with gzip.open(local_path, "rb") as handle:
                    state = pickle.load(handle)
                if state.get("signature") == checkpoint_signature():
                    print("CHECKPOINT_RESUME source=local", flush=True)
                    return state
            except Exception as exc:
                print(f"CHECKPOINT_WARNING local checkpoint ignored: {exc}", flush=True)
        return {"signature": checkpoint_signature(), "chronological": None, "folds": {}}

    base = _checkpoint_api_base()
    metadata_url = (f"{base}/contents/{BACKTEST_CHECKPOINT_PATH}"
                    f"?ref={BACKTEST_CHECKPOINT_BRANCH}")
    try:
        metadata = json.loads(_github_checkpoint_request(metadata_url).decode("utf-8"))
        raw_url = metadata.get("download_url")
        if not raw_url:
            raise RuntimeError("checkpoint download URL missing")
        compressed = _github_checkpoint_request(raw_url, raw=True)
        state = pickle.loads(gzip.decompress(compressed))
        if state.get("signature") != checkpoint_signature():
            print("CHECKPOINT_RESUME ignored_stale_checkpoint=true", flush=True)
            return {"signature": checkpoint_signature(), "chronological": None, "folds": {}}
        print(
            f"CHECKPOINT_RESUME source=github chronological={bool(state.get('chronological'))} "
            f"completed_folds={len(state.get('folds') or {})}",
            flush=True,
        )
        return state
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            print("CHECKPOINT_RESUME no_previous_checkpoint=true", flush=True)
            return {"signature": checkpoint_signature(), "chronological": None, "folds": {}}
        raise


def save_backtest_checkpoint(state, reason):
    state["signature"] = checkpoint_signature()
    local_path = ROOT / ".cache" / BACKTEST_CHECKPOINT_PATH
    local_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = local_path.with_suffix(local_path.suffix + ".tmp")
    with gzip.open(temp_path, "wb", compresslevel=6) as handle:
        pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temp_path, local_path)

    token = (os.environ.get("NAR_BACKTEST_CHECKPOINT_TOKEN")
             or os.environ.get("GITHUB_TOKEN"))
    if not token:
        print(f"CHECKPOINT_SAVED source=local reason={reason}", flush=True)
        return

    base = _checkpoint_api_base()
    branch_url = f"{base}/git/ref/heads/{BACKTEST_CHECKPOINT_BRANCH}"
    try:
        _github_checkpoint_request(branch_url)
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
        starting_sha = os.environ.get("GITHUB_SHA")
        if not starting_sha:
            raise RuntimeError("GITHUB_SHA is required to create checkpoint branch")
        create_url = f"{base}/git/refs"
        try:
            _github_checkpoint_request(create_url, method="POST", payload={
                "ref": f"refs/heads/{BACKTEST_CHECKPOINT_BRANCH}",
                "sha": starting_sha,
            })
        except urllib.error.HTTPError as create_exc:
            # Another retry/worker may have created the branch first.
            if create_exc.code != 422:
                raise

    contents_url = f"{base}/contents/{BACKTEST_CHECKPOINT_PATH}"
    existing_sha = None
    try:
        metadata = json.loads(_github_checkpoint_request(
            f"{contents_url}?ref={BACKTEST_CHECKPOINT_BRANCH}"
        ).decode("utf-8"))
        existing_sha = metadata.get("sha")
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise

    compressed = local_path.read_bytes()
    payload = {
        "message": f"Save NAR backtest checkpoint: {reason}",
        "content": base64.b64encode(compressed).decode("ascii"),
        "branch": BACKTEST_CHECKPOINT_BRANCH,
    }
    if existing_sha:
        payload["sha"] = existing_sha
    _github_checkpoint_request(contents_url, method="PUT", payload=payload)
    print(
        f"CHECKPOINT_SAVED source=github reason={reason} "
        f"bytes={len(compressed)}",
        flush=True,
    )

def norm(s):
    return re.sub(r"\s+", "", str(s or "")).replace("　", "")

def number(v):
    try:
        s = str(v if v is not None else "").replace(",", "").strip()
        return float(s) if s else None
    except (TypeError, ValueError):
        return None

def rank_value(v):
    n = number(v)
    return int(n) if n is not None and n > 0 and n < 99 else None

def decode(raw):
    for enc in ("utf-8-sig", "cp932", "shift_jis"):
        try: return raw.decode(enc)
        except UnicodeDecodeError: pass
    return raw.decode("utf-8", errors="replace")

def rows_from_bytes(raw):
    return list(csv.reader(io.StringIO(decode(raw))))

def race_key(venue, date, race):
    d = re.sub(r"\D", "", str(date or ""))
    if len(d) != 8: return None
    r = re.sub(r"\D", "", str(race or ""))
    if not r: return None
    return (norm(venue), d, str(int(r)))

def parse_corner_order(value):
    """Parse official NAR corner order; parenthesized horses share a rank."""
    text = str(value or "").strip()
    if not text: return {}
    groups = re.findall(r"\(([^()]*)\)|(\d+)", text)
    positions, rank = {}, 1
    for grouped, single in groups:
        nums = re.findall(r"\d+", grouped) if grouped else ([single] if single else [])
        nums = [str(int(x)) for x in nums if x]
        if not nums: continue
        shared_rank = rank + (len(nums)-1)/2
        for no in nums: positions[no] = shared_rank
        rank += len(nums)
    return positions


def corner_history_for_runner(horse, race):
    corners = race.get("corner_positions") or []
    available = [corner for corner in corners if corner and horse["no"] in corner]
    if not available: return {"first_corner_pct": None, "last_corner_pct": None}
    first, last = available[0][horse["no"]], available[-1][horse["no"]]
    size = number(race.get("field_size")) or max(len(available[0]), 1)
    denom = max(size-1, 1)
    return {"first_corner_pct": (first-1)/denom, "last_corner_pct": (last-1)/denom}


def pace_style(history_rows):
    recent = history_rows[-5:]
    early = [number(x.get("first_corner_pct")) for x in recent]
    early = [x for x in early if x is not None]
    late = [number(x.get("last_corner_pct")) for x in recent]
    late = [x for x in late if x is not None]
    if not early: return {"style":"unknown", "front_pct":None, "late_pct":None, "position_gain":None}
    front_pct = sum(early)/len(early)
    late_pct = sum(late)/len(late) if late else front_pct
    position_gain = front_pct-late_pct
    if front_pct <= 0.12 and late_pct <= 0.22:
        style = "逃げ"
    elif front_pct <= 0.38 and late_pct <= 0.45:
        style = "先行"
    elif late_pct <= 0.55 or position_gain >= 0.18:
        style = "差し"
    else:
        style = "追込"
    return {"style":style, "front_pct":front_pct, "late_pct":late_pct, "position_gain":position_gain}


def pace_adjustments(field, history):
    """Experimental low-weight pace adjustment using only prior corner positions."""
    profiles = {h["no"]:pace_style(history.get(h["name"], [])) for h in field}
    leaders = sum(1 for p in profiles.values() if p["front_pct"] is not None and p["front_pct"] <= 0.12)
    adjustments = {}
    for no, profile in profiles.items():
        style, delta = profile["style"], 0.0
        if style != "unknown":
            if leaders >= 3:
                delta = -0.018 if style == "逃げ" else 0.018 if style in ("差し", "追込") else 0.004
            elif leaders == 1:
                delta = 0.010 if style == "逃げ" else -0.006 if style in ("差し", "追込") else 0.0
        adjustments[no] = delta
    return profiles, leaders, adjustments


def month_iter(start, end):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        m += 1
        if m == 13: y, m = y + 1, 1

def download_month(y, m):
    url = "https://www.keiba.go.jp/KeibaWeb/DataDownload/RaceDataDownload?" + urllib.parse.urlencode({
        "type": "monthly", "k_year": y, "k_month": m
    })
    cache_dir = Path(os.environ.get("NAR_CACHE_DIR", ROOT / ".cache" / "nar-monthly"))
    cache_path = cache_dir / f"{y}-{m:02d}.zip"
    current = dt.date.today()
    if (y, m) < (current.year, current.month) and cache_path.is_file():
        data = cache_path.read_bytes()
        if data.startswith(b"PK"):
            return data, url
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Referer": "https://www.keiba.go.jp/"})
    last = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=90) as res:
                data = res.read()
            if not data.startswith(b"PK"):
                raise RuntimeError(f"{y}-{m:02d}: response is not a ZIP archive")
            if (y, m) < (current.year, current.month):
                cache_dir.mkdir(parents=True, exist_ok=True)
                cache_path.write_bytes(data)
            return data, url
        except Exception as exc:
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"{y}-{m:02d}: download failed: {last}")

def classify(name):
    n = norm(name).lower()
    if any(x in n for x in ("horselist", "horse_list", "出馬表")): return "horses"
    if any(x in n for x in ("racelist", "race_list", "レース一覧")): return "races"
    if any(x in n for x in ("payback", "payout", "払戻")): return "payouts"
    return None

def merge_payout_row(payouts, key, row):
    """Merge one official payout CSV row; a dead heat may produce several rows."""
    payout = payouts.setdefault(key, {
        "win": [], "place": [], "quinella": [], "exacta": [],
        "wide": [], "trio": [], "trifecta": [], "source_rows": 0,
    })
    payout["source_rows"] += 1

    def add_pay(target, values, amount):
        numbers = [re.sub(r"\D", "", str(value or "")) for value in values]
        pay = number(amount)
        if all(numbers) and pay is not None and pay > 0:
            item = (numbers, pay)
            if item not in payout[target]:
                payout[target].append(item)

    add_pay("win", [row[4]], row[5])
    for i in (7, 10, 13):
        add_pay("place", [row[i]], row[i+1])
    add_pay("quinella", [row[24], row[25]], row[26])
    add_pay("exacta", [row[28], row[29]], row[30])
    for i in (32, 36, 40):
        add_pay("wide", [row[i], row[i+1]], row[i+2])
    add_pay("trio", [row[44], row[45], row[46]], row[47])
    add_pay("trifecta", [row[49], row[50], row[51]], row[52])
    return payout


def load_final_odds_archives(start=None, end=None):
    """Load official monthly confirmed win odds for retrospective market benchmarking only."""
    start = start or dt.date(2026, 3, 1)
    end = end or END
    output = defaultdict(dict)
    months = list(month_iter(start, end))
    cache_dir = Path(os.environ.get("NAR_ODDS_CACHE_DIR", ROOT / ".cache" / "nar-odds-monthly"))
    current = dt.date.today()

    def download_odds_month(y, m):
        url = "https://www.keiba.go.jp/KeibaWeb/DataDownload/OddsDataDownload?" + urllib.parse.urlencode({
            "type": "monthly", "k_year": y, "k_month": m, "k_day": 1
        })
        cacheable = (y, m) < (current.year, current.month)
        cache_path = cache_dir / f"odds-{y}-{m:02d}.zip"
        if cacheable and cache_path.is_file():
            try:
                cached = cache_path.read_bytes()
                if cached.startswith(b"PK"):
                    return (y, m), cached, "hit"
            except OSError:
                pass
        req = urllib.request.Request(url, headers={"User-Agent": UA, "Referer": "https://www.keiba.go.jp/"})
        last = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=90) as res:
                    blob = res.read()
                if not blob.startswith(b"PK"):
                    raise RuntimeError(f"{y}-{m:02d}: odds response is not a ZIP archive")
                if cacheable:
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    temp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
                    temp_path.write_bytes(blob)
                    os.replace(temp_path, cache_path)
                return (y, m), blob, "miss" if cacheable else "current_month_refresh"
            except Exception as exc:
                last = exc
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"{y}-{m:02d}: odds download failed: {last}")

    downloaded = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        archives = list(pool.map(lambda ym: download_odds_month(*ym), months))
    for (year, month), blob, cache_status in archives:
        rows_seen = 0
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            for info in zf.infolist():
                if info.is_dir() or "odds" not in info.filename.lower():
                    continue
                for row in rows_from_bytes(zf.read(info)):
                    if len(row) < 10 or row[0] == "競馬場":
                        continue
                    if str(row[3]).strip() != "単勝":
                        continue
                    key = race_key(row[0], row[1], row[2])
                    if not key:
                        continue
                    date = dt.datetime.strptime(key[1], "%Y%m%d").date()
                    if date < start or date > end:
                        continue
                    horse_no = re.sub(r"\D", "", str(row[4] or ""))
                    odds = number(row[7])
                    if not horse_no or odds is None or odds <= 0:
                        continue
                    output[key][str(int(horse_no))] = odds
                    rows_seen += 1
        downloaded.append({"month": f"{year}-{month:02d}", "bytes": len(blob),
                           "win_odds_rows": rows_seen, "odds_cache": cache_status})
    return output, downloaded


def summarize_final_market_pair(bucket):
    return {
        **bucket,
        "top1_rate": round(100 * bucket["top1_hits"] / bucket["races"], 2) if bucket["races"] else None,
        "hit_rate": round(100 * bucket["hits"] / bucket["races"], 2) if bucket["races"] else None,
        **summarize_race_cluster_roi(bucket),
    }


PARSED_ARCHIVE_CACHE_VERSION = "nar-monthly-parsed-v1"

def parse_month_archive(blob, y, m):
    """Parse one official monthly archive into compact race/runner/payout objects."""
    races, horses, payouts = {}, defaultdict(list), {}
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        for info in zf.infolist():
            kind = classify(info.filename)
            if not kind or info.is_dir():
                continue
            for row in rows_from_bytes(zf.read(info)):
                if len(row) < 4:
                    continue
                key = race_key(row[0], row[1], row[2])
                if not key:
                    continue
                date = dt.datetime.strptime(key[1], "%Y%m%d").date()
                if date < START or date > END:
                    continue
                if kind == "races":
                    races[key] = {
                        "key": key, "date": date, "venue": row[0], "race": key[2],
                        "name": row[5] if len(row) > 5 else "",
                        "distance": number(row[23]) if len(row) > 23 else None,
                        "field_size": number(row[26]) if len(row) > 26 else None,
                        "surface": row[21] if len(row) > 21 else "",
                        "going": row[25] if len(row) > 25 else "",
                        "corner_positions": [parse_corner_order(row[i]) for i in range(58, min(len(row), 66))],
                    }
                elif kind == "horses" and len(row) > 35:
                    horse_no = re.sub(r"\D", "", str(row[5] or ""))
                    name = norm(row[6])
                    if not horse_no or not name:
                        continue
                    horses[key].append({
                        "no": horse_no, "name": name,
                        "frame_no": rank_value(row[4]) if len(row) > 4 else None,
                        "sex": norm(row[7]) if len(row) > 7 else "",
                        "age": number(row[8]),
                        "birth_date": re.sub(r"\\D", "", str(row[10] or "")) if len(row) > 10 else "",
                        "sire": norm(row[11]),
                        "dam": norm(row[12]) if len(row) > 12 else "",
                        "damsire": norm(row[13]) if len(row) > 13 else "",
                        "jockey": norm(row[14]), "carried": number(row[16]),
                        "finish": rank_value(row[31]),
                        "time_seconds": parse_race_time(row[32]) if len(row) > 32 else None,
                        "margin": row[33] if len(row) > 33 else "",
                        "last3f": number(row[34]) if len(row) > 34 else None,
                    })
                elif kind == "payouts" and len(row) > 52:
                    merge_payout_row(payouts, key, row)
    finalize_payout_indexes(payouts)
    return races, dict(horses), payouts


def load_parsed_month(ym):
    """Load a trusted, versioned parsed-month cache or download and parse one month."""
    y, m = ym
    current = dt.date.today()
    cache_dir = Path(os.environ.get("NAR_PARSED_CACHE_DIR", ROOT / ".cache" / "nar-parsed"))
    cache_path = cache_dir / f"{PARSED_ARCHIVE_CACHE_VERSION}-{y}-{m:02d}.pkl"
    cacheable = (y, m) < (current.year, current.month)
    if cacheable and cache_path.is_file():
        try:
            with cache_path.open("rb") as handle:
                cached = pickle.load(handle)
            if cached.get("version") == PARSED_ARCHIVE_CACHE_VERSION:
                return cached["races"], cached["horses"], cached["payouts"], {
                    "month": f"{y}-{m:02d}", "bytes": cached.get("archive_bytes", 0),
                    "parsed_cache": "hit",
                }
        except Exception:
            # A corrupt or incompatible cache is discarded and rebuilt below.
            try:
                cache_path.unlink()
            except OSError:
                pass

    blob, _ = download_month(y, m)
    races, horses, payouts = parse_month_archive(blob, y, m)
    if cacheable:
        cache_dir.mkdir(parents=True, exist_ok=True)
        temp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        payload = {
            "version": PARSED_ARCHIVE_CACHE_VERSION,
            "archive_bytes": len(blob),
            "races": races, "horses": horses, "payouts": payouts,
        }
        with temp_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temp_path, cache_path)
    return races, horses, payouts, {
        "month": f"{y}-{m:02d}", "bytes": len(blob),
        "parsed_cache": "miss" if cacheable else "current_month_refresh",
    }


def load_archives():
    """Load monthly archives in bounded parallel batches and reuse parsed results."""
    races, horses, payouts = {}, defaultdict(list), {}
    months = list(month_iter(START, END))
    downloaded = []
    # Bound memory: at most four parsed months are held before merging into the
    # aggregate dictionaries. The previous implementation retained every ZIP
    # archive in memory before parsing any of them.
    batch_size = 4
    for offset in range(0, len(months), batch_size):
        batch = months[offset:offset + batch_size]
        with ThreadPoolExecutor(max_workers=min(batch_size, len(batch))) as pool:
            parsed_months = list(pool.map(load_parsed_month, batch))
        for (month_year, month_num), (month_races, month_horses, month_payouts, metadata) in zip(batch, parsed_months):
            races.update(month_races)
            for key, rows in month_horses.items():
                horses[key].extend(rows)
            payouts.update(month_payouts)
            downloaded.append(metadata)
            print(
                f"Loaded {metadata['month']}: races={len(month_races)}, "
                f"horse_races={len(month_horses)}, parsed_cache={metadata['parsed_cache']}",
                flush=True,
            )
    finalize_payout_indexes(payouts)
    return races, horses, payouts, downloaded

def finalize_payout_indexes(payouts):
    """Build constant-time payout lookups while preserving dead-heat combinations."""
    for payout in payouts.values():
        index = {}
        for kind in ("win", "place", "quinella", "exacta", "wide", "trio", "trifecta"):
            kind_index = {}
            for numbers, pay in payout.get(kind, []):
                key = tuple(str(x) for x in numbers)
                if kind in ("quinella", "wide", "trio"):
                    key = tuple(sorted(key))
                kind_index.setdefault(key, pay)
            index[kind] = kind_index
        payout["_index"] = index
    return payouts

def parse_race_time(value):
    """Parse Japanese race time strings into seconds."""
    raw = str(value or "").strip()
    if not raw or raw in ("計不", "失格", "中止"):
        return None
    match = re.fullmatch(r"(\d+):(\d+(?:\.\d+)?)", raw)
    if match:
        return int(match.group(1)) * 60.0 + float(match.group(2))
    match = re.fullmatch(r"(\d+(?:\.\d+)?)", raw)
    return float(match.group(1)) if match else None


def margin_score(value):
    s = str(value or "").strip()
    if not s: return None
    if any(x in s for x in ("大差", "失格", "中止")): return 0.0
    if "ハナ" in s: return 1.0
    if "アタマ" in s: return 0.5
    if "クビ" in s: return 0.75
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    return float(m.group(1)) if m else None

def rank_quality(finish, field_size=None):
    rank = number(finish)
    if rank is None or rank <= 0:
        return None
    size = number(field_size)
    if size is not None and size > 1:
        return max(0.0, min(100.0, 100.0*(size-rank)/(size-1)))
    return max(5.0, 100.0-(rank-1)*8.0)


def smoothed_rate(successes, runs, prior_rate=0.50, prior_strength=10.0):
    """Shrink sparse sire/jockey rates toward a neutral prior to reduce small-sample noise."""
    successes = max(0.0, number(successes) or 0.0)
    runs = max(0.0, number(runs) or 0.0)
    prior_strength = max(0.0, number(prior_strength) or 0.0)
    prior_rate = min(1.0, max(0.0, number(prior_rate) if number(prior_rate) is not None else 0.50))
    if runs + prior_strength <= 0:
        return prior_rate
    return (successes + prior_rate * prior_strength) / (runs + prior_strength)


def field_performance_features(field):
    """Store race-relative closing-speed rank and margin quality for future races."""
    valid_last3f = [(h["no"], number(h.get("last3f"))) for h in field]
    valid_last3f = [(no, value) for no, value in valid_last3f if value is not None and value > 0]
    ordered = sorted(valid_last3f, key=lambda item: item[1])
    last3f_scores = {}
    if ordered:
        groups = defaultdict(list)
        for no, value in ordered:
            groups[value].append(no)
        position = 0
        total = len(ordered)
        for value in sorted(groups):
            group = groups[value]
            average_rank = position + (len(group) - 1) / 2
            score = 1.0 if total == 1 else 1.0 - average_rank / (total - 1)
            for no in group:
                last3f_scores[no] = score
            position += len(group)
    output = {}
    for h in field:
        raw = str(h.get("margin") or "").strip()
        if "ハナ" in raw: gap = 0.1
        elif "アタマ" in raw: gap = 0.2
        elif "クビ" in raw: gap = 0.3
        elif any(token in raw for token in ("大差", "失格", "中止")): gap = 10.0
        else:
            parsed = number(raw)
            gap = max(0.0, parsed) if parsed is not None else None
        margin_quality = math.exp(-gap / 2.0) if gap is not None else None
        output[h["no"]] = {
            "margin_quality": margin_quality,
            "last3f_rank_score": last3f_scores.get(h["no"]),
        }
    return output


def score_runner_performance(h, history, sire_stats, jockey_stats, race, model_weights,
                             margin_weight=0.10, closing_weight=0.10,
                             time_weight=0.0, include_recent_history=True,
                             going_weight=0.0, parts_cache=None, performance_cache=None):
    """B score blended with strictly prior margin/closing-speed form.

    parts_cache is scoped to one race. Feature values depend on prior history,
    not on the candidate weight vector, so challengers can reuse them safely.
    """
    if parts_cache is None:
        # Preserve the legacy call shape for wrappers and test instrumentation.
        base_score, parts = score_runner(h, history, sire_stats, jockey_stats, race,
                                         model_weights, include_recent_history)
    else:
        base_score, parts = score_runner(
            h, history, sire_stats, jockey_stats, race, model_weights,
            include_recent_history, parts_cache=parts_cache)
    performance_key = (str(h.get("no", "")), bool(include_recent_history))
    cached_performance = (performance_cache.get(performance_key)
                          if performance_cache is not None else None)
    if cached_performance is None:
        past = history.get(h["name"], [])[-5:] if include_recent_history else []
        def recent_average(key):
            values = [number(row.get(key)) for row in past]
            values = [value for value in values if value is not None]
            if not values:
                return 0.5
            weights = list(range(1, len(values) + 1))
            return sum(value * weight for value, weight in zip(values, weights)) / sum(weights)
        margin_form = recent_average("margin_quality")
        closing_form = recent_average("last3f_rank_score")
        # Going suitability is calculated only from this horse's earlier starts.
        # Sparse samples shrink toward neutral; no same-day/current-race outcome leaks in.
        target_going = norm(race.get("going"))
        target_surface = norm(race.get("surface"))
        going_rows = [row for row in past if target_going and norm(row.get("going")) == target_going
                      and (not target_surface or not norm(row.get("surface")) or norm(row.get("surface")) == target_surface)
                      and number(row.get("going_quality")) is not None]
        going_quality = [number(row.get("going_quality")) for row in going_rows]
        going_form = smoothed_rate(
            sum(going_quality), len(going_quality), prior_rate=0.5, prior_strength=4.0
        ) if going_quality else 0.5
        target_distance = number(race.get("distance"))
        comparable = [
            row for row in past
            if row.get("venue") == race.get("venue")
            and target_distance is not None
            and number(row.get("distance")) is not None
            and abs(number(row.get("distance")) - target_distance) <= 200
            and number(row.get("time_per_km")) is not None
        ]
        time_form = 0.5
        if len(comparable) >= 3:
            recent_times = [number(row["time_per_km"]) for row in comparable[-2:]]
            baseline_times = [number(row["time_per_km"]) for row in comparable[:-2]]
            if baseline_times:
                improvement = (sum(baseline_times)/len(baseline_times)
                               - sum(recent_times)/len(recent_times)) / max(sum(baseline_times)/len(baseline_times), 1.0)
                time_form = min(1.0, max(0.0, 0.5 + improvement * 8.0))
        cached_performance = (margin_form, closing_form, time_form, going_form)
        if performance_cache is not None:
            performance_cache[performance_key] = cached_performance
    margin_form, closing_form, time_form, going_form = cached_performance
    base_weight = max(0.0, 1.0 - margin_weight - closing_weight - time_weight - going_weight)
    score = (base_score * base_weight + margin_form * margin_weight
             + closing_form * closing_weight + time_form * time_weight
             + going_form * going_weight)
    return score, {**parts, "margin_form": margin_form, "closing_form": closing_form,
                   "time_form": time_form, "going_form": going_form,
                   "performance_weights": {"base": base_weight, "margin": margin_weight,
                                           "closing": closing_weight, "time": time_weight,
                                           "going": going_weight}}


def score_runner_d(h, history, sire_stats, jockey_stats, race, model_weights,
                   include_recent_history=True):
    """D1 compatibility wrapper: 80% B + 10% margin + 10% closing speed."""
    return score_runner_performance(h, history, sire_stats, jockey_stats, race,
                                    model_weights, 0.10, 0.10,
                                    include_recent_history=include_recent_history)


def score_runner(h, history, sire_stats, jockey_stats, race, model_weights,
                include_recent_history=True, parts_cache=None):
    # One cache is created per race and shared by every model candidate. The
    # feature vector is independent of model_weights; only the final dot product
    # changes between C and its challengers.
    cache_key = (str(h.get("no", "")), bool(include_recent_history))
    parts = parts_cache.get(cache_key) if parts_cache is not None else None
    if parts is None:
        past = history.get(h["name"], [])[-5:] if include_recent_history else []
        if past:
            quality_rows = [rank_quality(x.get("finish"), x.get("field_size")) for x in past]
            quality_rows = [q for q in quality_rows if q is not None]
            recency_weights = list(range(1, len(quality_rows)+1))
            weighted_form = (sum(q*w for q,w in zip(quality_rows,recency_weights))/sum(recency_weights)) if quality_rows else 45
            recent = past[-3:]
            top3 = sum(1 for x in recent if x.get("finish") and x["finish"] <= 3) / len(recent)
            form = 0.75*weighted_form + 0.25*top3*100 if quality_rows else 45
        else:
            form = 45
        venue_runs = [x for x in past if x.get("venue") == race["venue"] and x.get("finish")]
        venue_quality = [rank_quality(x.get("finish"), x.get("field_size")) for x in venue_runs]
        venue_quality = [q for q in venue_quality if q is not None]
        venue_score = sum(venue_quality)/len(venue_quality) if venue_quality else 48
        distance_runs = [x for x in past if race.get("distance") and x.get("distance") and abs(x["distance"]-race["distance"]) <= 200 and x.get("finish")]
        distance_quality = [rank_quality(x.get("finish"), x.get("field_size")) for x in distance_runs]
        distance_quality = [q for q in distance_quality if q is not None]
        distance_score = sum(distance_quality)/len(distance_quality) if distance_quality else 48
        sire = sire_stats[h["sire"]] if h["sire"] else {"runs": 0, "top3": 0}
        sire_score = 100 * smoothed_rate(sire["top3"], sire["runs"])
        j = jockey_stats[h["jockey"]] if h["jockey"] else {"runs": 0, "top3": 0}
        jockey_score = 100 * smoothed_rate(j["top3"], j["runs"])
        age = h.get("age")
        age_score = 72 if age is not None and 4 <= age <= 6 else 58 if age in (3,7) else 45
        parts = {"form":form/100,"venue":venue_score/100,"distance":distance_score/100,
                 "sire":sire_score/100,"jockey":jockey_score/100,"age":age_score/100}
        if parts_cache is not None:
            parts_cache[cache_key] = parts
    score = sum(model_weights.get(k, 0.0)*v for k,v in parts.items())
    return score, parts


def race_relative_scores(parts_by_no, model_weights):
    """Convert each feature to a within-race percentile before weighted scoring.

    This mirrors the race-relative normalization approach documented by public
    racing model builders. Ties receive their average rank; no outcome data is used.
    """
    horse_numbers = list(parts_by_no)
    if not horse_numbers:
        return {}
    n = len(horse_numbers)
    relative = {no: {} for no in horse_numbers}
    for feature in model_weights:
        # Missing features are neutral and must tie with an observed 0.5.
        ordered = sorted(
            (((number(parts_by_no[no].get(feature))
               if number(parts_by_no[no].get(feature)) is not None else 0.5), no)
             for no in horse_numbers),
            key=lambda item: item[0],
            reverse=True,
        )
        position = 0
        while position < n:
            value = ordered[position][0]
            end = position + 1
            while end < n and ordered[end][0] == value:
                end += 1
            average_rank = (position + end - 1) / 2.0
            percentile = 1.0 if n == 1 else 1.0 - average_rank / (n - 1)
            for _, no in ordered[position:end]:
                relative[no][feature] = percentile
            position = end
    return {
        no: sum(model_weights[key] * values.get(key, 0.5) for key in model_weights)
        for no, values in relative.items()
    }

def ablated_model_weights(model_weights, excluded_feature):
    """Drop one feature and renormalize the remaining fixed weights for diagnostics."""
    remaining = {key: value for key, value in model_weights.items()
                 if key != excluded_feature}
    total = sum(remaining.values())
    if not remaining or total <= 0:
        raise ValueError(f"Cannot ablate feature {excluded_feature!r}")
    return {key: value / total for key, value in remaining.items()}


FEATURE_WEIGHT_VARIANTS = {
    "weight_form_down_0_8": ("form", 0.80),
    "weight_form_up_1_2": ("form", 1.20),
    "weight_jockey_down_0_5": ("jockey", 0.50),
    "weight_jockey_up_1_5": ("jockey", 1.50),
    "weight_distance_down_0_7": ("distance", 0.70),
    "weight_distance_up_1_3": ("distance", 1.30),
    "weight_venue_down_0_7": ("venue", 0.70),
    "weight_venue_up_1_3": ("venue", 1.30),
}

# Combined challengers are research-only. All adjustments are applied together
# and normalized once, so interaction effects are measured on the same race set.
FEATURE_WEIGHT_COMBINATIONS = {
    "D1_jockey_venue": (("jockey", 1.50), ("venue", 0.70)),
    "D2_jockey_distance": (("jockey", 1.50), ("distance", 0.70)),
    "D3_venue_distance": (("venue", 0.70), ("distance", 0.70)),
    "D4_jockey_venue_distance": (("jockey", 1.50), ("venue", 0.70), ("distance", 0.70)),
}
FEATURE_WEIGHT_NAMES = tuple(FEATURE_WEIGHT_VARIANTS) + tuple(FEATURE_WEIGHT_COMBINATIONS)


def scaled_model_weights(model_weights, feature, multiplier):
    """Change one learned feature weight by a fixed factor and renormalize."""
    if feature not in model_weights or multiplier <= 0:
        raise ValueError("Invalid feature-weight experiment")
    adjusted = dict(model_weights)
    adjusted[feature] *= multiplier
    total = sum(adjusted.values())
    if total <= 0:
        raise ValueError("Feature weights must have a positive sum")
    return {key: value / total for key, value in adjusted.items()}


def combined_scaled_model_weights(model_weights, adjustments):
    """Apply multiple fixed feature multipliers together, then normalize once."""
    adjusted = dict(model_weights)
    if not adjustments:
        raise ValueError("At least one feature adjustment is required")
    for feature, multiplier in adjustments:
        if feature not in adjusted or multiplier <= 0:
            raise ValueError("Invalid combined feature-weight experiment")
        adjusted[feature] *= multiplier
    total = sum(adjusted.values())
    if total <= 0:
        raise ValueError("Feature weights must have a positive sum")
    return {key: value / total for key, value in adjusted.items()}


def market_rank_gap_bucket(gap):
    """Positive gap means LAB ranks a horse above the market."""
    gap = int(gap)
    if gap >= 5: return "lab_upgrade_5_plus"
    if gap >= 3: return "lab_upgrade_3_4"
    if gap >= 1: return "lab_upgrade_1_2"
    if gap <= -5: return "market_upgrade_5_plus"
    if gap <= -3: return "market_upgrade_3_4"
    return "near_alignment_minus2_to_plus2"


def new_market_rank_diagnostics():
    bands = ("lab_upgrade_5_plus", "lab_upgrade_3_4", "lab_upgrade_1_2",
             "near_alignment_minus2_to_plus2", "market_upgrade_3_4",
             "market_upgrade_5_plus")
    return {
        "races": 0, "matched_horses": 0, "top7_overlap_sum": 0,
        "rank_gap_bands": {name: {"horses": 0, "finish_top3": 0, "winner": 0}
                           for name in bands},
        "top_k_selection": {
            side: {str(k): {"selected": 0, "finish_top3": 0, "winner": 0,
                            "races_with_any_top3": 0, "races_with_all_top3": 0}
                   for k in (1, 3, 5, 7)}
            for side in ("lab", "market")
        },
    }


def update_market_rank_diagnostics(report, lab_ranked, field, odds_by_no):
    """Compare LAB and market ranks on exactly the same horses in a race."""
    field_by_no = {str(h.get("no")): h for h in field if h.get("no") is not None}
    market_rows = [(float(odds), str(no)) for no, odds in odds_by_no.items()
                   if str(no) in field_by_no and number(odds) is not None
                   and float(odds) > 0]
    market_rows.sort(key=lambda item: (item[0], int(item[1]) if item[1].isdigit() else item[1]))
    if len(market_rows) < 3:
        return
    market_order = [no for _, no in market_rows]
    market_set = set(market_order)
    lab_order = [str(h.get("no")) for h in lab_ranked if str(h.get("no")) in market_set]
    if len(lab_order) < 3:
        return
    market_rank = {no: i + 1 for i, no in enumerate(market_order)}
    lab_rank = {no: i + 1 for i, no in enumerate(lab_order)}
    report["races"] += 1
    report["matched_horses"] += len(lab_order)
    report["top7_overlap_sum"] += len(set(lab_order[:7]) & set(market_order[:7]))
    for no in lab_order:
        horse = field_by_no[no]
        finish = number(horse.get("finish"))
        bucket = report["rank_gap_bands"][market_rank_gap_bucket(market_rank[no] - lab_rank[no])]
        bucket["horses"] += 1
        bucket["finish_top3"] += int(finish is not None and finish <= 3)
        bucket["winner"] += int(finish == 1)
    for side, ranking in (("lab", lab_order), ("market", market_order)):
        for k in (1, 3, 5, 7):
            selected = [field_by_no[no] for no in ranking[:k] if no in field_by_no]
            bucket = report["top_k_selection"][side][str(k)]
            top3_count = sum(1 for h in selected
                             if number(h.get("finish")) is not None and number(h.get("finish")) <= 3)
            winner_count = sum(1 for h in selected if number(h.get("finish")) == 1)
            # Compare coverage only against actual top-three finishers in
            # the matched horse set; horses without valid market odds are
            # excluded from both LAB and market rankings.
            actual_top3 = sum(1 for no in market_set
                              if number(field_by_no[no].get("finish")) is not None
                              and number(field_by_no[no].get("finish")) <= 3)
            bucket["selected"] += len(selected)
            bucket["finish_top3"] += top3_count
            bucket["winner"] += winner_count
            bucket["races_with_any_top3"] += int(top3_count > 0)
            bucket["races_with_all_top3"] += int(actual_top3 > 0 and top3_count == actual_top3)


def summarize_market_rank_diagnostics(report):
    if not report["races"]:
        return {"races": 0, "matched_horses": 0, "avg_top7_overlap": None,
                "rank_gap_bands": {}, "top_k_selection": {}}
    bands = {}
    for name, row in report["rank_gap_bands"].items():
        bands[name] = {
            **row,
            "finish_top3_rate_pct": round(100 * row["finish_top3"] / row["horses"], 2)
                if row["horses"] else None,
            "winner_rate_pct": round(100 * row["winner"] / row["horses"], 2)
                if row["horses"] else None,
        }
    selection = {}
    for side, by_k in report["top_k_selection"].items():
        selection[side] = {}
        for k, row in by_k.items():
            selection[side][k] = {
                **row,
                "finish_top3_precision_pct": round(100 * row["finish_top3"] / row["selected"], 2)
                    if row["selected"] else None,
                "winner_precision_pct": round(100 * row["winner"] / row["selected"], 2)
                    if row["selected"] else None,
                "race_any_top3_rate_pct": round(100 * row["races_with_any_top3"] / report["races"], 2),
                "race_all_top3_rate_pct": round(100 * row["races_with_all_top3"] / report["races"], 2),
            }
    return {
        "races": report["races"], "matched_horses": report["matched_horses"],
        "avg_top7_overlap": round(report["top7_overlap_sum"] / report["races"], 2),
        "rank_gap_definition": "market rank minus LAB rank; positive means LAB ranks the horse higher",
        "rank_gap_bands": bands, "top_k_selection": selection,
        "comparison_policy": "LAB and market ranks are recomputed on the same horses with valid official final odds in each race; outcomes are official finishing positions.",
    }


def select_feature_weight_candidate(candidate_rows, baseline_top3_rate, baseline_trio_roi,
                                    roi_tolerance_pp=1.0, baseline_development_folds=None,
                                    baseline_top1_rate=None):
    """Select only challengers that improve top-1 and top-3 on pooled and each development fold."""
    eligible = []
    baseline_by_fold = {
        str(row.get("fold")): row for row in (baseline_development_folds or [])
        if row.get("fold") is not None
    }
    for name, row in candidate_rows.items():
        top3 = row.get("development_winner_top3_rate")
        roi = row.get("development_trio_roi")
        if top3 is None or roi is None or baseline_top3_rate is None or baseline_trio_roi is None:
            continue
        top1 = row.get("development_winner_top1_rate")
        if (top3 <= baseline_top3_rate or roi < baseline_trio_roi - roi_tolerance_pp
                or (baseline_top1_rate is not None and
                    (top1 is None or top1 <= baseline_top1_rate))):
            continue
        candidate_by_fold = {
            str(item.get("fold")): item for item in (row.get("development_folds") or [])
            if item.get("fold") is not None
        }
        # When fold detail is supplied, every development fold must independently
        # improve top-3 coverage and remain within the ROI tolerance.
        if baseline_by_fold:
            if set(candidate_by_fold) != set(baseline_by_fold):
                continue
            stable = True
            for fold_name, baseline in baseline_by_fold.items():
                candidate = candidate_by_fold[fold_name]
                candidate_top1 = candidate.get("winner_top1_rate")
                baseline_fold_top1 = baseline.get("winner_top1_rate")
                candidate_top3 = candidate.get("winner_top3_rate")
                baseline_fold_top3 = baseline.get("winner_top3_rate")
                candidate_roi = candidate.get("trio_roi")
                baseline_fold_roi = baseline.get("trio_roi")
                if (candidate_top3 is None or baseline_fold_top3 is None
                        or candidate_roi is None or baseline_fold_roi is None
                        or candidate_top3 <= baseline_fold_top3
                        or candidate_roi < baseline_fold_roi - roi_tolerance_pp
                        or (baseline_fold_top1 is not None and
                            (candidate_top1 is None or candidate_top1 <= baseline_fold_top1))):
                    stable = False
                    break
            if not stable:
                continue
        eligible.append(name)
    if not eligible:
        return None
    return max(eligible, key=lambda name: (
        candidate_rows[name]["development_winner_top3_rate"],
        candidate_rows[name].get("development_winner_top1_rate") or -1,
        candidate_rows[name].get("development_trio_roi") or -1,
    ))


def update_model(field, parts_by_no, model_weights):
    valid = [h for h in field if h.get("finish") and h["no"] in parts_by_no]
    if len(valid) < 3:
        return False

    # Use the exact pre-race score that produced the ranking when available.
    # The older fallback remains for callers/tests with legacy feature-only data.
    has_prediction_scores = all(
        number(parts_by_no[h["no"]].get("__prediction_score")) is not None
        for h in valid
    )
    if has_prediction_scores:
        scores = [number(parts_by_no[h["no"]]["__prediction_score"]) for h in valid]
    else:
        scores = [
            sum(model_weights[k] * parts_by_no[h["no"]][k] for k in model_weights)
            for h in valid
        ]

    peak = max(scores)
    expv = [math.exp((x - peak) * 8) for x in scores]
    denom = sum(expv) or 1
    probs = [x / denom for x in expv]
    winners = {i for i, h in enumerate(valid) if h["finish"] == 1}
    if not winners:
        return False

    target = 1.0 / len(winners)
    lr = 0.025
    for k in model_weights:
        grad = 0.0
        for i, horse in enumerate(valid):
            features = parts_by_no[horse["no"]]
            # Performance, frame and pace overlays affect the prediction score,
            # but only the base model weights are trainable here. Scale their
            # gradient by the base-score share of the actual prediction.
            scale = number(features.get("__model_gradient_scale"))
            if scale is None:
                scale = 1.0
            grad += ((target if i in winners else 0.0) - probs[i]) * features[k] * scale
        model_weights[k] = max(0.01, min(0.8, model_weights[k] + lr * grad))

    total = sum(model_weights.values()) or 1
    for k in model_weights:
        model_weights[k] /= total
    return True

def softmax_probabilities(scores, temperature):
    temperature = max(0.005, float(temperature or 0.18))
    peak = max(scores) if scores else 0.0
    values = [math.exp(max(-700.0, (score-peak)/temperature)) for score in scores]
    total = sum(values) or 1.0
    return [value/total for value in values]


def fit_temperature(records):
    usable = [r for r in records if r.get("scores") and r.get("winner_indices")]
    if not usable:
        return 0.18
    candidates = [0.02 + i*0.005 for i in range(97)]
    best_temperature, best_loss = 0.18, float("inf")
    for temperature in candidates:
        loss = 0.0
        for record in usable:
            probs = softmax_probabilities(record["scores"], temperature)
            mass = sum(probs[i] for i in record["winner_indices"] if 0 <= i < len(probs))
            loss -= math.log(max(1e-12, mass))
        loss /= len(usable)
        if loss < best_loss:
            best_temperature, best_loss = temperature, loss
    return best_temperature


def calibration_metrics(records, temperature):
    usable = [r for r in records if r.get("scores") and r.get("winner_indices")]
    if not usable:
        return {"races": 0, "temperature": temperature, "log_loss": None,
                "brier": None, "top1_accuracy": None, "ece": None}
    log_loss = brier = top1 = 0.0
    bins = [{"count":0,"confidence":0.0,"accuracy":0.0} for _ in range(10)]
    for record in usable:
        probs = softmax_probabilities(record["scores"], temperature)
        winners = {i for i in record["winner_indices"] if 0 <= i < len(probs)}
        if not winners:
            continue
        mass = sum(probs[i] for i in winners)
        log_loss -= math.log(max(1e-12, mass))
        target = 1.0 / len(winners)
        brier += sum((prob-(target if i in winners else 0.0))**2 for i,prob in enumerate(probs))
        best = max(range(len(probs)), key=lambda i: probs[i])
        correct = 1.0 if best in winners else 0.0
        top1 += correct
        confidence = probs[best]
        bucket = min(9, int(confidence*10))
        bins[bucket]["count"] += 1
        bins[bucket]["confidence"] += confidence
        bins[bucket]["accuracy"] += correct
    count = len(usable)
    ece = 0.0
    for bucket in bins:
        if bucket["count"]:
            ece += bucket["count"]/count * abs(
                bucket["confidence"]/bucket["count"] - bucket["accuracy"]/bucket["count"])
    confidence_buckets = []
    for index, bucket in enumerate(bins):
        bucket_count = bucket["count"]
        confidence_buckets.append({
            "range": f"{index * 10}-{(index + 1) * 10}%",
            "races": bucket_count,
            "mean_confidence_pct": round(100 * bucket["confidence"] / bucket_count, 2) if bucket_count else None,
            "hit_rate_pct": round(100 * bucket["accuracy"] / bucket_count, 2) if bucket_count else None,
            "calibration_gap_pp": round(100 * (bucket["accuracy"] - bucket["confidence"]) / bucket_count, 2) if bucket_count else None,
        })
    return {"races": count, "temperature": round(float(temperature), 4),
            "log_loss": round(log_loss/count, 5), "brier": round(brier/count, 5),
            "top1_accuracy": round(100*top1/count, 2), "ece": round(ece, 5),
            "confidence_buckets": confidence_buckets}


def split_nums(s):
    return [re.sub(r"\D", "", x) for x in re.split(r"[-－ー,／/ ]+", str(s or "")) if re.sub(r"\D", "", x)]

def same_unordered(a, b):
    return sorted(a) == sorted(b)

def payout_for(kind, ticket, p):
    """Return the official 100-yen payout for a ticket, including dead heats."""
    if not p:
        return 0
    index = p.get("_index")
    if index is not None:
        key = tuple(str(x) for x in ticket)
        if kind in ("quinella", "wide", "trio"):
            key = tuple(sorted(key))
        return index.get(kind, {}).get(key, 0)
    for numbers, pay in p.get(kind, []):
        if kind in ("win", "place"):
            matched = len(ticket) == 1 and ticket[0] == numbers[0]
        elif kind in ("quinella", "wide", "trio"):
            matched = same_unordered(ticket, numbers)
        else:  # exacta and trifecta are order-sensitive
            matched = ticket == numbers
        if matched:
            return pay or 0
    return 0

def top_k_winner_coverage(ranked, winners, ks=(5, 6, 7)):
    """Return winner-in-top-K indicators for identical race rankings."""
    winner_set = {str(no) for no in winners}
    return {str(k): int(any(str(h.get("no")) in winner_set for h in ranked[:k]))
            for k in ks}


def build_trifecta_formation(ranked, first_count=2, second_count=5, third_count=5):
    """Build an ordered 1st/2nd/3rd formation with independent candidate pools."""
    firsts = [str(h["no"]) for h in ranked[:first_count]]
    seconds = [str(h["no"]) for h in ranked[:second_count]]
    thirds = [str(h["no"]) for h in ranked[:third_count]]
    return [[first, second, third]
            for first in firsts
            for second in seconds if second != first
            for third in thirds if third not in (first, second)]


def build_trifecta_tickets(ranked, first_count=2, candidate_count=5):
    """Backward-compatible symmetric formation helper."""
    return build_trifecta_formation(ranked, first_count, candidate_count, candidate_count)


def build_shadow_portfolio_tickets(ranked):
    """Mirror build_auto_ticket_plans.shadow_tickets exactly; virtual stakes only."""
    unique = []
    seen = set()
    for horse in ranked or []:
        raw = re.sub(r"\D", "", str(horse.get("no") or ""))
        if raw:
            no = str(int(raw))
            if no not in seen:
                unique.append(no)
                seen.add(no)
    if len(unique) < 3:
        return []
    return [{"kind": "trio", "numbers": sorted(unique[:3], key=int)}]


def build_fixed_tickets(ranked):
    """Build the fixed research portfolio; trifecta remains top-2 x top-5 (24 tickets)."""
    nos = [str(h["no"]) for h in ranked[:5]]
    if len(nos) < 3:
        return {}
    first_candidates = nos[:2]
    second_candidates = nos[:3]
    return {
        "wide": [[nos[0],nos[1]],[nos[0],nos[2]],[nos[1],nos[2]]],
        "exacta": [[first,second] for first in first_candidates
                   for second in second_candidates if first != second],
        "trio": [list(combo) for combo in __import__("itertools").combinations(nos,3)],
        "trifecta": build_trifecta_tickets(ranked, 2, 5),
    }



def role_specific_features(scored, history, target_finish):
    """Compute blend-independent role form once per horse and finish position."""
    features = []
    for base_score, horse in scored:
        past = history.get(horse["name"], [])[-5:]
        values, weights = [], []
        for index, row in enumerate(past, start=1):
            finish = number(row.get("finish"))
            field_size = number(row.get("field_size"))
            if finish is None or field_size is None or field_size < 2:
                continue
            finish_fit = max(0.0, 1.0 - abs(finish - target_finish) / (field_size - 1))
            first_pos = number(row.get("first_corner_pct"))
            last_pos = number(row.get("last_corner_pct"))
            closing = number(row.get("last3f_rank_score"))
            if target_finish == 1:
                role_signal = 1.0 - first_pos if first_pos is not None else 0.5
            elif target_finish == 2:
                positions = [1.0 - value for value in (first_pos, last_pos) if value is not None]
                role_signal = sum(positions) / len(positions) if positions else 0.5
            else:
                gain = (0.5 + 0.5 * (first_pos - last_pos)
                        if first_pos is not None and last_pos is not None else None)
                signals = [
                    (0.40, 1.0 - last_pos if last_pos is not None else None),
                    (0.35, closing),
                    (0.25, gain),
                ]
                available = [(weight, value) for weight, value in signals if value is not None]
                weight_total = sum(weight for weight, _ in available)
                role_signal = (
                    sum(weight * value for weight, value in available) / weight_total
                    if weight_total else 0.5
                )
            values.append(0.70 * finish_fit + 0.30 * role_signal)
            weights.append(1.25 ** (index - 1))
        if values:
            role_form = sum(v*w for v,w in zip(values,weights)) / sum(weights)
            reliability = len(values) / (len(values) + 4.0)
            role_form = reliability * role_form + (1.0 - reliability) * 0.5
        else:
            role_form = 0.5
        features.append((base_score, role_form, horse))
    return features


def rank_role_specific_features(features, blend):
    """Apply a blend to precomputed role features and sort without rereading history."""
    ranked = [((1.0 - blend) * base_score + blend * role_form, horse)
              for base_score, role_form, horse in features]
    return [horse for _, horse in sorted(ranked, key=lambda item:item[0], reverse=True)]


def role_specific_ranking(scored, history, target_finish, blend):
    """Compatibility wrapper for callers that need a single role ranking."""
    return rank_role_specific_features(
        role_specific_features(scored, history, target_finish), blend)


def build_role_trifecta_formation(role_rankings, first_count, second_count, third_count):
    """Build ordered tickets from separate, role-specific candidate rankings."""
    firsts = [str(h["no"]) for h in role_rankings[1][:first_count]]
    seconds = [str(h["no"]) for h in role_rankings[2][:second_count]]
    thirds = [str(h["no"]) for h in role_rankings[3][:third_count]]
    return [[first, second, third]
            for first in firsts
            for second in seconds if second != first
            for third in thirds if third not in (first, second)]


def new_finish_position_diagnostics():
    """Measure whether each ranking predicts the actual horse for each finish position."""
    return {
        ranking: {
            str(role): {"races": 0, "topk": {str(k): 0 for k in (1, 2, 3, 5, 7)},
                        "rank_sum": 0.0, "reciprocal_rank_sum": 0.0,
                        "found": 0}
            for role in (1, 2, 3)
        }
        for ranking in ("ordinary", "role_specific")
    }


def update_finish_position_diagnostics(stats, ordinary_ranking, role_rankings, field):
    finishers = {
        role: {str(h["no"]) for h in field if h.get("finish") == role}
        for role in (1, 2, 3)
    }
    rankings = {"ordinary": {role: ordinary_ranking for role in (1, 2, 3)},
                "role_specific": role_rankings}
    for ranking_name, by_role in rankings.items():
        for role in (1, 2, 3):
            bucket = stats[ranking_name][str(role)]
            bucket["races"] += 1
            targets = finishers[role]
            ordered = [str(h["no"]) for h in by_role[role]]
            positions = [i + 1 for i, no in enumerate(ordered) if no in targets]
            if positions:
                best = min(positions)
                bucket["found"] += 1
                bucket["rank_sum"] += best
                bucket["reciprocal_rank_sum"] += 1.0 / best
            for k in (1, 2, 3, 5, 7):
                if any(position <= k for position in positions):
                    bucket["topk"][str(k)] += 1


def summarize_finish_position_diagnostics(stats):
    output = {}
    for ranking_name, by_role in stats.items():
        output[ranking_name] = {}
        for role, bucket in by_role.items():
            races = bucket["races"]
            output[ranking_name][role] = {
                "races": races,
                "topk_hits": bucket["topk"],
                "topk_rate": {
                    k: round(100 * value / races, 2) if races else None
                    for k, value in bucket["topk"].items()
                },
                "mean_rank_when_found": round(bucket["rank_sum"] / bucket["found"], 3)
                    if bucket["found"] else None,
                "mean_reciprocal_rank": round(bucket["reciprocal_rank_sum"] / races, 4)
                    if races else None,
                "found_count": bucket["found"],
            }
    return output


def new_role_trifecta_diagnostics():
    formations = ((1,3,4), (1,4,3), (1,4,4), (1,4,5), (1,5,4),
                  (2,3,3), (2,3,4), (2,4,3), (2,4,4), (2,5,3),
                  (2,3,5), (2,5,4), (3,4,4), (3,5,4))
    blends = (0.05, 0.10, 0.20, 0.30, 0.40)
    return {
        f"{blend:.2f}": {
            f"{a}-{b}-{c}": {"races":0,"tickets":0,"hits":0,"hit_races":0,
                              "stake":0,"return":0,"sum_return_sq":0,
                              "sum_stake_sq":0,"sum_stake_return":0}
            for a,b,c in formations
        } for blend in blends
    }


def update_role_trifecta_diagnostics(bucket, scored, history, payout,
                                       role_rankings_cache=None, role_features_cache=None):
    for blend_key, formations in bucket.items():
        blend = float(blend_key)
        if role_rankings_cache is not None and blend_key in role_rankings_cache:
            role_rankings = role_rankings_cache[blend_key]
        else:
            role_rankings = {}
            for role in (1, 2, 3):
                if role_features_cache is not None and role in role_features_cache:
                    role_features = role_features_cache[role]
                else:
                    role_features = role_specific_features(scored, history, role)
                    if role_features_cache is not None:
                        role_features_cache[role] = role_features
                role_rankings[role] = rank_role_specific_features(role_features, blend)
            if role_rankings_cache is not None:
                role_rankings_cache[blend_key] = role_rankings
        for formation, stats in formations.items():
            a,b,c = (int(x) for x in formation.split("-"))
            tickets = build_role_trifecta_formation(role_rankings, a, b, c)
            if not tickets:
                continue
            returned = sum(payout_for("trifecta", ticket, payout) or 0 for ticket in tickets)
            hits = sum(1 for ticket in tickets if payout_for("trifecta", ticket, payout))
            race_stake = 100 * len(tickets)
            stats["races"] += 1
            stats["tickets"] += len(tickets)
            stats["hits"] += hits
            stats["hit_races"] += int(hits > 0)
            stats["stake"] += race_stake
            stats["return"] += returned
            # Race-level ratio moments allow a conservative uncertainty estimate
            # without retaining raw race records in the published aggregate file.
            stats["sum_return_sq"] += returned * returned
            stats["sum_stake_sq"] += race_stake * race_stake
            stats["sum_stake_return"] += race_stake * returned


def summarize_role_trifecta_diagnostics(bucket):
    output = {}
    for blend, formations in bucket.items():
        output[blend] = {}
        for formation, stats in formations.items():
            races = stats["races"]
            stake = stats["stake"]
            returned = stats["return"]
            roi = 100 * returned / stake if stake else None
            ci_low = ci_high = None
            if races > 1 and roi is not None:
                ratio = returned / stake
                sum_z_sq = (stats["sum_return_sq"]
                            - 2 * ratio * stats["sum_stake_return"]
                            + ratio * ratio * stats["sum_stake_sq"])
                variance = max(0.0, sum_z_sq / (races - 1))
                mean_stake = stake / races
                standard_error = 100 * math.sqrt(variance / races) / mean_stake if mean_stake else 0.0
                ci_low = round(roi - 1.96 * standard_error, 2)
                ci_high = round(roi + 1.96 * standard_error, 2)
            enough = races >= 100 and stats["tickets"] >= 300
            output[blend][formation] = {
                **stats,
                "tickets_per_race": round(stats["tickets"] / races, 2) if races else None,
                "hit_rate": round(100 * stats["hits"] / stats["tickets"], 4) if stats["tickets"] else None,
                "race_hit_rate": round(100 * stats["hit_races"] / races, 2) if races else None,
                "roi": round(roi, 2) if roi is not None else None,
                "roi_ci95_normal_low": ci_low,
                "roi_ci95_normal_high": ci_high,
                "minimum_sample_met": enough,
                "candidate_pass": bool(enough and ci_low is not None and ci_low > 100),
                "profit": returned - stake,
            }
    return output

def new_trifecta_diagnostics():
    structures = {(first, second, third): {"races": 0, "tickets": 0, "hits": 0,
                                               "hit_races": 0, "stake": 0, "return": 0,
                                               "sum_return_sq": 0, "sum_stake_sq": 0,
                                               "sum_stake_return": 0}
                  for first, second, third in (
                      # Narrow formations: compare 6-12 ticket structures first.
                      (1,4,4), (1,4,5), (1,5,4),
                      (2,3,4), (2,4,3), (2,3,5), (2,5,3), (2,4,4),
                      # Existing reference structures.
                      (1,5,5), (2,4,5), (2,5,4), (2,5,5),
                      (2,5,6), (2,6,5), (3,5,5), (2,6,6), (2,7,7)
                  )}
    return {
        "races": 0,
        "winner_top1": 0, "winner_top2": 0, "winner_top3": 0,
        "second_top2": 0, "second_top3": 0, "second_top5": 0, "second_top7": 0,
        "third_top3": 0, "third_top5": 0, "third_top7": 0,
        "all_podium_top5": 0, "all_podium_top6": 0, "all_podium_top7": 0,
        "current_structure_exact_hit_races": 0,
        "structures": structures,
    }


def update_trifecta_diagnostics(bucket, ranked, field, payout):
    """Separate horse-role coverage from ordered-ticket hits and ticket-count effects."""
    if not ranked:
        return
    rank = {str(h["no"]): i + 1 for i, h in enumerate(ranked)}
    winners = {str(numbers[0]) for numbers, _ in payout.get("win", [])}
    second = {str(h["no"]) for h in field if h.get("finish") == 2}
    third = {str(h["no"]) for h in field if h.get("finish") == 3}
    def covered(group, top_n):
        return any(rank.get(no, 10**9) <= top_n for no in group)
    if winners and second and third:
        bucket["races"] += 1
        bucket["winner_top1"] += int(covered(winners, 1))
        bucket["winner_top2"] += int(covered(winners, 2))
        bucket["winner_top3"] += int(covered(winners, 3))
        bucket["second_top2"] += int(covered(second, 2))
        bucket["second_top3"] += int(covered(second, 3))
        bucket["second_top5"] += int(covered(second, 5))
        bucket["second_top7"] += int(covered(second, 7))
        bucket["third_top3"] += int(covered(third, 3))
        bucket["third_top5"] += int(covered(third, 5))
        bucket["third_top7"] += int(covered(third, 7))
        podium = winners | second | third
        bucket["all_podium_top5"] += int(all(covered({no}, 5) for no in podium))
        bucket["all_podium_top6"] += int(all(covered({no}, 6) for no in podium))
        bucket["all_podium_top7"] += int(all(covered({no}, 7) for no in podium))
        current = build_trifecta_tickets(ranked, 2, 5)
        if any(payout_for("trifecta", ticket, payout) for ticket in current):
            bucket["current_structure_exact_hit_races"] += 1
    # Ticket ROI is evaluated for every race with a valid payout, even if finish-role fields are incomplete.
    for (formation, stats) in bucket["structures"].items():
        first_count, second_count, third_count = formation
        tickets = build_trifecta_formation(ranked, first_count, second_count, third_count)
        if not tickets:
            continue
        race_stake = 100 * len(tickets)
        returned = sum(payout_for("trifecta", ticket, payout) or 0 for ticket in tickets)
        race_hits = sum(1 for ticket in tickets if payout_for("trifecta", ticket, payout))
        stats["races"] += 1
        stats["tickets"] += len(tickets)
        stats["stake"] += race_stake
        stats["return"] += returned
        stats["hits"] += race_hits
        stats["hit_races"] += int(race_hits > 0)
        stats["sum_return_sq"] += returned * returned
        stats["sum_stake_sq"] += race_stake * race_stake
        stats["sum_stake_return"] += race_stake * returned


def summarize_trifecta_diagnostics(bucket):
    n = bucket["races"]
    rate_keys = ("winner_top1","winner_top2","winner_top3","second_top2","second_top3",
                 "second_top5","second_top7","third_top3","third_top5","third_top7",
                 "all_podium_top5","all_podium_top6","all_podium_top7",
                 "current_structure_exact_hit_races")
    output = {key: bucket[key] for key in ("races", *rate_keys)}
    output["rates"] = {key: round(100 * bucket[key] / n, 2) if n else None for key in rate_keys}
    structures = {}
    for formation, stats in bucket["structures"].items():
        first_count, second_count, third_count = formation
        n = stats["races"]
        roi = 100 * stats["return"] / stats["stake"] if stats["stake"] else None
        ci_low = ci_high = None
        if n > 1 and roi is not None:
            ratio = roi / 100
            sum_z_sq = (stats["sum_return_sq"] - 2 * ratio * stats["sum_stake_return"]
                        + ratio * ratio * stats["sum_stake_sq"])
            variance = max(0.0, sum_z_sq / (n - 1))
            mean_stake = stats["stake"] / n
            standard_error = 100 * (variance / n) ** 0.5 / mean_stake if mean_stake else 0
            ci_low, ci_high = round(roi - 1.96 * standard_error, 2), round(roi + 1.96 * standard_error, 2)
        structures[f"{first_count}-{second_count}-{third_count}"] = {
            "races": n, "tickets": stats["tickets"], "hits": stats["hits"],
            "hit_races": stats["hit_races"], "stake": stats["stake"], "return": stats["return"],
            "hit_rate": round(100 * stats["hits"] / stats["tickets"], 4) if stats["tickets"] else None,
            "race_hit_rate": round(100 * stats["hit_races"] / n, 2) if n else None,
            "roi": round(roi, 2) if roi is not None else None,
            "roi_ci95_normal_low": ci_low, "roi_ci95_normal_high": ci_high,
            "roi_variance_components": {
                "sum_return_sq": stats["sum_return_sq"],
                "sum_stake_sq": stats["sum_stake_sq"],
                "sum_stake_return": stats["sum_stake_return"],
            },
            "profit": stats["return"] - stats["stake"],
            "tickets_per_race": round(stats["tickets"] / n, 2) if n else None,
        }
    output["structures"] = structures
    output["interpretation"] = "hit_rate is winning tickets divided by tickets; race_hit_rate is races with at least one winning ticket divided by races. ROI confidence intervals use a race-level normal ratio approximation and are diagnostic, not a promotion guarantee. Role coverage and ticket ROI are reported separately."
    return output


def build_trio_tickets(ranked, candidate_count=5):
    """Build unordered trio combinations from the top-N ranked runners."""
    import itertools
    nos = [str(h["no"]) for h in ranked[:candidate_count]]
    return [list(combo) for combo in itertools.combinations(nos, 3)] if len(nos) >= 3 else []


def narrow_trio_by_rank_sum(tickets, ranked, count):
    """Select a fixed number of existing 2-4-5 tickets by the LAB rank sum."""
    rank = {}
    for index, horse in enumerate(ranked or [], start=1):
        raw = re.sub(r"\D", "", str(horse.get("no") or ""))
        if raw:
            rank.setdefault(str(int(raw)), index)
    ordered = sorted(
        tickets or [],
        key=lambda ticket: (
            sum(rank.get(str(no), len(rank) + 1) for no in ticket),
            tuple(sorted((str(no) for no in ticket), key=int)),
        ),
    )
    return ordered[:max(0, min(int(count), len(ordered)))]


def build_trio_formation_2_4_5(ranked):
    """2-4-5 trio formation over seven ranked horses: top2 / top4 / ranks3-7.

    Trio is unordered, so duplicate permutations and repeated-horse tickets
    are removed. The canonical horse-number order makes every ticket unique.
    """
    unique = []
    seen_horses = set()
    for horse in ranked or []:
        raw = re.sub(r"\D", "", str(horse.get("no") or ""))
        if raw:
            no = str(int(raw))
            if no not in seen_horses:
                unique.append(no)
                seen_horses.add(no)
    firsts, seconds, thirds = unique[:2], unique[:4], unique[2:7]
    tickets, seen_tickets = [], set()
    for first in firsts:
        for second in seconds:
            if second == first:
                continue
            for third in thirds:
                if third in (first, second):
                    continue
                combo = tuple(sorted((first, second, third), key=int))
                if combo in seen_tickets:
                    continue
                seen_tickets.add(combo)
                tickets.append(list(combo))
    return tickets


FORMATION_ALLOCATION_STRENGTHS = {
    "equal": 0.0,
    "rank_weighted_mild": 0.18,
    "rank_weighted_strong": 0.35,
}


def allocate_formation_units(tickets, ranked, policy, budget_units=None):
    """Allocate 100-yen units; optionally preserve a fixed race budget when tickets are narrowed.

    This is deliberately rank-only: historical pre-off trio combination odds are
    not available for the full four-year holdout, so it is not an EV/Kelly model.
    """
    if policy not in FORMATION_ALLOCATION_STRENGTHS:
        raise ValueError(f"unknown formation allocation policy: {policy}")
    if not tickets:
        return []
    total_units = len(tickets) if budget_units is None else max(0, int(budget_units))
    if policy == "equal":
        base, remainder = divmod(total_units, len(tickets))
        return [base + (1 if index < remainder else 0) for index in range(len(tickets))]
    rank_by_no = {}
    for index, horse in enumerate(ranked or [], start=1):
        raw = re.sub(r"\D", "", str(horse.get("no") or ""))
        if raw:
            rank_by_no.setdefault(str(int(raw)), index)
    strength = FORMATION_ALLOCATION_STRENGTHS[policy]
    weights = []
    for ticket in tickets:
        rank_sum = sum(rank_by_no.get(str(no), len(ranked or []) + 1) for no in ticket)
        weights.append(math.exp(-strength * max(0, rank_sum - 6)))
    # Allocate one 100-yen unit at a time using a divisor method. The race
    # budget remains exactly equal to the equal-stake baseline; zero-unit
    # candidates are intentionally left unbet under concentrated policies.
    units = [0] * len(tickets)
    for _ in range(total_units):
        chosen = max(range(len(tickets)), key=lambda i: (weights[i] / (units[i] + 1), -i))
        units[chosen] += 1
    return units


def adaptive_trio_pool_size(ranked, scored, gap_z_threshold=0.60):
    """Choose top-4/5/6/7 pool from pre-race score separation only.

    A pronounced score gap at the 4/5, 5/6, or 6/7 boundary narrows the pool.
    Otherwise the pool expands to seven. The threshold is fixed before holdout;
    finish positions and payouts are never inputs to this decision.
    """
    available = min(len(ranked or []), 7)
    if available <= 3:
        return available
    score_by_no = {str(horse.get("no")): number(score)
                   for score, horse in (scored or [])}
    values = [score_by_no.get(str(horse.get("no"))) for horse in ranked[:available]]
    if any(value is None for value in values):
        return available
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    scale = math.sqrt(variance)
    if scale <= 1e-12:
        return available
    for boundary in (4, 5, 6):
        if available > boundary:
            gap_z = (values[boundary - 1] - values[boundary]) / scale
            if gap_z >= gap_z_threshold:
                return boundary
    return available


def select_balanced_trio_tickets(candidates, max_tickets=12, probability_weight=0.5, min_ev=1.05):
    """Mirror the Web/Supabase 50:50 probability-rank and EV-rank selector."""
    rows = []
    for ticket in candidates or []:
        probability = number(ticket.get("probability"))
        ev = number(ticket.get("expected_value", ticket.get("ev")))
        if probability is None or probability <= 0 or ev is None or ev < min_ev:
            continue
        rows.append((ticket, probability, ev))
    count = len(rows)
    if not count:
        return []
    probability_order = sorted(range(count), key=lambda i: -rows[i][1])
    ev_order = sorted(range(count), key=lambda i: -rows[i][2])
    probability_rank = {index: rank for rank, index in enumerate(probability_order)}
    ev_rank = {index: rank for rank, index in enumerate(ev_order)}
    weight = min(1.0, max(0.0, number(probability_weight) or 0.0))
    scored = []
    for index, (ticket, probability, ev) in enumerate(rows):
        probability_score = 1.0 if count == 1 else 1.0 - probability_rank[index] / (count - 1)
        ev_score = 1.0 if count == 1 else 1.0 - ev_rank[index] / (count - 1)
        scored.append(({
            **ticket,
            "probability_rank_score": probability_score,
            "ev_rank_score": ev_score,
            "selection_score": weight * probability_score + (1.0 - weight) * ev_score,
        }, probability, ev, index))
    scored.sort(key=lambda item: (-item[0]["selection_score"], -item[1], -item[2], item[3]))
    limit = max(0, min(12, int(max_tickets)))
    return [item[0] for item in scored[:limit]]


def new_trio_diagnostics():
    return {
        "races": 0,
        "podium_count_top7": {"0": 0, "1": 0, "2": 0, "3": 0},
        "adaptive_pool": {"races": 0, "tickets": 0, "hits": 0, "hit_races": 0,
                          "stake": 0, "return": 0, "selected_top_n": {
                              str(n): 0 for n in (4, 5, 6, 7)}},
        "all_podium_top4": 0, "all_podium_top5": 0,
        "all_podium_top6": 0, "all_podium_top7": 0,
        "structures": {
            pool: {"races": 0, "tickets": 0, "hits": 0, "hit_races": 0, "stake": 0, "return": 0,
                   "sum_return_sq": 0, "sum_stake_sq": 0, "sum_stake_return": 0}
            for pool in (4, 5, 6, 7)
        },
        "formation_2_4_5": {"races": 0, "tickets": 0, "hits": 0, "hit_races": 0,
                            "stake": 0, "return": 0, "sum_return_sq": 0,
                            "sum_stake_sq": 0, "sum_stake_return": 0},
        "formation_2_4_5_narrowed": {
            count: {"races": 0, "tickets": 0, "funded_tickets": 0,
                    "hits": 0, "hit_races": 0, "stake": 0, "return": 0}
            for count in (12, 10)
        },
        "formation_2_4_5_allocations": {
            policy: {"races": 0, "tickets": 0, "funded_tickets": 0,
                     "winning_candidates": 0, "hits": 0, "hit_races": 0,
                     "stake": 0, "return": 0}
            for policy in FORMATION_ALLOCATION_STRENGTHS
        },
    }


def update_trio_diagnostics(bucket, ranked, field, payout, scored=None):
    """Compare fixed and score-gap-adaptive unordered trio combinations."""
    if not ranked:
        return
    podium = {str(h["no"]) for h in field if h.get("finish") in (1, 2, 3)}
    rank = {str(h["no"]): i + 1 for i, h in enumerate(ranked)}
    if len(podium) == 3:
        bucket["races"] += 1
        captured_top7 = sum(1 for no in podium if rank.get(no, 10**9) <= 7)
        bucket["podium_count_top7"][str(captured_top7)] += 1
        for pool in (4, 5, 6, 7):
            bucket[f"all_podium_top{pool}"] += int(all(rank.get(no, 10**9) <= pool for no in podium))

    # Variable ticket count is selected using only the model's pre-race score gaps.
    adaptive_n = adaptive_trio_pool_size(ranked, scored)
    adaptive_tickets = build_trio_tickets(ranked, adaptive_n)
    if adaptive_tickets:
        adaptive = bucket["adaptive_pool"]
        race_return = sum(payout_for("trio", ticket, payout) or 0
                          for ticket in adaptive_tickets)
        race_hits = sum(1 for ticket in adaptive_tickets
                        if payout_for("trio", ticket, payout))
        adaptive["races"] += 1
        adaptive["tickets"] += len(adaptive_tickets)
        adaptive["hits"] += race_hits
        adaptive["hit_races"] += int(race_hits > 0)
        adaptive["stake"] += 100 * len(adaptive_tickets)
        adaptive["return"] += race_return
        adaptive["selected_top_n"][str(adaptive_n)] += 1

    # Ticket ROI is evaluated even when the finish-position fields are incomplete.
    for pool in (4, 5, 6, 7):
        stats = bucket["structures"][pool]
        tickets = build_trio_tickets(ranked, pool)
        if not tickets:
            continue
        race_stake = 100 * len(tickets)
        returned = sum(payout_for("trio", ticket, payout) or 0 for ticket in tickets)
        race_hits = sum(1 for ticket in tickets if payout_for("trio", ticket, payout))
        stats["races"] += 1
        stats["tickets"] += len(tickets)
        stats["stake"] += race_stake
        stats["return"] += returned
        stats["hits"] += race_hits
        stats["hit_races"] += int(race_hits > 0)
        stats["sum_return_sq"] += returned * returned
        stats["sum_stake_sq"] += race_stake * race_stake
        stats["sum_stake_return"] += race_stake * returned

    # Dedicated 2-4-5 formation: top2 / top4 / ranks3-7, 100 yen per
    # unique unordered combination. This is isolated from other ticket types.
    formation = bucket["formation_2_4_5"]
    tickets = build_trio_formation_2_4_5(ranked)
    if tickets:
        race_stake = 100 * len(tickets)
        returned = sum(payout_for("trio", ticket, payout) or 0 for ticket in tickets)
        race_hits = sum(1 for ticket in tickets if payout_for("trio", ticket, payout))
        formation["races"] += 1
        formation["tickets"] += len(tickets)
        formation["stake"] += race_stake
        formation["return"] += returned
        formation["hits"] += race_hits
        formation["hit_races"] += int(race_hits > 0)
        formation["sum_return_sq"] += returned * returned
        formation["sum_stake_sq"] += race_stake * race_stake
        formation["sum_stake_return"] += race_stake * returned

        ticket_returns = [payout_for("trio", ticket, payout) or 0 for ticket in tickets]
        for count, narrowed in bucket["formation_2_4_5_narrowed"].items():
            selected = narrow_trio_by_rank_sum(tickets, ranked, count)
            units = allocate_formation_units(selected, ranked, "equal", budget_units=19)
            selected_returns = [payout_for("trio", ticket, payout) or 0 for ticket in selected]
            funded_hits = sum(1 for unit, pay in zip(units, selected_returns) if unit > 0 and pay > 0)
            narrowed["races"] += 1
            narrowed["tickets"] += len(selected)
            narrowed["funded_tickets"] += sum(1 for unit in units if unit > 0)
            narrowed["hits"] += funded_hits
            narrowed["hit_races"] += int(funded_hits > 0)
            narrowed["stake"] += 1900 if selected else 0
            narrowed["return"] += sum(unit * pay for unit, pay in zip(units, selected_returns))
        for policy in FORMATION_ALLOCATION_STRENGTHS:
            allocation = bucket["formation_2_4_5_allocations"][policy]
            units = allocate_formation_units(tickets, ranked, policy)
            funded_wins = sum(1 for unit, pay in zip(units, ticket_returns) if unit > 0 and pay > 0)
            allocated_return = sum(unit * pay for unit, pay in zip(units, ticket_returns))
            allocation["races"] += 1
            allocation["tickets"] += len(tickets)
            allocation["funded_tickets"] += sum(1 for unit in units if unit > 0)
            allocation["winning_candidates"] += sum(1 for pay in ticket_returns if pay > 0)
            allocation["hits"] += funded_wins
            allocation["hit_races"] += int(funded_wins > 0)
            allocation["stake"] += 100 * sum(units)
            allocation["return"] += allocated_return


def summarize_trio_diagnostics(bucket):
    n = bucket["races"]
    keys = ("all_podium_top4", "all_podium_top5", "all_podium_top6", "all_podium_top7")
    output = {key: bucket[key] for key in ("races", *keys)}
    output["rates"] = {key: round(100 * bucket[key] / n, 2) if n else None for key in keys}
    output["podium_count_top7"] = {
        "counts": dict(bucket["podium_count_top7"]),
        "rates": {key: round(100 * value / n, 2) if n else None
                  for key, value in bucket["podium_count_top7"].items()},
        "average_podium_horses_captured": round(
            sum(int(key) * value for key, value in bucket["podium_count_top7"].items()) / n, 3
        ) if n else None,
    }
    adaptive = bucket["adaptive_pool"]
    adaptive_stake = adaptive["stake"]
    adaptive["hit_rate"] = round(100 * adaptive["hits"] / adaptive["tickets"], 4) if adaptive["tickets"] else None
    adaptive["race_hit_rate"] = round(100 * adaptive["hit_races"] / adaptive["races"], 2) if adaptive["races"] else None
    adaptive["roi"] = round(100 * adaptive["return"] / adaptive_stake, 2) if adaptive_stake else None
    adaptive["profit"] = adaptive["return"] - adaptive_stake
    adaptive["tickets_per_race"] = round(adaptive["tickets"] / adaptive["races"], 2) if adaptive["races"] else None
    adaptive["stake_per_ticket_yen"] = 100
    adaptive["selection_rule"] = "Choose top-4/5/6 when the adjacent score gap at ranks 4/5, 5/6, or 6/7 is at least 0.60 population standard deviations; otherwise top-7. Uses only pre-race scores."
    output["adaptive_pool"] = adaptive
    structures = {}
    for pool, stats in bucket["structures"].items():
        n = stats["races"]
        roi = 100 * stats["return"] / stats["stake"] if stats["stake"] else None
        ci_low = ci_high = None
        if n > 1 and roi is not None:
            ratio = roi / 100
            sum_z_sq = (stats["sum_return_sq"] - 2 * ratio * stats["sum_stake_return"]
                        + ratio * ratio * stats["sum_stake_sq"])
            variance = max(0.0, sum_z_sq / (n - 1))
            mean_stake = stats["stake"] / n
            standard_error = 100 * (variance / n) ** 0.5 / mean_stake if mean_stake else 0
            ci_low, ci_high = round(roi - 1.96 * standard_error, 2), round(roi + 1.96 * standard_error, 2)
        structures[f"top{pool}"] = {
            "races": n, "tickets": stats["tickets"], "hits": stats["hits"],
            "hit_races": stats["hit_races"], "stake": stats["stake"], "return": stats["return"],
            "hit_rate": round(100 * stats["hits"] / stats["tickets"], 4) if stats["tickets"] else None,
            "race_hit_rate": round(100 * stats["hit_races"] / n, 2) if n else None,
            "roi": round(roi, 2) if roi is not None else None,
            "roi_ci95_normal_low": ci_low, "roi_ci95_normal_high": ci_high,
            "roi_variance_components": {
                "sum_return_sq": stats["sum_return_sq"],
                "sum_stake_sq": stats["sum_stake_sq"],
                "sum_stake_return": stats["sum_stake_return"],
            },
            "profit": stats["return"] - stats["stake"],
            "tickets_per_race": round(stats["tickets"] / n, 2) if n else None,
        }
    output["structures"] = structures
    stats = bucket["formation_2_4_5"]
    n = stats["races"]
    roi = 100 * stats["return"] / stats["stake"] if stats["stake"] else None
    ci_low = ci_high = None
    if n > 1 and roi is not None:
        ratio = roi / 100
        sum_z_sq = (stats["sum_return_sq"] - 2 * ratio * stats["sum_stake_return"]
                    + ratio * ratio * stats["sum_stake_sq"])
        variance = max(0.0, sum_z_sq / (n - 1))
        mean_stake = stats["stake"] / n
        standard_error = 100 * (variance / n) ** 0.5 / mean_stake if mean_stake else 0
        ci_low, ci_high = round(roi - 1.96 * standard_error, 2), round(roi + 1.96 * standard_error, 2)
    output["formation_2_4_5"] = {
        **stats,
        "hit_rate": round(100 * stats["hits"] / stats["tickets"], 4) if stats["tickets"] else None,
        "race_hit_rate": round(100 * stats["hit_races"] / n, 2) if n else None,
        "roi": round(roi, 2) if roi is not None else None,
        "roi_ci95_normal_low": ci_low, "roi_ci95_normal_high": ci_high,
        "profit": stats["return"] - stats["stake"],
        "tickets_per_race": round(stats["tickets"] / n, 2) if n else None,
        "formation": "top2 / top4 / ranks3-7",
        "stake_per_ticket_yen": 100,
    }
    allocation_summary = {}
    for policy, allocation in bucket["formation_2_4_5_allocations"].items():
        stake = allocation["stake"]
        funded = allocation["funded_tickets"]
        races = allocation["races"]
        allocation_summary[policy] = {
            **allocation,
            "funded_ticket_hit_rate": round(100 * allocation["hits"] / funded, 4) if funded else None,
            "race_hit_rate": round(100 * allocation["hit_races"] / races, 2) if races else None,
            "roi": round(100 * allocation["return"] / stake, 2) if stake else None,
            "profit": allocation["return"] - stake,
            "funded_tickets_per_race": round(funded / races, 2) if races else None,
            "budget_per_race_matches_equal_policy": True,
        }
    output["formation_2_4_5"]["allocation_policies"] = allocation_summary
    narrowed_summary = {}
    for count, narrowed in bucket["formation_2_4_5_narrowed"].items():
        stake = narrowed["stake"]
        narrowed_summary[str(count)] = {
            **narrowed,
            "ticket_hit_rate": round(100 * narrowed["hits"] / narrowed["funded_tickets"], 4)
                if narrowed["funded_tickets"] else None,
            "race_hit_rate": round(100 * narrowed["hit_races"] / narrowed["races"], 2)
                if narrowed["races"] else None,
            "roi": round(100 * narrowed["return"] / stake, 2) if stake else None,
            "profit": narrowed["return"] - stake,
            "tickets_per_race": round(narrowed["tickets"] / narrowed["races"], 2)
                if narrowed["races"] else None,
            "budget_per_race_jpy": 1900,
            "selection": "lowest sum of the three LAB horse ranks among the original 19 unique 2-4-5 combinations",
            "allocation": "19 equal 100-yen units distributed across the selected tickets",
        }
    output["formation_2_4_5_narrowed"] = narrowed_summary
    output["formation_2_4_5"]["allocation_method"] = (
        "Exploratory rank-weighted allocation with the same total stake per race as "
        "100-yen equal staking. Historical pre-off trio combination odds are unavailable, "
        "so these policies are not expected-value or Kelly staking."
    )
    output["interpretation"] = "Trio tickets are unordered combinations. hit_rate is winning tickets divided by tickets; race_hit_rate is races with at least one winning ticket divided by races. ROI confidence intervals use a race-level normal ratio approximation and are diagnostic, not a promotion guarantee. Top5 is the unchanged fixed baseline. formation_2_4_5 is top2 / top4 / ranks3-7 with duplicate unordered combinations removed. The 12- and 10-ticket subsets are selected from those same 19 combinations by the lowest sum of LAB ranks; each keeps the same 1,900-yen race budget by distributing 19 equal 100-yen units. These are research comparisons, not live betting rules."
    return output


def frame_context_key(race, horse):
    distance = number(race.get("distance"))
    field_size = number(race.get("field_size"))
    frame_no = number(horse.get("frame_no"))
    if frame_no is None:
        return None
    distance_band = int(distance // 400) if distance is not None else "unknown"
    field_band = int(field_size // 4) if field_size is not None else "unknown"
    return (norm(race.get("venue")), norm(race.get("surface")), distance_band, field_band, int(frame_no))


def frame_form_score(race, horse, frame_stats):
    key = frame_context_key(race, horse)
    if key is None:
        return 0.5
    stats = frame_stats.get(key) or {"runs": 0, "top3": 0}
    return smoothed_rate(stats.get("top3", 0), stats.get("runs", 0),
                         prior_rate=0.5, prior_strength=8.0)


def blend_frame_score(base_score, race, horse, frame_stats, frame_score_cache=None):
    # Frame suitability depends on race context and prior frame history, not on
    # the model weight vector. Reuse it across challenger models in this race.
    weight = 0.08 if frame_stats else 0.0
    horse_no = str(horse.get("no", ""))
    frame_score = (frame_score_cache.get(horse_no)
                   if frame_score_cache is not None else None)
    if frame_score is None:
        frame_score = frame_form_score(race, horse, frame_stats)
        if frame_score_cache is not None:
            frame_score_cache[horse_no] = frame_score
    return base_score * (1.0 - weight) + frame_score * weight


def update_frame_stats(race, field, frame_stats):
    for horse in field:
        if not horse.get("finish") or horse.get("frame_no") is None:
            continue
        key = frame_context_key(race, horse)
        if key is None:
            continue
        frame_stats[key]["runs"] += 1
        frame_stats[key]["top3"] += int(horse["finish"] <= 3)


def run_backtest(races, horses, payouts, training_only=False):
    history = defaultdict(list)
    sire_stats = defaultdict(lambda: {"runs":0,"top3":0})
    jockey_stats = defaultdict(lambda: {"runs":0,"top3":0})
    frame_stats = defaultdict(lambda: {"runs": 0, "top3": 0})
    metrics = {k: {"races":0,"tickets":0,"hits":0,"stake":0,"return":0} for k in ("win","place","wide","quinella","exacta","trio","trifecta")}
    examples = []
    calibration_records = []
    model_weights = {"form":0.55,"venue":0.12,"distance":0.12,"sire":0.10,"jockey":0.07,"age":0.04}
    learning_updates = 0
    shadow_metrics = {kind: {"races": 0, "tickets": 0, "hits": 0, "stake": 0, "return": 0}
                      for kind in ("win", "place", "wide", "trio")}
    shadow_races = 0
    ordered = sorted((r for r in races.values()), key=lambda r:(r["date"],r["venue"],int(r["race"])))
    progress_every = max(1, len(ordered) // 10)
    print(f"CHRONOLOGICAL_PROGRESS start races={len(ordered)} dates={len({r['date'] for r in ordered})}", flush=True)
    daily_pending = []
    current_date = None

    def commit_day(items):
        nonlocal learning_updates
        for pending_field, pending_parts, pending_race in items:
            if update_model(pending_field, pending_parts, model_weights):
                learning_updates += 1
            performance = field_performance_features(pending_field)
            for runner in pending_field:
                if runner.get("finish"):
                    history[runner["name"]].append({
                        **performance.get(runner["no"], {}),
                        "date":pending_race["date"],"venue":pending_race["venue"],"distance":pending_race.get("distance"),"going":pending_race.get("going"),"surface":pending_race.get("surface"),"field_size":pending_race.get("field_size"),
                        "finish":runner["finish"],"margin":margin_score(runner.get("margin")),"last3f":runner.get("last3f"),
                        "going_quality": rank_quality(runner["finish"], pending_race.get("field_size")) / 100.0,
                        "time_seconds": runner.get("time_seconds"),
                        "time_per_km": (runner.get("time_seconds") * 1000 / pending_race["distance"]
                                        if runner.get("time_seconds") and pending_race.get("distance") else None),
                        "frame_no": runner.get("frame_no"),
                        **corner_history_for_runner(runner, pending_race)
                    })
                    if runner.get("sire"):
                        sire_stats[runner["sire"]]["runs"] += 1
                        sire_stats[runner["sire"]]["top3"] += int(runner["finish"] <= 3)
                    if runner.get("jockey"):
                        jockey_stats[runner["jockey"]]["runs"] += 1
                        jockey_stats[runner["jockey"]]["top3"] += int(runner["finish"] <= 3)
            update_frame_stats(pending_race, pending_field, frame_stats)

    for race_index, race in enumerate(ordered, 1):
        if race_index == 1 or race_index % progress_every == 0 or race_index == len(ordered):
            print(f"CHRONOLOGICAL_PROGRESS {race_index}/{len(ordered)} ({100*race_index/max(1,len(ordered)):.0f}%) date={race['date']}", flush=True)
        if current_date is not None and race["date"] != current_date:
            commit_day(daily_pending)
            daily_pending = []
        current_date = race["date"]
        key = race["key"]
        field = horses.get(key, [])
        if len(field) < 3: continue
        # Predict BEFORE adding any current-race results to history or trainer/jockey/sire aggregates.
        scored = []
        parts_by_no = {}
        for h in field:
            base_score, parts = score_runner_performance(
                h, history, sire_stats, jockey_stats, race, model_weights,
                margin_weight=0.10, closing_weight=0.10, time_weight=0.10,
                going_weight=0.10,
            )
            score = blend_frame_score(base_score, race, h, frame_stats)
            scored.append((score,h))
            parts_by_no[h["no"]] = parts
        _, _, pace_delta = pace_adjustments(field, history)
        scored = [(score + pace_delta.get(h["no"], 0.0), h) for score, h in scored]
        frame_gradient_scale = 0.92 if frame_stats else 1.0
        for predicted_score, horse in scored:
            feature_parts = parts_by_no[horse["no"]]
            base_share = number(
                (feature_parts.get("performance_weights") or {}).get("base")
            )
            if base_share is None:
                base_share = 1.0
            feature_parts["__prediction_score"] = predicted_score
            feature_parts["__model_gradient_scale"] = base_share * frame_gradient_scale
        winner_indices = [i for i,h in enumerate(field) if h.get("finish") == 1]
        if winner_indices and payouts.get(key) and payouts[key].get("win"):
            calibration_records.append({"date":race["date"].isoformat(), "venue":race["venue"],
                                        "race":race["race"], "race_key":"|".join(key),
                                        "scores":[score for score,_ in scored],
                                        "winner_indices":winner_indices})
        if training_only:
            # Walk-forward fold fitting needs only chronological model updates
            # and calibration scores. Skip ranking, ticket construction, payout
            # matching, and report aggregation; these do not affect learned weights.
            daily_pending.append((field, parts_by_no, race))
            continue
        ranked = [h for score,h in sorted(scored,key=lambda item:item[0],reverse=True)]
        shadow_tickets = build_shadow_portfolio_tickets(ranked)
        t = build_fixed_tickets(ranked)
        if not t: continue
        payout = payouts.get(key)
        if payout and payout.get("win"):
            if shadow_tickets:
                shadow_races += 1
                for ticket in shadow_tickets:
                    kind = ticket["kind"]
                    bucket = shadow_metrics[kind]
                    bucket["races"] += 1
                    bucket["tickets"] += 1
                    bucket["stake"] += 100
                    returned = payout_for(kind, ticket["numbers"], payout)
                    if returned:
                        bucket["hits"] += 1
                        bucket["return"] += returned
            for kind, tickets in t.items():
                m = metrics[kind]
                m["races"] += 1
                for ticket in tickets:
                    m["tickets"] += 1
                    m["stake"] += 100
                    ret = payout_for(kind,ticket,payout)
                    if ret:
                        m["hits"] += 1
                        m["return"] += ret
            if len(examples) < 25:
                examples.append({
                    "date": race["date"].isoformat(), "venue": race["venue"], "race": race["race"],
                    "name": race.get("name",""), "ranked": [{"no":h["no"],"name":h["name"]} for h in ranked[:6]],
                    "tickets": {k:["-".join(x) for x in v] for k,v in t.items()},
                    "winner": [numbers[0] for numbers, _ in payout["win"]],
                })
        # Keep today's results buffered: no race on the same date can learn from another race's result.
        daily_pending.append((field, parts_by_no, race))
    if daily_pending:
        commit_day(daily_pending)
    summary = {}
    for kind,m in metrics.items():
        summary[kind] = {**m, "hit_rate": round(100*m["hits"]/m["tickets"],2) if m["tickets"] else None,
                         "roi": round(100*m["return"]/m["stake"],2) if m["stake"] else None,
                         "profit": m["return"]-m["stake"]}
    shadow_total = {
        "races": shadow_races,
        "bet_types": {
            kind: {**bucket,
                   "hit_rate": round(100 * bucket["hits"] / bucket["tickets"], 2) if bucket["tickets"] else None,
                   "roi": round(100 * bucket["return"] / bucket["stake"], 2) if bucket["stake"] else None,
                   "profit": bucket["return"] - bucket["stake"]}
            for kind, bucket in shadow_metrics.items()
        },
        "policy": "Mirrors production shadowTickets: model rank 1 win; ranks 1-2 place; ranks 1-2 wide; ranks 1-3 trio; 100 JPY each; no odds-based real-bet selection.",
    }
    summary["shadow_portfolio"] = shadow_total
    return summary, examples, len(ordered), model_weights, learning_updates, calibration_records

def build_frame_stats(races, horses):
    stats = defaultdict(lambda: {"runs": 0, "top3": 0})
    ordered = sorted(races.values(), key=lambda r:(r["date"],r["venue"],int(r["race"])))
    for race in ordered:
        update_frame_stats(race, horses.get(race["key"], []), stats)
    return stats


def build_live_history(races, horses):
    horse_history = defaultdict(list)
    sire_stats = defaultdict(lambda: {"runs": 0, "top3": 0})
    jockey_stats = defaultdict(lambda: {"runs": 0, "top3": 0})
    ordered = sorted(races.values(), key=lambda r:(r["date"],r["venue"],int(r["race"])))
    for race in ordered:
        field = horses.get(race["key"], [])
        performance = field_performance_features(field)
        for h in field:
            finish = h.get("finish")
            if not finish:
                continue
            item = {"date": race["date"].isoformat(), "venue": race["venue"],
                    "distance": race.get("distance"), "going": race.get("going"),
                    "surface": race.get("surface"), "field_size": race.get("field_size"), "finish": finish,
                    "going_quality": rank_quality(finish, race.get("field_size")) / 100.0,
                    **performance.get(h["no"], {}),
                    "margin": h.get("margin", ""), "last3f": h.get("last3f"),
                    "time_seconds": h.get("time_seconds"),
                    "time_per_km": (h.get("time_seconds") * 1000 / race["distance"]
                                    if h.get("time_seconds") and race.get("distance") else None),
                    **corner_history_for_runner(h, race)}
            horse_history[norm(h["name"])].append(item)
            if h.get("sire"):
                sire_stats[norm(h["sire"])]["runs"] += 1
                sire_stats[norm(h["sire"])]["top3"] += int(finish <= 3)
            if h.get("jockey"):
                jockey_stats[norm(h["jockey"])]["runs"] += 1
                jockey_stats[norm(h["jockey"])]["top3"] += int(finish <= 3)
    return {
        "training_period": {"start": START.isoformat(), "end": END.isoformat()},
        "horse_history": {name: runs[-5:] for name, runs in horse_history.items()},
        "sire_stats": dict(sire_stats),
        "jockey_stats": dict(jockey_stats),
        "frame_stats": {"|".join(map(str, key)): value for key, value in build_frame_stats(races, horses).items()},
        "note": "Snapshot through period end; only use for target dates after training_period.end. Historical corner-position features are included where available."
    }

def race_segment_labels(race):
    """Stable, source-grounded condition buckets for out-of-sample diagnostics."""
    labels = {}
    for key in ("venue", "surface", "going"):
        value = norm(race.get(key))
        if value:
            labels[key] = value
    distance = number(race.get("distance"))
    if distance is not None and distance > 0:
        if distance <= 1200:
            band = "short_<=1200m"
        elif distance <= 1600:
            band = "mile_1201-1600m"
        elif distance <= 2000:
            band = "middle_1601-2000m"
        else:
            band = "long_>2000m"
        labels["distance_band"] = band
    field_size = number(race.get("field_size"))
    if field_size is not None and field_size > 0:
        if field_size <= 8:
            labels["field_size_band"] = "small_<=8"
        elif field_size <= 12:
            labels["field_size_band"] = "medium_9-12"
        else:
            labels["field_size_band"] = "large_>12"
    return labels


def new_condition_finish_bucket():
    return {
        str(role): {
            ranking: {"races": 0, "topk": {str(k): 0 for k in (1, 2, 3, 5, 7)}}
            for ranking in ("ordinary", "role_specific")
        }
        for role in (1, 2, 3)
    }


def update_condition_finish_bucket(bucket, ordinary_ranking, role_rankings, field):
    finishers = {
        role: {str(h["no"]) for h in field if h.get("finish") == role}
        for role in (1, 2, 3)
    }
    rankings = {
        "ordinary": {role: ordinary_ranking for role in (1, 2, 3)},
        "role_specific": role_rankings,
    }
    for role in (1, 2, 3):
        targets = finishers[role]
        # Do not count a race in a role's denominator when the official
        # finishing position is absent (e.g. incomplete/void result rows).
        if not targets:
            continue
        for ranking_name, by_role in rankings.items():
            stats = bucket[str(role)][ranking_name]
            stats["races"] += 1
            positions = [i + 1 for i, horse in enumerate(by_role[role])
                         if str(horse["no"]) in targets]
            for k in (1, 2, 3, 5, 7):
                stats["topk"][str(k)] += int(any(position <= k for position in positions))


def summarize_condition_finish_segments(stats):
    output = {}
    for segment, roles in sorted(stats.items()):
        output[segment] = {}
        for role, rankings in roles.items():
            output[segment][role] = {}
            for ranking, bucket in rankings.items():
                races = bucket["races"]
                output[segment][role][ranking] = {
                    "races": races,
                    "topk_hits": bucket["topk"],
                    "topk_rate": {
                        k: round(100 * value / races, 2) if races else None
                        for k, value in bucket["topk"].items()
                    },
                }
    return output


CONFIDENCE_THRESHOLDS = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)

def new_confidence_filter_stats():
    return {
        f"{threshold:.2f}": {
            "races": 0, "winner_top1": 0, "winner_top3": 0,
            "bet_types": {
                kind: {"tickets": 0, "hits": 0, "stake": 0, "return": 0}
                for kind in ("win", "place", "wide", "quinella", "exacta", "trio", "trifecta")
            }
        }
        for threshold in CONFIDENCE_THRESHOLDS
    }


def update_confidence_filter_stats(stats, top_probability, ranked, winners, ticket_map, payout):
    """Evaluate fixed tickets only when calibrated top-win probability clears a preset threshold."""
    for threshold in CONFIDENCE_THRESHOLDS:
        if top_probability + 1e-12 < threshold:
            continue
        bucket = stats[f"{threshold:.2f}"]
        bucket["races"] += 1
        bucket["winner_top1"] += int(bool(ranked) and ranked[0]["no"] in winners)
        bucket["winner_top3"] += int(any(h["no"] in winners for h in ranked[:3]))
        for kind, tickets in ticket_map.items():
            target = bucket["bet_types"][kind]
            for ticket in tickets:
                target["tickets"] += 1
                target["stake"] += 100
                returned = payout_for(kind, ticket, payout) or 0
                if returned:
                    target["hits"] += 1
                    target["return"] += returned


def summarize_confidence_filter_stats(stats):
    output = {}
    for threshold, bucket in stats.items():
        bet_types = {}
        for kind, values in bucket["bet_types"].items():
            bet_types[kind] = {
                **values,
                "hit_rate": round(100 * values["hits"] / values["tickets"], 2) if values["tickets"] else None,
                "roi": round(100 * values["return"] / values["stake"], 2) if values["stake"] else None,
                "profit": values["return"] - values["stake"],
            }
        output[threshold] = {
            "races": bucket["races"],
            "winner_top1_rate": round(100 * bucket["winner_top1"] / bucket["races"], 2) if bucket["races"] else None,
            "winner_top3_rate": round(100 * bucket["winner_top3"] / bucket["races"], 2) if bucket["races"] else None,
            "bet_types": bet_types,
        }
    return output



def summarize_condition_model_segments(stats, kinds=("wide", "exacta", "trio")):
    """Summarize baseline/challenger condition metrics for each requested bet type."""
    segments = {
        kind: {"baseline": {}, "trained_no_recent_history": {}}
        for kind in kinds
    }
    for (model_name, kind, dimension, label), bucket in sorted(stats.items()):
        if kind not in segments or model_name not in segments[kind]:
            continue
        roi = round(100 * bucket["return"] / bucket["stake"], 2) if bucket["stake"] else None
        hit_rate = round(100 * bucket["hits"] / bucket["tickets"], 2) if bucket["tickets"] else None
        segments[kind][model_name][f"{dimension}:{label}"] = {
            **bucket, "roi": roi, "hit_rate": hit_rate,
            "profit": bucket["return"] - bucket["stake"]
        }
    comparisons = {}
    for kind in kinds:
        comparison = []
        baseline_segments = segments[kind]["baseline"]
        candidate_segments = segments[kind]["trained_no_recent_history"]
        for segment_key in sorted(set(baseline_segments) | set(candidate_segments)):
            baseline_bucket = baseline_segments.get(segment_key)
            candidate_bucket = candidate_segments.get(segment_key)
            baseline_roi = baseline_bucket.get("roi") if baseline_bucket else None
            candidate_roi = candidate_bucket.get("roi") if candidate_bucket else None
            comparison.append({
                "segment": segment_key,
                "baseline_races": baseline_bucket.get("races", 0) if baseline_bucket else 0,
                "candidate_races": candidate_bucket.get("races", 0) if candidate_bucket else 0,
                "baseline_tickets": baseline_bucket.get("tickets", 0) if baseline_bucket else 0,
                "candidate_tickets": candidate_bucket.get("tickets", 0) if candidate_bucket else 0,
                "baseline_hits": baseline_bucket.get("hits", 0) if baseline_bucket else 0,
                "candidate_hits": candidate_bucket.get("hits", 0) if candidate_bucket else 0,
                "baseline_hit_rate": baseline_bucket.get("hit_rate") if baseline_bucket else None,
                "candidate_hit_rate": candidate_bucket.get("hit_rate") if candidate_bucket else None,
                "baseline_roi": baseline_roi, "candidate_roi": candidate_roi,
                "roi_delta": round(candidate_roi - baseline_roi, 2)
                    if baseline_roi is not None and candidate_roi is not None else None,
                "baseline_profit": baseline_bucket.get("profit") if baseline_bucket else None,
                "candidate_profit": candidate_bucket.get("profit") if candidate_bucket else None,
            })
        comparisons[kind] = comparison
    return segments, comparisons



def summarize_race_cluster_roi(stats):
    """Estimate ROI uncertainty using races, not individual tickets, as clusters."""
    races = int(stats.get("race_count") or stats.get("races") or 0)
    tickets = int(stats.get("tickets") or 0)
    stake = float(stats.get("stake") or 0)
    returned = float(stats.get("return") or 0)
    roi = 100.0 * returned / stake if stake else None
    ci_low = ci_high = None
    if races > 1 and stake > 0 and roi is not None:
        ratio = returned / stake
        sum_z_sq = (
            float(stats.get("sum_return_sq") or 0)
            - 2.0 * ratio * float(stats.get("sum_stake_return") or 0)
            + ratio * ratio * float(stats.get("sum_stake_sq") or 0)
        )
        variance = max(0.0, sum_z_sq / (races - 1))
        mean_stake = stake / races
        standard_error = 100.0 * math.sqrt(variance / races) / mean_stake
        ci_low = round(roi - 1.96 * standard_error, 2)
        ci_high = round(roi + 1.96 * standard_error, 2)
    enough_races = races >= 100
    enough_tickets = tickets >= 300
    roi_above_100 = roi is not None and roi > 100.0
    ci_lower_above_100 = ci_low is not None and ci_low > 100.0
    pass_conditions = {
        "roi_above_100": roi_above_100,
        "races_at_least_100": enough_races,
        "tickets_at_least_300": enough_tickets,
        "roi_ci95_lower_above_100": ci_lower_above_100,
    }
    candidate_pass = all(pass_conditions.values())
    return {
        "race_count": races,
        "roi": round(roi, 2) if roi is not None else None,
        "roi_ci95_method": "race_cluster_normal_approximation",
        "roi_ci95_normal_low": ci_low,
        "roi_ci95_normal_high": ci_high,
        "minimum_sample_met": bool(enough_races and enough_tickets),
        "pass_conditions": pass_conditions,
        "candidate_pass": candidate_pass,
        "profit": returned - stake,
    }

def evaluate_holdout_fold(races, horses, payouts, train_dates, test_dates, final_odds=None):
    train_set = set(train_dates)
    test_set = set(test_dates)
    train_races = {k:r for k,r in races.items() if r["date"] in train_set}
    test_races = {k:r for k,r in races.items() if r["date"] in test_set}
    train_horses = {k:v for k,v in horses.items() if k in train_races}
    train_payouts = {k:v for k,v in payouts.items() if k in train_races}
    _, _, _, learned, updates, train_calibration_records = run_backtest(
        train_races, train_horses, train_payouts, training_only=True)
    temperature = fit_temperature(train_calibration_records)
    baseline = {"form":0.55,"venue":0.12,"distance":0.12,"sire":0.10,"jockey":0.07,"age":0.04}
    # Keep the learned B weights normalized to 1.0. The performance scorer
    # applies the 0.80/0.10/0.10 blend itself; pre-scaling these weights would
    # accidentally reduce the B contribution to 0.64.
    ablation_features = ("form", "venue", "distance", "sire", "jockey", "age")
    models = {"A_baseline":baseline, "B_condition":learned, "C_performance_pace":learned,
              "C_performance_pace_no_frame":learned,
              "baseline":baseline, "trained":learned, "trained_no_recent_history":learned,
              "trained_race_relative":learned,
              "trained_with_pace":learned, "trained_with_performance":learned, "trained_with_time":learned,
              "trained_performance_with_pace":learned, "trained_with_conditions":learned,
              "trained_with_conditions_and_pace":learned,
              "trained_margin_only":learned, "trained_closing_only":learned,
              "trained_performance_light":learned, "trained_performance_with_time":learned}
    # One-feature-at-a-time ablations use the learned B weights with only the
    # named feature removed and the remaining weights renormalized. This is a
    # diagnostic sensitivity test, not a separately refitted candidate model.
    models.update({f"ablate_{feature}": learned for feature in ablation_features})
    weight_variant_names = FEATURE_WEIGHT_NAMES
    models.update({
        name: scaled_model_weights(learned, feature, multiplier)
        for name, (feature, multiplier) in FEATURE_WEIGHT_VARIANTS.items()
    })
    models.update({
        name: combined_scaled_model_weights(learned, adjustments)
        for name, adjustments in FEATURE_WEIGHT_COMBINATIONS.items()
    })
    performance_blends = {
        "trained_with_performance": (0.10, 0.10, 0.00),
        "trained_performance_with_pace": (0.10, 0.10, 0.00),
        "trained_margin_only": (0.10, 0.00, 0.00),
        "trained_closing_only": (0.00, 0.10, 0.00),
        "trained_performance_light": (0.05, 0.05, 0.00),
        "trained_with_time": (0.00, 0.00, 0.10),
        "trained_performance_with_time": (0.07, 0.08, 0.10),
    }
    kinds = ("win","place","wide","quinella","exacta","trio","trifecta")
    results = {name:{"races":0,"winner_top1":0,"winner_top3":0,
                     "winner_topk":{str(k):0 for k in (5,6,7)},
                     "bet_types":{kind:{"tickets":0,"hits":0,"stake":0,"return":0,
                                         "race_count":0,"sum_return_sq":0,
                                         "sum_stake_sq":0,"sum_stake_return":0}
                                  for kind in kinds},
                     "shadow_portfolio":{"races":0,"bet_types":{kind:{"tickets":0,"hits":0,"stake":0,"return":0}
                                  for kind in ("win","place","wide","trio")}},
                     "trifecta_diagnostics":new_trifecta_diagnostics(),
                     "role_trifecta_diagnostics":new_role_trifecta_diagnostics(),
                     "finish_position_diagnostics":new_finish_position_diagnostics(),
                     "trio_diagnostics":new_trio_diagnostics()} for name in models}

    history = defaultdict(list)
    sire_stats = defaultdict(lambda:{"runs":0,"top3":0})
    jockey_stats = defaultdict(lambda:{"runs":0,"top3":0})
    # Build frame context in the same pass as the other fold training history.
    # build_frame_stats() used to make a separate full traversal of every race.
    frame_stats = defaultdict(lambda: {"runs": 0, "top3": 0})
    train_ordered = sorted(train_races.values(),
                           key=lambda r:(r["date"],r["venue"],int(r["race"])))
    train_progress_every = max(1, len(train_ordered) // 10)
    print(f"HOLDOUT_PROGRESS training start races={len(train_ordered)} dates={len(train_dates)}", flush=True)
    for train_index, train_race in enumerate(train_ordered, 1):
        if train_index == 1 or train_index % train_progress_every == 0 or train_index == len(train_ordered):
            print(f"HOLDOUT_PROGRESS training {train_index}/{len(train_ordered)} ({100*train_index/max(1,len(train_ordered)):.0f}%) date={train_race['date']}", flush=True)
        train_field = train_horses.get(train_race["key"], [])
        update_frame_stats(train_race, train_field, frame_stats)
        train_performance = field_performance_features(train_field)
        for h in train_field:
            if not h.get("finish"):
                continue
            history[h["name"]].append({"date":train_race["date"],"venue":train_race["venue"],
                **train_performance.get(h["no"], {}),
                "distance":train_race.get("distance"),"field_size":train_race.get("field_size"),"finish":h["finish"],
                "margin":margin_score(h.get("margin")),"last3f":h.get("last3f"),
                "time_seconds": h.get("time_seconds"),
                "time_per_km": (h.get("time_seconds") * 1000 / train_race["distance"]
                                if h.get("time_seconds") and train_race.get("distance") else None),
                **corner_history_for_runner(h, train_race)})
            if h.get("sire"):
                sire_stats[h["sire"]]["runs"] += 1
                sire_stats[h["sire"]]["top3"] += int(h["finish"] <= 3)
            if h.get("jockey"):
                jockey_stats[h["jockey"]]["runs"] += 1
                jockey_stats[h["jockey"]]["top3"] += int(h["finish"] <= 3)

    ordered = sorted(test_races.values(), key=lambda r:(r["date"],r["venue"],int(r["race"])))
    test_progress_every = max(1, len(ordered) // 10)
    print(f"HOLDOUT_PROGRESS evaluation start races={len(ordered)} dates={len(test_dates)} models={len(models)}", flush=True)
    pending=[]; current=None
    test_calibration_records=[]
    confidence_filter_stats = new_confidence_filter_stats()
    # E: pre-registered integrated challenger. The confidence gate and role blend
    # are fixed before holdout evaluation; no test-fold threshold tuning is allowed.
    market_comparison_names = ("C_model_top_pick", "final_odds_favorite",
                               *(f"ablate_{feature}" for feature in ablation_features),
                               *FEATURE_WEIGHT_NAMES)
    final_market_stats = {name: {"races":0,"top1_hits":0,"hits":0,"tickets":0,"stake":0,"return":0,"race_count":0,"sum_return_sq":0,"sum_stake_sq":0,"sum_stake_return":0} for name in market_comparison_names}
    market_rank_diagnostics = {name: new_market_rank_diagnostics() for name in ("C_performance_pace", "D4_jockey_venue_distance")}
    integrated_stats = {
        "confidence_threshold": 0.30,
        "role_blend": 0.10,
        "trifecta_formation": [2, 3, 4],
        "races": 0,
        "bet_types": {
            kind: {"tickets": 0, "hits": 0, "stake": 0, "return": 0,
                   "race_count": 0, "sum_return_sq": 0,
                   "sum_stake_sq": 0, "sum_stake_return": 0}
            for kind in ("wide", "exacta", "trio", "trifecta")
        },
        # Same qualifying races and same 2-3-4 ticket shape, but with the
        # ordinary single ranking. This isolates the effect of role ranking.
        "trifecta_same_races_baseline_reference": {
            "tickets": 0, "hits": 0, "stake": 0, "return": 0,
            "race_count": 0, "sum_return_sq": 0,
            "sum_stake_sq": 0, "sum_stake_return": 0,
        },
    }
    segment_stats = defaultdict(lambda: {
        "races": 0, "winner_top1": 0, "winner_top3": 0,
        "bet_types": {kind: {"tickets": 0, "hits": 0, "stake": 0, "return": 0}
                      for kind in ("win","place","wide","quinella","exacta","trio","trifecta")}
    })
    # Compare the baseline and challenger on wide, exacta (馬単), and trio
    # (3連複), stratified by venue/surface/going/distance.
    condition_model_segment_stats = defaultdict(lambda: {
        "races": 0, "tickets": 0, "hits": 0, "stake": 0, "return": 0
    })
    condition_bet_types = ("wide", "exacta", "trio", "trifecta")
    condition_finish_stats = defaultdict(new_condition_finish_bucket)
    def commit(items):
        for field,race in items:
            performance = field_performance_features(field)
            for h in field:
                if not h.get("finish"): continue
                history[h["name"]].append({"date":race["date"],"venue":race["venue"],
                    **performance.get(h["no"], {}),
                    "distance":race.get("distance"),"going":race.get("going"),"surface":race.get("surface"),
                    "going_quality": rank_quality(h["finish"], race.get("field_size")) / 100.0,
                    "field_size":race.get("field_size"),"finish":h["finish"],
                    "margin":margin_score(h.get("margin")),"last3f":h.get("last3f"),
                    "time_seconds": h.get("time_seconds"),
                    "time_per_km": (h.get("time_seconds") * 1000 / race["distance"]
                                    if h.get("time_seconds") and race.get("distance") else None),
                    **corner_history_for_runner(h, race)})
                if h.get("sire"):
                    sire_stats[h["sire"]]["runs"]+=1
                    sire_stats[h["sire"]]["top3"]+=int(h["finish"]<=3)
                if h.get("jockey"):
                    jockey_stats[h["jockey"]]["runs"]+=1
                    jockey_stats[h["jockey"]]["top3"]+=int(h["finish"]<=3)
            update_frame_stats(race, field, frame_stats)

    for race_index, race in enumerate(ordered, 1):
        if race_index == 1 or race_index % test_progress_every == 0 or race_index == len(ordered):
            print(f"HOLDOUT_PROGRESS evaluation {race_index}/{len(ordered)} ({100*race_index/max(1,len(ordered)):.0f}%) date={race['date']}", flush=True)
        if current is not None and race["date"] != current:
            commit(pending); pending=[]
        current=race["date"]
        key=race["key"]; field=horses.get(key,[])
        # Add every race to the daily buffer, regardless of payout availability.
        pending.append((field,race))
        if len(field)<3: continue
        payout=payouts.get(key)
        if not payout or not payout.get("win"): continue
        winners={numbers[0] for numbers, _ in payout.get("win", [])}
        if not winners: continue
        race_feature_cache = {}
        race_performance_cache = {}
        race_frame_score_cache = {}
        # Pace profiles depend on the race and strictly prior history, not on
        # model weights. Calculate once and reuse across all challenger models.
        _, _, race_pace_delta = pace_adjustments(field, history)
        for model_name,weights in models.items():
            include_recent_history = model_name != "trained_no_recent_history"
            if model_name == "trained_race_relative":
                relative_parts = {
                    h["no"]: score_runner(h,history,sire_stats,jockey_stats,race,weights,
                                          include_recent_history=include_recent_history,
                                          parts_cache=race_feature_cache)[1]
                    for h in field
                }
                relative_scores = race_relative_scores(relative_parts, weights)
                scored = [(relative_scores[h["no"]], h) for h in field]
            elif model_name.startswith("ablate_"):
                excluded = model_name.removeprefix("ablate_")
                reduced_weights = ablated_model_weights(weights, excluded)
                scored = [
                    (score_runner_performance(
                        h, history, sire_stats, jockey_stats, race, reduced_weights,
                        0.10, 0.10, 0.10, include_recent_history, going_weight=0.10,
                        parts_cache=race_feature_cache, performance_cache=race_performance_cache)[0], h)
                    for h in field
                ]
            elif model_name in ("B_condition", "trained_with_conditions", "trained_with_conditions_and_pace"):
                scored=[(score_runner_performance(h,history,sire_stats,jockey_stats,race,weights,
                            0.0,0.0,0.0,include_recent_history,going_weight=0.10,
                            parts_cache=race_feature_cache, performance_cache=race_performance_cache)[0],h) for h in field]
            elif model_name in ("C_performance_pace", "C_performance_pace_no_frame") or model_name in FEATURE_WEIGHT_NAMES:
                scored=[]
                for h in field:
                    performance_score = score_runner_performance(
                        h, history, sire_stats, jockey_stats, race, weights,
                        0.10, 0.10, 0.10, include_recent_history, going_weight=0.10,
                        parts_cache=race_feature_cache, performance_cache=race_performance_cache)[0]
                    if model_name == "C_performance_pace" or model_name in FEATURE_WEIGHT_NAMES:
                        performance_score = blend_frame_score(
                            performance_score, race, h, frame_stats,
                            frame_score_cache=race_frame_score_cache)
                    scored.append((performance_score, h))
            elif model_name in performance_blends:
                margin_weight, closing_weight, time_weight = performance_blends[model_name]
                scored=[(score_runner_performance(h,history,sire_stats,jockey_stats,race,weights,
                            margin_weight,closing_weight,time_weight,include_recent_history,
                            parts_cache=race_feature_cache, performance_cache=race_performance_cache)[0],h) for h in field]
            else:
                scored=[(score_runner(h,history,sire_stats,jockey_stats,race,weights,
                            include_recent_history=include_recent_history,
                            parts_cache=race_feature_cache)[0],h) for h in field]
            if model_name in ("C_performance_pace", "C_performance_pace_no_frame", "trained_with_pace", "trained_performance_with_pace", "trained_with_conditions_and_pace") or model_name.startswith("ablate_") or model_name in FEATURE_WEIGHT_NAMES:
                scored = [(score + race_pace_delta.get(h["no"], 0.0), h) for score, h in scored]
            if model_name == "trained":
                winner_indices=[i for i,h in enumerate(field) if h.get("finish") == 1]
                if winner_indices:
                    test_calibration_records.append({"scores":[score for score,_ in scored],
                                                     "winner_indices":winner_indices})
            ranked=[h for _,h in sorted(scored,key=lambda x:x[0],reverse=True)]
            if (model_name == "C_performance_pace" or model_name.startswith("ablate_") or model_name in FEATURE_WEIGHT_NAMES) and final_odds:
                odds_for_race = final_odds.get(key, {})
                field_numbers = {str(h["no"]) for h in field}
                available_market = [(float(odds), str(no)) for no, odds in odds_for_race.items()
                                    if str(no) in field_numbers]
                if available_market:
                    if model_name in ("C_performance_pace", "D4_jockey_venue_distance"):
                        update_market_rank_diagnostics(
                            market_rank_diagnostics[model_name], ranked, field, odds_for_race)
                    favorite_no = min(available_market, key=lambda item: (item[0], int(item[1])))[1]
                    selected_label = ("C_model_top_pick" if model_name == "C_performance_pace"
                                      else model_name)
                    selected_no = str(ranked[0]["no"])
                    selected_bucket = final_market_stats[selected_label]
                    selected_bucket["races"] += 1
                    selected_bucket["tickets"] += 1
                    selected_bucket["stake"] += 100
                    selected_bucket["race_count"] += 1
                    is_hit = selected_no in winners
                    selected_bucket["top1_hits"] += int(is_hit)
                    selected_bucket["hits"] += int(is_hit)
                    returned = payout_for("win", (selected_no,), payout) or 0
                    selected_bucket["return"] += returned
                    selected_bucket["sum_return_sq"] += returned * returned
                    selected_bucket["sum_stake_sq"] += 10000
                    selected_bucket["sum_stake_return"] += 100 * returned
                    # Count the market favorite once, on the shared C-model race set.
                    if model_name == "C_performance_pace":
                        market_bucket = final_market_stats["final_odds_favorite"]
                        market_bucket["races"] += 1
                        market_bucket["tickets"] += 1
                        market_bucket["stake"] += 100
                        market_bucket["race_count"] += 1
                        market_hit = favorite_no in winners
                        market_bucket["top1_hits"] += int(market_hit)
                        market_bucket["hits"] += int(market_hit)
                        market_return = payout_for("win", (favorite_no,), payout) or 0
                        market_bucket["return"] += market_return
                        market_bucket["sum_return_sq"] += market_return * market_return
                        market_bucket["sum_stake_sq"] += 10000
                        market_bucket["sum_stake_return"] += 100 * market_return

            if len(ranked)<3: continue
            if model_name == "trained":
                calibrated_probabilities = softmax_probabilities([score for score, _ in scored], temperature)
                top_probability = max(calibrated_probabilities) if calibrated_probabilities else 0.0
            else:
                top_probability = None
            m=results[model_name]; m["races"]+=1
            shadow = m["shadow_portfolio"]
            shadow_tickets = build_shadow_portfolio_tickets(ranked)
            if shadow_tickets:
                shadow["races"] += 1
                for ticket in shadow_tickets:
                    bucket = shadow["bet_types"][ticket["kind"]]
                    bucket["tickets"] += 1
                    bucket["stake"] += 100
                    returned = payout_for(ticket["kind"], ticket["numbers"], payout)
                    if returned:
                        bucket["hits"] += 1
                        bucket["return"] += returned
            if ranked[0]["no"] in winners: m["winner_top1"]+=1
            if any(h["no"] in winners for h in ranked[:3]): m["winner_top3"]+=1
            for k, hit in top_k_winner_coverage(ranked, winners).items():
                m["winner_topk"][k] += hit
            ticket_map=build_fixed_tickets(ranked)
            update_trifecta_diagnostics(m["trifecta_diagnostics"], ranked, field, payout)
            # Collect role-specific trifecta diagnostics for every candidate model.
            # Walk-forward selection may choose baseline or any trained variant;
            # restricting this to "trained" leaves the selected model's report empty.
            role_rankings_cache = {}
            role_features_cache = {}
            update_role_trifecta_diagnostics(
                m["role_trifecta_diagnostics"], scored, history, payout,
                role_rankings_cache=role_rankings_cache,
                role_features_cache=role_features_cache)
            role_rankings_for_positions = role_rankings_cache["0.10"]
            update_finish_position_diagnostics(
                m["finish_position_diagnostics"], ranked, role_rankings_for_positions, field)
            update_trio_diagnostics(m["trio_diagnostics"], ranked, field, payout, scored)
            if model_name == "trained":
                role_segment_rankings = role_rankings_cache["0.10"]
                for dimension, label in race_segment_labels(race).items():
                    update_condition_finish_bucket(
                        condition_finish_stats[f"{dimension}:{label}"],
                        ranked, role_segment_rankings, field)
            if model_name in ("baseline", "trained_no_recent_history"):
                for dimension, label in race_segment_labels(race).items():
                    for condition_kind in condition_bet_types:
                        condition_bucket = condition_model_segment_stats[
                            (model_name, condition_kind, dimension, label)]
                        condition_bucket["races"] += 1
                        for condition_ticket in ticket_map.get(condition_kind, []):
                            condition_bucket["tickets"] += 1
                            condition_bucket["stake"] += 100
                            condition_return = payout_for(
                                condition_kind, condition_ticket, payout) or 0
                            if condition_return:
                                condition_bucket["hits"] += 1
                                condition_bucket["return"] += condition_return
            if model_name == "trained":
                update_confidence_filter_stats(confidence_filter_stats, top_probability,
                                               ranked, winners, ticket_map, payout)
                # Integrated E: apply the fixed confidence gate to B's calibrated
                # probability, retain C's role-specific trifecta pools, and use
                # the same wide/exacta/trio ticket definitions as the baseline.
                if top_probability + 1e-12 >= integrated_stats["confidence_threshold"]:
                    role_blend = integrated_stats["role_blend"]
                    role_rankings = (
                        role_rankings_cache[f"{role_blend:.2f}"]
                        if f"{role_blend:.2f}" in role_rankings_cache else {
                            target: rank_role_specific_features(
                                role_features_cache[target], role_blend)
                            for target in (1, 2, 3)
                        }
                    )
                    e_tickets = dict(ticket_map)
                    e_tickets["trifecta"] = build_role_trifecta_formation(
                        role_rankings, *integrated_stats["trifecta_formation"])
                    # Fair same-race reference: same confidence gate, same 2-3-4
                    # formation and 100-yen unit stake, changing only the ranking.
                    reference_tickets = build_trifecta_formation(
                        ranked, *integrated_stats["trifecta_formation"])
                    reference_bucket = integrated_stats["trifecta_same_races_baseline_reference"]
                    reference_stake = 100 * len(reference_tickets)
                    reference_return = sum(
                        payout_for("trifecta", ticket, payout) or 0
                        for ticket in reference_tickets)
                    reference_hits = sum(
                        1 for ticket in reference_tickets
                        if payout_for("trifecta", ticket, payout))
                    reference_bucket["tickets"] += len(reference_tickets)
                    reference_bucket["hits"] += reference_hits
                    reference_bucket["stake"] += reference_stake
                    reference_bucket["return"] += reference_return
                    reference_bucket["race_count"] += 1
                    reference_bucket["sum_return_sq"] += reference_return * reference_return
                    reference_bucket["sum_stake_sq"] += reference_stake * reference_stake
                    reference_bucket["sum_stake_return"] += reference_stake * reference_return
                    integrated_stats["races"] += 1
                    for e_kind, e_ticket_list in e_tickets.items():
                        e_bucket = integrated_stats["bet_types"][e_kind]
                        e_race_stake = 100 * len(e_ticket_list)
                        e_race_return = 0
                        for e_ticket in e_ticket_list:
                            e_return = payout_for(e_kind, e_ticket, payout) or 0
                            e_bucket["tickets"] += 1
                            e_bucket["stake"] += 100
                            if e_return:
                                e_bucket["hits"] += 1
                                e_bucket["return"] += e_return
                                e_race_return += e_return
                        e_bucket["race_count"] += 1
                        e_bucket["sum_return_sq"] += e_race_return * e_race_return
                        e_bucket["sum_stake_sq"] += e_race_stake * e_race_stake
                        e_bucket["sum_stake_return"] += e_race_stake * e_race_return
                winners_in_race = winners
                labels = race_segment_labels(race)
                for dimension, label in labels.items():
                    segment = segment_stats[(dimension, label)]
                    segment["races"] += 1
                    segment["winner_top1"] += int(ranked[0]["no"] in winners_in_race)
                    segment["winner_top3"] += int(any(h["no"] in winners_in_race for h in ranked[:3]))
                    for segment_kind, segment_tickets in ticket_map.items():
                        bucket = segment["bet_types"][segment_kind]
                        for segment_ticket in segment_tickets:
                            bucket["tickets"] += 1
                            bucket["stake"] += 100
                            segment_return = payout_for(segment_kind, segment_ticket, payout) or 0
                            if segment_return:
                                bucket["hits"] += 1
                                bucket["return"] += segment_return
            for kind,tickets in ticket_map.items():
                bm=m["bet_types"][kind]
                race_stake = 100 * len(tickets)
                race_return = 0
                for ticket in tickets:
                    bm["tickets"]+=1; bm["stake"]+=100
                    ret=payout_for(kind,ticket,payout) or 0
                    if ret:
                        bm["hits"]+=1; bm["return"]+=ret
                        race_return += ret
                bm["race_count"] += 1
                bm["sum_return_sq"] += race_return * race_return
                bm["sum_stake_sq"] += race_stake * race_stake
                bm["sum_stake_return"] += race_stake * race_return

    if pending: commit(pending)
    out={}
    for name,m in results.items():
        n=m["races"] or 1
        bet_summary={}
        for kind,bm in m["bet_types"].items():
            bet_summary[kind]={**bm,
                "hit_rate":round(100*bm["hits"]/bm["tickets"],2) if bm["tickets"] else None,
                **summarize_race_cluster_roi(bm)}
        out[name]={**m,"bet_types":bet_summary,
          "trifecta_diagnostics":summarize_trifecta_diagnostics(m["trifecta_diagnostics"]),
          "role_trifecta_diagnostics":summarize_role_trifecta_diagnostics(m["role_trifecta_diagnostics"]),
          "finish_position_diagnostics":summarize_finish_position_diagnostics(m["finish_position_diagnostics"]),
          "trio_diagnostics":summarize_trio_diagnostics(m["trio_diagnostics"]),
          "winner_top1_rate":round(100*m["winner_top1"]/n,2),
          "winner_top3_rate":round(100*m["winner_top3"]/n,2),
          "winner_topk_rate":{k:round(100*v/n,2) for k,v in m["winner_topk"].items()},
          "win_stake":bet_summary["win"]["stake"],"win_return":bet_summary["win"]["return"],
          "place_stake":bet_summary["place"]["stake"],"place_return":bet_summary["place"]["return"],
          "wide_stake":bet_summary["wide"]["stake"],"wide_return":bet_summary["wide"]["return"],
          "win_roi":bet_summary["win"]["roi"],"place_roi":bet_summary["place"]["roi"],
          "wide_roi":bet_summary["wide"]["roi"]}
    segment_output = {}
    for (dimension, label), segment in sorted(segment_stats.items()):
        n = segment["races"] or 1
        bet_types = {}
        for kind, bucket in segment["bet_types"].items():
            bet_types[kind] = {
                **bucket,
                "hit_rate": round(100 * bucket["hits"] / bucket["tickets"], 2) if bucket["tickets"] else None,
                "roi": round(100 * bucket["return"] / bucket["stake"], 2) if bucket["stake"] else None,
                "profit": bucket["return"] - bucket["stake"],
            }
        segment_output[f"{dimension}:{label}"] = {
            "dimension": dimension, "label": label, "races": segment["races"],
            "winner_top1_rate": round(100 * segment["winner_top1"] / n, 2),
            "winner_top3_rate": round(100 * segment["winner_top3"] / n, 2),
            "bet_types": bet_types,
        }
    condition_segments, condition_comparisons = summarize_condition_model_segments(
        condition_model_segment_stats, condition_bet_types)
    condition_finish_segments = summarize_condition_finish_segments(condition_finish_stats)
    integrated_bets = {}
    for kind, bucket in integrated_stats["bet_types"].items():
        integrated_bets[kind] = {
            **bucket,
            "hit_rate": round(100 * bucket["hits"] / bucket["tickets"], 2)
                if bucket["tickets"] else None,
            **summarize_race_cluster_roi(bucket),
        }
    reference_bucket = integrated_stats["trifecta_same_races_baseline_reference"]
    reference_summary = {
        **reference_bucket,
        "hit_rate": round(100 * reference_bucket["hits"] / reference_bucket["tickets"], 2)
            if reference_bucket["tickets"] else None,
        **summarize_race_cluster_roi(reference_bucket),
        "tickets_per_race": round(reference_bucket["tickets"] / reference_bucket["race_count"], 2)
            if reference_bucket["race_count"] else None,
        "formation": [2, 3, 4],
        "ranking": "ordinary single ranking",
        "same_races_as_E": True,
    }
    e_trifecta = integrated_bets.get("trifecta", {})
    integrated_summary = {
        "status": "evaluated" if integrated_stats["races"] else "no_races_passed_confidence_gate",
        "confidence_threshold": integrated_stats["confidence_threshold"],
        "role_blend": integrated_stats["role_blend"],
        "trifecta_formation": integrated_stats["trifecta_formation"],
        "races": integrated_stats["races"],
        "bet_types": integrated_bets,
        "trifecta_same_races_baseline_reference": reference_summary,
        "trifecta_role_ranking_roi_delta": round(
            (e_trifecta.get("roi") or 0) - (reference_summary.get("roi") or 0), 2)
            if e_trifecta.get("roi") is not None and reference_summary.get("roi") is not None else None,
        "policy": "Fixed 0.30 calibrated top-win probability gate; role-specific trifecta ranking blend 0.10; 2-3-4 formation. The reference uses the same qualifying races and formation with ordinary ranking, isolating the ranking change. Parameters are not selected on the holdout fold.",
    }
    return {"train_start":min(train_dates).isoformat(),"train_end":max(train_dates).isoformat(),
            "test_start":min(test_dates).isoformat(),"test_end":max(test_dates).isoformat(),
            "train_races":len(train_races),"test_races":len(ordered),
            "training_updates":updates,"trained_weights":learned,"temperature":round(temperature,4),
            "calibration":calibration_metrics(test_calibration_records,temperature),"models":out,
            "trained_condition_segments":segment_output,
            "condition_model_segments":condition_segments,
            "condition_model_comparisons":condition_comparisons,
            "condition_finish_position_segments":condition_finish_segments,
            "confidence_filter_backtest":summarize_confidence_filter_stats(confidence_filter_stats),
            "integrated_strategy_E": integrated_summary,
            "market_rank_diagnostics": {
                name: summarize_market_rank_diagnostics(bucket)
                for name, bucket in market_rank_diagnostics.items()
            },
            "final_odds_market_comparison": {name: summarize_final_market_pair(bucket) for name, bucket in final_market_stats.items()}}


def load_odds_snapshot_records():
    directory = ROOT / "data" / "odds-snapshots"
    records = []
    files = sorted(directory.glob("*.jsonl.gz")) if directory.exists() else []
    for path in files:
        try:
            with gzip.open(path, "rt", encoding="utf-8") as source:
                for line in source:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            continue
    return files, records


def load_odds_snapshots():
    """Select the closest valid pre-off snapshot for each race; discard post-off records."""
    _, records = load_odds_snapshot_records()
    selected = {}
    for record in records:
        parts = str(record.get("race_key") or "").split("|")
        if len(parts) != 3:
            continue
        key = (parts[0], parts[1], parts[2])
        minutes = number(record.get("minutes_to_post"))
        if minutes is None:
            try:
                captured = dt.datetime.fromisoformat(str(record["captured_at"]))
                scheduled = dt.datetime.fromisoformat(str(record["scheduled_post_time"]))
                minutes = (scheduled-captured).total_seconds()/60
            except (KeyError, TypeError, ValueError):
                continue
        # Keep only snapshots captured in the production policy window:
        # from 30 minutes down to (and including) 2 minutes before post time.
        if minutes < 2 or minutes > 30:
            continue
        previous = selected.get(key)
        previous_minutes = number(previous.get("minutes_to_post")) if previous else None
        if previous and previous_minutes is None:
            try:
                previous_minutes = (dt.datetime.fromisoformat(str(previous["scheduled_post_time"]))-
                                     dt.datetime.fromisoformat(str(previous["captured_at"]))).total_seconds()/60
            except (KeyError, TypeError, ValueError):
                previous_minutes = float("inf")
        if previous is None or minutes < previous_minutes:
            selected[key] = record
    return selected


def odds_snapshot_inventory():
    files, records = load_odds_snapshot_records()
    selected = load_odds_snapshots()
    dates = [str(r.get("date") or "").replace("-", "") for r in selected.values() if r.get("date")]
    captured = [str(r.get("captured_at")) for r in records if r.get("captured_at")]
    return {
        "snapshot_files": len(files),
        "snapshot_records": len(records),
        "races": len(selected),
        "predicted_races": sum(1 for r in selected.values() if r.get("predictions")),
        "odds_rows": sum(len(r.get("odds") or []) for r in selected.values()),
        "first_date": min(dates) if dates else None,
        "last_date": max(dates) if dates else None,
        "last_captured_at": max(captured) if captured else None,
        "model_versions": sorted({str(r.get("model_version")) for r in selected.values() if r.get("model_version")}),
        "note": "The closest valid snapshot from 30 to 2 minutes before post time is used; snapshots inside the final 2 minutes, post-off records, and monthly confirmed odds are excluded."
    }



def preoff_value_backtest(payouts):
    records = list(load_odds_snapshots().values())
    thresholds = (1.00, 1.05, 1.10, 1.15, 1.20, 1.30)
    eligible_records = []
    model_versions = set()
    for record in records:
        parts = str(record.get("race_key") or "").split("|")
        if len(parts) != 3:
            continue
        key = race_key(parts[0], parts[1], parts[2])
        if not key:
            continue
        try:
            captured = dt.datetime.fromisoformat(str(record["captured_at"]))
            scheduled = dt.datetime.fromisoformat(str(record["scheduled_post_time"]))
            minutes = float(record["minutes_to_post"])
        except (KeyError, TypeError, ValueError):
            continue
        if (captured.tzinfo is None or scheduled.tzinfo is None or captured >= scheduled
                or not 2 <= minutes <= 30):
            continue
        payout = payouts.get(key)
        predictions = record.get("predictions") or []
        if not payout or not payout.get("win") or not predictions:
            continue
        eligible_records.append((str(record.get("date") or key[1]), record, payout))
        if record.get("model_version"):
            model_versions.add(str(record["model_version"]))

    def summarize(subset):
        stats = {threshold:{"races_with_bets":0,"tickets":0,"hits":0,"stake":0,
                            "return":0,"ev_sum":0.0,"odds_sum":0.0,"race_results":[]}
                 for threshold in thresholds}
        for _, record, payout in subset:
            predictions = record.get("predictions") or []
            for threshold in thresholds:
                bucket = stats[threshold]
                race_stake = race_return = race_bets = 0
                for prediction in predictions:
                    ev = number(prediction.get("expected_value"))
                    odds_value = number(prediction.get("win_odds"))
                    horse_no = re.sub(r"\D", "", str(prediction.get("horse_no") or ""))
                    if not horse_no or ev is None or odds_value is None or ev < threshold:
                        continue
                    race_bets += 1
                    bucket["tickets"] += 1
                    bucket["stake"] += 100
                    bucket["ev_sum"] += ev
                    bucket["odds_sum"] += odds_value
                    race_stake += 100
                    returned = payout_for("win", [str(int(horse_no))], payout) or 0
                    if returned:
                        bucket["hits"] += 1
                        bucket["return"] += returned
                        race_return += returned
                if race_bets:
                    bucket["races_with_bets"] += 1
                    bucket["race_results"].append((race_stake, race_return))
        output = {}
        for threshold, bucket in stats.items():
            tickets = bucket["tickets"]
            race_results = bucket.pop("race_results")
            ci_low = ci_high = None
            if len(race_results) >= 30:
                rng = random.Random(20261103 + int(round(threshold*100)))
                samples = []
                count = len(race_results)
                for _ in range(1000):
                    chosen = [race_results[rng.randrange(count)] for _ in range(count)]
                    stake = sum(x[0] for x in chosen)
                    returned = sum(x[1] for x in chosen)
                    if stake:
                        samples.append(100*returned/stake)
                samples.sort()
                if samples:
                    ci_low = round(samples[int(0.025*(len(samples)-1))], 2)
                    ci_high = round(samples[int(0.975*(len(samples)-1))], 2)
            enough = bucket["races_with_bets"] >= 100 and tickets >= 300
            output[f"{threshold:.2f}"] = {
                **bucket,
                "hit_rate": round(100*bucket["hits"]/tickets, 2) if tickets else None,
                "roi": round(100*bucket["return"]/bucket["stake"], 2) if bucket["stake"] else None,
                "profit": bucket["return"]-bucket["stake"],
                "average_expected_value": round(bucket["ev_sum"]/tickets, 4) if tickets else None,
                "average_snapshot_odds": round(bucket["odds_sum"]/tickets, 2) if tickets else None,
                "roi_ci95_low": ci_low,
                "roi_ci95_high": ci_high,
                "minimum_sample_met": enough,
                "candidate_pass": bool(enough and ci_low is not None and ci_low > 100),
            }
        return output

    eligible_records.sort(key=lambda item:(item[0],str(item[1].get("scheduled_post_time") or "")))
    overall = summarize(eligible_records)
    dates = sorted({item[0] for item in eligible_records})
    development = {}
    holdout = {}
    selected_threshold = None
    selected_holdout = None
    if len(dates) >= 10:
        split = max(1, min(len(dates)-1, int(len(dates)*0.70)))
        development_dates = set(dates[:split])
        development_records = [item for item in eligible_records if item[0] in development_dates]
        holdout_records = [item for item in eligible_records if item[0] not in development_dates]
        development = summarize(development_records)
        holdout = summarize(holdout_records)
        eligible_thresholds = [(float(key), value) for key, value in development.items()
                               if value["candidate_pass"]]
        if eligible_thresholds:
            selected_threshold = f"{max(eligible_thresholds, key=lambda item:item[1]['roi'] or 0)[0]:.2f}"
            selected_holdout = holdout.get(selected_threshold)
    if selected_threshold is None:
        validation_status = "awaiting_sample_size" if len(dates) >= 10 else "awaiting_more_dates"
    elif selected_holdout and selected_holdout.get("candidate_pass"):
        validation_status = "ready"
    elif selected_holdout and selected_holdout.get("minimum_sample_met"):
        validation_status = "holdout_failed"
    else:
        validation_status = "awaiting_holdout_sample"
    return {
        "status": "ok" if eligible_records else ("awaiting_settled_snapshots" if records else "awaiting_snapshots"),
        "snapshot_races": len(records),
        "settled_evaluated_races": len(eligible_records),
        "date_count": len(dates),
        "development_date_count": len(dates[:max(1, int(len(dates)*0.70))]) if len(dates) >= 10 else 0,
        "holdout_date_count": len(dates)-len(dates[:max(1, int(len(dates)*0.70))]) if len(dates) >= 10 else 0,
        "model_versions": sorted(model_versions),
        "thresholds": overall,
        "development": development,
        "holdout": holdout,
        "selected_threshold": selected_threshold,
        "selected_holdout": selected_holdout,
        "validation_status": validation_status,
        "minimum_sample_rule": "At least 100 races with bets and 300 tickets per threshold; a candidate additionally needs the race-cluster bootstrap 95% lower ROI bound above 100%. Threshold selection uses the first 70% of dates only; the last 30% remains untouched holdout.",
        "policy": "Flat 100-yen win bets only when saved calibrated win probability multiplied by timestamped pre-off win odds meets the threshold. Returns use official settled 100-yen payouts.",
        "note": "Only the closest valid pre-off snapshot from 30 to 2 minutes before post time per race is evaluated. No monthly confirmed odds are substituted."
    }


def preoff_ticket_type_backtest(payouts):
    """Evaluate each bet type on timestamped pre-off odds from the current model only."""
    records = []
    legacy_snapshot_races_excluded = 0
    expected_model_version = "nar-chronological-v9-D-frame-preoff"
    for record in load_odds_snapshots().values():
        if record.get("model_version") != expected_model_version:
            legacy_snapshot_races_excluded += 1
            continue
        parts = str(record.get("race_key") or "").split("|")
        if len(parts) != 3:
            continue
        key = race_key(parts[0], parts[1], parts[2])
        if not key:
            continue
        try:
            captured = dt.datetime.fromisoformat(str(record["captured_at"]))
            scheduled = dt.datetime.fromisoformat(str(record["scheduled_post_time"]))
            minutes = float(record["minutes_to_post"])
        except (KeyError, TypeError, ValueError):
            continue
        payout = payouts.get(key)
        tickets = record.get("combination_predictions") or []
        if (captured.tzinfo is None or scheduled.tzinfo is None or captured >= scheduled
                or not 2 <= minutes <= 30 or not payout or not payout.get("win") or not tickets):
            continue
        records.append((str(record.get("date") or key[1]), tickets, payout))

    kinds = ("win","place","wide","quinella","exacta","trio","trifecta")
    thresholds = (1.00,1.05,1.10,1.15,1.20,1.30,1.50)

    def summarize(subset, kind):
        output = {}
        for threshold in thresholds:
            races_with_bets = tickets_count = hits = stake = returned = 0
            max_tickets_per_race = 0
            race_results = []
            for _, tickets, payout in subset:
                race_stake = race_return = race_count = 0
                candidates = [ticket for ticket in tickets if ticket.get("bet_type") == kind]
                if kind == "trio":
                    selected_tickets = select_balanced_trio_tickets(
                        candidates, 12, 0.5, max(1.05, threshold))
                else:
                    selected_tickets = [
                        ticket for ticket in candidates
                        if number(ticket.get("expected_value")) is not None
                        and number(ticket.get("expected_value")) >= threshold
                    ]
                for ticket in selected_tickets:
                    ev = number(ticket.get("expected_value"))
                    numbers = [str(x) for x in ticket.get("numbers") or []]
                    if ev is None or not numbers:
                        continue
                    race_count += 1
                    race_stake += 100
                    pay = payout_for(kind, numbers, payout) or 0
                    if pay:
                        hits += 1
                        race_return += pay
                if race_count:
                    races_with_bets += 1
                    max_tickets_per_race = max(max_tickets_per_race, race_count)
                    tickets_count += race_count
                    stake += race_stake
                    returned += race_return
                    race_results.append((race_stake,race_return))
            ci_low = ci_high = None
            if len(race_results) >= 30:
                rng = random.Random(20261103 + int(round(threshold*100)) + sum(ord(c) for c in kind))
                samples = []
                count = len(race_results)
                for _ in range(1000):
                    sample = [race_results[rng.randrange(count)] for _ in range(count)]
                    sample_stake = sum(x[0] for x in sample)
                    sample_return = sum(x[1] for x in sample)
                    if sample_stake:
                        samples.append(100*sample_return/sample_stake)
                samples.sort()
                if samples:
                    ci_low = round(samples[int(0.025*(len(samples)-1))],2)
                    ci_high = round(samples[int(0.975*(len(samples)-1))],2)
            enough = races_with_bets >= 100 and tickets_count >= 300
            output[f"{threshold:.2f}"] = {
                "races_with_bets":races_with_bets,"tickets":tickets_count,"hits":hits,
                "max_tickets_per_race":max_tickets_per_race,
                "average_tickets_per_bet_race":round(tickets_count/races_with_bets,2) if races_with_bets else None,
                "stake":stake,"return":returned,
                "hit_rate":round(100*hits/tickets_count,2) if tickets_count else None,
                "roi":round(100*returned/stake,2) if stake else None,
                "profit":returned-stake,"roi_ci95_low":ci_low,"roi_ci95_high":ci_high,
                "minimum_sample_met":enough,
                "candidate_pass":bool(enough and ci_low is not None and ci_low > 100),
            }
        return output

    records.sort(key=lambda x:x[0])
    dates = sorted({x[0] for x in records})
    split = max(1,min(len(dates)-1,int(len(dates)*0.70))) if len(dates)>=10 else 0
    dev_dates = set(dates[:split])
    development_records = [x for x in records if x[0] in dev_dates]
    holdout_records = [x for x in records if x[0] not in dev_dates] if split else []
    by_type = {}
    for kind in kinds:
        overall = summarize(records,kind)
        development = summarize(development_records,kind) if split else {}
        holdout = summarize(holdout_records,kind) if split else {}
        eligible = [(float(k),v) for k,v in development.items() if v["candidate_pass"]]
        selected = max(eligible,key=lambda x:x[1]["roi"] or 0)[0] if eligible else None
        selected_key = f"{selected:.2f}" if selected is not None else None
        selected_holdout = holdout.get(selected_key) if selected_key else None
        if selected_key is None:
            type_validation_status = "awaiting_sample_size" if split else "awaiting_more_dates"
        elif selected_holdout and selected_holdout.get("candidate_pass"):
            type_validation_status = "ready"
        elif selected_holdout and selected_holdout.get("minimum_sample_met"):
            type_validation_status = "holdout_failed"
        else:
            type_validation_status = "awaiting_holdout_sample"
        by_type[kind] = {
            "overall":overall,"development":development,"holdout":holdout,
            "selected_threshold":selected_key,
            "selected_holdout":selected_holdout,
            "validation_status":type_validation_status,
        }
    return {
        "status":"ok" if records else "awaiting_snapshots",
        "snapshot_races":len(records),"date_count":len(dates),
        "legacy_snapshot_races_excluded":legacy_snapshot_races_excluded,
        "required_model_version":expected_model_version,
        "development_date_count":len(dev_dates),"holdout_date_count":len(dates)-len(dev_dates),
        "selection_policy":"Each bet type is evaluated independently. Trio first filters EV at max(1.05, threshold), then ranks eligible candidates by 50% probability percentile + 50% EV percentile and selects at most 12; fewer or zero are allowed. Other bet types retain threshold-only selection. Threshold selection uses only the first 70% of dates; the final 30% is untouched holdout. Candidate pass requires at least 100 races, 300 tickets, and a race-cluster bootstrap 95% lower ROI bound above 100%.",
        "probability_model":"Plackett-Luce top-three finish probabilities derived from calibrated win probabilities; combination odds are taken from the same timestamped official pre-off snapshot.",
        "bet_types":by_type,
    }


def strategy_risk_diagnostics(race_results):
    """Summarize drawdown and losing-streak risk from race outcomes."""
    equity = peak = max_drawdown = 0
    losing_streak = max_losing_streak = 0
    losing_streak_loss = max_losing_streak_loss = 0
    losing_races = total_stake = total_return = 0
    rolling = []
    worst_10_race_roi = None
    bet_races = 0
    for stake_value, return_value in race_results:
        stake_value = max(0, int(stake_value))
        return_value = max(0, int(return_value))
        if stake_value <= 0:
            continue
        bet_races += 1
        total_stake += stake_value
        total_return += return_value
        net = return_value - stake_value
        equity += net
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
        if net < 0:
            losing_races += 1
            losing_streak += 1
            losing_streak_loss += -net
            max_losing_streak = max(max_losing_streak, losing_streak)
            max_losing_streak_loss = max(max_losing_streak_loss, losing_streak_loss)
        else:
            losing_streak = 0
            losing_streak_loss = 0
        rolling.append((stake_value, return_value))
        if len(rolling) > 10:
            rolling.pop(0)
        if len(rolling) == 10:
            rolling_stake = sum(x[0] for x in rolling)
            rolling_return = sum(x[1] for x in rolling)
            roi = 100 * rolling_return / rolling_stake if rolling_stake else None
            if roi is not None and (worst_10_race_roi is None or roi < worst_10_race_roi):
                worst_10_race_roi = roi
    return {
        "bet_races": bet_races,
        "losing_races": losing_races,
        "losing_race_rate": round(100 * losing_races / bet_races, 2) if bet_races else None,
        "max_drawdown_jpy": max_drawdown,
        "max_losing_streak_races": max_losing_streak,
        "max_losing_streak_loss_jpy": max_losing_streak_loss,
        "worst_10_consecutive_race_roi_percent": round(worst_10_race_roi, 2) if worst_10_race_roi is not None else None,
        "net_profit_jpy": total_return - total_stake,
        "note": "Drawdown is measured from the running peak of cumulative net profit, starting at zero; no assumed starting bankroll is used."
    }


def race_budget_auto_strategy_backtest(payouts, type_validation):
    """Choose one validated bet type per race, allow skips, and cap stake at JPY 2,000.

    Type thresholds are selected on the development dates only. The integrated
    per-race type choice and ROI are then evaluated on the untouched final 30%
    of snapshot dates. Only genuine 2-30 minute pre-off snapshots from the
    current C-preoff model version are eligible.
    """
    required_version = "nar-chronological-v9-D-frame-preoff"
    kinds = ("win", "place", "wide", "quinella", "exacta", "trio", "trifecta")
    selected_thresholds = {}
    type_validation = (type_validation or {}).get("bet_types", {})
    for kind in kinds:
        item = type_validation.get(kind) or {}
        threshold = item.get("selected_threshold")
        development = item.get("development") or {}
        candidate = development.get(str(threshold)) if threshold is not None else None
        if candidate and candidate.get("candidate_pass"):
            selected_thresholds[kind] = float(threshold)

    records = []
    for record in load_odds_snapshots().values():
        if record.get("model_version") != required_version:
            continue
        parts = str(record.get("race_key") or "").split("|")
        if len(parts) != 3:
            continue
        key = race_key(parts[0], parts[1], parts[2])
        if not key:
            continue
        try:
            captured = dt.datetime.fromisoformat(str(record["captured_at"]))
            scheduled = dt.datetime.fromisoformat(str(record["scheduled_post_time"]))
            minutes = float(record["minutes_to_post"])
        except (KeyError, TypeError, ValueError):
            continue
        payout = payouts.get(key)
        tickets = record.get("combination_predictions") or []
        if (captured.tzinfo is None or scheduled.tzinfo is None or captured >= scheduled
                or not 2 <= minutes <= 30 or not payout or not payout.get("win") or not tickets):
            continue
        records.append((str(record.get("date") or key[1]), tickets, payout))

    records.sort(key=lambda item: item[0])
    dates = sorted({item[0] for item in records})
    split = max(1, min(len(dates)-1, int(len(dates)*0.70))) if len(dates) >= 10 else 0
    development_dates = set(dates[:split]) if split else set()
    holdout = [item for item in records if item[0] not in development_dates] if split else []

    races_with_bets = tickets_count = hits = stake = returned = skipped = 0
    race_results = []
    selected_type_counts = {kind: 0 for kind in kinds}
    skip_reasons = {"no_validated_bet_type": 0, "no_qualifying_ticket": 0}
    max_race_stake = 0
    for _, tickets, payout in holdout:
        if not selected_thresholds:
            skipped += 1
            skip_reasons["no_validated_bet_type"] += 1
            continue
        by_type = {}
        for kind, threshold in selected_thresholds.items():
            eligible = []
            for ticket in tickets:
                if ticket.get("bet_type") != kind:
                    continue
                ev = number(ticket.get("expected_value"))
                probability = number(ticket.get("probability"))
                odds_value = number(ticket.get("odds"))
                numbers = [str(x) for x in ticket.get("numbers") or []]
                if (ev is None or ev < threshold or probability is None or probability <= 0
                        or odds_value is None or odds_value <= 0 or not numbers):
                    continue
                eligible.append(ticket)
            if eligible:
                eligible.sort(key=lambda x: (
                    number(x.get("expected_value")) or 0,
                    number(x.get("probability")) or 0
                ), reverse=True)
                mean_ev = sum(number(x.get("expected_value")) for x in eligible) / len(eligible)
                by_type[kind] = (mean_ev, eligible)
        if not by_type:
            skipped += 1
            skip_reasons["no_qualifying_ticket"] += 1
            continue

        chosen_kind, (_, candidates) = max(
            by_type.items(),
            key=lambda item: (item[1][0],
                              sum(number(x.get("probability")) or 0 for x in item[1][1]) /
                              len(item[1][1]),
                              -kinds.index(item[0]))
        )
        # Match the app: up to twenty qualified tickets, with a 100-yen
        # minimum unit and EV x sqrt(probability) weighting. The target spend
        # is 100 yen per qualified ticket, capped at 2,000 yen; unused budget
        # is never forced into a race.
        selected = candidates[:20]
        weights = [
            max(0.0001, (number(ticket.get("expected_value")) - 1.0) *
                math.sqrt(max(number(ticket.get("probability")) or 0.000001, 0.000001)))
            for ticket in selected
        ]
        weight_total = sum(weights)
        target_stake = min(2000, len(selected) * 100)
        allocations = [
            (ticket, math.floor((target_stake * weight / weight_total) / 100) * 100
             if weight_total else 0)
            for ticket, weight in zip(selected, weights)
        ]
        allocations = [(ticket, amount) for ticket, amount in allocations if amount >= 100]
        race_stake = sum(amount for _, amount in allocations)
        index = 0
        while allocations and race_stake + 100 <= target_stake:
            ticket, amount = allocations[index]
            allocations[index] = (ticket, amount + 100)
            race_stake += 100
            index = (index + 1) % len(allocations)
        race_return = 0
        race_hits = 0
        for ticket, amount in allocations:
            pay = payout_for(chosen_kind, [str(x) for x in ticket.get("numbers") or []], payout) or 0
            if pay:
                race_hits += 1
                race_return += int(pay * amount / 100)
        if race_stake > 2000:
            raise AssertionError("per-race stake exceeded JPY 2,000")
        if race_stake:
            races_with_bets += 1
            tickets_count += len(allocations)
            hits += race_hits
            stake += race_stake
            returned += race_return
            max_race_stake = max(max_race_stake, race_stake)
            selected_type_counts[chosen_kind] += 1
            race_results.append((race_stake, race_return))
        else:
            skipped += 1
            skip_reasons["no_qualifying_ticket"] += 1

    ci_low = ci_high = None
    if len(race_results) >= 30:
        rng = random.Random(20261103 + len(race_results))
        samples = []
        for _ in range(2000):
            sample = [race_results[rng.randrange(len(race_results))] for _ in race_results]
            sample_stake = sum(x[0] for x in sample)
            sample_return = sum(x[1] for x in sample)
            if sample_stake:
                samples.append(100 * sample_return / sample_stake)
        samples.sort()
        if samples:
            ci_low = round(samples[int(0.025 * (len(samples)-1))], 2)
            ci_high = round(samples[int(0.975 * (len(samples)-1))], 2)

    minimum_sample_met = races_with_bets >= 100 and tickets_count >= 300
    if not records:
        status = "awaiting_current_model_snapshots"
    elif len(dates) < 10:
        status = "awaiting_more_snapshot_dates"
    elif not selected_thresholds:
        status = "awaiting_development_validation"
    elif not holdout:
        status = "awaiting_holdout_dates"
    elif not minimum_sample_met:
        status = "awaiting_holdout_sample"
    else:
        status = "profitable_candidate" if ci_low is not None and ci_low > 100 else "not_validated_profitable"

    return {
        "status": status,
        "required_model_version": required_version,
        "snapshot_races": len(records),
        "snapshot_dates": len(dates),
        "development_dates": len(development_dates),
        "holdout_dates": len(dates) - len(development_dates) if split else 0,
        "development_selected_thresholds": selected_thresholds,
        "holdout_races_with_bets": races_with_bets,
        "skipped_races": skipped,
        "skip_reasons": skip_reasons,
        "tickets": tickets_count,
        "hits": hits,
        "hit_rate": round(100 * hits / tickets_count, 2) if tickets_count else None,
        "stake_jpy": stake,
        "return_jpy": returned,
        "profit_jpy": returned - stake,
        "roi_percent": round(100 * returned / stake, 2) if stake else None,
        "roi_ci95_low": ci_low,
        "roi_ci95_high": ci_high,
        "minimum_sample_met": minimum_sample_met,
        "max_race_stake_jpy": max_race_stake,
        "race_stake_cap_jpy": 2000,
        "selected_bet_type_race_counts": selected_type_counts,
        "risk_diagnostics": strategy_risk_diagnostics(race_results),
        "stake_rule": "Up to 20 qualified tickets; target stake is 100 JPY per qualified ticket (max 2,000 JPY), allocated proportionally by max(0.0001, (EV-1) * sqrt(probability)) in 100-yen units. Unused budget is not forced into a race.",
        "selection_rule": "Use only bet-type thresholds that passed development-date validation. For each holdout race, choose the eligible bet type with the highest mean expected value using same-snapshot pre-off odds; if none qualifies, skip the race.",
        "profitability_rule": "At least 100 holdout races with bets and 300 tickets, plus a race-cluster bootstrap 95% ROI lower bound above 100%.",
        "note": "ROI is a historical simulation using official settled payouts, not proof of future profit. If current-version snapshots are too few, the workflow reports insufficient evidence rather than inventing results."
    }


BET_TYPE_SELECTION_RULES = {
    # Wide/exacta/trio candidates must beat baseline by at least 2.5 points
    # in every available prior fold. This is a model-selection hurdle, not a
    # profitability gate; ROI must still clear 100% on independent validation.
    "wide": {"min_tickets": 300, "min_roi_gain": 2.5},
    "exacta": {"min_tickets": 300, "min_roi_gain": 2.5},
    "trio": {"min_tickets": 300, "min_roi_gain": 2.5},
    # Trifecta has the largest combination space and remains deliberately strict.
    "trifecta": {"min_tickets": 5000, "min_roi_gain": 12.0},
}


def select_models_from_prior_folds(prior_folds, min_tickets=300, min_roi_gain=5.0):
    """Select candidates only when the edge is repeated across every prior fold.

    The trifecta uses a stricter 5,000-ticket / 12-point hurdle. All other types
    use the caller's default hurdle. Baseline is always the fallback.
    """
    prior_folds = list(prior_folds or [])
    if not prior_folds:
        return {}
    selected = {}
    kinds = ("win", "place", "wide", "quinella", "exacta", "trio", "trifecta")
    for kind in kinds:
        rule = BET_TYPE_SELECTION_RULES.get(kind, {})
        required_tickets = int(rule.get("min_tickets", min_tickets))
        required_gain = float(rule.get("min_roi_gain", min_roi_gain))
        model_names = set()
        for fold in prior_folds:
            model_names.update((fold.get("models") or {}).keys())
        model_names.discard("baseline")
        candidates = []
        baseline_history = []
        for fold in prior_folds:
            baseline_bucket = ((fold.get("models") or {}).get("baseline", {})
                               .get("bet_types") or {}).get(kind, {})
            baseline_history.append(number(baseline_bucket.get("roi")))
        for model_name in model_names:
            edges, model_rois, model_tickets = [], [], []
            valid = True
            for fold, baseline_roi in zip(prior_folds, baseline_history):
                model_bucket = ((fold.get("models") or {}).get(model_name, {})
                                .get("bet_types") or {}).get(kind, {})
                model_roi = number(model_bucket.get("roi"))
                tickets = int(model_bucket.get("tickets") or 0)
                if (baseline_roi is None or model_roi is None or tickets < required_tickets
                        or model_roi < baseline_roi + required_gain):
                    valid = False
                    break
                edges.append(model_roi - baseline_roi)
                model_rois.append(model_roi)
                model_tickets.append(tickets)
            if valid and edges:
                candidates.append((sum(edges) / len(edges), model_name,
                                   model_rois, model_tickets, edges))
        baseline_roi = baseline_history[-1]
        baseline_bucket = ((prior_folds[-1].get("models") or {}).get("baseline", {})
                           .get("bet_types") or {}).get(kind, {})
        if candidates:
            edge, model_name, rois, tickets, edges = max(
                candidates, key=lambda item: (item[0], min(item[3])))
            selected[kind] = {
                "model": model_name, "development_roi": rois[-1],
                "development_tickets": tickets[-1], "baseline_roi": baseline_roi,
                "development_roi_history": rois, "edge_history": [round(x, 2) for x in edges],
                "required_min_tickets": required_tickets,
                "required_min_roi_gain": required_gain,
            }
        else:
            selected[kind] = {
                "model": "baseline", "development_roi": baseline_roi,
                "development_tickets": int(baseline_bucket.get("tickets") or 0),
                "baseline_roi": baseline_roi, "development_roi_history": [baseline_roi],
                "edge_history": [], "required_min_tickets": required_tickets,
                "required_min_roi_gain": required_gain,
            }
    return selected


def select_models_from_prior_fold(prior_fold, min_tickets=300, min_roi_gain=5.0):
    """Backward-compatible single-fold selector used by tests and diagnostics."""
    return select_models_from_prior_folds([prior_fold], min_tickets, min_roi_gain)


def evaluate_walk_forward_selection(folds, min_tickets=300, min_roi_gain=5.0):
    """Select from all available prior folds, then evaluate on the next untouched fold."""
    evaluations = []
    for index in range(1, len(folds)):
        prior = folds[index - 1]
        current = folds[index]
        choices = select_models_from_prior_folds(folds[:index], min_tickets, min_roi_gain)
        per_type = {}
        for kind, choice in choices.items():
            model_name = choice["model"]
            actual = current.get("models", {}).get(model_name, {}).get("bet_types", {}).get(kind, {})
            baseline = current.get("models", {}).get("baseline", {}).get("bet_types", {}).get(kind, {})
            per_type[kind] = {
                **choice, "validation_fold": current.get("fold"),
                "validation_roi": actual.get("roi"), "validation_tickets": actual.get("tickets", 0),
                "validation_hits": actual.get("hits", 0), "validation_stake": actual.get("stake", 0),
                "validation_return": actual.get("return", 0),
                "validation_race_count": actual.get("race_count", 0),
                "validation_sum_return_sq": actual.get("sum_return_sq", 0),
                "validation_sum_stake_sq": actual.get("sum_stake_sq", 0),
                "validation_sum_stake_return": actual.get("sum_stake_return", 0),
                "validation_profit": actual.get("profit", 0),
                "baseline_validation_roi": baseline.get("roi"),
                "roi_delta_vs_baseline": (round(actual["roi"] - baseline["roi"], 2)
                    if actual.get("roi") is not None and baseline.get("roi") is not None else None),
            }
        evaluations.append({"selection_fold": prior.get("fold"),
                            "prior_folds_used": [fold.get("fold") for fold in folds[:index]],
                            "validation_fold": current.get("fold"), "bet_types": per_type})
    selected_portfolio = {}
    for kind in ("wide", "exacta", "trio", "trifecta"):
        chosen = [entry["bet_types"][kind] for entry in evaluations]
        totals = {
            "tickets": sum(int(item.get("validation_tickets") or 0) for item in chosen),
            "hits": sum(int(item.get("validation_hits") or 0) for item in chosen),
            "stake": sum(int(item.get("validation_stake") or 0) for item in chosen),
            "return": sum(int(item.get("validation_return") or 0) for item in chosen),
            "race_count": sum(int(item.get("validation_race_count") or 0) for item in chosen),
            "sum_return_sq": sum(float(item.get("validation_sum_return_sq") or 0) for item in chosen),
            "sum_stake_sq": sum(float(item.get("validation_sum_stake_sq") or 0) for item in chosen),
            "sum_stake_return": sum(float(item.get("validation_sum_stake_return") or 0) for item in chosen),
        }
        totals.update(summarize_race_cluster_roi(totals))
        totals["hit_rate"] = round(100 * totals["hits"] / totals["tickets"], 2) if totals["tickets"] else None
        totals["validation_folds"] = [
            {"fold": item.get("validation_fold"), "model": item.get("model"),
             "roi": item.get("validation_roi"), "tickets": item.get("validation_tickets")}
            for item in chosen
        ]
        selected_portfolio[kind] = totals
    # Select role-specific trifecta blend/formation using only prior folds,
    # then score that exact choice on the next untouched fold.
    selected_role_strategy_rows = []
    selected_role_portfolio_stats = {
        "races": 0, "tickets": 0, "hits": 0, "hit_races": 0,
        "stake": 0, "return": 0, "sum_return_sq": 0,
        "sum_stake_sq": 0, "sum_stake_return": 0,
    }
    for index in range(1, len(folds)):
        prior_folds = folds[:index]
        current = folds[index]
        choices = select_models_from_prior_folds(prior_folds, min_tickets, min_roi_gain)
        model_name = (choices.get("trifecta") or {}).get("model", "baseline")
        candidates = []
        for blend, formations in new_role_trifecta_diagnostics().items():
            for formation in formations:
                prior_stats = []
                valid = True
                for prior_fold in prior_folds:
                    prior_model = ((prior_fold.get("models") or {}).get(model_name) or {})
                    stats = ((prior_model.get("role_trifecta_diagnostics") or {})
                             .get(blend) or {}).get(formation)
                    if (not stats or int(stats.get("races") or 0) < 100
                            or int(stats.get("tickets") or 0) < 300
                            or not int(stats.get("stake") or 0)):
                        valid = False
                        break
                    prior_stats.append(stats)
                if not valid:
                    continue
                stake = sum(int(x.get("stake") or 0) for x in prior_stats)
                returned = sum(int(x.get("return") or 0) for x in prior_stats)
                if stake:
                    candidates.append({
                        "blend": blend, "formation": formation,
                        "development_roi": 100 * returned / stake,
                        "development_races": sum(int(x.get("races") or 0) for x in prior_stats),
                        "development_tickets": sum(int(x.get("tickets") or 0) for x in prior_stats),
                    })
        selected_strategy = max(
            candidates,
            key=lambda x: (x["development_roi"], x["development_tickets"]),
        ) if candidates else None
        row = {
            "selection_folds": [x.get("fold") for x in prior_folds],
            "validation_fold": current.get("fold"),
            "model": model_name,
            "status": "selected_from_prior_folds" if selected_strategy else "awaiting_development_sample",
            "strategy": selected_strategy,
            "validation": None,
        }
        if selected_strategy:
            current_model = ((current.get("models") or {}).get(model_name) or {})
            validation_stats = (((current_model.get("role_trifecta_diagnostics") or {})
                                 .get(selected_strategy["blend"]) or {})
                                .get(selected_strategy["formation"]))
            if validation_stats:
                validation_summary = summarize_role_trifecta_diagnostics({
                    "selected": {selected_strategy["formation"]: validation_stats}
                })["selected"][selected_strategy["formation"]]
                row["validation"] = validation_summary
                for key in selected_role_portfolio_stats:
                    selected_role_portfolio_stats[key] += int(validation_stats.get(key) or 0)
        selected_role_strategy_rows.append(row)

    selected_role_strategy_portfolio = summarize_role_trifecta_diagnostics({
        "selected": {"walk_forward": selected_role_portfolio_stats}
    })["selected"]["walk_forward"]

    # Aggregate all role-specific combinations for the model selected from
    # prior folds, evaluated on the next untouched fold. This diagnostic grid
    # is descriptive; strategy promotion uses the walk-forward selection above.
    selected_role_stats = new_role_trifecta_diagnostics()
    selected_role_models = []
    fold_by_number = {fold.get("fold"): fold for fold in folds}
    for entry in evaluations:
        trifecta_choice = (entry.get("bet_types") or {}).get("trifecta") or {}
        model_name = trifecta_choice.get("model")
        validation_fold = fold_by_number.get(entry.get("validation_fold"))
        model_bucket = ((validation_fold or {}).get("models") or {}).get(model_name or "", {})
        diagnostics = model_bucket.get("role_trifecta_diagnostics") or {}
        if not diagnostics:
            continue
        selected_role_models.append({
            "validation_fold": entry.get("validation_fold"),
            "model": model_name,
        })
        for blend, formations in diagnostics.items():
            for formation, source_stats in formations.items():
                target = selected_role_stats[blend][formation]
                for key in ("races", "tickets", "hits", "hit_races", "stake", "return",
                            "sum_return_sq", "sum_stake_sq", "sum_stake_return"):
                    target[key] += int(source_stats.get(key) or 0)
    return {
        "status": "ok" if evaluations else "awaiting_next_fold",
        "selection_method": "Each candidate must clear its type-specific sample and ROI-edge hurdle in every available prior fold. Selection uses no validation-fold outcomes; baseline is the fallback. Selection hurdles are not profitability gates.",
        "selected_portfolio": selected_portfolio,
        "selected_role_trifecta_models_by_validation_fold": selected_role_models,
        "selected_role_trifecta_strategy_by_validation_fold": selected_role_strategy_rows,
        "selected_role_trifecta_strategy_portfolio": selected_role_strategy_portfolio,
        "selected_role_trifecta_diagnostics": summarize_role_trifecta_diagnostics(selected_role_stats),
        "min_tickets": min_tickets, "min_roi_gain_percentage_points": min_roi_gain,
        "selection_rules_by_bet_type": {
            kind: {"min_tickets": int(BET_TYPE_SELECTION_RULES.get(kind, {}).get("min_tickets", min_tickets)),
                   "min_roi_gain_percentage_points": float(BET_TYPE_SELECTION_RULES.get(kind, {}).get("min_roi_gain", min_roi_gain))}
            for kind in ("win", "place", "wide", "quinella", "exacta", "trio", "trifecta")
        },
        "evaluations": evaluations,
        "promotion_policy": "No model is automatically promoted from a single fold. Require repeated next-fold gains, adequate samples, and independent market-odds validation before production use."
    }


def holdout_validation(races, horses, payouts, final_odds=None, checkpoint_state=None):
    dates=sorted({r["date"] for r in races.values()})
    if len(dates)<60:
        return {"status":"insufficient_dates","available_dates":len(dates)}
    # Four years of source data allow a clean comparison: the first year is
    # warm-up/training, the following two years are evaluated as two sequential
    # one-year walk-forward windows, and the latest year is the final holdout.
    recent_start = END - dt.timedelta(days=364)
    previous_two_start = recent_start - dt.timedelta(days=730)
    previous_split = previous_two_start + dt.timedelta(days=365)
    fold_specs = [
        ("previous_year_2", previous_two_start, previous_split),
        ("previous_year_1", previous_split, recent_start),
        ("recent_year", recent_start, END + dt.timedelta(days=1)),
    ]
    folds=[]
    fold_runtime_seconds = {}
    print(f"HOLDOUT_PROGRESS folds={len(fold_specs)} available_dates={len(dates)}", flush=True)
    for index,(label,test_start,test_end) in enumerate(fold_specs,1):
        train_dates=[d for d in dates if d < test_start]
        test_dates=[d for d in dates if test_start <= d < test_end]
        if not train_dates or not test_dates:
            continue
        saved_folds = (checkpoint_state or {}).get("folds") or {}
        if label in saved_folds:
            saved_fold = saved_folds[label]
            folds.append(saved_fold)
            fold_runtime_seconds[label] = saved_fold.get("runtime_seconds", 0)
            print(f"CHECKPOINT_RESUME fold={label} reused=true", flush=True)
            continue
        fold_started = time.perf_counter()
        print(f"HOLDOUT_PROGRESS fold_start={label} train_dates={len(train_dates)} test_dates={len(test_dates)}", flush=True)
        fold_result = evaluate_holdout_fold(
            races,horses,payouts,train_dates,test_dates,final_odds)
        fold_elapsed = round(time.perf_counter() - fold_started, 3)
        fold_runtime_seconds[label] = fold_elapsed
        print(f"PERF_PROFILE holdout_fold_{label}={fold_elapsed}s "
              f"train_dates={len(train_dates)} test_dates={len(test_dates)}", flush=True)
        completed_fold = {"fold":index,"period_label":label,
                          "runtime_seconds":fold_elapsed, **fold_result}
        folds.append(completed_fold)
        if checkpoint_state is not None:
            checkpoint_state.setdefault("folds", {})[label] = completed_fold
            save_backtest_checkpoint(checkpoint_state, f"fold-{label}")
    if len(folds)<3:
        return {"status":"insufficient_folds","folds":folds,
                "fold_runtime_seconds":fold_runtime_seconds,
                "required_periods":["previous_year_2","previous_year_1","recent_year"]}

    def summarize_period(period_folds):
        model_names = period_folds[0]["models"].keys()
        period_models = {}
        for model_name in model_names:
            source_models = [fold["models"][model_name] for fold in period_folds]
            race_total = sum(int(m.get("races") or 0) for m in source_models)
            winner_top1 = sum(int(m.get("winner_top1") or 0) for m in source_models)
            winner_top3 = sum(int(m.get("winner_top3") or 0) for m in source_models)
            bet_types = {}
            for kind in ("win","place","wide","quinella","exacta","trio","trifecta"):
                totals = {key:sum(int((m.get("bet_types") or {}).get(kind,{}).get(key) or 0)
                                  for m in source_models)
                          for key in ("tickets","hits","stake","return","race_count",
                                      "sum_return_sq","sum_stake_sq","sum_stake_return")}
                bet_types[kind] = {
                    **totals,
                    "hit_rate":round(100*totals["hits"]/totals["tickets"],2) if totals["tickets"] else None,
                    **summarize_race_cluster_roi(totals),
                }
            shadow_races = sum(int((m.get("shadow_portfolio") or {}).get("races") or 0)
                                for m in source_models)
            shadow_bet_types = {}
            for kind in ("win", "place", "wide", "trio"):
                shadow_totals = {key: sum(int((((m.get("shadow_portfolio") or {}).get("bet_types") or {}).get(kind) or {}).get(key) or 0)
                                          for m in source_models)
                                 for key in ("tickets", "hits", "stake", "return")}
                shadow_bet_types[kind] = {
                    **shadow_totals,
                    "hit_rate": round(100 * shadow_totals["hits"] / shadow_totals["tickets"], 2)
                        if shadow_totals["tickets"] else None,
                    "roi": round(100 * shadow_totals["return"] / shadow_totals["stake"], 2)
                        if shadow_totals["stake"] else None,
                    "profit": shadow_totals["return"] - shadow_totals["stake"],
                }
            period_models[model_name] = {
                "races":race_total,
                "winner_top1_rate":round(100*winner_top1/race_total,2) if race_total else None,
                "winner_top3_rate":round(100*winner_top3/race_total,2) if race_total else None,
                "bet_types":bet_types,
                "shadow_portfolio": {"races": shadow_races, "bet_types": shadow_bet_types},
            }
        return period_models

    previous_folds = [fold for fold in folds if fold["period_label"] in ("previous_year_2","previous_year_1")]
    recent_folds = [fold for fold in folds if fold["period_label"] == "recent_year"]
    previous_two_year_models = summarize_period(previous_folds)
    recent_year_models = summarize_period(recent_folds)
    recent_year_validation = {
        "status":"evaluated" if recent_folds else "insufficient_dates",
        "period":{"start":recent_start.isoformat(),"end":END.isoformat()},
        "train_end":recent_folds[0]["train_end"] if recent_folds else None,
        "test_races_expected":sum(int(fold.get("test_races") or 0) for fold in recent_folds),
        "models":recent_year_models,
        "fold":recent_folds[0] if recent_folds else None,
    }
    previous_two_year_validation = {
        "status":"evaluated" if len(previous_folds)==2 else "insufficient_dates",
        "period":{"start":previous_two_start.isoformat(),"end":(recent_start-dt.timedelta(days=1)).isoformat()},
        "test_races":sum(int(fold.get("test_races") or 0) for fold in previous_folds),
        "folds":[{"period_label":fold["period_label"],"train_start":fold["train_start"],
                  "train_end":fold["train_end"],"test_start":fold["test_start"],
                  "test_end":fold["test_end"],"test_races":fold["test_races"]}
                 for fold in previous_folds],
        "models":previous_two_year_models,
    }
    period_comparison = {
        "previous_two_years":previous_two_year_validation,
        "recent_one_year":recent_year_validation,
        "method":"Previous two years are aggregated from two sequential one-year out-of-sample windows. Each year's model is trained only on dates before that year's test window. The recent year is a separate final holdout trained on the preceding three years. Hit rate is winning tickets / all tickets; ROI is official payout / stake.",
    }

    kinds=("win","place","wide","quinella","exacta","trio","trifecta")
    aggregate={}
    for model_name in ("A_baseline","B_condition","C_performance_pace",
                       "baseline","trained","trained_no_recent_history","trained_race_relative",
                       "trained_with_pace","trained_with_performance","trained_performance_with_pace","trained_with_conditions","trained_with_conditions_and_pace",
                       "trained_margin_only","trained_closing_only","trained_performance_light",
                       "trained_with_time","trained_performance_with_time", *FEATURE_WEIGHT_NAMES):
        total={"races":0,"winner_top1":0,"winner_top3":0,
               "winner_topk":{str(k):0 for k in (5,6,7)},
               "bet_types":{kind:{"tickets":0,"hits":0,"stake":0,"return":0,
                                   "race_count":0,"sum_return_sq":0,
                                   "sum_stake_sq":0,"sum_stake_return":0}
                            for kind in kinds}}
        for fold in folds:
            fm=fold["models"][model_name]
            total["races"]+=fm["races"]
            total["winner_top1"]+=fm["winner_top1"]
            total["winner_top3"]+=fm["winner_top3"]
            for k in ("5","6","7"):
                total["winner_topk"][k] += fm["winner_topk"][k]
            for kind in kinds:
                for key in ("tickets","hits","stake","return","race_count",
                            "sum_return_sq","sum_stake_sq","sum_stake_return"):
                    total["bet_types"][kind][key]+=fm["bet_types"][kind][key]
        bet_summary={}
        for kind,bm in total["bet_types"].items():
            bet_summary[kind]={**bm,
                "hit_rate":round(100*bm["hits"]/bm["tickets"],2) if bm["tickets"] else None,
                **summarize_race_cluster_roi(bm)}
        n=total["races"] or 1
        aggregate[model_name]={**total,"bet_types":bet_summary,
            "winner_top1_rate":round(100*total["winner_top1"]/n,2),
            "winner_top3_rate":round(100*total["winner_top3"]/n,2),
            "winner_topk_rate":{k:round(100*v/n,2) for k,v in total["winner_topk"].items()},
            "selected_bet_types":{kind:bet_summary[kind] for kind in ("wide","exacta","trio","trifecta")},
            "win_roi":bet_summary["win"]["roi"],"place_roi":bet_summary["place"]["roi"],
            "wide_roi":bet_summary["wide"]["roi"]}
    history_comparison = {}
    without = aggregate["trained_no_recent_history"]
    with_history = aggregate["trained"]
    for kind in kinds:
        before, after = without["bet_types"][kind], with_history["bet_types"][kind]
        history_comparison[kind] = {
            "before_hit_rate": before["hit_rate"], "after_hit_rate": after["hit_rate"],
            "hit_rate_delta": round(after["hit_rate"]-before["hit_rate"], 2) if after["hit_rate"] is not None and before["hit_rate"] is not None else None,
            "before_roi": before["roi"], "after_roi": after["roi"],
            "roi_delta": round(after["roi"]-before["roi"], 2) if after["roi"] is not None and before["roi"] is not None else None,
            "before_tickets": before["tickets"], "after_tickets": after["tickets"],
        }
    history_comparison["_method"] = "Same chronological holdout races and same trained weights; before excludes each horse's last-five-run form/venue/distance inputs, after includes them. Sire, jockey and age features remain identical."
    auto_selection = evaluate_walk_forward_selection(folds)
    # Aggregate E only from untouched walk-forward test folds. Raw race-cluster
    # moments are summed before confidence intervals are recomputed.
    e_total = {
        "races": 0,
        "bet_types": {
            kind: {"tickets": 0, "hits": 0, "stake": 0, "return": 0,
                   "race_count": 0, "sum_return_sq": 0,
                   "sum_stake_sq": 0, "sum_stake_return": 0}
            for kind in ("wide", "exacta", "trio", "trifecta")
        },
    }
    for fold in folds:
        e_fold = fold.get("integrated_strategy_E") or {}
        e_total["races"] += int(e_fold.get("races") or 0)
        for kind, bucket in e_fold.get("bet_types", {}).items():
            for key in ("tickets", "hits", "stake", "return", "race_count",
                        "sum_return_sq", "sum_stake_sq", "sum_stake_return"):
                e_total["bet_types"][kind][key] += bucket.get(key, 0)
    e_bets = {}
    for kind, bucket in e_total["bet_types"].items():
        e_bets[kind] = {
            **bucket,
            "hit_rate": round(100 * bucket["hits"] / bucket["tickets"], 2)
                if bucket["tickets"] else None,
            **summarize_race_cluster_roi(bucket),
        }
    integrated_e_oos = {
        "status": "evaluated" if e_total["races"] else "no_races_passed_confidence_gate",
        "confidence_threshold": 0.30,
        "role_blend": 0.10,
        "trifecta_formation": [2, 3, 4],
        "races": e_total["races"],
        "bet_types": e_bets,
        "folds": [
            {"fold": fold["fold"],
             "races": fold.get("integrated_strategy_E", {}).get("races", 0),
             "bet_types": {
                 kind: {
                     "roi": fold.get("integrated_strategy_E", {}).get("bet_types", {}).get(kind, {}).get("roi"),
                     "hit_rate": fold.get("integrated_strategy_E", {}).get("bet_types", {}).get(kind, {}).get("hit_rate"),
                     "tickets": fold.get("integrated_strategy_E", {}).get("bet_types", {}).get(kind, {}).get("tickets", 0),
                 } for kind in ("wide", "exacta", "trio", "trifecta")
             }} for fold in folds
        ],
        "policy": "Pre-registered integrated challenger; no parameter selection on the validation folds.",
    }
    return {"status":"ok","auto_improvement_loop":auto_selection,
            "integrated_strategy_E": integrated_e_oos,"method":"three rolling-origin folds with non-overlapping test windows; weights frozen within each test window; prior results added only after all races on a date are scored; market-value bets use timestamped official pre-off odds snapshots",
            "folds":folds,
            "fold_runtime_seconds":fold_runtime_seconds,
            "recent_year_validation":recent_year_validation,
            "previous_two_year_validation":previous_two_year_validation,
            "period_comparison":period_comparison,
            "test_races":sum(x["models"]["trained"]["races"] for x in folds),
            "test_start":folds[0]["test_start"],"test_end":folds[-1]["test_end"],
            "train_end":folds[-1]["train_end"],"models":aggregate,
            "recent_history_comparison":history_comparison,
            "trained_weights":folds[-1]["trained_weights"],
            "folds_above_100_roi":{kind:sum(1 for fold in folds if (fold["models"]["trained"]["bet_types"][kind]["roi"] or 0)>=100) for kind in kinds},
            "selection_rule":"Each fold uses only strictly earlier training data; test weights remain fixed and race results are added to history only after the entire date is scored.",
            "promotion_gate": {
                "roi_strictly_above_100": True,
                "minimum_independent_test_races": 100,
                "minimum_tickets": 300,
                "race_cluster_95_percent_ci_lower_strictly_above_100": True,
                "all_conditions_required": True,
                "ci_method": "race_cluster_normal_approximation",
                "scope": "Aggregated untouched walk-forward validation folds; development folds are not counted toward the validation sample."
            }}

def frame_effect_audit(races, horses):
    """Measure historical frame-number outcome rates; diagnostic only, not a scoring feature."""
    buckets = defaultdict(lambda: {"starts": 0, "wins": 0, "top3": 0})
    venue_buckets = defaultdict(lambda: {"starts": 0, "wins": 0, "top3": 0})
    for key, race in races.items():
        venue = norm(race.get("venue"))
        for runner in horses.get(key, []):
            frame = rank_value(runner.get("frame_no"))
            finish = rank_value(runner.get("finish"))
            if frame is None or finish is None:
                continue
            for bucket in (buckets[frame], venue_buckets[(venue, frame)]):
                bucket["starts"] += 1
                bucket["wins"] += int(finish == 1)
                bucket["top3"] += int(finish <= 3)
    def summarize(source):
        result = {}
        for key, bucket in source.items():
            starts = bucket["starts"]
            result[str(key)] = {
                **bucket,
                "win_rate_pct": round(100 * bucket["wins"] / starts, 2) if starts else None,
                "top3_rate_pct": round(100 * bucket["top3"] / starts, 2) if starts else None,
            }
        return result
    return {
        "overall_by_frame_number": summarize(buckets),
        "venue_by_frame_number": summarize(venue_buckets),
        "method": "Official frame number versus finish position; descriptive only. No frame effect is added to live model scores unless a separate chronological holdout proves improvement.",
    }


def data_quality_audit(races, horses, payouts):
    """Report source coverage before interpreting model metrics."""
    race_rows = list(races.values())
    runner_rows = [runner for key in races for runner in horses.get(key, [])]
    total_races = len(race_rows)
    total_runners = len(runner_rows)
    def present(value):
        return value is not None and str(value).strip() != ""
    def pct(count, total):
        return round(100.0 * count / total, 2) if total else None
    return {
        "races": total_races,
        "runners": total_runners,
        "races_with_payout": sum(1 for key in races if payouts.get(key) and payouts[key].get("win")),
        "race_field_coverage_pct": pct(sum(1 for key in races if len(horses.get(key, [])) >= 3), total_races),
        "distance_coverage_pct": pct(sum(1 for race in race_rows if present(race.get("distance"))), total_races),
        "surface_coverage_pct": pct(sum(1 for race in race_rows if present(race.get("surface"))), total_races),
        "going_coverage_pct": pct(sum(1 for race in race_rows if present(race.get("going"))), total_races),
        "runner_age_coverage_pct": pct(sum(1 for h in runner_rows if present(h.get("age"))), total_runners),
        "runner_jockey_coverage_pct": pct(sum(1 for h in runner_rows if present(h.get("jockey"))), total_runners),
        "runner_sire_coverage_pct": pct(sum(1 for h in runner_rows if present(h.get("sire"))), total_runners),
        "finish_coverage_pct": pct(sum(1 for h in runner_rows if present(h.get("finish"))), total_runners),
        "frame_no_coverage_pct": pct(sum(1 for h in runner_rows if present(h.get("frame_no"))), total_runners),
        "frame_effect": frame_effect_audit(races, horses),
        "last3f_coverage_pct": pct(sum(1 for h in runner_rows if present(h.get("last3f"))), total_runners),
        "policy": "Coverage diagnostics only; missing values are not imputed as observed facts and this report does not imply predictive value."
    }



# Pre-off ranking research is evaluated only from timestamped snapshots.
def preoff_ranking_strategy_backtest(payouts):
    """Compare candidate ranking rules on timestamped pre-off odds, with date holdout."""
    top_n = {"wide":3, "exacta":4, "trio":10, "trifecta":24}
    strategies = ("expected_value", "probability", "odds")
    records = []
    legacy_snapshot_races_excluded = 0
    expected_model_version = "nar-chronological-v9-D-frame-preoff"
    for record in load_odds_snapshots().values():
        if record.get("model_version") != expected_model_version:
            legacy_snapshot_races_excluded += 1
            continue
        parts = str(record.get("race_key") or "").split("|")
        if len(parts) != 3:
            continue
        key = race_key(parts[0], parts[1], parts[2])
        if not key:
            continue
        try:
            captured = dt.datetime.fromisoformat(str(record["captured_at"]))
            scheduled = dt.datetime.fromisoformat(str(record["scheduled_post_time"]))
            minutes = float(record["minutes_to_post"])
        except (KeyError, TypeError, ValueError):
            continue
        payout = payouts.get(key)
        candidates = record.get("candidate_combinations") or []
        if (captured.tzinfo is None or scheduled.tzinfo is None or captured >= scheduled
                or not 2 <= minutes <= 30 or not payout or not payout.get("win")
                or not candidates):
            continue
        records.append((str(record.get("date") or key[1]), candidates, payout))
    records.sort(key=lambda item:item[0])
    dates = sorted({item[0] for item in records})
    split = max(1, min(len(dates)-1, int(len(dates)*0.70))) if len(dates) >= 10 else 0
    development_dates = set(dates[:split])
    development = [item for item in records if item[0] in development_dates]
    holdout = [item for item in records if split and item[0] not in development_dates]

    def evaluate(subset, kind, strategy):
        races = tickets = hits = stake = returned = 0
        race_results = []
        for _, candidates, payout in subset:
            eligible = [ticket for ticket in candidates
                        if ticket.get("bet_type") == kind
                        and number(ticket.get("probability")) is not None
                        and number(ticket.get("odds")) is not None
                        and number(ticket.get("odds")) > 0]
            eligible.sort(key=lambda ticket: (
                number(ticket.get(strategy)) if number(ticket.get(strategy)) is not None else -1,
                number(ticket.get("expected_value")) or -1,
                number(ticket.get("probability")) or -1), reverse=True)
            selected = eligible[:top_n[kind]]
            if not selected:
                continue
            race_stake = 100 * len(selected)
            race_return = sum(payout_for(kind, [str(x) for x in ticket.get("numbers") or []], payout) or 0
                              for ticket in selected)
            race_hits = sum(1 for ticket in selected
                            if payout_for(kind, [str(x) for x in ticket.get("numbers") or []], payout))
            races += 1
            tickets += len(selected)
            hits += race_hits
            stake += race_stake
            returned += race_return
            race_results.append((race_stake, race_return))
        ci_low = ci_high = None
        if len(race_results) >= 30:
            rng = random.Random(20261103 + sum(ord(c) for c in kind + strategy))
            samples = []
            for _ in range(1000):
                sample = [race_results[rng.randrange(len(race_results))] for _ in race_results]
                sample_stake = sum(x[0] for x in sample)
                sample_return = sum(x[1] for x in sample)
                if sample_stake:
                    samples.append(100 * sample_return / sample_stake)
            samples.sort()
            if samples:
                ci_low = round(samples[int(0.025*(len(samples)-1))], 2)
                ci_high = round(samples[int(0.975*(len(samples)-1))], 2)
        enough = races >= 100 and tickets >= 300
        return {
            "races_with_bets":races, "tickets":tickets, "hits":hits, "stake":stake,
            "return":returned, "hit_rate":round(100*hits/tickets,2) if tickets else None,
            "roi":round(100*returned/stake,2) if stake else None,
            "profit":returned-stake, "roi_ci95_low":ci_low, "roi_ci95_high":ci_high,
            "minimum_sample_met":enough,
            "candidate_pass":bool(enough and ci_low is not None and ci_low > 100),
        }

    result = {"status":"ok" if records else "awaiting_C_model_preoff_snapshots",
              "snapshot_races":len(records),
              "legacy_snapshot_races_excluded":legacy_snapshot_races_excluded,
              "required_model_version":expected_model_version,
              "date_count":len(dates),
              "development_date_count":len(development_dates),
              "holdout_date_count":len(dates)-len(development_dates) if split else 0,
              "top_n_per_race":top_n, "strategies":{}, "selected_by_development":{}}
    for kind in top_n:
        result["strategies"][kind] = {}
        best = None
        for strategy in strategies:
            overall = evaluate(records, kind, strategy)
            dev = evaluate(development, kind, strategy)
            test = evaluate(holdout, kind, strategy)
            result["strategies"][kind][strategy] = {
                "overall":overall, "development":dev, "holdout":test}
            if dev["minimum_sample_met"] and (best is None or (dev["roi"] or 0) > best[1]):
                best = (strategy, dev["roi"] or 0)
        selected = best[0] if best else None
        result["selected_by_development"][kind] = {
            "strategy":selected,
            "holdout":result["strategies"][kind][selected]["holdout"] if selected else None,
            "status":"selected_for_holdout" if selected else "awaiting_development_sample",
        }
    result["policy"] = ("For each bet type, rank the full pre-off candidate universe by expected value, "
                        "model hit probability, or market odds; buy the same fixed top-N count. "
                        "Select a ranking only on the first 70% of dates and evaluate it on the untouched last 30%. "
                        "A strategy is not validated unless its holdout has adequate sample and race-cluster "
                        "bootstrap 95% ROI lower bound above 100%.")
    return result

def build_trifecta_research_fold(fold):
    """Export every condition diagnostic needed by the research report."""
    trained = (fold.get("models") or {}).get("trained") or {}
    return {
        "fold": fold.get("fold"),
        "train_end": fold.get("train_end"),
        "test_start": fold.get("test_start"),
        "test_end": fold.get("test_end"),
        "races": trained.get("races"),
        "finish_position_diagnostics": trained.get("finish_position_diagnostics"),
        "role_trifecta_diagnostics": trained.get("role_trifecta_diagnostics"),
        "trifecta_diagnostics": trained.get("trifecta_diagnostics"),
        "condition_finish_position_segments": fold.get("condition_finish_position_segments", {}),
        "condition_model_segments": fold.get("condition_model_segments", {}),
        "condition_model_comparisons": fold.get("condition_model_comparisons", {}),
        "trained_condition_segments": fold.get("trained_condition_segments", {}),
    }


def main():
    print(f"Backtest range: {START} through {END}", flush=True)
    profile_started = time.perf_counter()
    profile_last = profile_started
    performance_profile = {}

    def mark_profile(stage):
        nonlocal profile_last
        now = time.perf_counter()
        elapsed = round(now - profile_last, 3)
        performance_profile[stage] = elapsed
        profile_last = now
        print(f"PERF_PROFILE {stage}={elapsed}s total={round(now-profile_started, 3)}s", flush=True)

    checkpoint_state = load_backtest_checkpoint()
    races, horses, payouts, months = load_archives()
    mark_profile("load_archives")
    saved_chronological = checkpoint_state.get("chronological")
    if saved_chronological:
        metrics = saved_chronological["metrics"]
        examples = saved_chronological["examples"]
        race_count = saved_chronological["race_count"]
        learned_weights = saved_chronological["learned_weights"]
        learning_updates = saved_chronological["learning_updates"]
        calibration_records = saved_chronological["calibration_records"]
        print("CHECKPOINT_RESUME stage=chronological_run_backtest reused=true", flush=True)
    else:
        metrics, examples, race_count, learned_weights, learning_updates, calibration_records = run_backtest(races, horses, payouts)
        checkpoint_state["chronological"] = {
            "metrics": metrics, "examples": examples, "race_count": race_count,
            "learned_weights": learned_weights, "learning_updates": learning_updates,
            "calibration_records": calibration_records,
        }
        save_backtest_checkpoint(checkpoint_state, "chronological-complete")
    mark_profile("chronological_run_backtest")
    production_temperature = fit_temperature(calibration_records)
    production_calibration = calibration_metrics(calibration_records, production_temperature)
    mark_profile("production_calibration")
    final_odds, final_odds_downloads = load_final_odds_archives(dt.date(2026, 3, 1), END)
    print(f"Official monthly final-odds races: {len(final_odds)}", flush=True)
    mark_profile("final_odds_archive_load")
    validation = checkpoint_state.get("validation")
    if validation:
        print("CHECKPOINT_RESUME stage=walk_forward_holdout_validation reused=true", flush=True)
    else:
        validation = holdout_validation(races, horses, payouts, final_odds, checkpoint_state)
        checkpoint_state["validation"] = validation
        save_backtest_checkpoint(checkpoint_state, "walk-forward-validation-complete")
    mark_profile("walk_forward_holdout_validation")
    preoff_type_report = preoff_ticket_type_backtest(payouts)
    mark_profile("preoff_ticket_type_backtest")
    race_budget_report = race_budget_auto_strategy_backtest(payouts, preoff_type_report)
    mark_profile("race_budget_strategy_backtest")
    preoff_ranking_report = preoff_ranking_strategy_backtest(payouts)
    mark_profile("preoff_ranking_strategy_backtest")
    result = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "period": {"start":START.isoformat(),"end":END.isoformat()},
        "source": "NAR official monthly race data (approximately four years, including a final one-year holdout); raw archives are not retained",
        "downloaded_months": months,
        "race_count": race_count,
        "horse_rows": sum(len(v) for v in horses.values()),
        "payout_races": sum(1 for p in payouts.values() if p.get("win")),
        "payout_rows": sum(p.get("source_rows", 0) for p in payouts.values()),
        "data_quality_audit": data_quality_audit(races, horses, payouts),
        "multi_row_payout_races": sum(1 for p in payouts.values() if p.get("source_rows", 0) > 1),
        "evaluated": sum(v["races"] for v in metrics.values()),
        "method": {
            "prediction_order": "date-batched chronological; every race on a date is predicted before any result from that date updates history or model weights",
            "odds_feature": False,
            "production_parity": "Primary chronological backtest uses the same D performance, going, pace, temperature and learned frame-context scoring blocks as live predictions; odds are overlaid only from genuine timestamped pre-off snapshots.",
            "frame_feature": "Prior top-three rate by venue, surface, 400m distance band, field-size band and frame number; 8% blend with D performance score, neutral prior for sparse/missing buckets; updates only after all races on a date are predicted.",
            "market_value_policy": "Simulate win bets only when calibrated model probability multiplied by timestamped official pre-off win odds meets a fixed threshold; thresholds are reported independently and not selected on test folds.",
            "probability_calibration": "temperature fitted on chronological prequential training predictions; out-of-sample calibration reported per walk-forward fold",
            "ticket_policy": "Research ticket portfolio is reported separately from the production shadow portfolio; each virtual ticket is 100 JPY and is not a betting recommendation.",
            "shadow_ticket_policy": "Exact production shadowTickets policy: model rank 1 win; ranks 1-2 place; ranks 1-2 wide; ranks 1-3 trio; 100 JPY each, evaluated separately from actual BET decisions.",
            "ticket_counts_per_race": {"wide":3,"exacta":4,"trio":10,"trifecta":24},
            "ticket_generation": "Active research portfolio only: wide all pairs among top 3 (3); exacta first-place top 2 x second-place top 3 excluding same horse (4); trio all combinations from top 5 (10); trifecta first-place top 2 x second-place top 5 excluding first x third-place remaining top 5 excluding first and second (24). Win, place and quinella are excluded from candidate selection and ticket evaluation. Each ticket costs 100 JPY.",
            "limitations": ["No historical pre-race odds feature; NAR monthly odds are confirmed odds and are not used.",
                            "All payout rows for dead-heat races are retained and matched across every winning combination.",
                            "Ticket ranks use field-size-normalized historical finishes; probabilities are temperature-calibrated and evaluated in rolling holdout folds.",
                            "Missing or incomplete historical rows are excluded where necessary."]
        },
        "metrics": metrics,
        "out_of_sample_validation": validation,
        "final_odds_market_benchmark": {
            "source": "NAR official monthly confirmed win odds; retrospective market benchmark only, not pre-off purchasable odds",
            "downloaded_months": final_odds_downloads,
            "races_with_win_odds": len(final_odds),
            "comparison": "C model top-ranked horse vs lowest final win odds (market favorite), same eligible races, one 100 JPY win ticket per race",
            "recent_year_fold": ((validation.get("recent_year_validation") or {}).get("fold") or {}).get("final_odds_market_comparison", {})
        },
        "strategy_comparison": {
            "A_baseline": "out_of_sample_validation.models.A_baseline (fixed reference weights)",
            "B_condition": "out_of_sample_validation.models.B_condition (learned base weights + prior track/going suitability)",
            "C_performance_pace": "out_of_sample_validation.models.C_performance_pace (B + prior margin/last3f/time + race pace + smoothed prior frame-context effect)",
            "D_preoff_odds": "preoff_ranking_strategy_backtest (timestamped 2-30 minute snapshots only; separately reports snapshot coverage)",
            "comparison_design": "A-C are evaluated on identical chronological holdout races with identical ticket construction and 100 JPY per ticket. B and C share the same learned base weights intentionally so their difference isolates the added feature blocks; A is the fixed baseline. D is evaluated only on races with valid genuine pre-off snapshots and is never imputed for missing snapshots.",
            "E_integrated": "out_of_sample_validation.integrated_strategy_E",
            "comparison_policy": "A-E are reported separately. D is marked unevaluable until enough genuine pre-off snapshots exist. No strategy is declared profitable unless sample minimums are met and the 95% race-cluster ROI lower bound exceeds 100%."
        },
        "odds_snapshot_archive": odds_snapshot_inventory(),
        "preoff_value_backtest": preoff_value_backtest(payouts),
        "preoff_ticket_type_backtest": preoff_type_report,
        "race_budget_auto_strategy_backtest": race_budget_report,
        "preoff_ranking_strategy_backtest": preoff_ranking_report,
        "performance_profile": {
            "stage_seconds": performance_profile,
            "total_seconds_before_report_serialization": round(time.perf_counter() - profile_started, 3),
            "note": "Wall-clock stage timings for identifying the next safe optimization target; timings do not affect model scores or selection."
        },
        "online_learning": {"updates": learning_updates, "final_weights": learned_weights,
                            "temperature": round(production_temperature,4), "calibration": production_calibration,
                            "note": "Daily batch learning: all races on a date are scored before that date's results update weights for later dates."},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    mark_profile("report_serialization_and_publish_views")

    # Publish three distinct backtest reports from the same cached source data
    # and the same untouched final-year holdout. No extra download or duplicate
    # model fitting is performed solely to create these views.
    recent_validation = validation.get("recent_year_validation", {}) if isinstance(validation, dict) else {}
    recent_fold = recent_validation.get("fold") or {}
    recent_models = recent_fold.get("models") or {}
    canonical_models = ("A_baseline", "B_condition", "C_performance_pace")
    ablation_names = [f"ablate_{feature}" for feature in ("form", "venue", "distance", "sire", "jockey", "age")]
    weight_variant_names = list(FEATURE_WEIGHT_NAMES)
    ablation_report = {
        "generated_at": result["generated_at"],
        "holdout_period": recent_validation.get("period"),
        "train_end": recent_validation.get("train_end"),
        "eligible_races": (recent_models.get("C_performance_pace") or {}).get("races"),
        "baseline_model": {
            "name": "C_performance_pace",
            "winner_top1_rate": (recent_models.get("C_performance_pace") or {}).get("winner_top1_rate"),
            "winner_top3_rate": (recent_models.get("C_performance_pace") or {}).get("winner_top3_rate"),
            "winner_topk_rate": (recent_models.get("C_performance_pace") or {}).get("winner_topk_rate"),
        },
        "frame_ablation": {
            "with_frame": recent_models.get("C_performance_pace"),
            "without_frame": recent_models.get("C_performance_pace_no_frame"),
            "comparison_policy": "Same chronological holdout races, same learned base weights, same D performance/pace features and identical tickets. Only the prior frame-context 8% blend is removed.",
        },
        "feature_ablations": {
            name: {
                "races": (recent_models.get(name) or {}).get("races"),
                "winner_top1_rate": (recent_models.get(name) or {}).get("winner_top1_rate"),
                "winner_top3_rate": (recent_models.get(name) or {}).get("winner_top3_rate"),
                "winner_topk_rate": (recent_models.get(name) or {}).get("winner_topk_rate"),
                "win_roi": (recent_models.get(name) or {}).get("win_roi"),
                "trio_roi": (((recent_models.get(name) or {}).get("bet_types") or {}).get("trio") or {}).get("roi"),
                "trio_hit_rate": (((recent_models.get(name) or {}).get("bet_types") or {}).get("trio") or {}).get("hit_rate"),
                "trio_profit": (((recent_models.get(name) or {}).get("bet_types") or {}).get("trio") or {}).get("profit"),
            } for name in ablation_names
        },
        "feature_weight_experiment": {},
        "market_comparison": recent_fold.get("final_odds_market_comparison", {}),
        "market_rank_diagnostics": recent_fold.get("market_rank_diagnostics", {}),
        "method": {
            "type": "feature_ablation_plus_single_and_combined_chronological_weight_challengers",
            "weight_policy": "Ablations remove one learned B-model feature and renormalize. Single-feature challengers scale one learned weight; D1-D4 apply the predefined jockey/venue/distance multipliers together and normalize once. Candidate selection uses the two earlier chronological development folds and the same 12-ticket, 1,900-yen trio portfolio, then assesses the selected candidate on the untouched final-year holdout.",
            "market_data": "Official final confirmed odds are retrospective only and were not available before the race. This is a market benchmark, not a live pre-off comparison.",
            "selection": "All model and market picks are compared on the same holdout races with available official final odds.",
            "production_impact": "Diagnostic only. No live model weights, predictions, or G4 preregistered value thresholds are changed.",
            "interpretation": "Compare winner_top1_rate first. A feature whose removal improves the top-1 rate is a candidate for deeper review, not automatic deletion. Check sample size and uncertainty before any change.",
        },
    }
    # Tune only on the two earlier chronological validation folds. The latest
    # year is untouched and is used once to assess the selected weight variant.
    prior_feature_folds = [
        fold for fold in (validation.get("folds") or [])
        if fold.get("period_label") in ("previous_year_2", "previous_year_1")
    ]
    baseline_dev_races = sum(int(((fold.get("models") or {}).get("C_performance_pace") or {}).get("races") or 0)
                             for fold in prior_feature_folds)
    baseline_dev_top1 = sum(int(((fold.get("models") or {}).get("C_performance_pace") or {}).get("winner_top1") or 0)
                            for fold in prior_feature_folds)
    baseline_dev_top3 = sum(int(((fold.get("models") or {}).get("C_performance_pace") or {}).get("winner_top3") or 0)
                            for fold in prior_feature_folds)
    def pooled_trio_amount(model_name, field_name):
        total = 0
        for fold in prior_feature_folds:
            metrics = ((fold.get("models") or {}).get(model_name) or {})
            narrowed = (((metrics.get("trio_diagnostics") or {}).get(
                "formation_2_4_5_narrowed") or {}).get("12") or {})
            total += int(narrowed.get(field_name) or 0)
        return total

    baseline_dev_trio_stake = pooled_trio_amount("C_performance_pace", "stake")
    baseline_dev_trio_return = pooled_trio_amount("C_performance_pace", "return")
    baseline_dev_top3_rate = round(100 * baseline_dev_top3 / baseline_dev_races, 2) if baseline_dev_races else None
    baseline_dev_top1_rate = round(100 * baseline_dev_top1 / baseline_dev_races, 2) if baseline_dev_races else None
    baseline_dev_trio_roi = round(100 * baseline_dev_trio_return / baseline_dev_trio_stake, 2) if baseline_dev_trio_stake else None
    baseline_dev_fold_rows = []
    for fold in prior_feature_folds:
        metrics = ((fold.get("models") or {}).get("C_performance_pace") or {})
        trio = (((metrics.get("trio_diagnostics") or {}).get(
            "formation_2_4_5_narrowed") or {}).get("12") or {})
        baseline_dev_fold_rows.append({
            "fold": fold.get("period_label"),
            "races": metrics.get("races"),
            "winner_top1_rate": metrics.get("winner_top1_rate"),
            "winner_top3_rate": metrics.get("winner_top3_rate"),
            "trio_roi": trio.get("roi"),
        })
    feature_weight_rows = {}
    for name in weight_variant_names:
        single_spec = FEATURE_WEIGHT_VARIANTS.get(name)
        feature, multiplier = single_spec if single_spec else ("combined", None)
        adjustments = FEATURE_WEIGHT_COMBINATIONS.get(name)
        fold_rows = []
        for fold in prior_feature_folds:
            metrics = ((fold.get("models") or {}).get(name) or {})
            trio = (((metrics.get("trio_diagnostics") or {}).get(
                "formation_2_4_5_narrowed") or {}).get("12") or {})
            fold_rows.append({
                "fold": fold.get("period_label"),
                "races": metrics.get("races"),
                "winner_top1_rate": metrics.get("winner_top1_rate"),
                "winner_top3_rate": metrics.get("winner_top3_rate"),
                "trio_roi": trio.get("roi"),
            })
        total_races = sum(int(row.get("races") or 0) for row in fold_rows)
        top1_hits = sum(int(((fold.get("models") or {}).get(name) or {}).get("winner_top1") or 0)
                        for fold in prior_feature_folds)
        top3_hits = sum(int(((fold.get("models") or {}).get(name) or {}).get("winner_top3") or 0)
                        for fold in prior_feature_folds)
        trio_stake = pooled_trio_amount(name, "stake")
        trio_return = pooled_trio_amount(name, "return")
        feature_weight_rows[name] = {
            "feature": feature, "multiplier": multiplier,
            "combined_adjustments": (
                [{"feature": item[0], "multiplier": item[1]} for item in adjustments]
                if adjustments else None
            ),
            "development_folds": fold_rows,
            "development_races": total_races,
            "development_winner_top1_rate": round(100 * top1_hits / total_races, 2) if total_races else None,
            "development_winner_top3_rate": round(100 * top3_hits / total_races, 2) if total_races else None,
            "development_trio_roi": round(100 * trio_return / trio_stake, 2) if trio_stake else None,
            "holdout": recent_models.get(name),
        }
    selected_weight_name = select_feature_weight_candidate(
        feature_weight_rows, baseline_dev_top3_rate, baseline_dev_trio_roi,
        roi_tolerance_pp=1.0, baseline_development_folds=baseline_dev_fold_rows,
        baseline_top1_rate=baseline_dev_top1_rate
    ) if feature_weight_rows and prior_feature_folds else None
    selected_weight = feature_weight_rows.get(selected_weight_name) if selected_weight_name else None
    baseline_c = recent_models.get("C_performance_pace") or {}
    selected_holdout = (selected_weight or {}).get("holdout") or {}
    holdout_trio_base = (((baseline_c.get("trio_diagnostics") or {}).get(
        "formation_2_4_5_narrowed") or {}).get("12") or {})
    holdout_trio_candidate = (((selected_holdout.get("trio_diagnostics") or {}).get(
        "formation_2_4_5_narrowed") or {}).get("12") or {})
    feature_weight_experiment = {
        "candidate_variants": feature_weight_rows,
        "baseline_development": {"races": baseline_dev_races, "winner_top1_rate": baseline_dev_top1_rate,
                                 "winner_top3_rate": baseline_dev_top3_rate, "trio_roi": baseline_dev_trio_roi,
                                 "folds": baseline_dev_fold_rows},
        "selected_on_prior_folds": selected_weight_name,
        "selection_rule": "A candidate must improve pooled top-1 and top-3 coverage over C, stay within 1.0 ROI percentage point for the 12-ticket 2-4-5 narrowed trio portfolio (19 x 100-yen units; 1,900 yen per race), and independently improve both top-1 and top-3 while staying within the same ROI tolerance in every prior development fold. Among eligible candidates, rank by pooled top3, then top1, then trio ROI. The latest one-year holdout is untouched.",
        "development_roi_tolerance_pp": 1.0,
        "trio_comparison_policy": "12 unique 2-4-5 narrowed tickets per race, 19 x 100-yen units allocated equally, fixed 1,900-yen race budget; identical race set for every model.",
        "holdout_comparison": {
            "baseline_C": {
                "races": baseline_c.get("races"),
                "winner_top1_rate": baseline_c.get("winner_top1_rate"),
                "winner_top3_rate": baseline_c.get("winner_top3_rate"),
                "trio_roi": holdout_trio_base.get("roi"),
                "trio_roi_ci95_normal": {
                    "low": holdout_trio_base.get("roi_ci95_normal_low"),
                    "high": holdout_trio_base.get("roi_ci95_normal_high"),
                },
            },
            "selected_candidate": {
                "name": selected_weight_name,
                "races": selected_holdout.get("races"),
                "winner_top1_rate": selected_holdout.get("winner_top1_rate"),
                "winner_top3_rate": selected_holdout.get("winner_top3_rate"),
                "trio_roi": holdout_trio_candidate.get("roi"),
                "trio_roi_ci95_normal": {
                    "low": holdout_trio_candidate.get("roi_ci95_normal_low"),
                    "high": holdout_trio_candidate.get("roi_ci95_normal_high"),
                },
            },
            "winner_top1_delta_pp": round((selected_holdout.get("winner_top1_rate") or 0) - (baseline_c.get("winner_top1_rate") or 0), 2) if selected_holdout else None,
            "winner_top3_delta_pp": round((selected_holdout.get("winner_top3_rate") or 0) - (baseline_c.get("winner_top3_rate") or 0), 2) if selected_holdout else None,
            "trio_roi_delta_pp": round((holdout_trio_candidate.get("roi") or 0) - (holdout_trio_base.get("roi") or 0), 2) if selected_holdout and holdout_trio_candidate else None,
            "promotion_ready": False,
            "selection_guardrail_passed": bool(selected_weight_name),
            "note": "Candidate remains research-only. Promotion requires repeatable improvement in ranking and 12-ticket trio ROI on independent future holdout data; this single selected holdout does not change live weights.",
        },
    }
    ablation_report["feature_weight_experiment"] = feature_weight_experiment
    (OUT.parent / "backtest-5-feature-ablation.json").write_text(
        json.dumps(ablation_report, ensure_ascii=False, indent=2), encoding="utf-8")

    model_report = {
        "generated_at": result["generated_at"],
        "period": result["period"],
        "holdout": {
            "status": recent_validation.get("status"),
            "period": recent_validation.get("period"),
            "train_end": recent_validation.get("train_end"),
            "test_races": recent_validation.get("test_races_expected"),
            "method": validation.get("method") if isinstance(validation, dict) else None,
        },
        "models": {name: recent_models.get(name) for name in canonical_models},
        "D_preoff_odds": result.get("preoff_ranking_strategy_backtest"),
        "comparison_policy": result["strategy_comparison"]["comparison_design"],
    }
    final_market_report = {
        "generated_at": result["generated_at"],
        "source": result["final_odds_market_benchmark"],
        "holdout_period": recent_validation.get("period"),
        "comparison": result["final_odds_market_benchmark"]["recent_year_fold"],
        "interpretation": "Final confirmed odds are published after race completion and are used only as a retrospective market benchmark. This is not a live betting strategy and must not be mixed into model E."
    }
    (OUT.parent / "backtest-4-final-market-comparison.json").write_text(
        json.dumps(final_market_report, ensure_ascii=False, indent=2), encoding="utf-8")

    (OUT.parent / "backtest-1-model-comparison.json").write_text(
        json.dumps(model_report, ensure_ascii=False, indent=2), encoding="utf-8")

    ticket_report = {
        "generated_at": result["generated_at"],
        "holdout_period": recent_validation.get("period"),
        "train_end": recent_validation.get("train_end"),
        "ranking_model": "C_performance_pace",
        "ticket_policy": result["method"]["ticket_generation"],
        "stake_per_ticket_jpy": 100,
        "bet_types": (recent_models.get("C_performance_pace") or {}).get("bet_types", {}),
        "trio_formation_comparison": (recent_models.get("C_performance_pace") or {}).get("trio_diagnostics", {}),
        "production_shadow_portfolio": (recent_models.get("C_performance_pace") or {}).get("shadow_portfolio", {}),
        "race_budget_auto_strategy": result.get("race_budget_auto_strategy_backtest", {}),
        "all_model_bet_type_comparison": {
            name: (recent_models.get(name) or {}).get("bet_types", {})
            for name in canonical_models
        },
        "interpretation": "The fixed-ticket comparison remains separate from the integrated live-policy simulation. race_budget_auto_strategy uses only current-version timestamped pre-off snapshots, development-selected thresholds, one bet type per race, a maximum of JPY 2,000 per race, and skips races with no qualified ticket.",
    }
    (OUT.parent / "backtest-2-ticket-strategy.json").write_text(
        json.dumps(ticket_report, ensure_ascii=False, indent=2), encoding="utf-8")

    condition_report = {
        "generated_at": result["generated_at"],
        "holdout_period": recent_validation.get("period"),
        "train_end": recent_validation.get("train_end"),
        "test_races": recent_validation.get("test_races_expected"),
        "venue_surface_going_distance_comparisons": recent_fold.get("condition_model_comparisons", {}),
        "condition_finish_position_segments": recent_fold.get("condition_finish_position_segments", {}),
        "trained_condition_segments": recent_fold.get("trained_condition_segments", {}),
        "interpretation": "Condition segments are diagnostics within the untouched final-year holdout. Small segments must be treated as descriptive, not as evidence of a profitable betting edge.",
    }
    (OUT.parent / "backtest-3-condition-robustness.json").write_text(
        json.dumps(condition_report, ensure_ascii=False, indent=2), encoding="utf-8")

    # Publish a compact, directly retrievable report for the trifecta-only
    # research loop. Keep every fold separate so candidate comparisons cannot
    # silently pool tuning and evaluation data.
    validation_folds = validation.get("folds", []) if isinstance(validation, dict) else []
    research_folds = [build_trifecta_research_fold(fold) for fold in validation_folds]
    research_report = {
        "generated_at": result["generated_at"],
        "period": result["period"],
        "source_races": race_count,
        "out_of_sample_races": validation.get("test_races") if isinstance(validation, dict) else None,
        "folds": research_folds,
        "walk_forward_selection": (validation.get("auto_improvement_loop") or {}) if isinstance(validation, dict) else {},
        "interpretation": "Fold-specific results are preserved. Formation/blend grids are exploratory unless selected using prior-fold data and evaluated on a later untouched fold. No result is a betting recommendation.",
    }
    (OUT.parent / "trifecta-research.json").write_text(
        json.dumps(research_report, ensure_ascii=False, indent=2), encoding="utf-8")

    validated_weights = learned_weights
    live_model = {
        "model_version": "nar-chronological-v9-D-frame-preoff",
        "generated_at": result["generated_at"],
        "training_period": result["period"],
        "updates": learning_updates,
        "validation": validation,
        "feature_schema": ["form", "venue", "distance", "sire", "jockey", "age"],
        "condition_feature_schema": ["going", "surface", "going_quality"],
        "frame_feature_schema": {
            "fields": ["venue", "surface", "distance_band_400m", "field_size_band_4", "frame_no"],
            "score_weight": 0.08,
            "prior_rate": 0.50,
            "prior_strength": 8.0,
            "history_source": "official historical horse rows",
            "update_policy": "Only earlier dates are used for each prediction; all races on a date are scored before that date's results update frame statistics."
        },
        "experimental_condition_variants": {
            "C1_going_match": {"going_weight": 0.10, "minimum_prior_starts": 1,
                               "smoothing_prior_strength": 4, "promotion_requires_holdout_gain": True},
            "C2_going_plus_pace": {"going_weight": 0.10, "pace_adjustment": True,
                                   "promotion_requires_holdout_gain": True}
        },
        "experimental_d_features": ["recent_margin_quality", "recent_race_relative_last3f_rank", "prior_same_going_quality"],
        "experimental_performance_variants": {
            "D1_combined": {"margin_weight": 0.10, "closing_weight": 0.10},
            "D2_combined_with_pace": {"margin_weight": 0.10, "closing_weight": 0.10, "pace": True},
            "D3_margin_only": {"margin_weight": 0.10, "closing_weight": 0.00},
            "D4_closing_only": {"margin_weight": 0.00, "closing_weight": 0.10},
            "D5_light_combined": {"margin_weight": 0.05, "closing_weight": 0.05}
        },
        "pace_feature_schema": ["first_corner_pct", "last_corner_pct"],
        "pace_feature_note": "Experimental trained_with_pace model is holdout-only; production weights remain unchanged until validation passes.",
        "condition_feature_note": "Going suitability uses only prior starts on the same official going label, with shrinkage toward neutral for sparse samples. The C1 variant remains experimental until untouched holdout validation supports promotion.",
        "weights": validated_weights,
        "temperature": production_temperature,
        "calibration": production_calibration,
        "prediction_policy": "Scores use only races strictly earlier than the target race date; current-race finish/popularity/payout are excluded. All races on a date are predicted before any results from that date update the model and frame statistics."
    }
    (OUT.parent / "live-model.json").write_text(json.dumps(live_model,ensure_ascii=False,indent=2),encoding="utf-8")
    live_history = build_live_history(races, horses)
    (OUT.parent / "live-history.json").write_text(json.dumps(live_history,ensure_ascii=False,separators=(",",":")),encoding="utf-8")
    print(json.dumps({"races":race_count,"horse_rows":result["horse_rows"],"payout_races":sum(1 for p in payouts.values() if p.get("win")),"metrics":metrics,"learning_updates":learning_updates,"final_weights":learned_weights},ensure_ascii=False),flush=True)
    if race_count == 0 or not any(p.get("win") for p in payouts.values()):
        raise RuntimeError("No usable historical race/payout data parsed; refusing to publish an empty backtest.")

if __name__ == "__main__":
    main()

# Re-run historical validation on 2026-11-03; no model behavior changed.
