"""Fetch 883926 constituents as a **time-varying** universe (iFinD data_pool p03473).

p03473 live-verified schema (2026-07-05); historical ``iv_date`` confirmed working
(sampled 2024-01-02 / 2025-06-03 / 2026-07-01, each returns 100 rows):
* ``p03473_f001`` = as-of date ('YYYY-MM-DD')
* ``p03473_f002`` = constituent code in **iFinD** format ('000536.SZ')
* ``p03473_f003`` = name (Chinese)

**Time-varying universe (T-day pool, no lookahead)** — user decision 2026-07-10:
"883926 股池 T 日盘前更新；T 日的观察股池 = 883926 的 T 日股池".

Mechanism: for each constituent code, find its continuous membership segments
``[d_in, d_out]`` from daily snapshots and register them **as-is** (no shift).
A T-day ``D.features`` query returns instruments with ``start <= T <= end`` ⇔
exactly the T-day membership. The 883926 index is updated pre-market on day T,
so the T-day pool is known before the 9:41 decision → no lookahead.
``Strategy``/``Exchange`` need zero changes — qlib filters by the per-instrument
date range natively.

**History**: 2026-07-05 to 2026-07-09 used T-1 lag (``shift_T1`` +1 trading day)
under the assumption that the 883926 pool was T-1 updated. Verified 2026-07-10
that it is actually T-day pre-market updated → ``dump_universe`` now uses raw
segments. ``shift_T1`` + ``FAR_FUTURE`` retained for reference / rollback.

Snapshots are cached to ``data/universe_snapshots.csv`` (long format, resumable);
re-running ``dump_universe`` after a successful backfill hits zero iFinD API.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DAY_CAL, INDEX_CODE_IFIND, PROJECT_ROOT, QLIB_DATA, UNIVERSE_MARKET
from . import ifind

P03473_FIELDS = ["p03473_f001", "p03473_f002", "p03473_f003"]
P03473_FIELD_MAP = {"p03473_f001": "date", "p03473_f002": "code_ifind", "p03473_f003": "name"}

ALL_INSTRUMENTS = QLIB_DATA / "instruments" / "all.txt"

# Resumable daily-snapshot cache (long: date, code_ifind, code_qlib, name).
SNAPSHOT_CACHE = PROJECT_ROOT / "data" / "universe_snapshots.csv"

# T-1 shift sentinel for a segment whose d_out is the calendar's last day (still
# in the index as of the window end). Matches qlib IndexBase "open end" semantics.
FAR_FUTURE = "2099-12-31"


def ifind_to_qlib(code_ifind: str) -> str:
    """``000536.SZ``→``SZ000536``; ``600519.SH``→``SH600519``; ``430090.BJ``→``BJ430090``."""
    body, suffix = code_ifind.split(".", 1)
    return suffix.upper() + body


def load_all_instruments(path: Path = ALL_INSTRUMENTS) -> dict[str, tuple[str, str]]:
    """Read qlib_data instruments/all.txt → {qlib_code: (start_date, end_date)}."""
    out: dict[str, tuple[str, str]] = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) >= 3:
            out[parts[0]] = (parts[1], parts[2])
    return out


def load_day_calendar(path: Path = DAY_CAL) -> list[str]:
    """Load ``calendars/day.txt`` → list of 'YYYY-MM-DD' (full 26y calendar)."""
    return [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()]


def load_day_calendar_window(start: str, end: str, path: Path = DAY_CAL) -> list[str]:
    """Trading days within [start, end] inclusive → ['YYYY-MM-DD', ...]."""
    return [d for d in load_day_calendar(path) if start <= d <= end]


def fetch_constituents(code_ifind: str = INDEX_CODE_IFIND, iv_date: str | None = None) -> pd.DataFrame:
    """Fetch p03473 constituents for the index on a single ``iv_date``.

    ``iv_date`` is 'YYYYMMDD'; defaults to the calendar's last day (most recent
    snapshot). Returns DF [date, code_ifind, name, code_qlib].
    """
    if iv_date is None:
        iv_date = load_day_calendar()[-1].replace("-", "")
    df = ifind.fetch_data_pool(
        reportname="p03473",
        functionpara={"iv_date": iv_date, "iv_zsdm": code_ifind},
        outputpara=P03473_FIELDS,
        field_map=P03473_FIELD_MAP,
    )
    if df.empty:
        raise RuntimeError(f"p03473 returned no constituents for {code_ifind} @ {iv_date}")
    df["code_qlib"] = df["code_ifind"].map(ifind_to_qlib)
    return df


# --- time-varying pipeline ---------------------------------------------------
def fetch_history_snapshots(start: str, end: str,
                            code_ifind: str = INDEX_CODE_IFIND,
                            cache_path: Path = SNAPSHOT_CACHE) -> pd.DataFrame:
    """Walk [start, end] trading days, fetch daily p03473 snapshots, resumable.

    Caches to ``cache_path`` (long CSV). Days already in the cache are skipped;
    only missing days hit the API. Flushes every 50 days so a mid-run crash keeps
    progress. Returns long DF [date, code_ifind, code_qlib, name].
    """
    dates = load_day_calendar_window(start, end)
    cached_df = pd.read_csv(cache_path) if cache_path.exists() else pd.DataFrame()
    have = set(cached_df["date"].astype(str).unique()) if not cached_df.empty else set()
    todo = [d for d in dates if d not in have]
    print(f"universe snapshots: {len(dates)} days in window, {len(have)} cached, {len(todo)} to fetch")

    new_frames: list[pd.DataFrame] = []
    for i, d in enumerate(todo, 1):
        df = fetch_constituents(code_ifind, iv_date=d.replace("-", ""))
        df["date"] = d
        df = df[["date", "code_ifind", "code_qlib", "name"]]
        new_frames.append(df)
        if i % 50 == 0:
            print(f"  …{i}/{len(todo)} days fetched (flushing cache)")
            _flush_cache(cache_path, cached_df, new_frames)
    out = _flush_cache(cache_path, cached_df, new_frames)
    if todo:
        print(f"  ✓ {len(todo)} new days fetched → {cache_path}")
    return out


def _flush_cache(cache_path: Path, cached_df: pd.DataFrame,
                 new_frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concat cached + new frames, write CSV, return the merged DF."""
    frames = [cached_df] + new_frames if not cached_df.empty else new_frames
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["date", "code_ifind", "code_qlib", "name"])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache_path, index=False)
    return out


def snapshots_to_segments(df: pd.DataFrame) -> dict[str, list[tuple[str, str]]]:
    """Long snapshot DF → {code_qlib: [(d_in, d_out), ...]} continuous segments.

    A code may have multiple segments (entered, left, re-entered). Continuity is
    over the window's trading-day calendar (consecutive calendar indices).
    """
    if df.empty:
        return {}
    cal = sorted(df["date"].astype(str).unique())
    idx_of = {d: i for i, d in enumerate(cal)}
    segments: dict[str, list[tuple[str, str]]] = {}
    for code, day_set in df.groupby("code_qlib")["date"].apply(lambda s: set(s.astype(str))).items():
        idxs = sorted(idx_of[d] for d in day_set)
        segs: list[tuple[str, str]] = []
        seg_start, prev = idxs[0], idxs[0]
        for i in idxs[1:]:
            if i == prev + 1:
                prev = i
            else:
                segs.append((cal[seg_start], cal[prev]))
                seg_start, prev = i, i
        segs.append((cal[seg_start], cal[prev]))
        segments[code] = segs
    return segments


def shift_T1(segments: dict[str, list[tuple[str, str]]],
             calendar: list[str] | None = None) -> dict[str, list[tuple[str, str]]]:
    """Shift every segment +1 trading day: (d_in,d_out) → (next(d_in), next(d_out)).

    Uses the **full** day calendar (not just the window) so next(d) for the
    window's last day still resolves. d at calendar end → FAR_FUTURE sentinel.
    """
    cal = calendar if calendar is not None else load_day_calendar()
    nxt = {d: (cal[i + 1] if i + 1 < len(cal) else FAR_FUTURE) for i, d in enumerate(cal)}
    return {code: [(nxt[s], nxt[e]) for (s, e) in segs] for code, segs in segments.items()}


def dump_universe(start: str = "2024-01-01", end: str = "2026-07-02",
                  code_ifind: str = INDEX_CODE_IFIND, market: str = UNIVERSE_MARKET,
                  cache_path: Path = SNAPSHOT_CACHE) -> tuple[Path, list[str]]:
    """Time-varying end-to-end: daily snapshots → segments → instruments file (T-day pool).

    Writes ``instruments/<market>.txt`` (TSV ``code\\tstart\\tend``, one row per
    code×segment). Segments are used **as-is** (no T-1 shift): the 883926 pool is
    updated pre-market on day T, so qlib's ``start <= T <= end`` returns exactly
    the T-day membership — known before the 9:41 decision → no lookahead.
    Returns (path, sorted list of every historical code seen — the superset
    build_overlay must link+materialize so backtests reaching into the train
    window see bins for stocks no longer in the index today).
    """
    from .overlay import write_market_file
    snapshots = fetch_history_snapshots(start, end, code_ifind, cache_path)
    segments = snapshots_to_segments(snapshots)
    # T-day pool: 883926 updated pre-market on T → no shift needed (was shift_T1
    # before 2026-07-10; see module docstring History section).
    records = [(code, s, e) for code, segs in segments.items() for (s, e) in segs]
    path = write_market_file(market, records)
    all_codes = sorted(segments.keys())
    n_seg = len(records)
    print(f"universe: {len(all_codes)} historical codes, {n_seg} segments "
          f"(avg {n_seg / max(len(all_codes), 1):.1f} seg/code) → {path}")
    return path, all_codes


def build_market(records, all_instruments: dict[str, tuple[str, str]] | None = None,
                 market: str = UNIVERSE_MARKET, fallback_start: str = "2020-01-01"):
    """Static fallback path: write instruments/<market>.txt from a list of qlib codes
    using each code's full (start, end) from all.txt. **Not used by the time-varying
    dump_universe** — kept for ad-hoc static pools / debugging.
    """
    from .overlay import write_market_file
    all_inst = all_instruments if all_instruments is not None else load_all_instruments()
    out_codes, out_records = [], []
    missing = []
    for code in records:
        if code in all_inst:
            s, e = all_inst[code]
        else:
            s, e = fallback_start, "2099-12-31"
            missing.append(code)
        out_records.append((code, s, e))
        out_codes.append(code)
    path = write_market_file(market, out_records)
    if missing:
        print(f"⚠️  {len(missing)} constituents not in qlib_data all.txt (used fallback dates): {missing}")
    return path, out_codes
