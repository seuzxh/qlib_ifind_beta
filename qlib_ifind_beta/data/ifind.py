"""Thin iFinD HTTP client: flat ``.ifind_token`` cache + history_data + data_pool.

Reuses the canonical token-refresh pattern from ``qlib_data/scripts/ifind`` (flat
``access_token=.../expired_time=...`` file at ``/home/zxh/qlib_data/.ifind_token``)
and the battle-tested payload/parse shapes from the sibling ``2.qlib_ifind_hot_concept``
project (LIVE-VERIFIED 2026-06-20).

Only the two endpoints the MVP needs:

* ``history_data`` (quant gateway, CPS=0 raw price) — dump SH883926 benchmark.
* ``data_pool`` (quant gateway) — p03473 = current 883926 constituents.

Secrets discipline: the refresh token is read from the environment or parsed
at runtime from an existing ``qlib_data`` script — it is NEVER hardcoded,
printed, or committed here.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import requests

from .config import IFIND_TOKEN_FILE

# quant gateway
IFIND_TOKEN_URL = "https://quantapi.51ifind.com/api/v1/get_access_token"
IFIND_HISTORY_URL = "https://quantapi.51ifind.com/api/v1/history_data"
IFIND_DATAPOOL_URL = "https://quantapi.51ifind.com/api/v1/data_pool"

# Existing qlib_data credential sources (parsed at runtime, never copied here).
_REFRESH_TOKEN_PATHS = (
    Path("/home/zxh/qlib_data/scripts/daily_update.py"),
    Path("/home/zxh/qlib_data/scripts/verify_data.py"),
    Path("/home/zxh/qlib_data/scripts/qlib_dumper/instrument_source.py"),
)

# history_data indicator → output column (matches sibling _HD_FIELD_MAP; CPS=0 = raw)
HD_FIELD_MAP = {
    "pre_close": "pre_close",
    "open": "open", "high": "high", "low": "low", "close": "close",
    "vwap": "vwap", "chg": "chg", "pct_chg": "pct_chg",
    "volume": "volume", "amt": "amount", "turn": "turnover_ratio",
}
HISTORY_DATA_INDICATORS = list(HD_FIELD_MAP.keys())


class IfindError(RuntimeError):
    """Raised for permanent iFinD failures (bad params, quota exhausted, non-JSON)."""


# --- refresh_token loader (mirrors qlib_data canonical _load_ifind_refresh_token) ---
def load_refresh_token() -> str:
    """Load ``IFIND_REFRESH_TOKEN`` without copying it into this repository."""
    token = os.environ.get("IFIND_REFRESH_TOKEN", "").strip()
    if token:
        return token
    for path in _REFRESH_TOKEN_PATHS:
        if not path.exists():
            continue
        with path.open() as source:
            for line in source:
                stripped = line.strip()
                if stripped.startswith("IFIND_REFRESH_TOKEN") and "=" in stripped:
                    _, rhs = stripped.split("=", 1)
                    token = rhs.strip().strip('"').strip("'")
                    if token:
                        return token
    searched = ", ".join(str(path) for path in _REFRESH_TOKEN_PATHS)
    raise RuntimeError(f"IFIND_REFRESH_TOKEN not found in environment or: {searched}")


# --- token cache (flat file: access_token=... / expired_time=...) ---
def _read_token_cache(token_file: Path = IFIND_TOKEN_FILE) -> tuple[str | None, str | None]:
    if not token_file.exists():
        return None, None
    tok = exp = None
    for line in token_file.read_text().splitlines():
        if line.startswith("access_token="):
            tok = line.split("=", 1)[1].strip()
        elif line.startswith("expired_time="):
            exp = line.split("=", 1)[1].strip()
    return tok, exp


def _is_expired(expired_time: str | None) -> bool:
    if not expired_time:
        return True
    try:
        # refresh a bit before the wire edge
        return time.strptime(expired_time, "%Y-%m-%d %H:%M:%S") <= time.localtime()
    except (ValueError, TypeError):
        return True


def _refresh_access_token(refresh_token: str, token_file: Path = IFIND_TOKEN_FILE) -> str:
    r = requests.post(
        IFIND_TOKEN_URL,
        headers={"Content-Type": "application/json"},
        json={"refresh_token": refresh_token},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    if str(data.get("errorcode")) not in ("0", "None", "", "nan", None):
        raise IfindError(f"token refresh failed: errorcode={data.get('errorcode')} {data.get('errmsg')}")
    access = data["data"]["access_token"]
    exp = data["data"]["expired_time"]
    token_file.write_text(f"access_token={access}\nexpired_time={exp}\n")
    return access


def get_access_token(token_file: Path = IFIND_TOKEN_FILE, refresh_token: str | None = None) -> str:
    tok, exp = _read_token_cache(token_file)
    if tok and not _is_expired(exp):
        return tok
    if refresh_token is None:
        refresh_token = load_refresh_token()
    return _refresh_access_token(refresh_token, token_file)


# --- HTTP core ----------------------------------------------------------------
def _post(url: str, payload: dict, *, timeout: int = 240, retries: int = 3,
          token_file: Path = IFIND_TOKEN_FILE, refresh_token: str | None = None) -> dict:
    """POST + JSON decode + token refresh-on-errorcode + transient backoff.

    Permanent errors (HTTP 4xx except 429 / non-JSON 2xx / bad errorcode after refresh)
    raise :class:`IfindError` immediately — fail-fast rather than burn the pipeline.
    """
    access = get_access_token(token_file, refresh_token)
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            r = requests.post(url, json=payload,
                              headers={"Content-Type": "application/json", "access_token": access},
                              timeout=timeout)
            if 400 <= r.status_code < 500 and r.status_code != 429:
                raise IfindError(f"iFinD HTTP {r.status_code} (permanent) @ {url}: {r.text[:200]!r}")
            if r.status_code >= 500 or r.status_code == 429:
                raise requests.HTTPError(f"iFinD HTTP {r.status_code} (transient) @ {url}: {r.text[:200]!r}")
            data = r.json()
            # token-expired signature: errorcode != 0 → refresh once, retry once
            if str(data.get("errorcode")) not in ("0", "None", "", "nan", None):
                access = _refresh_access_token(refresh_token or load_refresh_token(), token_file)
                data = requests.post(url, json=payload,
                                     headers={"Content-Type": "application/json", "access_token": access},
                                     timeout=timeout).json()
            if str(data.get("errorcode")) not in ("0", "None", "", "nan", None):
                raise IfindError(f"iFinD errorcode={data.get('errorcode')} @ {url}: errormsg={data.get('errmsg')!r}")
            return data
        except IfindError:
            raise
        except (ValueError, requests.RequestException) as e:
            last_exc = e
            if attempt < retries:
                time.sleep(min(2 ** attempt, 8))
    raise IfindError(f"iFinD request failed after {retries + 1} attempts @ {url}: {last_exc}")


# --- parsers ------------------------------------------------------------------
def parse_tables_long(resp: dict, field_map: dict | None = None) -> pd.DataFrame:
    """Parse history_data / date_sequence ``tables:[{thscode,time,table}]`` → long DF.

    Output: ``[symbol, date, <fields>]`` (one row per code×date).
    """
    tables = resp.get("tables") or resp.get("result", {}).get("tables", [])
    frames = []
    for t in tables:
        code = t.get("thscode") or t.get("code") or t.get("symbol")
        times = t.get("time") or t.get("times")
        tbl = t.get("table") or t.get("columns") or {}
        if code is None or times is None or not tbl:
            continue
        col = {"symbol": code, "date": list(times)}
        for ind, vals in tbl.items():
            col[field_map.get(ind, ind) if field_map else ind] = list(vals)
        frames.append(pd.DataFrame(col))
    if not frames:
        return pd.DataFrame(columns=["symbol", "date"])
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df


def parse_data_pool(resp: dict, field_map: dict | None = None) -> pd.DataFrame:
    """Parse data_pool parallel-array response ``tables[0].table = {field: [vals]}``."""
    tables = resp.get("tables") or resp.get("result", {}).get("tables", [])
    if not tables:
        return pd.DataFrame()
    tbl = tables[0].get("table") or tables[0].get("columns") or {}
    out = {}
    for k, vals in tbl.items():
        out[field_map.get(k, k) if field_map else k] = list(vals)
    return pd.DataFrame(out)


# --- public fetchers ----------------------------------------------------------
def fetch_history_data(code: str, start: str, end: str,
                       indicators: list[str] | None = None,
                       cps: str = "0", **kwargs) -> pd.DataFrame:
    """history_data (quant gateway, CPS=0 raw). Returns long DF [symbol,date,...].

    Single code per call (benchmark use case); history_data silently truncates >10
    codes per the sibling's LIVE-VERIFIED note. For multi-code, call per-code or chunk.
    """
    payload = {"reqBody": {
        "codes": code,
        "startdate": start, "enddate": end,
        "indicators": ",".join(indicators or HISTORY_DATA_INDICATORS),
        "functionpara": {"CPS": cps},
    }}
    return parse_tables_long(_post(IFIND_HISTORY_URL, payload, **kwargs), HD_FIELD_MAP)


def fetch_data_pool(reportname: str, functionpara: dict, outputpara: str | list[str],
                    field_map: dict | None = None, **kwargs) -> pd.DataFrame:
    """data_pool report endpoint. ``outputpara`` may be a comma-string or list."""
    if isinstance(outputpara, (list, tuple)):
        outputpara = ",".join(outputpara)
    payload = {"reportname": reportname,
               "functionpara": functionpara,
               "outputpara": outputpara}
    return parse_data_pool(_post(IFIND_DATAPOOL_URL, payload, **kwargs), field_map)
