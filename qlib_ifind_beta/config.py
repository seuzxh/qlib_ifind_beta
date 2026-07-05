"""Project paths and constants.

All paths are absolute. The readonly qlib_data tree is the single source of truth
for the 7 base fields; our overlay (`data/qlib_root/`) layers derived bins on top.
"""
from __future__ import annotations

from pathlib import Path

# --- filesystem layout -------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
QLIB_DATA = Path("/home/zxh/qlib_data")              # readonly source (26y daily, 7 fields)
OVERLAY_ROOT = PROJECT_ROOT / "data" / "qlib_root"    # provider_uri target (our farm)

FEATURES_SRC = QLIB_DATA / "features"                 # source feature bins (readonly)
FEATURES_DST = OVERLAY_ROOT / "features"              # overlay feature dir (symlinks + derived)
INSTRUMENTS_DST = OVERLAY_ROOT / "instruments"
CALENDAR_DST = OVERLAY_ROOT / "calendars"

# --- iFinD -------------------------------------------------------------------
IFIND_TOKEN_FILE = QLIB_DATA / ".ifind_token"
IFIND_BASE = "https://quantapi.51ifind.com/api/v1"
IFIND_HISTORY_URL = f"{IFIND_BASE}/history_data"     # historical quotes (883926 etc.)
IFIND_DATAPOOL_URL = f"{IFIND_BASE}/data_pool"       # report-style data (p03473 constituents)

# --- 883926 high-beta index --------------------------------------------------
INDEX_CODE_IFIND = "883926.TI"        # iFinD code (`.TI` = 同花顺指数)
INDEX_CODE_QLIB = "SH883926"          # qlib storage code if we ever dump it
UNIVERSE_MARKET = "highbeta883926"    # instruments market name for qrun
# Benchmark: SH000300 (CSI300), reused read-only from qlib_data (26y clean bins)
# via overlay.link_stock + materialize. 883926 deferred: iFinD history_data returns
# an incoherent 883926.TI close series (20k→2.8M drift then −99.96% snap on
# 2026-05-25; vwap decoupled by 2-3 orders) — first-principles probe 2026-07-05,
# functionpara/CPS has zero effect. User decision: use 000300 for now.
BENCHMARK = "SH000300"

# --- fields ------------------------------------------------------------------
BASE_FIELDS = ("open", "high", "low", "close", "volume", "factor", "vwap")
DERIVED_FIELDS = ("change", "limit_up", "limit_down")
FREQ = "day"

# calendar
DAY_CAL = QLIB_DATA / "calendars" / "day.txt"
