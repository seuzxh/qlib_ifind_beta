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

# --- minute-frequency factors (T-day 9:30-9:40, materialized as day.bin) ------
# Source: /home/zxh/cn_data_1min (readonly, 1min bins, 242 slots/day).
# Slot map (probe-verified 2026-07-06): slot 0 (09:30) is universally NaN
# pool-wide — every day, every stock (call-auction placeholder, 0/604 non-NaN).
# First REAL bar = slot 1 (covers [09:30,09:31), the open continuous-bid minute); daily
# open == minute_open[slot 1] exactly (8/8 days verified). 240 real bars/day
# (242 − slot 0 − slot 121); slot 121 (13:00 mid-day break) is also universally NaN;
# slot 241 (15:00) has real data (602/604 days).
CN_DATA_1MIN = Path("/home/zxh/cn_data_1min")
FEATURES_1MIN_SRC = CN_DATA_1MIN / "features"
MIN_CAL = CN_DATA_1MIN / "calendars" / "1min.txt"
SLOTS_PER_DAY = 242          # cn_data_1min calendar: 242 slots/day (09:30-15:00)
FIRST_FEATURE_SLOT = 1       # slot 1 = first REAL bar (covers [09:30,09:31) open auction);
                             # slot 0 (09:30) is universally NaN pool-wide (probe 2026-07-06)
FEATURE_SLOT_COUNT = 10      # slots 1-10 = 09:31-09:40 factor input (10 real bars)
BUY_SLOT = 11                # slot 11 = 09:41 close → $price_941 buy price

# 14 minute factors materialized as <name>.day.bin per stock.
MINUTE_FACTOR_FIELDS = (
    "startup_mom_1m", "startup_mom_3m", "startup_mom_5m", "startup_total",
    "accel_1m", "accel_3m", "accel_5m",
    "close_pos_1m", "close_pos_3m", "close_pos_5m",
    "vol_ratio_1m", "vol_ratio_3m", "vol_ratio_5m",
    "vol_vs_yest",
)
# 15th materialized bin: T-day 9:41 close (deal_price for buy, NOT a feature).
MINUTE_DEAL_PRICE_FIELD = "price_941"
