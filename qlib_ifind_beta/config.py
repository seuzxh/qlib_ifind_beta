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

# Index sources referenced by ChangeInstrument in daily/index factors (NOT benchmark).
# 上证综指 SH000001 = 用户语义「上证指数」（共振 + 冰点/沸点因子引用源）。区别于 benchmark
# (SH000300，回测基准冻结)。这些指数的 7 base bin 在 build_overlay step6 被 link 进 overlay
# （仅 link，不 materialize——指数不交易，无需 change/limit 衍生）。qlib_data/features/sh000001/
# 已实测与 sh000300 同构（7 base bin × 6296 bytes，26 年深度；amount 8 bytes 空文件不用）。
INDEX_FACTOR_SOURCES = ("SH000001",)

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
REAL_BARS_PER_DAY = SLOTS_PER_DAY - 2   # 240: 242 slots − slot 0 (09:30) − slot 121 (13:00),
                             # both universally NaN pool-wide (probe 2026-07-06). Exact, not
                             # approximate: 240 real bars/day (used as vol_vs_yest denominator).
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
# 16th materialized bin: 9:41 时刻涨跌幅（不复权）vs T-1 不复权收盘 —— v2 涨跌停 buy 表达式用。
# (price_941[T]/factor[T]) / (close[T-1]/factor[T-1]) - 1；与 materialize.compute_change 同源、
# 仅把"全天 close"换成"9:41 close"。必须不复权（除权日 factor 跳变会误判涨跌停）。
MINUTE_CHANGE_941_FIELD = "change_941"

# 17th-20th materialized bins — enhanced champion 的 4 extra 因子（2026-07-07）。
# MINUTE_FACTOR_FIELDS（14）不变 → m14 复现保留；这 4 extra 被 MinuteEnhancedHandler 消费
#（14 baseline + 4 extra = 18 因子；详见 backtest-log §22 champion = enhanced@n_drop=15）。
#   - vol_vs_yest_t2/t3/t5：vol_vs_yest 多日族，T 日 9:30-9:40 累积量 / (T-k 日全天分钟量/240)，
#     k∈{2,3,5}（与 vol_vs_yest 同形，分母 shift 改 2/3/5）。min-cal 空间，随 fac scatter。
#   - overnight_gap：不复权开盘跳空 = (open[T]/factor[T])/(close[T-1]/factor[T-1]) - 1。day-cal 空间，
#     与 change_941 同源（不复权，除权日跳空反映真实开盘情绪；后复权会抵消除权缺口 → 口径错）。
MINUTE_FACTOR_EXTRA_FIELDS = (
    "vol_vs_yest_t2", "vol_vs_yest_t3", "vol_vs_yest_t5",
    "overnight_gap",
)

# 21st-25th materialized bins — T-1 尾盘分钟族（2026-07-07，goal 因子优化迭代）。
# 破局 Step B「IC +12% 不传导组合」：此前 T-1 全天分钟结构完全未用作因子（vol_vs_yest 分母
# 仅把 T-1 全天量压成聚合标量，丢结构）。尾盘 slot 232-241 是次日惯性/高潮/启动最强领先信号。
# 与早盘 slot 1-10 严格对称（slot 映射：slot N 下午覆盖 [13:00+(N-122), 13:00+(N-121))，
# 故 slot 232 = [14:50,14:51) = 14:51 时刻 bar，slot 241 = [14:59,15:00) = 15:00 时刻 bar，
# slot 232-241 覆盖 [14:50,15:00) 共 10 根）。物化时 shift 1（min-cal 空间）→ T 行 bin =
# T-1 尾盘，handler $field 直接消费（与 14 分钟因子零 Ref 模式统一）。无前视（T-1 15:00
# 收盘，T 日 9:41 决策已知）。详见 backtest-log（goal 设计）。
TAIL_FIRST_SLOT = 232         # 尾盘 10min 窗首 slot（14:51 时刻 bar，覆盖 [14:50,14:51)）
TAIL_SLOT_COUNT = 10          # slots 232-241 = 14:51-15:00 时刻 bar，覆盖 [14:50,15:00)
MINUTE_FACTOR_TAIL_FIELDS = (
    "tail_mom_t1",            # 尾盘 10 根整体动量 close[9]/close[0]-1 → 惯性冲高/高潮
    "tail_mom_last5_t1",      # 14:55-15:00 收盘竞价动量 close[9]/close[5]-1
    "tail_close_pos_t1",      # 收盘在尾盘振幅位置 → 收盘强弱
    "tail_vol_ratio_t1",      # 尾盘量比 vol[9]/vol[0:9].mean → 主力介入/出逃
    "tail_accel_t1",          # 尾盘加速度 → 高潮见顶/惯性持续
)
