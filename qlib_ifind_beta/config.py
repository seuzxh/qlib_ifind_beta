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

# --- champion FROZEN 推理口径（2026-07-09 实战对接 P1）-------------------------
# champion = enhanced(18)@topk10/nd8, commit 24b18dd, recorder caf649ca（params.pkl = LGBModel）。
# P1 inference 复刻此 fit 段（FROZEN），仅 end_time 扩到 T。spike 2026-07-09 验证零偏离
#（predict_day("2026-07-02") vs champion pred.pkl 同日 max|diff|=0、top10 10/10 完全一致）。
CHAMPION_RECORDER_ID = "caf649ca6aa44aac8dec8c4e5a252aef"
CHAMPION_EXPERIMENT = "minute_enhanced_tk10_nd8"
CHAMPION_DATA_START = "2024-01-01"        # handler start_time（含 train 段供 learned processor fit）
CHAMPION_FIT_START = "2024-01-01"         # FROZEN = champion train 段
CHAMPION_FIT_END = "2025-12-31"           # FROZEN = champion train 段
CHAMPION_LABEL_EXPR = "Ref($close, -1) / $price_941 - 1"   # FROZEN label（P1 仅 fetch 结构，不读值）
CHAMPION_TOPK = 10

# --- rolling retrain（2026-07-10，item 3 每日滚动重训）-------------------------
# 用 qlib 原生 RollingGen（task.gen）+ task_train + OnlineToolR（online utils）实现。
# 每个交易日生成新任务（step=1），滑动窗口（ROLL_SD = train/valid/test 同步前移），
# 训练后将新 recorder 标记为 online（旧模型自动 offline）。inference use_online=True
# 时从最新 online recorder 加载模型，替代 FROZEN CHAMPION_RECORDER_ID。
# 详见 scripts/retrain.py + qlib workflow.online 文档。
ROLLING_EXPERIMENT = "minute_enhanced_rolling"
ROLLING_STEP = 1                    # 每日滚动重训（step=1 交易日）
ROLLING_RTYPE = "sliding"           # RollingGen.ROLL_SD（滑动窗口）

# --- fields ------------------------------------------------------------------
BASE_FIELDS = ("open", "high", "low", "close", "volume", "factor", "vwap")
DERIVED_FIELDS = ("change", "limit_up", "limit_down")
FREQ = "day"

# calendar
DAY_CAL = QLIB_DATA / "calendars" / "day.txt"

# --- minute-frequency factors (T-day 9:30-9:40, materialized as day.bin) ------
# Source: /home/zxh/cn_data_1min (readonly, 1min bins, 240 slots/day).
# Slot map (probe-verified 2026-07-11): 240 real bars/day, 09:31~15:00, no NaN
# placeholder slots. cn_data_1min was rebuilt (was 242 slots with slot 0 = 09:30 NaN
# and slot 121 = 13:00 NaN placeholders; now 240 pure real bars).
# First REAL bar = slot 0 (09:31); daily open == minute_open[slot 0].
CN_DATA_1MIN = Path("/home/zxh/cn_data_1min")
FEATURES_1MIN_SRC = CN_DATA_1MIN / "features"
MIN_CAL = CN_DATA_1MIN / "calendars" / "1min.txt"
SLOTS_PER_DAY = 240          # cn_data_1min calendar: 240 slots/day (09:31-15:00, all real)
REAL_BARS_PER_DAY = 240      # 240 real bars/day (no NaN placeholders since 2026-07-11 rebuild;
                             # was 242−2=240 before, value unchanged)
FIRST_FEATURE_SLOT = 0       # slot 0 = 09:31 = first real bar (continuous-bid open)
FEATURE_SLOT_COUNT = 10      # slots 0-9 = 09:31-09:40 factor input (10 real bars)
BUY_SLOT = 10                # slot 10 = 09:41 close → $price_941 buy price

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

# 21st-22nd materialized bins — amount（成交额）量能因子（item 7，2026-07-12）。
# amount = volume × vwap（两个字段都在 cn_data_1min），无需新数据源。
# 价格加权的量能信号：自动校正高低价股的 volume 不可比性，且部分吸收送股/拆分的结构性量变。
#   - amt_ratio_5m：开盘后段成交额比前段 = mean(vol[5:10]×vwap[5:10]) / mean(vol[0:4]×vwap[0:4])。
#     对标 vol_ratio_5m（champion gain 4.0%），测价格加权是否比 raw volume 更干净。
#   - amt_vs_yest：开盘成交额 vs 昨日全天均 = sum(vol[0:10]×vwap[0:10]) / (T-1 全天 amt / 240)。
#     对标 vol_vs_yest（champion gain 24.3%，alpha #1），测价格加权能否提升核心信号。
MINUTE_FACTOR_AMT_FIELDS = (
    "amt_ratio_5m",
    "amt_vs_yest",
)
