# 883926 高贝塔策略 · 分钟因子实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有日频 Alpha158 baseline 之上叠加 14 个 T 日 9:30~9:40 分钟因子（物化为 day.bin）+ `$price_941` 撮合价字段，子类 `HighBetaAlpha158` 把 14 因子拼进特征集，label/deal_price 改为 9:41 买 / T+1 收盘卖。

**Architecture:** 频率融合放数据层（最轻、与项目已验证的 `change/limit_up/limit_down` 物化模式一致）—— 独立脚本读 `cn_data_1min` 的 1min.bin，按日切窗（每日 242 槽，index 0-10 因子输入、index 11 = 9:41 买价），算 14 因子 + 9:41 close，写成 day.bin 落进 overlay；qlib 仍只消费日频 overlay（`provider_uri=data/qlib_root, freq=day`），不挂双频。`HighBetaAlpha158(Alpha158)` 覆写 `get_feature_config()` 追加 14 个 `$field`。YAML 改 handler class / label / deal_price 三处。

**Tech Stack:** Python 3.12（conda env `qlib_ifind_beta`）/ pyqlib 0.9.7 / numpy 2.4.6 / pytest 9.0.3 / LightGBM 4.6.0。所有命令 `conda run -n qlib_ifind_beta` 前缀（CLAUDE.md 硬约束）。

**Source of truth:** [docs/superpowers/specs/2026-07-06-minute-factors-design.md](../specs/2026-07-06-minute-factors-design.md)（定稿）。本计划是它的 TDD 拆解，不引入新设计决策。

---

## File Structure

| 文件 | 责任 | 动作 |
|---|---|---|
| `qlib_ifind_beta/config.py` | 路径/字段常量 | **改**：追加 cn_data_1min 路径 + 14 因子名表 + 槽位常量 |
| `qlib_ifind_beta/minute_factors.py` | 14 因子纯函数（无 IO） | **建**：`compute_day_factors(c,o,h,l,vol,prev_day_volume)` |
| `qlib_ifind_beta/materialize_minute.py` | 读 1min.bin + 日频对齐 → 写 15 day.bin | **建**：`materialize_minute_instrument(code)` + 日历 helper |
| `qlib_ifind_beta/highbeta_handler.py` | `HighBetaAlpha158(Alpha158)` 子类 | **建**：覆写 `get_feature_config()` |
| `scripts/materialize_minute.py` | 全 universe 物化 CLI（不依赖 iFinD） | **建**：循环 instruments，调用 `materialize_minute_instrument` |
| `scripts/build_overlay.py` | 端到端 overlay 编排 | **改**：循环里加分钟物化调用（fresh build 含分钟因子） |
| `qrun/run.py` | qrun 入口 | **改**：插入项目根到 sys.path（handler `module_path` 依赖） |
| `qrun/workflow.yaml` | 全量回测配置 | **改**：handler class → HighBetaAlpha158 + label + deal_price |
| `qrun/workflow_smoke.yaml` | 烟雾测试配置 | **改**：同上三处 |
| `tests/conftest.py` | pytest sys.path 引导 | **建**：插入项目根 |
| `tests/test_minute_factors.py` | 14 因子纯函数单测（手算期望值） | **建** |
| `tests/test_1min_format.py` | 1min bin 格式 + 日历对齐（A3） | **建** |
| `tests/test_materialize_minute.py` | 物化正确性交叉核验（A1/A2/A4） | **建** |
| `tests/test_highbeta_handler.py` | 特征数 158+14（A5 单测部分） | **建** |

**为什么不挂双频 / 为什么物化**：见 spec §物化架构（方案 A）。qlib 原生 `NestedDataLoader`/`QlibDataLoader(freq=dict)` 混出来是分钟级行（一天 242 行），本设计要日级二维表送 LGBModel + 日级回测，qlib 无现成「分钟序列→每日标量」聚合类，物化最轻。

---

## Task 1: config.py 追加分钟因子常量

**Files:**
- Modify: `qlib_ifind_beta/config.py:44`（在 `DAY_CAL = ...` 之后追加）

- [ ] **Step 1: 追加常量块**

在 `qlib_ifind_beta/config.py` 末尾（`DAY_CAL = QLIB_DATA / "calendars" / "day.txt"` 之后）追加：

```python

# --- minute-frequency factors (T-day 9:30-9:40, materialized as day.bin) ------
# Source: /home/zxh/cn_data_1min (readonly, 1min bins, 242 slots/day).
# Slot map (probe-verified): index 0-10 = 9:30-9:40 (factor input),
# index 11 = 9:41 (buy-price bar, NOT used in factors). See spec §数据基础.
CN_DATA_1MIN = Path("/home/zxh/cn_data_1min")
FEATURES_1MIN_SRC = CN_DATA_1MIN / "features"
MIN_CAL = CN_DATA_1MIN / "calendars" / "1min.txt"
SLOTS_PER_DAY = 242          # cn_data_1min: 9:30-15:00 = 242 1min bars/trading day
FACTOR_INPUT_SLOTS = 11      # slots 0-10 (9:30-9:40) feed the 14 factors
PRICE_941_SLOT = 11          # slot 11 = 9:41 close → $price_941 buy price

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
```

- [ ] **Step 2: 验证 import**

Run: `conda run -n qlib_ifind_beta python -c "from qlib_ifind_beta.config import MINUTE_FACTOR_FIELDS, MINUTE_DEAL_PRICE_FIELD, FEATURES_1MIN_SRC, MIN_CAL, SLOTS_PER_DAY; print(len(MINUTE_FACTOR_FIELDS), MINUTE_DEAL_PRICE_FIELD, FEATURES_1MIN_SRC.exists(), MIN_CAL.exists(), SLOTS_PER_DAY)"`
Expected: `15 price_941 True True 242`

- [ ] **Step 3: Commit**

```bash
git add qlib_ifind_beta/config.py
git commit -m "feat(config): add cn_data_1min paths + 14 minute-factor field names

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 2: minute_factors.py 纯函数 + 单测（TDD）

14 因子的数学核心，无 IO，可独立单测。符号约定见 spec §符号约定：`c_i/o_i/h_i/l_i/vol_i` = index i 的分钟 close/open/high/low/volume。

**Files:**
- Create: `qlib_ifind_beta/minute_factors.py`
- Test: `tests/test_minute_factors.py`

- [ ] **Step 1: 写 conftest.py（让 tests/ 能 import 包）**

Create `tests/conftest.py`：

```python
import sys
from pathlib import Path

# 项目根本地化（无 pyproject/editable install，测试需手动加路径）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

- [ ] **Step 2: 写失败测试**

Create `tests/test_minute_factors.py`：

```python
"""14 minute-factor formulas — hand-computed expected values.

Slots 0-10 = 9:30-9:40 (factor input), slot 11 = 9:41 (price_941, NOT a factor).
See spec §因子集 for formula derivation.
"""
import numpy as np
import pytest

from qlib_ifind_beta.minute_factors import compute_day_factors


def _synthetic():
    """12-slot arrays with a clean linear ramp so hand-math is exact."""
    c = np.array([10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 11.0, 11.1])
    o = np.array([9.9, 10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 11.0])
    h = c + 0.05        # high = close + 0.05
    l = o - 0.05        # low  = open - 0.05
    vol = np.array([100.0] * 11 + [50.0])   # slot 11 vol irrelevant
    return c, o, h, l, vol


def test_startup_momentum():
    c, o, h, l, vol = _synthetic()
    f = compute_day_factors(c, o, h, l, vol, prev_day_volume=24000.0)
    assert f["startup_mom_1m"] == pytest.approx(11.0 / 10.9 - 1)
    assert f["startup_mom_3m"] == pytest.approx(11.0 / 10.7 - 1)
    assert f["startup_mom_5m"] == pytest.approx(11.0 / 10.5 - 1)
    assert f["startup_total"] == pytest.approx(11.0 / 9.9 - 1)


def test_acceleration():
    c, o, h, l, vol = _synthetic()
    f = compute_day_factors(c, o, h, l, vol, prev_day_volume=24000.0)
    assert f["accel_1m"] == pytest.approx((11.0 / 10.9 - 1) - (10.0 / 9.9 - 1))
    assert f["accel_3m"] == pytest.approx((11.0 / 10.7 - 1) - (10.2 / 9.9 - 1))
    assert f["accel_5m"] == pytest.approx((11.0 / 10.5 - 1) - (10.4 / 9.9 - 1))


def test_close_position():
    c, o, h, l, vol = _synthetic()
    f = compute_day_factors(c, o, h, l, vol, prev_day_volume=24000.0)
    # window index10
    assert f["close_pos_1m"] == pytest.approx((11.0 - 10.85) / (11.05 - 10.85))
    # window index8-10: max h = 11.05, min l = 10.65
    assert f["close_pos_3m"] == pytest.approx((11.0 - 10.65) / (11.05 - 10.65))
    # window index6-10: max h = 11.05, min l = 10.45
    assert f["close_pos_5m"] == pytest.approx((11.0 - 10.45) / (11.05 - 10.45))


def test_volume_ratio():
    c, o, h, l, vol = _synthetic()
    f = compute_day_factors(c, o, h, l, vol, prev_day_volume=24000.0)
    # vol[10]=50, mean(vol[0:10])=100
    assert f["vol_ratio_1m"] == pytest.approx(50.0 / 100.0)
    # mean(vol[8:11])=(100+100+50)/3, mean(vol[0:3])=100
    assert f["vol_ratio_3m"] == pytest.approx(((100 + 100 + 50) / 3) / 100.0)
    # mean(vol[6:11])=(100*4+50)/5=90, mean(vol[0:5])=100
    assert f["vol_ratio_5m"] == pytest.approx(90.0 / 100.0)


def test_vol_vs_yest():
    c, o, h, l, vol = _synthetic()
    # prev_day_volume=24000 → per-min denom = 100; numerator = sum(vol[0:11]) = 1100
    f = compute_day_factors(c, o, h, l, vol, prev_day_volume=24000.0)
    assert f["vol_vs_yest"] == pytest.approx(1100.0 / 100.0)
    # no prev volume → NaN (first trading day)
    f0 = compute_day_factors(c, o, h, l, vol, prev_day_volume=None)
    assert np.isnan(f0["vol_vs_yest"])


def test_price_941():
    c, o, h, l, vol = _synthetic()
    f = compute_day_factors(c, o, h, l, vol, prev_day_volume=24000.0)
    assert f["price_941"] == pytest.approx(11.1)   # c[11]
```

- [ ] **Step 3: 跑测试确认失败**

Run: `conda run -n qlib_ifind_beta python -m pytest tests/test_minute_factors.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'qlib_ifind_beta.minute_factors'`

- [ ] **Step 4: 实现 minute_factors.py**

Create `qlib_ifind_beta/minute_factors.py`：

```python
"""14 minute-bar factors + $price_941, computed for ONE trading day.

Inputs are per-day 1min slot arrays (length >= 12). Slot map (probe-verified,
cn_data_1min): index 0-10 = 9:30-9:40 (factor input), index 11 = 9:41 (buy-price
bar, NOT used in factors). See docs/superpowers/specs/2026-07-06-minute-factors-design.md.

Pure (no IO) — the materialize layer (materialize_minute.py) calls this per day
and writes the results as day.bin. Hand-unit-tested in tests/test_minute_factors.py.
"""
from __future__ import annotations

import numpy as np


def compute_day_factors(c, o, h, l, vol, prev_day_volume=None) -> dict:
    """Compute 14 factors + price_941 for one trading day.

    Args:
        c, o, h, l, vol: 1D arrays length >= 12 (slots 0-11). Float-castable.
        prev_day_volume: previous trading day's TOTAL daily volume (scalar), used
            as the vol_vs_yest denominator (Ref($volume,1)/240). None/<=0 → NaN
            (first trading day or missing daily bin).

    Returns:
        dict with keys = 14 factor names + "price_941". NaN where undefined.
    """
    c = np.asarray(c, dtype=np.float64)
    o = np.asarray(o, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    l = np.asarray(l, dtype=np.float64)
    vol = np.asarray(vol, dtype=np.float64)

    out = {}
    # A. startup momentum (last-N-bar return from slot 10)
    out["startup_mom_1m"] = c[10] / c[9] - 1.0
    out["startup_mom_3m"] = c[10] / c[7] - 1.0
    out["startup_mom_5m"] = c[10] / c[5] - 1.0
    out["startup_total"] = c[10] / o[0] - 1.0

    # B. acceleration = back-seg return − front-seg return (seg = seg-open → seg-close)
    out["accel_1m"] = (c[10] / o[10] - 1.0) - (c[0] / o[0] - 1.0)
    out["accel_3m"] = (c[10] / o[8] - 1.0) - (c[2] / o[0] - 1.0)
    out["accel_5m"] = (c[10] / o[6] - 1.0) - (c[4] / o[0] - 1.0)

    # C. close position in window = (c10 - low_W) / (high_W - low_W)
    out["close_pos_1m"] = (c[10] - l[10]) / (h[10] - l[10])
    h3, l3 = h[8:11].max(), l[8:11].min()
    out["close_pos_3m"] = (c[10] - l3) / (h3 - l3)
    h5, l5 = h[6:11].max(), l[6:11].min()
    out["close_pos_5m"] = (c[10] - l5) / (h5 - l5)

    # D. volume ratio = back-seg mean vol / front-seg mean vol
    out["vol_ratio_1m"] = vol[10] / vol[0:10].mean()
    out["vol_ratio_3m"] = vol[8:11].mean() / vol[0:3].mean()
    out["vol_ratio_5m"] = vol[6:11].mean() / vol[0:5].mean()

    # E. cross-day volume = first-11-bar total / (prev day total vol / 240)
    if prev_day_volume is not None and prev_day_volume > 0:
        out["vol_vs_yest"] = vol[0:11].sum() / (prev_day_volume / 240.0)
    else:
        out["vol_vs_yest"] = np.nan

    # buy-price bar (slot 11) — not a feature, materialized as $price_941
    out["price_941"] = float(c[11])
    return out
```

- [ ] **Step 5: 跑测试确认通过**

Run: `conda run -n qlib_ifind_beta python -m pytest tests/test_minute_factors.py -v`
Expected: PASS (6 tests)

- [ ] **Step 6: Commit**

```bash
git add tests/conftest.py tests/test_minute_factors.py qlib_ifind_beta/minute_factors.py
git commit -m "feat(minute-factors): 14 pure factor functions + hand-calc unit tests

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 3: 1min bin 格式 + 日历对齐验证（第一性原理 gate）

最高风险假设：1min.bin 的 `start_index` 也是 float32（与 day.bin 同 `FileFeatureStorage` 格式），且股票首根 1min bar 落在某个交易日的 slot 0（day-aligned，reshape 前提）。这个测试用真数据钉死它。

**Files:**
- Test: `tests/test_1min_format.py`

- [ ] **Step 1: 写测试**

Create `tests/test_1min_format.py`：

```python
"""First-principles: pin the cn_data_1min 1min.bin format before materialization.

Guards two assumptions the materialize layer depends on:
  1. start_index is a float32 calendar-row index (same FileFeatureStorage format as
     day.bin). If it were uint32-reinterpreted, the value would be ~1e9 and fail
     the `< len(calendar)` check.
  2. A stock's first 1min bar is slot 0 of some trading day (day-aligned), so the
     bin can be reshaped to (n_days, 242) with day boundaries intact.
"""
from datetime import datetime, time
from pathlib import Path

from qlib_ifind_beta.binio import read_bin
from qlib_ifind_beta.config import FEATURES_1MIN_SRC, MIN_CAL, SLOTS_PER_DAY


def _load_1min_cal():
    dts = []
    with open(MIN_CAL) as fp:
        for line in fp:
            line = line.strip()
            if line:
                dts.append(datetime.strptime(line, "%Y-%m-%d %H:%M:%S"))
    return dts


def test_start_index_is_float32_calendar_row():
    """binio.read_bin (float32) yields a start_index inside the 1min calendar."""
    dts = _load_1min_cal()
    p = Path(FEATURES_1MIN_SRC) / "sh600519" / "close.1min.bin"
    assert p.exists(), f"missing test fixture {p}"
    si, arr = read_bin(p)
    assert si is not None and si >= 0
    assert si < len(dts), f"start_index {si} >= calendar len {len(dts)} (uint32 misread?)"
    assert arr.size + si <= len(dts), "bin overflows 1min calendar"


def test_first_bar_day_aligned_near_open():
    """Stock's first 1min bar is slot 0 of a trading day, near market open."""
    dts = _load_1min_cal()
    p = Path(FEATURES_1MIN_SRC) / "sh600519" / "close.1min.bin"
    si, arr = read_bin(p)
    first = dts[si]
    # slot 0 = market open (9:30 or 9:31 depending on marking convention)
    assert first.time() in (time(9, 30), time(9, 31)), f"first bar not open: {first}"
    # consecutive 1-min spacing within the first day
    assert (dts[si + 1] - dts[si]).total_seconds() == 60


def test_slot11_is_941():
    """index 11 (buy-price bar) lands on 9:41 — spec §数据基础 slot map."""
    dts = _load_1min_cal()
    p = Path(FEATURES_1MIN_SRC) / "sh600519" / "close.1min.bin"
    si, _ = read_bin(p)
    # spec: index 0-10 = 9:30-9:40, index 11 = 9:41
    slot11_time = dts[si + 11].time()
    assert slot11_time == time(9, 41), (
        f"slot 11 = {slot11_time}, expected 9:41. "
        "If this fails, the spec slot-map is wrong — do NOT silently change; "
        "re-probe cn_data_1min and update spec + PRICE_941_SLOT."
    )


def test_slots_per_day_is_242():
    """242 rows per trading day in the 1min calendar (reshape precondition)."""
    dts = _load_1min_cal()
    # count rows in the first full trading day
    first_date = dts[0].date()
    count = sum(1 for d in dts if d.date() == first_date)
    assert count == SLOTS_PER_DAY, f"first day has {count} slots, expected {SLOTS_PER_DAY}"
```

- [ ] **Step 2: 跑测试**

Run: `conda run -n qlib_ifind_beta python -m pytest tests/test_1min_format.py -v`
Expected: PASS (4 tests)。

> **如果 `test_slot11_is_941` 失败**：说明 spec 的槽位→时刻映射错了（可能是 end-time 标记导致整体偏移一格）。**不要静默改断言** —— 先 `conda run -n qlib_ifind_beta python -c "from qlib_ifind_beta.config import MIN_CAL; print(open(MIN_CAL).readline().strip())"` 看 1min.txt 第一行真实时刻，再回头订正 spec §数据基础 + `config.PRICE_941_SLOT` + `minute_factors` 的 slot 索引。这是第一性原理 gate，必须查清楚。

- [ ] **Step 3: Commit**

```bash
git add tests/test_1min_format.py
git commit -m "test(1min-format): pin bin start_index + slot-11=9:41 + 242 slots/day

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 4: materialize_minute.py + 物化正确性交叉核验（A1）

物化层。读 5 个 1min.bin（close/open/high/low/volume）+ 日频 close/volume.bin（对齐 + vol_vs_yest 分母），向量化算 14 因子 + price_941，写 15 个 day.bin（start_index/length 与该票日频 close.bin 完全对齐，保证 `$price_941[T]` 行对齐 `$close[T]`）。

**Files:**
- Create: `qlib_ifind_beta/materialize_minute.py`
- Test: `tests/test_materialize_minute.py`

- [ ] **Step 1: 写失败测试（A1 交叉核验）**

Create `tests/test_materialize_minute.py`：

```python
"""Materialization correctness (spec A1) — cross-check materialized day.bin against
an independent raw read of the 1min source for SH600519."""
from pathlib import Path

import numpy as np

from qlib_ifind_beta import materialize_minute as mm
from qlib_ifind_beta.binio import read_bin
from qlib_ifind_beta.config import (
    FEATURES_1MIN_SRC, FEATURES_DST, FEATURES_SRC, SLOTS_PER_DAY,
)


def _stock_arrays(code="SH600519"):
    """Read 1min + daily arrays for independent hand-recompute."""
    si_m, c1m = read_bin(Path(FEATURES_1MIN_SRC) / code.lower() / "close.1min.bin")
    _, v1m = read_bin(Path(FEATURES_1MIN_SRC) / code.lower() / "volume.1min.bin")
    si_dc, close_d = read_bin(Path(FEATURES_SRC) / code.lower() / "close.day.bin")
    si_dv, vol_d = read_bin(Path(FEATURES_SRC) / code.lower() / "volume.day.bin")
    n_days_m = c1m.size // SLOTS_PER_DAY
    c2d = c1m[: n_days_m * SLOTS_PER_DAY].reshape(n_days_m, SLOTS_PER_DAY)
    v2d = v1m[: n_days_m * SLOTS_PER_DAY].reshape(n_days_m, SLOTS_PER_DAY)
    return si_m, c2d, v2d, si_dc, close_d, si_dv, vol_d, n_days_m


def test_materialize_returns_true_for_liquid_stock():
    assert mm.materialize_minute_instrument("SH600519") is True


def test_materialize_writes_15_bins():
    mm.materialize_minute_instrument("SH600519")
    d = Path(FEATURES_DST) / "sh600519"
    from qlib_ifind_beta.config import MINUTE_FACTOR_FIELDS, MINUTE_DEAL_PRICE_FIELD
    for name in MINUTE_FACTOR_FIELDS:
        assert (d / f"{name}.day.bin").exists(), name
    assert (d / f"{MINUTE_DEAL_PRICE_FIELD}.day.bin").exists()


def test_day_bin_aligned_to_daily_close():
    """materialized bin start_index == stock's daily close.bin start_index."""
    si_dc, _ = read_bin(Path(FEATURES_SRC) / "sh600519" / "close.day.bin")
    si_out, _ = read_bin(Path(FEATURES_DST) / "sh600519" / "price_941.day.bin")
    assert si_out == si_dc


def test_startup_mom_and_price941_crosscheck():
    """A1: materialized startup_mom_1m / price_941 == raw 1min hand-compute."""
    mm.materialize_minute_instrument("SH600519")
    si_out, startup = read_bin(Path(FEATURES_DST) / "sh600519" / "startup_mom_1m.day.bin")
    _, p941 = read_bin(Path(FEATURES_DST) / "sh600519" / "price_941.day.bin")

    si_m, c2d, v2d, si_dc, close_d, si_dv, vol_d, n_days_m = _stock_arrays()
    min_dates, _ = mm._load_min_calendar()
    _, date_to_row = mm._load_day_calendar_lookup()

    # find a recent 1min day with valid slots 9,10,11
    km = None
    for k in range(n_days_m - 1, -1, -1):
        if np.all(np.isfinite(c2d[k, [9, 10, 11]])):
            km = k
            break
    assert km is not None, "no valid 1min day for SH600519"

    first_row = si_m + km * SLOTS_PER_DAY
    day_row = int(date_to_row[min_dates[first_row]])
    out_row = day_row - si_out
    assert 0 <= out_row < startup.size

    expected_startup = c2d[km, 10] / c2d[km, 9] - 1.0
    expected_p941 = c2d[km, 11]
    assert abs(startup[out_row] - expected_startup) < 1e-4
    assert abs(p941[out_row] - expected_p941) < 1e-2


def test_vol_vs_yest_crosscheck():
    """A1: vol_vs_yest (dual-source factor) == raw hand-compute."""
    mm.materialize_minute_instrument("SH600519")
    si_out, vol_vs_yest = read_bin(Path(FEATURES_DST) / "sh600519" / "vol_vs_yest.day.bin")

    si_m, c2d, v2d, si_dc, close_d, si_dv, vol_d, n_days_m = _stock_arrays()
    min_dates, _ = mm._load_min_calendar()
    _, date_to_row = mm._load_day_calendar_lookup()

    km = None
    for k in range(n_days_m - 1, -1, -1):
        if np.all(np.isfinite(v2d[k, 0:11])) and k > 0:
            km = k
            break
    assert km is not None
    first_row = si_m + km * SLOTS_PER_DAY
    day_row = int(date_to_row[min_dates[first_row]])
    out_row = day_row - si_out
    prev_day_vol = vol_d[day_row - 1 - si_dv]
    expected = v2d[km, 0:11].sum() / (prev_day_vol / 240.0)
    assert abs(vol_vs_yest[out_row] - expected) < 1e-3


def test_no_lookahead_price941_is_slot11_only():
    """A4: price_941 uses ONLY slot 11 (9:41), never later slots. Sanity: value
    equals c[11] not c[10] or c[12] — already covered by crosscheck; here we
    additionally assert the buy price is strictly the 9:41 bar by checking it
    differs from the 9:40 close for the chosen day."""
    si_m, c2d, v2d, *_ = _stock_arrays()
    # pick a day where slot 10 != slot 11 (price moved)
    for k in range(c2d.shape[0] - 1, -1, -1):
        if np.all(np.isfinite(c2d[k, [10, 11]])) and c2d[k, 10] != c2d[k, 11]:
            assert c2d[k, 11] != c2d[k, 10]
            return
    # extremely unlikely no such day exists; if so, skip silently
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n qlib_ifind_beta python -m pytest tests/test_materialize_minute.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'qlib_ifind_beta.materialize_minute'`

- [ ] **Step 3: 实现 materialize_minute.py**

Create `qlib_ifind_beta/materialize_minute.py`：

```python
"""Materialize 14 minute factors + $price_941 as day.bin into the overlay.

Reads cn_data_1min 1min bins (close/open/high/low/volume) per stock, slices each
trading day's first 12 slots (9:30-9:41), computes the 14 factors + 9:41 close,
and writes them as day.bin aligned to qlib_data's day calendar (same start_index
and length as the stock's daily close.bin, so $price_941[T] row-aligns with
$close[T]).

vol_vs_yest denominator = previous trading day's TOTAL daily volume / 240 (read
from qlib_data daily volume.bin) — the only factor that crosses minute↔daily.

Vectorized per stock: reshape the dense 1min array to (n_days, 242), take
[:, :12], compute factors column-wise, scatter into the day-aligned output.

Run: see scripts/materialize_minute.py (full universe) or call
materialize_minute_instrument(code) directly.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

from .binio import read_bin, write_bin
from .config import (
    DAY_CAL, FEATURES_1MIN_SRC, FEATURES_DST, FEATURES_SRC, FREQ,
    MIN_CAL, MINUTE_DEAL_PRICE_FIELD, MINUTE_FACTOR_FIELDS, SLOTS_PER_DAY,
)

_1MIN_FIELDS = ("close", "open", "high", "low", "volume")
_cal_cache: dict = {}


def _load_min_calendar():
    """Return (min_dates_ord, min_slots) as int arrays over the 1min calendar.

    min_dates_ord[i] = date.toordinal() of 1min row i; min_slots[i] = slot-within-day.
    Cached (1.55M rows, ~once per process).
    """
    if "min" not in _cal_cache:
        dts = []
        with open(MIN_CAL) as fp:
            for line in fp:
                line = line.strip()
                if line:
                    dts.append(datetime.strptime(line, "%Y-%m-%d %H:%M:%S"))
        dates = np.empty(len(dts), dtype=np.int64)
        slots = np.empty(len(dts), dtype=np.int16)
        cur_date = None
        slot = 0
        for i, dt in enumerate(dts):
            if dt.date() != cur_date:
                cur_date = dt.date()
                slot = 0
            dates[i] = dt.toordinal()
            slots[i] = slot
            slot += 1
        _cal_cache["min"] = (dates, slots)
    return _cal_cache["min"]


def _load_day_calendar_lookup():
    """Return (n_days, date_to_row) where date_to_row[ordinal] = day-calendar row.

    date_to_row is a numpy int32 array indexed by date.toordinal(); -1 if the date
    is not a trading day. Sized to the max ordinal in day.txt.
    """
    if "day" not in _cal_cache:
        with open(DAY_CAL) as fp:
            rows = [line.strip() for line in fp if line.strip()]
        ordinals = [
            datetime.strptime(s, "%Y-%m-%d").toordinal() for s in rows
        ]
        max_ord = max(ordinals)
        lookup = np.full(max_ord + 1, -1, dtype=np.int32)
        for i, ord_ in enumerate(ordinals):
            lookup[ord_] = i
        _cal_cache["day"] = (len(rows), lookup)
    return _cal_cache["day"]


def _read_1min_fields(code: str):
    """Return (start_index, dict field→float32 array) or (None, None) if missing/misaligned."""
    d = Path(FEATURES_1MIN_SRC) / code.lower()
    arrs = {}
    si = None
    for f in _1MIN_FIELDS:
        p = d / f"{f}.1min.bin"
        if not p.exists():
            return None, None
        s, a = read_bin(p)
        if a.size == 0 or s is None:
            return None, None
        arrs[f] = a
        if si is None:
            si = s
        elif s != si:
            return None, None   # fields disagree on start_index
    return si, arrs


def materialize_minute_instrument(code: str) -> bool:
    """Read 1min + daily bins for `code`, write 14 factor day.bins + price_941.day.bin.

    Returns True on success; False if the 1min source is missing/misaligned, or the
    daily close/volume bins are missing/misaligned (caller treats as "no minute
    data for this stock" — qlib reads NaN).
    """
    si_m, m = _read_1min_fields(code)
    if si_m is None:
        return False

    # daily close.bin = alignment reference (output start_index + length);
    # daily volume.bin = vol_vs_yest denominator. Must share start_index.
    ddir = Path(FEATURES_SRC) / code.lower()
    si_dc, close_d = read_bin(ddir / f"close.{FREQ}.bin")
    si_dv, vol_d = read_bin(ddir / f"volume.{FREQ}.bin")
    if close_d.size == 0 or si_dc is None or si_dc != si_dv:
        return False

    # day-alignment precondition (verified for liquid stocks in test_1min_format);
    # if a stock's 1min bin doesn't start at slot 0, the reshape below is invalid.
    _, min_slots = _load_min_calendar()
    if si_m >= min_slots.size or min_slots[si_m] != 0:
        return False

    min_dates, _ = _load_min_calendar()
    _, date_to_row = _load_day_calendar_lookup()

    n_m = m["close"].size
    n_days_m = n_m // SLOTS_PER_DAY
    take = n_days_m * SLOTS_PER_DAY       # drop trailing partial day if any

    def win(field):
        return m[field][:take].reshape(n_days_m, SLOTS_PER_DAY)[:, :12]

    c, o, h, l, v = win("close"), win("open"), win("high"), win("low"), win("volume")

    # per-1min-day date → day-calendar row → output-relative index.
    # clip ordinal before fancy-index (defensive: a stray 1min date outside the
    # day calendar must not crash the whole stock — mark it invalid instead).
    first_rows = si_m + np.arange(n_days_m) * SLOTS_PER_DAY
    day_dates = min_dates[first_rows]
    max_ord = date_to_row.size - 1
    safe = np.clip(day_dates, 0, max_ord)
    oob = (day_dates < 0) | (day_dates > max_ord)
    day_rows = np.where(oob, -1, date_to_row[safe])   # -1 where not a trading day
    rel = day_rows - si_dc

    # vectorized factors over n_days_m
    fac = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        fac["startup_mom_1m"] = c[:, 10] / c[:, 9] - 1.0
        fac["startup_mom_3m"] = c[:, 10] / c[:, 7] - 1.0
        fac["startup_mom_5m"] = c[:, 10] / c[:, 5] - 1.0
        fac["startup_total"] = c[:, 10] / o[:, 0] - 1.0
        fac["accel_1m"] = (c[:, 10] / o[:, 10] - 1.0) - (c[:, 0] / o[:, 0] - 1.0)
        fac["accel_3m"] = (c[:, 10] / o[:, 8] - 1.0) - (c[:, 2] / o[:, 0] - 1.0)
        fac["accel_5m"] = (c[:, 10] / o[:, 6] - 1.0) - (c[:, 4] / o[:, 0] - 1.0)
        fac["close_pos_1m"] = (c[:, 10] - l[:, 10]) / (h[:, 10] - l[:, 10])
        h3, l3 = h[:, 8:11].max(axis=1), l[:, 8:11].min(axis=1)
        fac["close_pos_3m"] = (c[:, 10] - l3) / (h3 - l3)
        h5, l5 = h[:, 6:11].max(axis=1), l[:, 6:11].min(axis=1)
        fac["close_pos_5m"] = (c[:, 10] - l5) / (h5 - l5)
        fac["vol_ratio_1m"] = v[:, 10] / v[:, 0:10].mean(axis=1)
        fac["vol_ratio_3m"] = v[:, 8:11].mean(axis=1) / v[:, 0:3].mean(axis=1)
        fac["vol_ratio_5m"] = v[:, 6:11].mean(axis=1) / v[:, 0:5].mean(axis=1)
        # vol_vs_yest: prev-day daily volume / 240 (NaN where prev row out of range)
        prev_rel = rel - 1
        prev_in_range = (prev_rel >= 0) & (prev_rel < vol_d.size)
        prev_vol = np.where(prev_in_range, vol_d[np.clip(prev_rel, 0, vol_d.size - 1)], np.nan)
        fac["vol_vs_yest"] = v[:, 0:11].sum(axis=1) / (prev_vol / 240.0)
    price_941 = c[:, 11]

    # scatter into day-aligned output (length = daily close.bin length)
    n_out = close_d.size
    out = {name: np.full(n_out, np.nan, dtype=np.float32) for name in MINUTE_FACTOR_FIELDS}
    out_p941 = np.full(n_out, np.nan, dtype=np.float32)
    valid = (rel >= 0) & (rel < n_out) & (day_rows >= 0)
    rel_v = rel[valid]
    for name in MINUTE_FACTOR_FIELDS:
        out[name][rel_v] = fac[name][valid].astype(np.float32)
    out_p941[rel_v] = price_941[valid].astype(np.float32)

    dst_dir = Path(FEATURES_DST) / code.lower()
    for name in MINUTE_FACTOR_FIELDS:
        write_bin(dst_dir / f"{name}.{FREQ}.bin", si_dc, out[name])
    write_bin(dst_dir / f"{MINUTE_DEAL_PRICE_FIELD}.{FREQ}.bin", si_dc, out_p941)
    return True
```

- [ ] **Step 4: 跑测试确认通过**

Run: `conda run -n qlib_ifind_beta python -m pytest tests/test_materialize_minute.py -v`
Expected: PASS (6 tests)。

> 若 `test_vol_vs_yest_crosscheck` 失败：检查 `vol_d` 索引方向 —— 分母应是 **T-1 日**（前一交易日）的日总量。`prev_rel = rel - 1` 对应 daily volume.bin 的前一格。确认 `si_dv == si_dc`（daily 字段同 start）。

- [ ] **Step 5: Commit**

```bash
git add tests/test_materialize_minute.py qlib_ifind_beta/materialize_minute.py
git commit -m "feat(materialize-minute): vectorized 14-factor day.bin materialization + A1 crosscheck

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 5: 全 universe 物化 CLI + 接入 build_overlay（A2/A3）

两个落点：(a) 独立 CLI `scripts/materialize_minute.py` 只读现有 instruments 文件循环物化（不依赖 iFinD，改公式后重跑用）；(b) `build_overlay.py` 循环里加调用，保证 fresh build 含分钟因子。

**Files:**
- Create: `scripts/materialize_minute.py`
- Modify: `scripts/build_overlay.py:24,48-58`

- [ ] **Step 1: 建独立 CLI**

Create `scripts/materialize_minute.py`：

```python
"""Re-materialize 14 minute factors + $price_941 without redoing the full overlay.

Use after tweaking qlib_ifind_beta/minute_factors.py formulas — avoids re-pulling
the iFinD universe (unlike scripts/build_overlay.py). Reads the existing
instruments/highbeta883926.txt + ensures each stock's overlay dir exists, then
calls materialize_minute_instrument per code.

Run: conda run -n qlib_ifind_beta python scripts/materialize_minute.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qlib_ifind_beta import materialize_minute, overlay
from qlib_ifind_beta.config import INSTRUMENTS_DST, UNIVERSE_MARKET


def _load_codes():
    p = INSTRUMENTS_DST / f"{UNIVERSE_MARKET}.txt"
    codes = []
    with open(p) as fp:
        for line in fp:
            parts = line.strip().split("\t")
            if parts and parts[0]:
                codes.append(parts[0])
    return sorted(set(codes))


def main():
    codes = _load_codes()
    print(f"▶ materializing minute factors for {len(codes)} codes")
    ok, miss = [], []
    for i, code in enumerate(codes, 1):
        overlay.link_stock(code)                       # ensure feature dir + 7 base bins
        if materialize_minute.materialize_minute_instrument(code):
            ok.append(code)
        else:
            miss.append(code)
        if i % 100 == 0:
            print(f"  …{i}/{len(codes)}  (ok={len(ok)}, miss={len(miss)})")
    print(f"✓ minute factors: {len(ok)} ok, {len(miss)} missing")
    if miss:
        print(f"  missing sample: {miss[:12]}")
        print(f"  (expected ~76 — delisted/suspended stocks with no 1min/daily bins)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 接入 build_overlay.py**

Modify `scripts/build_overlay.py`（两处 Edit，锚点已对齐真实文件）：

**(2a) import 行（line 24）**，把
```python
from qlib_ifind_beta import overlay, universe, materialize
```
改为
```python
from qlib_ifind_beta import overlay, universe, materialize, materialize_minute
```

**(2b) per-stock 循环（lines 46-58）**，把
```python
    # 4. per-stock overlay + derived bins
    ok, miss = [], []
    for i, code in enumerate(codes, 1):
        overlay.link_stock(code)               # 7 base bins (symlink)
        if materialize.materialize_instrument(code):  # change/limit_up/limit_down
            ok.append(code)
        else:
            miss.append(code)
        if i % 25 == 0:
            print(f"  …{i}/{len(codes)} stock dirs done")
    summary["materialized_ok"] = len(ok)
    summary["materialized_missing"] = miss
    print(f"✓ derived bins: {len(ok)} ok" + (f", {len(miss)} missing: {miss}" if miss else ""))
```
改为
```python
    # 4. per-stock overlay + derived bins (daily change/limit + minute factors)
    ok, miss = [], []
    min_ok, min_miss = 0, 0
    for i, code in enumerate(codes, 1):
        overlay.link_stock(code)               # 7 base bins (symlink)
        if materialize.materialize_instrument(code):  # change/limit_up/limit_down
            ok.append(code)
        else:
            miss.append(code)
        if materialize_minute.materialize_minute_instrument(code):  # 14 minute factors + price_941
            min_ok += 1
        else:
            min_miss += 1
        if i % 25 == 0:
            print(f"  …{i}/{len(codes)} stock dirs done")
    summary["materialized_ok"] = len(ok)
    summary["materialized_missing"] = miss
    summary["minute_ok"] = min_ok
    summary["minute_missing"] = min_miss
    print(f"✓ derived bins: {len(ok)} ok" + (f", {len(miss)} missing: {miss}" if miss else ""))
    print(f"✓ minute factors: {min_ok} ok, {min_miss} missing (delisted/no-1min)")
```

- [ ] **Step 3: 跑全 universe 物化（用独立 CLI，不碰 iFinD）**

Run: `conda run -n qlib_ifind_beta python scripts/materialize_minute.py`
Expected:
- 进度行 `…100/N  (ok=..., miss=...)` 滚动；
- 末尾 `✓ minute factors: ~5040 ok, ~76 missing`（A2：missing ≈ 76，与日频缺 bin 集一致，不崩）；
- 耗时约 5-10 分钟（5040 票 × 5 个 1min.bin 读，~156GB IO，SSD + page cache）。

- [ ] **Step 4: A3 日历对齐抽样验证**

Run:
```bash
conda run -n qlib_ifind_beta python -c "
from pathlib import Path
from qlib_ifind_beta.binio import read_bin
from qlib_ifind_beta.config import FEATURES_SRC, FEATURES_DST, MINUTE_FACTOR_FIELDS
for code in ['sh600519','sz300750','sh688981']:
    si_dc, _ = read_bin(Path(FEATURES_SRC)/code/'close.day.bin')
    si_out, _ = read_bin(Path(FEATURES_DST)/code/'startup_mom_1m.day.bin')
    si_p941, n = read_bin(Path(FEATURES_DST)/code/'price_941.day.bin')
    print(f'{code}: daily_start={si_dc} minute_start={si_out} price941_start={si_p941} len={n} align={si_dc==si_out==si_p941}')
"
```
Expected: 三票均 `align=True`，`len` 与各自 daily close.bin 长度一致（A3 通过）。

- [ ] **Step 5: Commit**

```bash
git add scripts/materialize_minute.py scripts/build_overlay.py
git commit -m "feat(materialize-minute): standalone CLI + wire into build_overlay

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 6: HighBetaAlpha158 子类 + 特征数单测（A5）

Handler 子类追加 14 个 `$field` 到 Alpha158 原生 158 特征。`get_feature_config` 不依赖实例状态（Alpha158.get_feature_config 忽略 self，直接调 Alpha158DL），故单测可用 `__new__` 绕开 fetch。

**Files:**
- Create: `qlib_ifind_beta/highbeta_handler.py`
- Test: `tests/test_highbeta_handler.py`

- [ ] **Step 1: 写失败测试**

Create `tests/test_highbeta_handler.py`：

```python
"""HighBetaAlpha158 appends 14 $minute-fields to Alpha158's native 158 (A5 unit part).

get_feature_config ignores instance state (Alpha158 delegates to Alpha158DL), so we
can test it via __new__ without qlib.init / data fetch. The full fetch shape (172
columns in the real df) is verified by the smoke run (Task 8).
"""
from qlib_ifind_beta.config import MINUTE_FACTOR_FIELDS
from qlib_ifind_beta.highbeta_handler import HighBetaAlpha158


def test_feature_count_is_158_plus_14():
    h = HighBetaAlpha158.__new__(HighBetaAlpha158)
    fields, names = h.get_feature_config()
    assert len(names) == 158 + 14
    assert len(fields) == len(names)


def test_minute_fields_present():
    h = HighBetaAlpha158.__new__(HighBetaAlpha158)
    fields, names = h.get_feature_config()
    for mf in MINUTE_FACTOR_FIELDS:
        assert f"${mf}" in names, mf
        assert f"${mf}" in fields, mf
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n qlib_ifind_beta python -m pytest tests/test_highbeta_handler.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'qlib_ifind_beta.highbeta_handler'`

- [ ] **Step 3: 实现 highbeta_handler.py**

Create `qlib_ifind_beta/highbeta_handler.py`：

```python
"""HighBetaAlpha158 — Alpha158 + 14 materialized minute factors.

Overrides get_feature_config to append 14 $-prefixed minute fields (materialized
as day.bin by materialize_minute.py) to Alpha158's native 158. qlib reads them
like any other field; no frequency mixing at the Handler layer.

label / deal_price are passed via qrun YAML (handler.kwargs.label,
exchange_kwargs.deal_price), not here — see qrun/workflow.yaml.
"""
from __future__ import annotations

from qlib.contrib.data.handler import Alpha158

from .config import MINUTE_FACTOR_FIELDS


class HighBetaAlpha158(Alpha158):
    def get_feature_config(self):
        fields, names = super().get_feature_config()   # native 158
        min_fields = [f"${n}" for n in MINUTE_FACTOR_FIELDS]
        return fields + min_fields, names + min_fields
```

- [ ] **Step 4: 跑测试确认通过**

Run: `conda run -n qlib_ifind_beta python -m pytest tests/test_highbeta_handler.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add tests/test_highbeta_handler.py qlib_ifind_beta/highbeta_handler.py
git commit -m "feat(handler): HighBetaAlpha158 subclass appends 14 minute factors

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 7: run.py sys.path 修复 + YAML 三处改动

`python qrun/run.py` 把 `qrun/`（脚本目录）放 sys.path[0]，项目根不在路径上 → handler 的 `module_path: qlib_ifind_beta.highbeta_handler` import 失败。run.py 需显式插入项目根。YAML 改 handler class / label / deal_price 三处（workflow.yaml + workflow_smoke.yaml 各一份）。

**Files:**
- Modify: `qrun/run.py:18-22`（imports 前插 sys.path）
- Modify: `qrun/workflow.yaml:33-34,62,88-89`
- Modify: `qrun/workflow_smoke.yaml:22-23,46,71-72`

- [ ] **Step 1: run.py 插入项目根到 sys.path**

在 `qrun/run.py` 的 `import sys` / `from pathlib import Path` 之后、`from ruamel.yaml import YAML` 之前插入：

```python
# 让 handler 的 module_path: qlib_ifind_beta.highbeta_handler 可 import。
# `python qrun/run.py` 把 qrun/（脚本目录）放 sys.path[0]，项目根不在路径上。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

具体：把
```python
import os
import sys
from pathlib import Path

# 必须在 import qlib / mlflow 前设置
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

from ruamel.yaml import YAML
```
改为
```python
import os
import sys
from pathlib import Path

# 必须在 import qlib / mlflow 前设置
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

# 让 handler 的 module_path: qlib_ifind_beta.highbeta_handler 可 import。
# `python qrun/run.py` 把 qrun/（脚本目录）放 sys.path[0]，项目根不在路径上。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ruamel.yaml import YAML
```

- [ ] **Step 2: workflow.yaml 改 handler class**

把 `qrun/workflow.yaml` 的
```yaml
                class: Alpha158            # qlib contrib 原生，零自定义因子，仅依赖 7 字段
                module_path: qlib.contrib.data.handler
```
改为
```yaml
                class: HighBetaAlpha158    # Alpha158 + 14 分钟因子（qlib_ifind_beta/highbeta_handler.py）
                module_path: qlib_ifind_beta.highbeta_handler
```

- [ ] **Step 3: workflow.yaml 改 label**

把 `qrun/workflow.yaml` 的
```yaml
    # Label（用户指定 2026-07-05）：T+1 开盘买入、T+2 收盘卖出的收益。
    # 默认 Alpha158 label = Ref($close,-2)/Ref($close,-1)-1（close[T+1]→close[T+2]）；
    # 此处分母改 $open：T 日收盘出信号 → T+1 开盘成交（买价）、T+2 收盘卖出。
    # deal_price 已同步为 ["$open","$close"]（见下 exchange_kwargs.deal_price，qlib 原生支持
    #   买卖不同价 exchange.py:44/157-164）：买入 open[T+1]、卖出 close[T+2]，与本 label 完全对齐。
    label:
        - Ref($close, -2) / Ref($open, -1) - 1
```
改为
```yaml
    # Label（分钟因子版 2026-07-06）：T 日 9:41 买入、T+1 收盘卖出的收益。
    # 分钟特征 ≤ T 日 9:40、买入价 $price_941[T]（9:41）、卖出 $close[T+1]，全程无前视。
    # deal_price 同步为 ["$price_941","$close"]（见下 exchange_kwargs.deal_price）。
    label:
        - Ref($close, -1) / $price_941 - 1
```

- [ ] **Step 4: workflow.yaml 改 deal_price**

把 `qrun/workflow.yaml` 的
```yaml
            deal_price: ["$open", "$close"]
                                 # 买卖不同价（exchange.py:44/157-164 原生支持二元 deal_price）：
                                 # 买入 open[T+1]、卖出 close[T+2]，与 label 完全对齐。
```
改为
```yaml
            deal_price: ["$price_941", "$close"]
                                 # 买卖不同价（exchange.py:44/157-164 原生支持二元 deal_price）：
                                 # 买入 $price_941[T]（T 日 9:41 close）、卖出 $close[T+1]，与 label 完全对齐。
```

- [ ] **Step 5: workflow_smoke.yaml 同样三处改动**

(5a) handler class：把
```yaml
                class: Alpha158
                module_path: qlib.contrib.data.handler
```
改为
```yaml
                class: HighBetaAlpha158
                module_path: qlib_ifind_beta.highbeta_handler
```

(5b) label：把
```yaml
    # Label：T+1 开盘买、T+2 收盘卖（与 workflow.yaml 一致）。
    label:
        - Ref($close, -2) / Ref($open, -1) - 1
```
改为
```yaml
    # Label：T 日 9:41 买、T+1 收盘卖（与 workflow.yaml 一致）。
    label:
        - Ref($close, -1) / $price_941 - 1
```

(5c) deal_price：把
```yaml
            deal_price: ["$open", "$close"]
                                 # 买卖不同价（见 workflow.yaml 详解）：买 open[T+1]、卖 close[T+2]，与 label 完全对齐
```
改为
```yaml
            deal_price: ["$price_941", "$close"]
                                 # 买卖不同价（见 workflow.yaml 详解）：买 $price_941[T]（9:41）、卖 $close[T+1]，与 label 对齐
```

- [ ] **Step 6: YAML 解析校验**

Run: `conda run -n qlib_ifind_beta python -c "from ruamel.yaml import YAML; c=YAML(typ='safe',pure=True).load(open('qrun/workflow.yaml')); h=c['task']['dataset']['kwargs']['handler']; print(h['class'], h['module_path'], c['data_handler_config']['label'], c['port_analysis_config']['backtest']['exchange_kwargs']['deal_price'])"`
Expected: `HighBetaAlpha158 qlib_ifind_beta.highbeta_handler ['Ref($close, -1) / $price_941 - 1'] ['$price_941', '$close']`

- [ ] **Step 7: Commit**

```bash
git add qrun/run.py qrun/workflow.yaml qrun/workflow_smoke.yaml
git commit -m "feat(qrun): HighBetaAlpha158 + 9:41 label/deal_price + run.py sys.path fix

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 8: 烟雾测试全链路（B1 回测跑通 + B2 撮合生效 + A5 fetch shape）

用 2025 子窗口快速跑通新链路（Handler 拼表 → 训练 → 回测），确认 172 特征 fetch、`$price_941` 进 quote_df、买入价 = 9:41 close。

**Files:** 无（运行验证）

- [ ] **Step 1: 跑烟雾测试**

Run: `conda run -n qlib_ifind_beta python qrun/run.py qrun/workflow_smoke.yaml`
Expected:
- `[run.py] limit_threshold coerce: PortAnaRecord: list(2) → tuple`
- `[run.py] experiment=workflow provider_uri=.../data/qlib_root`
- 训练日志（LGBModel，~17s）
- `SignalRecord` → `SigAnaRecord`（打印 IC/Rank IC）→ `PortAnaRecord`
- `[run.py] done. recorder_id=<id>`
- 无 `KeyError` / `module not found` / `$price_941` 缺列错误。

> 若报 `KeyError: '$price_941'` 或 `Feature ... not provided`：Task 5 的全 universe 物化没覆盖到烟雾窗口里的票，或某票 price_941.day.bin 没写。重跑 Task 5 Step 3，再查 `ls data/qlib_root/features/sh600519/ | grep price_941`。

- [ ] **Step 2: A5 — 确认 fetch 出 172 feature + 1 label**

Run（用最新 recorder，或直接构造 Handler fetch）：
```bash
conda run -n qlib_ifind_beta python -c "
import qlib
from qlib.utils import init_instance_by_config
qlib.init(provider_uri='data/qlib_root', region='cn')
cfg = {'class':'HighBetaAlpha158','module_path':'qlib_ifind_beta.highbeta_handler',
       'kwargs':{'start_time':'2025-11-01','end_time':'2025-11-10','fit_start_time':'2025-01-01',
                 'fit_end_time':'2025-09-30','instruments':'highbeta883926',
                 'label':['Ref(\$close, -1) / \$price_941 - 1']}}
h = init_instance_by_config(cfg)
df = h.fetch()
print('shape:', df.shape, '| feature cols:', len(df.columns)-1)
# 分钟因子列名原样带 \$（DataLoaderLP names 即我们传的 \$field）
print('minute cols present:', [c for c in df.columns if c in ['\$startup_mom_1m','\$vol_vs_yest','\$price_941']])
print('any non-null in last col (label):', df.iloc[:, -1].notna().any())
"
```
Expected:
- `shape: (N, 173)`（172 feature + 1 label）；
- `minute cols present: ['$startup_mom_1m', '$vol_vs_yest']`（`$price_941` 是 deal_price 字段，不在 feature 里——它在 quote_df，不在 handler feature 表）；
- label 列（最后一列）有非 NaN 值。

- [ ] **Step 3: B2 — 确认 deal_price=$price_941 生效（买入价 = 9:41 close）**

Run（读最新 recorder 的 report_normal，看买入价是否 ≈ 当日 close 而非 open）：
```bash
conda run -n qlib_ifind_beta python -c "
import qlib, glob, pandas as pd
from qlib.workflow import R
qlib.init(provider_uri='data/qlib_root', region='cn')
exp = R.get_exp(experiment_name='workflow')
rec = max(exp.list_recorders(), key=lambda r: r.info.get('start_time', ''))
rep = rec.load_object('portfolio_analysis/report_normal_1day.pkl')
print(rec.recorder_id)
print(rep.head(3).to_string())
"
```
Expected: `report_normal_1day.pkl` 存在；交易记录的 buy 价格在合理区间（9:41 close ≈ 当日 vwap 附近，介于 open 与 close 之间，**不等于** T 日 open——那是旧 deal_price）。

> B2 是定性核验：只要跑通 + deal_price 配置生效（无报错 + 报告产出）即视为通过；精确逐笔买入价 vs 9:41 close 的对账用 Task 9 全量跑后的 positions_normal 抽样。

- [ ] **Step 4: Commit（无代码变更，跳过；如烟雾跑中发现需修的 bug，单列 commit）**

烟雾测试是验证步骤，本身无产物进 git。**只有**当本步暴露出代码 bug 并修复时才 commit，消息写明修了什么。

---

## Task 9: 全量回测 + baseline IC 对比 + 特征重要性（B3/B4）

核心判据。加 14 分钟因子后 IC 必须比 baseline（IC=-0.0078）提升（B3）；特征重要性 top-30 里至少几个分钟因子（B4）。

**Files:** 无（运行 + 对比）

- [ ] **Step 1: 全量回测**

Run: `conda run -n qlib_ifind_beta python qrun/run.py qrun/workflow.yaml`
Expected:
- 全程 ~30-90 秒（172 特征 × 2.5 年，比 baseline 158 特征略重）；
- 完整产出：`pred.pkl` / `label.pkl` / `sig_analysis/{ic,ric}.pkl` / `portfolio_analysis/*.pkl`；
- `[run.py] done. recorder_id=<id>`。

- [ ] **Step 2: B3 — IC vs baseline 对比**

Run:
```bash
conda run -n qlib_ifind_beta python -c "
import qlib, pandas as pd
from qlib.workflow import R
qlib.init(provider_uri='data/qlib_root', region='cn')
exp = R.get_exp(experiment_name='workflow')
recs = exp.list_recorders()
rec = max(recs, key=lambda r: r.info.get('start_time',''))
ic = rec.load_object('sig_analysis/ic.pkl')
ric = rec.load_object('sig_analysis/ric.pkl')
print('recorder:', rec.recorder_id)
print(f'IC    = {ic.mean():.4f} ± {ic.std():.4f}   ICIR = {ic.mean()/ic.std():.4f}')
print(f'RankIC= {ric.mean():.4f} ± {ric.std():.4f}')
print('baseline was: IC=-0.0078 ± 0.140  ICIR=-0.055  RankIC=-0.0072')
"
```
Expected & 判据：
- IC / RankIC / ICIR **三项均高于 baseline**（baseline 全 ≈ 0 / 负）→ B3 通过，设计有效；
- 若 IC 仍 ≈ 0 或更负 → B3 失败，**回炉**（不是调参，是因子设计没抓到信号，重想公式）。

- [ ] **Step 3: B4 — 特征重要性 top-30 含分钟因子**

Run:
```bash
conda run -n qlib_ifind_beta python -c "
import qlib, pandas as pd
from qlib.workflow import R
qlib.init(provider_uri='data/qlib_root', region='cn')
exp = R.get_exp(experiment_name='workflow')
rec = max(exp.list_recorders(), key=lambda r: r.info.get('start_time',''))
# trainer.py:50 存的 key 是 'params.pkl'，反序列化回 LGBModel 实例，
# 其 .model 属性是 lightgbm Booster（gbdt.py:77 self.model = lgb.train(...)）。
lgbm = rec.load_object('params.pkl')
booster = lgbm.model
imp = pd.Series(booster.feature_importance(importance_type='gain'),
                index=booster.feature_name()).sort_values(ascending=False)
top = imp.head(30)
from qlib_ifind_beta.config import MINUTE_FACTOR_FIELDS
minute = [n for n in top.index if n in MINUTE_FACTOR_FIELDS]
print('top-30:'); print(top.to_string())
print(f'minute factors in top-30: {minute}')
"
```
Expected & 判据：
- top-30 里**至少 2-3 个**分钟因子 → B4 通过（设计抓到了信号）；
- 若 14 个分钟因子**一个都没进** top-30 → B4 失败，**重想因子公式**（spec §B4 「定生死」）。

- [ ] **Step 4: 记录结论 + 决定下一步**

把 B3/B4 结果记入项目主文档笔记（Get笔记 note_id=1914664125624050528），追加 `## [2026-07-06] 分钟因子 v1 全量回测结果`，含：IC/RankIC/ICIR vs baseline、top-30 含哪些分钟因子、是否通过 B3/B4 判据、下一步（通过→调参/扩因子；失败→回炉公式）。**secret 不入正文**。

> 若 B3/B4 均通过：本计划完成，pipeline 可用。若失败：不要在本计划里改设计——回 brainstorming 重审因子公式（spec §已知妥协 列出的候选：卖出价优化、更多窗口周期、HFLGBModel 二分类等）。

---

## 自查（spec 覆盖 / 占位符扫描 / 类型一致性）

**1. Spec 覆盖**：

| spec 条目 | 实现任务 |
|---|---|
| §因子集 A 启动动量 4 个 | Task 2（公式）+ Task 4（物化） |
| §因子集 B 加速度 3 个 | Task 2 + Task 4 |
| §因子集 C 拉升形态 3 个 | Task 2 + Task 4 |
| §因子集 D 量能放量 3 个 | Task 2 + Task 4 |
| §因子集 E 跨日量能 vol_vs_yest（双源） | Task 2（prev_day_volume 参数）+ Task 4（读日频 volume.bin 分母） |
| §物化架构 方案 A（独立读 1min.bin，qlib 不挂双频） | Task 4 + Task 5 |
| §Handler 子类 HighBetaAlpha158 | Task 6 |
| §Label `Ref($close,-1)/$price_941-1` | Task 7（workflow.yaml + smoke） |
| §deal_price `["$price_941","$close"]` | Task 7 |
| §验证 A1 物化正确性 | Task 4 测试（startup/price_941/vol_vs_yest 交叉核验） |
| §验证 A2 缺失容错 | Task 5 Step 3（~76 missing 不崩） |
| §验证 A3 日历对齐 | Task 3（格式）+ Task 5 Step 4（三票 start_index 对齐） |
| §验证 A4 无前视 | Task 4 `test_no_lookahead_*` + spec §时序模型 已论证（特征 ≤9:40、撮合 9:41、日频 lag1） |
| §验证 A5 混频拼表 172 feature | Task 6 单测（158+14）+ Task 8 Step 2（fetch shape 173） |
| §验证 B1 链路跑通 | Task 8 + Task 9 |
| §验证 B2 deal_price 生效 | Task 8 Step 3 |
| §验证 B3 IC 对比（命门） | Task 9 Step 2 |
| §验证 B4 特征重要性（定生死） | Task 9 Step 3 |

无遗漏。

**2. 占位符扫描**：无 TBD/TODO/"implement later"。所有代码块完整。YAML 改动给全了 old/new。失败-实现-通过 TDD 闭环。

**3. 类型一致性**：
- `MINUTE_FACTOR_FIELDS`（config，14 元组）在 Task 1/2/4/6 引用一致；
- `compute_day_factors` 返回 dict 含 14 因子名 + `price_941`（Task 2 定义，Task 4 向量化重算同名因子，未调 compute_day_factors —— 见下注）；
- `materialize_minute_instrument(code) -> bool`（Task 4 定义，Task 5 CLI 与 build_overlay 调用一致）；
- `_load_min_calendar() / _load_day_calendar_lookup()`（Task 4 定义，Task 4 测试与 Task 5 间接共用）。

> **注（设计说明，非占位符）**：Task 4 物化层**向量化重算** 14 因子（性能：5040 票），未调用 Task 2 的 `compute_day_factors`（那是逐日标量版，供单测手算核验）。两者公式逐字相同（Task 2 单测 + Task 4 交叉核验共同保证一致）。若未来想 DRY 合并，可让物化层用 numpy 向量化版、单测用标量版，共一份公式常量 —— 但当前 YAGNI，两版各自被测覆盖即可。

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-07-06-minute-factors.md`. Two execution options:**

**1. Subagent-Driven（推荐）** — 每个 Task 派一个 fresh subagent，两阶段 review，快速迭代。

**2. Inline Execution** — 本会话内按 executing-plans 批量执行，checkpoint review。

**选哪种？**
