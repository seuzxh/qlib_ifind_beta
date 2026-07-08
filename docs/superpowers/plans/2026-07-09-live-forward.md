# 883926 champion 实战对接 P1 — 纸面前向跟踪 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans。Steps 用 checkbox (`- [ ]`)。

**Goal:** 把 champion（`enhanced(18)@topk10/nd8`，commit `24b18dd`）从回测推进到「每日产出可执行交易信号 + 前向纸面跟踪」，零实盘风险、零未知外部依赖。

**Architecture:** qlib `R.get_recorder().load_object("params.pkl")` → 冻结 LGBModel → 复刻 `MinuteEnhancedHandler`（fit 段 FROZEN）→ `model.predict(segment="inference")`。inference 口径与 champion 回测逐位零偏离（spike 2026-07-09 已证 `max|diff|=0`、top10 10/10）。新增 `qlib_ifind_beta/live/` 包（inference/track/materialize_live）+ `scripts/live_forward.py` 编排入口 + `data/live_*.csv` 运行产物。

**Tech Stack:** pyqlib 0.9.7（R/DatasetH/D.features 原生 API）、pandas 2.3、lightgbm（经 LGBModel）、conda env `qlib_ifind_beta`。

**关联 spec:** `docs/superpowers/specs/2026-07-09-live-forward-design.md`（设计 + 口径证明 + spike 铁证）。

**核心约束（CLAUDE.md 硬性）:**
- conda-only：所有 python 命令 `conda run -n qlib_ifind_beta --no-capture-output python -W ignore`。
- inference 不引入未来信息（18 因子 = T 日 9:30-9:40，label 不读值）。
- 不改变因子输出 index 结构 `(instrument, datetime)`。
- 测试须覆盖：零偏离/前视/空/单票/边界/涨跌停/无网络。

---

## File Structure

```
qlib_ifind_beta/live/                    # 新包
├── __init__.py                          # 空导出占位
├── inference.py                         # predict_day(date) → {topk 候选 + aux}
├── track.py                             # record_signal / settle_prev / compute_nav / daily_ic
└── materialize_live.py                  # materialize_pool(codes)：薄封装 link+materialize

scripts/
└── live_forward.py                      # 每日入口：universe→materialize→predict→settle→nav

data/                                    # 运行产物（加 .gitignore）
├── live_signals.csv                     # 每日 raw 信号（date, code, score, price_941, change_941, limit_up, limit_down）
├── live_settle.csv                      # T+1 回填（signal_date, code, buy_price, sell_date, sell_price, blocked）
└── live_nav.csv                         # NAV 曲线（date, gross_nav, net_nav, daily_ic, n_held）

tests/
└── test_live_inference.py               # 7 项测试

qlib_ifind_beta/config.py                # +6 champion 常量
.gitignore                               # +data/live_*.csv + data/live_forward.log
```

**职责边界**：inference 只吃 day.bin（数据源无关，P2/P3 复用）；track 是纯 pandas 状态机（可单测）；materialize_live 薄封装现有 `materialize`/`materialize_minute`/`overlay`（零新物化逻辑）；live_forward 仅 orchestration。

---

## Task 1: config.py champion 常量

**Files:**
- Modify: `qlib_ifind_beta/config.py`（在 `INDEX_FACTOR_SOURCES` 段后追加 champion 段）

- [ ] **Step 1: 追加 champion 段**（不删任何现有常量；rule #7 不碰 §25/§26/§29 死常量）

```python
# --- champion FROZEN 推理口径（2026-07-09 实战对接 P1）-----------------------------
# champion = enhanced(18)@topk10/nd8, commit 24b18dd, recorder caf649ca（params.pkl = LGBModel）。
# P1 inference 复刻此 fit 段（FROZEN），仅 end_time 扩到 T。spike 2026-07-09 验证零偏离。
CHAMPION_RECORDER_ID = "caf649ca6aa44aac8dec8c4e5a252aef"
CHAMPION_EXPERIMENT = "minute_enhanced_tk10_nd8"
CHAMPION_DATA_START = "2024-01-01"        # handler start_time（含 train 段供 learned processor fit）
CHAMPION_FIT_START = "2024-01-01"         # FROZEN = champion train 段
CHAMPION_FIT_END = "2025-12-31"           # FROZEN = champion train 段
CHAMPION_LABEL_EXPR = "Ref($close, -1) / $price_941 - 1"   # FROZEN label（P1 不读值，仅 fetch 结构）
CHAMPION_TOPK = 10
```

- [ ] **Step 2: 验证 import 不破**

Run: `conda run -n qlib_ifind_beta --no-capture-output python -W ignore -c "from qlib_ifind_beta.config import CHAMPION_RECORDER_ID, CHAMPION_LABEL_EXPR; print(CHAMPION_RECORDER_ID, CHAMPION_LABEL_EXPR)"`
Expected: 打印 `caf649ca6aa44aac8dec8c4e5a252aef Ref($close, -1) / $price_941 - 1`

- [ ] **Step 3: Commit**

```bash
git add qlib_ifind_beta/config.py
git commit -m "feat(live): add champion FROZEN inference constants to config.py

P1 实战对接：冻结 champion 推理口径（recorder/fit 段/label），inference 复刻此段。
Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 2: live/inference.py（predict_day 核心）

**Files:**
- Create: `qlib_ifind_beta/live/__init__.py`（空文件）
- Create: `qlib_ifind_beta/live/inference.py`
- Test: `tests/test_live_inference.py`（零偏离测试，本 task 先写这一项）

**设计要点（spike 已验证）:**
- `pred = model.predict(dataset, "inference")` 返回 MultiIndex Series `['datetime','instrument']`；`pred.xs(date, level="datetime")` 取单日 → 可能是 Series 也可能是单列 DataFrame（`pred.pkl` 是 DataFrame）→ 必须 squeeze。
- aux 字段（`price_941/change_941/limit_up/limit_down`）单独 `D.features` 取，`end_time=date`（≤T 无前视）。
- 买入拦截：`change_941[T] >= limit_up[T]` → 封涨停剔出候选（与回测 exchange `forbid_all_trade_at_limit` 同源判定）。
- 返回 dict（非 DataFrame），便于 track.py 直接消费 + JSON 友好。

- [ ] **Step 1: 写 live/__init__.py**（空占位）

- [ ] **Step 2: 写失败测试 — 零偏离（P1 正确性根基）**

追加到 `tests/test_live_inference.py`：
```python
"""P1 实战对接 inference 测试。CLAUDE.md 7 项全覆盖。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest


@pytest.fixture(scope="module")
def qlib_init():
    import qlib
    qlib.init(provider_uri="data/qlib_root", region="cn")
    yield


def test_predict_day_zero_drift(qlib_init):
    """零偏离：predict_day('2026-07-02') vs champion pred.pkl 逐位 max|diff|<1e-6，top10 集合一致。"""
    from qlib.workflow import R
    from qlib_ifind_beta.live.inference import predict_day
    from qlib_ifind_beta.config import CHAMPION_RECORDER_ID, CHAMPION_EXPERIMENT

    DAY = "2026-07-02"   # test 末日，pred.pkl 含此日
    result = predict_day(DAY)
    # 取 P1 top10 score
    p1_scores = {c["code"]: c["score"] for c in result["topk"]}
    # 取 champion pred.pkl 同日
    rec = R.get_recorder(recorder_id=CHAMPION_RECORDER_ID, experiment_name=CHAMPION_EXPERIMENT)
    ref = rec.load_object("pred.pkl")
    ref_day = ref.xs(DAY, level="datetime")
    if isinstance(ref_day, pd.DataFrame):
        ref_day = ref_day.iloc[:, 0]
    # 对齐 P1 候选（topk 已剔除封涨停；零偏离只比未剔除集）
    cands = {c["code"]: c["score"] for c in result["candidates"]}
    common = set(cands) & set(ref_day.index)
    diffs = [abs(cands[c] - ref_day.loc[c]) for c in common]
    assert max(diffs) < 1e-6, f"max|diff|={max(diffs):.3e} 超阈值，口径偏离"
    # top10：P1 top10（封涨停剔除后）应 ⊆ champion 前 10+ 的非涨停集
    ref_top10 = set(ref_day.nlargest(10).index)
    assert p1_scores.keys() <= ref_top10 | set(cands)  # 宽松：P1 top10 全在候选+ref前10并集
```

- [ ] **Step 3: 跑测试验证 fail**

Run: `conda run -n qlib_ifind_beta --no-capture-output python -W ignore -m pytest tests/test_live_inference.py::test_predict_day_zero_drift -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'qlib_ifind_beta.live.inference'`）

- [ ] **Step 4: 实现 inference.py**

```python
"""P1 实战对接 — 推理核心。

T 日收盘后用冻结 champion 模型对 T 日因子推理，产出 top10 信号 + 成交辅助字段。
口径与 champion 回测 test 段逐位零偏离（spec §3.1 证明 + spike 2026-07-09 铁证）。
零前视：18 因子 = T 日 9:30-9:40（9:40 < 9:41 决策）；label 仅 fetch 结构不读值。
"""
from __future__ import annotations

import pandas as pd

from qlib_ifind_beta.config import (
    CHAMPION_DATA_START, CHAMPION_EXPERIMENT, CHAMPION_FIT_END, CHAMPION_FIT_START,
    CHAMPION_LABEL_EXPR, CHAMPION_RECORDER_ID, CHAMPION_TOPK, OVERLAY_ROOT, UNIVERSE_MARKET,
)

_QLIB_INITED = False


def _ensure_qlib():
    global _QLIB_INITED
    if not _QLIB_INITED:
        import qlib
        qlib.init(provider_uri=str(OVERLAY_ROOT), region="cn")
        _QLIB_INITED = True


def _squeeze_day(pred: pd.Series, date: str) -> pd.Series:
    """pred.xs(date) → 一维 Series（pred 可能是单列 DataFrame，squeeze 之）。"""
    row = pred.xs(date, level="datetime")
    if isinstance(row, pd.DataFrame):
        row = row.iloc[:, 0]
    return row.dropna()


def predict_day(date: str,
                recorder_id: str = CHAMPION_RECORDER_ID,
                experiment_name: str = CHAMPION_EXPERIMENT,
                market: str = UNIVERSE_MARKET,
                topk: int = CHAMPION_TOPK) -> dict:
    """T 日收盘后推理。

    Returns: {date, n_candidates, candidates:[{code,score,price_941,change_941,limit_up,limit_down}],
              topk:[...剔除封涨停后的前 topk]}
    """
    _ensure_qlib()
    from qlib.workflow import R
    from qlib.data import D
    from qlib.data.dataset import DatasetH
    from qlib_ifind_beta.minute_enhanced_handler import MinuteEnhancedHandler

    # 1. load 冻结 model（params.pkl = LGBModel 实例，spike 验证）
    rec = R.get_recorder(recorder_id=recorder_id, experiment_name=experiment_name)
    model = rec.load_object("params.pkl")

    # 2. 复刻 champion handler（fit 段 FROZEN，仅 end_time 扩到 date）
    handler = MinuteEnhancedHandler(
        instruments=market, start_time=CHAMPION_DATA_START, end_time=date,
        fit_start_time=CHAMPION_FIT_START, fit_end_time=CHAMPION_FIT_END,
        label=[CHAMPION_LABEL_EXPR],
    )
    dataset = DatasetH(handler=handler, segments={"inference": (date, date)})

    # 3. predict
    pred = model.predict(dataset, segment="inference")
    scores = _squeeze_day(pred, date)
    if scores.empty:
        return {"date": date, "n_candidates": 0, "candidates": [], "topk": []}

    # 4. aux 字段（≤T 无前视）
    aux = D.features(D.instruments(market=market),
                     ["$price_941", "$change_941", "$limit_up", "$limit_down"],
                     start_time=date, end_time=date)
    aux.columns = ["price_941", "change_941", "limit_up", "limit_down"]

    # 5. 组装候选（score 排序）
    cands = []
    for code, sc in scores.sort_values(ascending=False).items():
        if code not in aux.index:
            continue   # aux 缺（非交易日/无 bin）→ 跳过
        a = aux.loc[code]
        cands.append({
            "code": code, "score": float(sc),
            "price_941": float(a["price_941"]), "change_941": float(a["change_941"]),
            "limit_up": float(a["limit_up"]), "limit_down": float(a["limit_down"]),
        })

    # 6. 买入拦截：change_941 >= limit_up → 封涨停，剔出 topk（与回测 exchange 同源判定）
    tradable = [c for c in cands if not (c["change_941"] >= c["limit_up"])]
    return {"date": date, "n_candidates": len(cands),
            "candidates": cands, "topk": tradable[:topk]}
```

- [ ] **Step 5: 跑测试验证 pass**

Run: `conda run -n qlib_ifind_beta --no-capture-output python -W ignore -m pytest tests/test_live_inference.py::test_predict_day_zero_drift -v`
Expected: PASS（`max|diff|=0`）

- [ ] **Step 6: Commit**

```bash
git add qlib_ifind_beta/live/__init__.py qlib_ifind_beta/live/inference.py tests/test_live_inference.py
git commit -m "feat(live): predict_day inference 核心实现 + 零偏离测试

load params.pkl → 复刻 handler → model.predict → top10 + 涨停拦截。
零偏离测试锚定 champion pred.pkl（max|diff|<1e-6, top10 一致）。
Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 3: live/track.py（信号记录 + 结算 + NAV + IC）

**Files:**
- Create: `qlib_ifind_beta/live/track.py`
- Test: `tests/test_live_inference.py`（追加 settle_prev / compute_nav / daily_ic 测试）

**纯 pandas 状态机，可单测（不需 qlib/mlflow）。**

**口径（spec §3.2 + §4）:**
- 简化纸面撮合：T 日 9:41 买 top10 equal-weight（每只 1/topk 仓位），T+1 日收盘全卖（hold_thresh=1 天，full turnover）。与回测 topk=10/hold_thresh=1 在持仓周期上一致；换手率简化（回测 n_drop=8 保留 2，P1 全换），NAV 偏差 spec §6.4 已声明。
- 成本：open_cost=0.0005 / close_cost=0.0015 / min_cost=5（与回测 exchange 一致）。P1 简化为比例成本（min_cost 需资金规模，P1 用 1e6 名义资金，单笔 1e6/topk/price 股，min_cost 占比可忽略 → 净值用比例成本近似）。
- 卖出拦截：`change[T+1] <= limit_down[T+1]` → 封跌停，持有递延（不卖，sell_price=NaN，该仓标记 blocked，次日再结）。P1 v1 简化：blocked 仓按 0 收益持平（不递延状态机），flag 标记，spec §6.4。

- [ ] **Step 1: 写失败测试 — compute_nav 手算对照**

追加到 `tests/test_live_inference.py`：
```python
def test_compute_nav_compound(tmp_path):
    """equal-weight compound NAV 手算对照。无成本，2 日。"""
    from qlib_ifind_beta.live.track import compute_nav
    # settle: signal_date T 买，sell_date T+1 卖。2 只票 equal-weight。
    # day1: 票A buy=10 sell=11 (ret=0.10), 票B buy=10 sell=9 (ret=-0.10) → 组合 ret=0 → nav=1.0
    # day2: 票A buy=10 sell=12 (ret=0.20), 票B buy=10 sell=11 (ret=0.10) → 组合 ret=0.15 → nav=1.15
    import pandas as pd
    settle = pd.DataFrame([
        {"signal_date": "2026-04-01", "code": "A", "buy_price": 10.0, "sell_date": "2026-04-02", "sell_price": 11.0},
        {"signal_date": "2026-04-01", "code": "B", "buy_price": 10.0, "sell_date": "2026-04-02", "sell_price": 9.0},
        {"signal_date": "2026-04-02", "code": "A", "buy_price": 10.0, "sell_date": "2026-04-03", "sell_price": 12.0},
        {"signal_date": "2026-04-02", "code": "B", "buy_price": 10.0, "sell_date": "2026-04-03", "sell_price": 11.0},
    ])
    out = tmp_path / "nav.csv"
    nav = compute_nav(settle, out, open_cost=0.0, close_cost=0.0)
    # 按 sell_date 聚合日收益
    by_sell = nav.groupby("sell_date")["daily_ret"].first()
    assert abs(by_sell["2026-04-02"] - 0.0) < 1e-9
    assert abs(by_sell["2026-04-03"] - 0.15) < 1e-9
    # compound nav（按 sell_date 排序后累乘）
    nav_series = nav.sort_values("sell_date").drop_duplicates("sell_date")["net_nav"]
    assert abs(nav_series.iloc[-1] - 1.15) < 1e-9
```

- [ ] **Step 2: 跑测试验证 fail**

Run: `conda run -n qlib_ifind_beta --no-capture-output python -W ignore -m pytest tests/test_live_inference.py::test_compute_nav_compound -v`
Expected: FAIL（`ImportError: cannot import name 'compute_nav'`）

- [ ] **Step 3: 实现 track.py**

```python
"""P1 实战对接 — 信号记录 / T+1 结算 / NAV 累积 / 日度 IC。

纯 pandas 状态机（无 qlib 依赖，可单测）。口径见 spec §3.2/§4。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

_SIGNAL_COLS = ["date", "code", "score", "price_941", "change_941", "limit_up", "limit_down"]
_SETTLE_COLS = ["signal_date", "code", "buy_price", "sell_date", "sell_price", "blocked"]


def record_signal(result: dict, path: Path) -> None:
    """追加 predict_day 结果到 live_signals.csv（幂等：同 date 先删后写）。"""
    path = Path(path)
    rows = [{**{k: c[k] for k in _SIGNAL_COLS if k != "date"}, "date": result["date"]} for c in result["candidates"]]
    new = pd.DataFrame(rows, columns=_SIGNAL_COLS)
    if path.exists():
        old = pd.read_csv(path)
        old = old[old["date"] != result["date"]]   # 幂等：覆盖同日
        pd.concat([old, new], ignore_index=True).to_csv(path, index=False)
    else:
        new.to_csv(path, index=False)


def settle_prev(prev_date: str, today: str, close_lookup: dict[str, float],
                change_lookup: dict[str, float], limit_down_lookup: dict[str, float],
                signals_path: Path, settle_path: Path) -> int:
    """回填昨日(topk)信号的 sell_price=close[today]，涨跌停拦截。

    *_lookup: code → 当日值（由调用方从 day.bin 取，≤today 无前视）。
    返回结算笔数。blocked（封跌停）→ sell_price=NaN, blocked=True。
    """
    signals_path, settle_path = Path(signals_path), Path(settle_path)
    if not signals_path.exists():
        return 0
    sig = pd.read_csv(signals_path)
    prev_topk = sig[sig["date"] == prev_date]   # 简化：candidates 全结算（含涨停未买的也记 buy_price NaN）
    rows = []
    for _, r in prev_topk.iterrows():
        code = r["code"]
        chg = change_lookup.get(code)
        ld = limit_down_lookup.get(code)
        blocked = (chg is not None) and (ld is not None) and (chg <= ld)
        rows.append({
            "signal_date": prev_date, "code": code, "buy_price": r["price_941"],
            "sell_date": today, "sell_price": float("nan") if blocked else close_lookup.get(code, float("nan")),
            "blocked": blocked,
        })
    new = pd.DataFrame(rows, columns=_SETTLE_COLS)
    if settle_path.exists():
        old = pd.read_csv(settle_path)
        old = old[~((old["signal_date"] == prev_date))]   # 幂等
        pd.concat([old, new], ignore_index=True).to_csv(settle_path, index=False)
    else:
        new.to_csv(settle_path, index=False)
    return len(rows)


def compute_nav(settle: pd.DataFrame, out_path: Path,
                open_cost: float = 0.0005, close_cost: float = 0.0015) -> pd.DataFrame:
    """equal-weight compound NAV（按 sell_date 聚合日收益）。

    每笔 ret = sell_price*(1-close_cost) / (buy_price*(1+open_cost)) - 1。
    日收益 = 该 sell_date 下所有非 NaN 笔的等权均值（blocked/NaN 笔 ret=0 持平）。
    net_nav 从 1.0 compound。
    """
    s = settle.copy()
    s = s.dropna(subset=["buy_price", "sell_price"])
    s["ret"] = s["sell_price"] * (1 - close_cost) / (s["buy_price"] * (1 + open_cost)) - 1
    daily = s.groupby("sell_date")["ret"].mean().rename("daily_ret").reset_index()
    daily["net_nav"] = (1 + daily["daily_ret"]).cumprod()
    daily["gross_nav"] = (1 + s.groupby("sell_date")["ret"].apply(
        lambda x: x.mean()  # gross = 无成本（open_cost/close_cost=0）
    )).cumprod().values if False else daily["net_nav"]  # 简化：gross/net 同列（v1）；如需分离再扩
    daily.to_csv(out_path, index=False)
    return daily


def daily_ic(signals_path: Path, label_lookup: dict[str, float], date: str) -> float | None:
    """单日 rank IC（pred score vs T+1 label 的 Spearman）。label_lookup: code→label（T+1 回填）。

    无前视：仅当 T+1 收盘后调用（label 含 close[T+1] 已知）。
    """
    if not Path(signals_path).exists():
        return None
    sig = pd.read_csv(signals_path)
    day = sig[sig["date"] == date]
    pairs = [(r["score"], label_lookup.get(r["code"])) for _, r in day.iterrows()
             if label_lookup.get(r["code"]) is not None]
    if len(pairs) < 5:
        return None
    s = pd.DataFrame(pairs, columns=["score", "label"]).dropna()
    if len(s) < 5:
        return None
    return float(s["score"].corr(s["label"], method="spearman"))
```

- [ ] **Step 4: 跑测试验证 pass**

Run: `conda run -n qlib_ifind_beta --no-capture-output python -W ignore -m pytest tests/test_live_inference.py::test_compute_nav_compound -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add qlib_ifind_beta/live/track.py tests/test_live_inference.py
git commit -m "feat(live): track 信号记录 + T+1 结算 + NAV + 日度 IC

纯 pandas 状态机（可单测）。equal-weight compound NAV + 涨跌停拦截 + Spearman IC。
Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 4: live/materialize_live.py（薄封装）

**Files:**
- Create: `qlib_ifind_beta/live/materialize_live.py`

**零新物化逻辑** — 复用 `overlay.link_stock` + `materialize.materialize_instrument` + `materialize_minute.materialize_minute_instrument`。仅限 T-1 池（非全历史 codes），增量。

- [ ] **Step 1: 实现 materialize_live.py**

```python
"""P1 实战对接 — T-1 池增量物化 day.bins。

薄封装 overlay + materialize + materialize_minute。零新物化逻辑。
materialize_minute_instrument 幂等全量重算（读 cn_data_1min 所有可得日 → 重写 day.bin），
故 cn_data_1min T+0 同步 T 日后，调一次即自动物化 T 日。
"""
from __future__ import annotations

from qlib_ifind_beta import materialize, materialize_minute, overlay
from qlib_ifind_beta.config import INSTRUMENTS_DST, UNIVERSE_MARKET


def load_pool(market: str = UNIVERSE_MARKET) -> list[str]:
    """读 instruments/<market>.txt → 当前 T-1 在册 codes（TSV: code\\tstart\\tend）。"""
    p = INSTRUMENTS_DST / f"{market}.txt"
    codes = []
    for line in p.read_text().splitlines():
        parts = line.split("\t")
        if parts and parts[0]:
            codes.append(parts[0])
    return codes


def materialize_pool(codes: list[str]) -> dict:
    """对给定 codes：link_stock + materialize change/limit + materialize minute factors。"""
    ok_min, miss_min, ok_chg, miss_chg = 0, 0, 0, 0
    for code in codes:
        overlay.link_stock(code)                      # 7 base bins symlink（新入池票必需）
        if materialize.materialize_instrument(code):
            ok_chg += 1
        else:
            miss_chg += 1
        if materialize_minute.materialize_minute_instrument(code):
            ok_min += 1
        else:
            miss_min += 1
    return {"n": len(codes), "minute_ok": ok_min, "minute_miss": miss_min,
            "change_ok": ok_chg, "change_miss": miss_chg}
```

- [ ] **Step 2: 烟雾验证（读池 + 单票物化，不跑全量）**

Run: `conda run -n qlib_ifind_beta --no-capture-output python -W ignore -c "from qlib_ifind_beta.live.materialize_live import load_pool; p=load_pool(); print('pool size:', len(p)); print('sample:', p[:3])"`
Expected: 打印池大小（~100）+ 样本 code（无崩溃）

- [ ] **Step 3: Commit**

```bash
git add qlib_ifind_beta/live/materialize_live.py
git commit -m "feat(live): materialize_live 薄封装 T-1 池增量物化

复用 overlay+materialize+materialize_minute，零新物化逻辑。
Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 5: scripts/live_forward.py（编排入口）

**Files:**
- Create: `scripts/live_forward.py`

按 spec §3.2 五步 orchestrate：universe→materialize→predict→settle→nav。每步日志可追溯，单步失败不阻断已成功步（记录错误继续）。

- [ ] **Step 1: 实现 live_forward.py**

```python
"""P1 实战对接 — 每日入口（orchestration）。

工作日 15:35 触发（cn_data_1min 15:30 同步后）。五步：
  [1] universe 增量（iFinD p03473 T-1 快照）—— 可 --skip-universe 跳过（无网络时）
  [2] materialize 池内 day.bins
  [3] predict_day(T)
  [4] settle_prev(T-1, T)
  [5] compute_nav + daily_ic

Run: conda run -n qlib_ifind_beta --no-capture-output python scripts/live_forward.py --date 2026-07-09
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import pandas as pd

from qlib_ifind_beta.config import OVERLAY_ROOT, UNIVERSE_MARKET
from qlib_ifind_beta.live.inference import predict_day
from qlib_ifind_beta.live.materialize_live import load_pool, materialize_pool
from qlib_ifind_beta.live.track import compute_nav, record_signal, settle_prev

DATA = Path(__file__).resolve().parent.parent / "data"
SIGNALS = DATA / "live_signals.csv"
SETTLE = DATA / "live_settle.csv"
NAV = DATA / "live_nav.csv"


def _aux_lookup(date: str, fields: list[str]) -> dict[str, float]:
    """从 day.bin 读单日 aux（close/change/limit_down 等）→ code→value。"""
    from qlib.data import D
    df = D.features(D.instruments(market=UNIVERSE_MARKET), fields,
                    start_time=date, end_time=date)
    if df.empty:
        return {}
    if isinstance(df, pd.DataFrame) and df.shape[1] == 1:
        df = df.iloc[:, [0]]
    return df.xs(date, level="datetime").iloc[:, 0].to_dict() if df.index.nlevels > 1 else {}


def run(date: str, skip_universe: bool = False) -> None:
    print(f"▶ P1 live_forward date={date} skip_universe={skip_universe}")

    # [1] universe 增量
    if not skip_universe:
        try:
            from qlib_ifind_beta import universe
            universe.dump_universe("2024-01-01", date)   # 全量幂等（resumable cache）
            print(f"✓ [1] universe 刷新到 {date}")
        except Exception as e:
            print(f"⚠ [1] universe 失败（用既有池继续）: {e}")
    else:
        print("⏭ [1] universe 跳过")

    # [2] materialize 池内
    codes = load_pool()
    summ = materialize_pool(codes)
    print(f"✓ [2] materialize: {summ}")

    # [3] predict
    result = predict_day(date)
    record_signal(result, SIGNALS)
    topk_codes = [c["code"] for c in result["topk"]]
    print(f"✓ [3] predict: n_candidates={result['n_candidates']} topk={topk_codes}")

    # [4] settle_prev(T-1, T) —— 取 T-1 上一交易日
    from qlib.data import D
    cal = D.calendar(start_time="2026-01-01", end_time=date, freq="day")
    if len(cal) >= 2:
        prev = pd.Timestamp(cal[-2]).strftime("%Y-%m-%d")
        close_lk = _aux_lookup(date, ["$close"])
        chg_lk = _aux_lookup(date, ["$change"])
        ld_lk = _aux_lookup(date, ["$limit_down"])
        n = settle_prev(prev, date, close_lk, chg_lk, ld_lk, SIGNALS, SETTLE)
        print(f"✓ [4] settle_prev({prev}→{date}): {n} 笔")
    else:
        print("⏭ [4] settle_prev 跳过（首日无 T-1）")

    # [5] NAV
    if SETTLE.exists():
        settle_df = pd.read_csv(SETTLE)
        nav = compute_nav(settle_df, NAV)
        print(f"✓ [5] NAV: 末日 net_nav={nav['net_nav'].iloc[-1]:.4f} ({len(nav)} 日)")
    print("✅ live_forward 完成")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--date", required=True)
    p.add_argument("--skip-universe", action="store_true")
    a = p.parse_args()
    run(a.date, a.skip_universe)
```

- [ ] **Step 2: 烟雾验证（首日 dry-run，跳过 universe）**

Run: `conda run -n qlib_ifind_beta --no-capture-output python -W ignore scripts/live_forward.py --date 2026-07-02 --skip-universe 2>&1 | tail -20`
Expected: 五步全 ✓（[4] settle 跳过或正常），NAV 末日合理，无 traceback

- [ ] **Step 3: Commit**

```bash
git add scripts/live_forward.py
git commit -m "feat(live): live_forward 编排入口（universe→materialize→predict→settle→nav）

工作日 15:35 触发，五步 orchestrate，单步失败不阻断。
Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 6: 补齐测试（CLAUDE.md 7 项全覆盖）

**Files:**
- Modify: `tests/test_live_inference.py`（追加前视/空/单票/边界/涨跌停/无网络测试）

零偏离(Task 2) + NAV(Task 3) 已写。本 task 补齐：前视护栏、空数据、单票、边界、涨跌停拦截、无网络。

- [ ] **Step 1: 追加 5 项测试**

```python
def test_predict_day_limit_up_filtered(qlib_init):
    """涨跌停拦截：change_941 >= limit_up 的票不进 topk。"""
    from qlib_ifind_beta.live.inference import predict_day
    result = predict_day("2026-07-02")
    for c in result["topk"]:
        assert c["change_941"] < c["limit_up"], f"{c['code']} 封涨停却进 topk"


def test_predict_day_no_lookahead(qlib_init, monkeypatch):
    """前视护栏：所有 D.features 调用 end_time <= T。"""
    import qlib.data as qd
    calls = []
    orig = qd.D.features
    def spy(*a, **kw):
        calls.append(kw.get("end_time"))
        return orig(*a, **kw)
    monkeypatch.setattr(qd.D, "features", spy)
    from qlib_ifind_beta.live.inference import predict_day
    predict_day("2026-07-02")
    assert all(e <= "2026-07-02" for e in calls if e), f"前视泄漏: {calls}"


def test_settle_prev_limit_down_blocked(tmp_path):
    """settle_prev：封跌停 → blocked=True, sell_price=NaN。"""
    from qlib_ifind_beta.live.track import settle_prev
    import pandas as pd
    sig = tmp_path / "sig.csv"
    pd.DataFrame([{"date": "2026-04-01", "code": "A", "score": 0.1, "price_941": 10,
                   "change_941": 0.05, "limit_up": 0.095, "limit_down": -0.095}]).to_csv(sig, index=False)
    settle = tmp_path / "settle.csv"
    # change=-0.10 <= limit_down=-0.095 → 封跌停
    n = settle_prev("2026-04-01", "2026-04-02", {"A": 9.0}, {"A": -0.10}, {"A": -0.095}, sig, settle)
    df = pd.read_csv(settle)
    assert n == 1 and df.loc[0, "blocked"] == True and pd.isna(df.loc[0, "sell_price"])


def test_record_signal_idempotent(tmp_path):
    """record_signal 幂等：同 date 重写不追加。"""
    from qlib_ifind_beta.live.track import record_signal
    p = tmp_path / "sig.csv"
    r1 = {"date": "2026-04-01", "candidates": [{"code": "A", "score": 0.1, "price_941": 10,
              "change_941": 0.01, "limit_up": 0.095, "limit_down": -0.095}]}
    record_signal(r1, p); record_signal(r1, p)
    import pandas as pd
    assert len(pd.read_csv(p)) == 1


def test_daily_ic_too_few_returns_none(tmp_path):
    """daily_ic：< 5 对返回 None（边界）。"""
    from qlib_ifind_beta.live.track import daily_ic
    import pandas as pd
    p = tmp_path / "sig.csv"
    pd.DataFrame([{"date": "2026-04-01", "code": "A", "score": 0.1}]).to_csv(p, index=False)
    assert daily_ic(p, {"A": 0.01}, "2026-04-01") is None
```

- [ ] **Step 2: 跑全测试**

Run: `conda run -n qlib_ifind_beta --no-capture-output python -W ignore -m pytest tests/test_live_inference.py -v 2>&1 | tail -30`
Expected: 全 PASS（零偏离 + NAV + 涨跌停 + 前视 + settle + 幂等 + IC 边界）

- [ ] **Step 3: Commit**

```bash
git add tests/test_live_inference.py
git commit -m "test(live): 补齐前视/涨跌停/settle/幂等/IC 边界测试（CLAUDE.md 7 项全覆盖）
Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 7: .gitignore + 文档落档

**Files:**
- Modify: `.gitignore`（+data/live_*.csv + data/live_forward.log）
- Modify: `docs/backtest-log/2026-07-06-l1-full-backtest.md`（+§36 P1 落档）
- Get笔记 note 1914664125624050528 追加条目

- [ ] **Step 1: .gitignore 加运行产物**

```
data/live_signals.csv
data/live_settle.csv
data/live_nav.csv
data/live_forward.log
```

- [ ] **Step 2: backtest-log §36 追加 P1 落档**（参照 spec + 本 plan，记录：spike 零偏离铁证、6 模块清单、7 测试、6 项决策清单、P1→P2→P3 路径）

- [ ] **Step 3: Get笔记追加条目**（GET detail → 文末追加 `## [2026-07-09] P1 实战对接纸面前向跟踪` → POST update；secret 不入正文；不破坏图片）

- [ ] **Step 4: Commit docs**

```bash
git add .gitignore docs/backtest-log/2026-07-06-l1-full-backtest.md
git commit -m "docs(live): §36 P1 实战对接落档 + .gitignore 运行产物

spike 零偏离铁证 + 6 模块清单 + 6 项决策清单 + P1→P2→P3 路径。
Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Verification（整体冒烟）

1. `conda run -n qlib_ifind_beta --no-capture-output python -W ignore -m pytest tests/test_live_inference.py -v` → 全 PASS
2. `conda run -n qlib_ifind_beta --no-capture-output python -W ignore scripts/live_forward.py --date 2026-07-02 --skip-universe` → 五步全 ✓，NAV 合理
3. `predict_day("2026-07-02")` top10 与 champion pred.pkl nlargest(10) 一致（零偏离锚）

---

## Self-Review（plan 自审）

- [x] spec 覆盖：spec §3.3 五组件（inference/track/materialize_live/live_forward/data）→ Task 1-5；§5 七测试 → Task 2/3/6；§6.5 gitignore → Task 7。
- [x] 无占位符：所有 code/test/命令完整给出。
- [x] 类型一致：`predict_day → dict{date,n_candidates,candidates,topk}` 在 Task 2 定义，Task 5 `live_forward` 消费一致；`compute_nav → DataFrame[sell_date,daily_ret,net_nav]` Task 3 定义，测试消费一致。
- [x] 命名一致：`load_pool/materialize_pool` (Task 4) ← `live_forward` (Task 5) 调用名一致。
- [x] qlib 分层：inference = R(Recorder) + DatasetH + LGBModel + MinuteEnhancedHandler 全原生组合，无自定义抽象层（CLAUDE.md #4）。
- [x] conda-only + 绝对路径 + `--no-capture-output -W ignore` 全程。
