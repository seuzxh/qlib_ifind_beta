# 全自动影子模拟盘实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把回放状态机转为无人值守模拟盘——先回放补齐 7/21→9/11，再由 cron 每交易日 16:30 自动续跑并累积净值。

**Architecture:** 提取 `replay_intraday_shadow.run()` 的日循环体为模块级 `run_paper_day()`；新 `scripts/paper_shadow.py` 做账本结转 + 数据门禁 + nav 追加；账本 = `data/paper_shadow/` 下最新日期目录，无 state.json。前向日纯 bin 源（15:30 同步后），零实时代码。

**Tech Stack:** Python 3.12 / pandas / pyqlib 0.9.7（既有栈，零新依赖）

**Spec:** `docs/superpowers/specs/2026-09-13-paper-shadow-trading-design.md`

## Global Constraints

- 只使用 conda 环境 `qlib_ifind_beta`（`conda run -n qlib_ifind_beta ...`），禁止系统 Python
- 测试离线可运行，网络调用一律 mock（本计划新增测试全部纯本地）
- 禁止未来数据；`broker_orders_submitted` 恒 false；不自动晋升候选模型
- 评分恒用冻结 Champion `93d435e0`（`load_model_bundle(date, use_online=False)`）
- `data/`、`logs/`、`mlruns/` 是运行资产，不提交 Git
- 只读源：`/home/zxh/.qlib/qlib_data/cn_data{,_1min}`（config 已有 `DAY_CAL`/`MIN_CAL`/`FEATURES_1MIN_SRC`）

---

### Task 1: 提取 `run_paper_day`（纯重构）

**Files:**
- Modify: `scripts/replay_intraday_shadow.py`

**Interfaces:**
- Produces（后续 Task 依赖的精确签名）:
  ```python
  def run_paper_day(source, date: str, positions: pd.DataFrame, cash: float,
                    output_root: Path, bundle, model_path: Path, qlib_version: str,
                    snapshots: pd.DataFrame, stored: "pd.Series | None" = None,
                    require_stored_parity: bool = True
                    ) -> tuple[pd.DataFrame, float, dict]
  # 返回 (positions_after_close, cash_after_buy, day_result_row)
  ```

- [ ] **Step 1: 重构 `replay_intraday_shadow.py`**

在 `_metrics` 函数之后、`run` 之前新增模块级函数：

```python
def run_paper_day(source, date: str, positions: pd.DataFrame, cash: float,
                  output_root: Path, bundle, model_path: Path, qlib_version: str,
                  snapshots: pd.DataFrame, stored: "pd.Series | None" = None,
                  require_stored_parity: bool = True
                  ) -> tuple[pd.DataFrame, float, dict]:
    """跑一个完整影子日：结转→bars→特征→评分→计划→撮合→估值。

    返回 (positions_after_close, cash_after_buy, day_result_row)。
    """
```

`run_paper_day` 函数体 = 现 `run()` 循环体（`for number, date in enumerate(dates, 1):` 之内、`daily_results.append(row)` 之前的全部代码）逐行搬入（开头三行持仓结转原样保留），做且仅做以下替换：

1. `paths = DayPaths.create(date, output_root)`（原样）
2. universe 快照：函数体内改为
   `day_snapshot = snapshots.loc[snapshots["date"] == date].copy()`
   （`snapshots` 由参数传入，含 `date/code/eligible` 列；`run()` 传全量段帧）
3. parity 块：`stored_dates` 改为
   `stored_dates = set(pd.to_datetime(stored.index.get_level_values("datetime"))) if stored is not None else set()`；
   `reference_available = pd.Timestamp(date) in stored_dates`
4. `make_active_manifest(..., qlib_version=qlib_version)`（原用闭包 `qlib.__version__`）
5. 函数返回 `return positions, cash, row`（`row` 即写进 `shadow_day_result.json` 的 dict）
6. 循环内的 `print(...)` 移出——由调用方打印

`run()` 改为：初始化段（snapshots/source/recorder/stored/bundle/model_matches/positions/cash）不动，循环体替换为：

```python
    for number, date in enumerate(dates, 1):
        positions, cash, row = run_paper_day(
            source, date, positions, cash, output_root, bundle, model_matches[0],
            qlib.__version__, snapshots, stored=stored,
            require_stored_parity=require_stored_parity)
        daily_results.append(row)
        parity_text = (
            f"{row['max_abs_score_diff']:.1e}/common_top10={row['common_universe_top10_overlap']}"
            if row["score_reference_available"] else "forward-unreferenced"
        )
        print(f"[{number:02d}/{len(dates)}] {date} PASS features={row['feature_count']} "
              f"parity={parity_text} drift={row['universe_drift_count']} "
              f"nav={row['nav']:,.2f}", flush=True)
```

- [ ] **Step 2: 重构回归——2 日真实回放（本 Task 的守卫）**

Run:
```bash
conda run -n qlib_ifind_beta --no-capture-output python -W ignore \
  scripts/replay_intraday_shadow.py --start 2026-04-01 --end 2026-04-02 \
  --output /tmp/replay_refactor_check
```
Expected: `trading_days: 2`、`status: PASS`、`all_reconciled: true`、
`max_referenced_score_difference == 0.0`（与重构前行为一致）

- [ ] **Step 3: 全量测试 + 提交**

```bash
conda run -n qlib_ifind_beta python -m pytest -q
git add scripts/replay_intraday_shadow.py
git commit -m "refactor(replay): 提取 run_paper_day — 日循环体模块级复用"
```

---

### Task 2: `paper_shadow.py` 纯助手（账本读取 + nav 追加）

**Files:**
- Create: `scripts/paper_shadow.py`
- Test: `tests/test_paper_shadow.py`（追加）

**Interfaces:**
- Consumes: 无（纯 pandas/pathlib）
- Produces:
  ```python
  ROOT: Path            # data/paper_shadow
  NAV: Path             # data/paper_shadow/nav.csv
  def load_latest_state(root: Path, before: str) -> tuple[str, pd.DataFrame, float]
  def append_nav(nav_path: Path, row: dict) -> None
  ```

- [ ] **Step 1: 写失败测试（新建 tests/test_paper_shadow.py）**

```python
import json

import pandas as pd


def _mk_day(root, date, cash, code="SH600000", qty=100):
    d = root / date
    d.mkdir(parents=True)
    pd.DataFrame([{"code": code, "quantity": qty,
                   "sellable_quantity": 0, "hold_days": 0}]
                 ).to_csv(d / "positions_after_close.csv", index=False)
    (d / "cash_after_buy.json").write_text(json.dumps({"cash_after_buy": cash}))


def test_load_latest_state_picks_latest_before(tmp_path):
    from scripts.paper_shadow import load_latest_state

    _mk_day(tmp_path, "2026-09-10", 100.0)
    _mk_day(tmp_path, "2026-09-11", 200.0)
    day, positions, cash = load_latest_state(tmp_path, "2026-09-14")
    assert day == "2026-09-11"
    assert cash == 200.0
    assert positions.iloc[0]["code"] == "SH600000"


def test_append_nav_appends_and_is_idempotent_per_date(tmp_path):
    from scripts.paper_shadow import append_nav

    nav = tmp_path / "nav.csv"
    append_nav(nav, {"date": "2026-09-10", "cash": 1.0, "market_value": 2.0, "nav": 3.0})
    append_nav(nav, {"date": "2026-09-11", "cash": 1.0, "market_value": 2.0, "nav": 4.0})
    append_nav(nav, {"date": "2026-09-11", "cash": 1.0, "market_value": 2.0, "nav": 5.0})
    frame = pd.read_csv(nav)
    assert frame["date"].tolist() == ["2026-09-10", "2026-09-11"]
    assert frame["nav"].tolist() == [3.0, 5.0]
```

- [ ] **Step 2: 运行确认失败**

Run: `conda run -n qlib_ifind_beta python -m pytest tests/test_paper_shadow.py -q`
Expected: FAIL（`ModuleNotFoundError: scripts.paper_shadow` 或导入错误）

- [ ] **Step 3: 创建 `scripts/paper_shadow.py`（本 Task 只写助手 + 常量）**

```python
"""Unattended bin-sourced paper shadow account (spec 2026-09-13).

回放补齐（replay_intraday_shadow.py --output data/paper_shadow）之后，
每个交易日 16:30 由 cron 调 `day` 续跑一日并追加净值；账本 = 最新日期目录。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from qlib_ifind_beta.config import DAY_CAL, MIN_CAL, OVERLAY_ROOT, PROJECT_ROOT

ROOT = PROJECT_ROOT / "data" / "paper_shadow"
NAV = ROOT / "nav.csv"


def load_latest_state(root: Path, before: str) -> tuple[str, pd.DataFrame, float]:
    """最新 <before 日期目录 即账本；返回 (date, positions, cash)。"""
    days = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name < before)
    if not days:
        raise RuntimeError(f"no ledger day before {before} under {root}")
    last = root / days[-1]
    positions = pd.read_csv(last / "positions_after_close.csv")
    cash = float(json.loads((last / "cash_after_buy.json").read_text())["cash_after_buy"])
    return days[-1], positions, cash


def append_nav(nav_path: Path, row: dict) -> None:
    """追加一行 (date, cash, market_value, nav)；同日覆盖（幂等）。"""
    frame = pd.DataFrame([{k: row[k] for k in ("date", "cash", "market_value", "nav")}])
    if nav_path.exists():
        old = pd.read_csv(nav_path)
        frame = pd.concat([old.loc[old["date"] != row["date"]], frame], ignore_index=True)
    frame.to_csv(nav_path, index=False)


def main() -> None:  # Task 3 填充子命令
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 测试通过 + 提交**

```bash
conda run -n qlib_ifind_beta python -m pytest tests/test_paper_shadow.py -q
git add scripts/paper_shadow.py tests/test_paper_shadow.py
git commit -m "feat(paper): 账本读取 load_latest_state + nav 追加（幂等）"
```

---

### Task 3: `day` / `report` 子命令（门禁 + 编排）

**Files:**
- Modify: `scripts/paper_shadow.py`
- Modify: `tests/test_paper_shadow.py`（追加门禁单测）

**Interfaces:**
- Consumes: Task 1 的 `run_paper_day`、Task 2 的 `load_latest_state`/`append_nav`/`ROOT`/`NAV`；
  既有 `_eligible_snapshots`/`_metrics`（`scripts.replay_intraday_shadow`）、
  `HistoricalReplaySource`、`load_model_bundle`
- Produces: CLI `paper_shadow.py day --date T` / `report`

- [ ] **Step 1: 写 `_bars_ready` 的失败测试（追加）**

```python
def test_bars_ready_requires_calendar_and_bar(tmp_path, monkeypatch):
    from scripts import paper_shadow

    day_cal = tmp_path / "day.txt"
    day_cal.write_text("2026-09-11\n")
    min_cal = tmp_path / "1min.txt"
    min_cal.write_text("2026-09-11 09:41:00\n")
    monkeypatch.setattr(paper_shadow, "DAY_CAL", str(day_cal))
    monkeypatch.setattr(paper_shadow, "MIN_CAL", str(min_cal))

    def fake_read_bin(path):
        return 0, pd.array([100.0], dtype="float32")  # si=0, 有限值

    monkeypatch.setattr(paper_shadow, "_read_bin", fake_read_bin)
    assert paper_shadow._bars_ready("2026-09-11", probe=str(tmp_path / "v.bin"))
    assert not paper_shadow._bars_ready("2026-09-14", probe=str(tmp_path / "v.bin"))
```

- [ ] **Step 2: 运行确认失败**

Run: `conda run -n qlib_ifind_beta python -m pytest tests/test_paper_shadow.py -q`
Expected: FAIL（`_bars_ready`/`_read_bin` 不存在）

- [ ] **Step 3: 实现**

在 `scripts/paper_shadow.py` 中补充（放在 `append_nav` 之后）：

```python
from qlib_ifind_beta.data.binio import read_bin as _read_bin
from qlib_ifind_beta.config import CHAMPION_EXPERIMENT, CHAMPION_RECORDER_ID, FEATURES_1MIN_SRC


def _snapshot_has(date: str) -> bool:
    cache = PROJECT_ROOT / "data" / "universe_snapshots.csv"
    return cache.exists() and f"{date}," in cache.read_text()


def _bars_ready(date: str, probe: str | None = None) -> bool:
    """T 已进 day 日历，且样本股 1min bin 的 09:41 行有有限值（防 daily 先于 1min 完成）。"""
    cal = {line.strip() for line in Path(DAY_CAL).read_text().splitlines() if line.strip()}
    if date not in cal:
        return False
    lines = [line.strip() for line in Path(MIN_CAL).read_text().splitlines() if line.strip()]
    wanted = f"{date} 09:41:00"
    if wanted not in lines:
        return False
    path = Path(probe) if probe else (
        FEATURES_1MIN_SRC / "sh600004" / "volume.1min.bin")
    si, vol = _read_bin(path)
    idx = lines.index(wanted) - si
    return 0 <= idx < vol.size and pd.notna(vol[idx])


def day(date: str) -> dict:
    ROOT.mkdir(parents=True, exist_ok=True)
    # 交易日门禁：幂等补拉当日快照（update_universe 内部对节假日静默跳过）
    subprocess.run([sys.executable, "-m", "scripts.update_universe", "--end", date],
                   cwd=PROJECT_ROOT, check=False)
    if not _snapshot_has(date):
        print(f"{date}: no constituent snapshot (holiday?) — skip")
        return {"date": date, "status": "SKIPPED_HOLIDAY"}
    # 数据门禁：轮询等 15:30 同步落 bin（超时=同步事故，exit 非 0）
    deadline = time.time() + 60 * 60
    while not _bars_ready(date):
        if time.time() >= deadline:
            raise RuntimeError(f"sync gate timeout for {date}")
        time.sleep(60)
    import qlib
    qlib.init(provider_uri=str(OVERLAY_ROOT), region="cn")
    from qlib_ifind_beta.live.historical_replay import HistoricalReplaySource
    from qlib_ifind_beta.experiment.model_ensemble import load_model_bundle
    from scripts.replay_intraday_shadow import _eligible_snapshots, run_paper_day

    _, positions, cash = load_latest_state(ROOT, date)
    snapshots, _ = _eligible_snapshots(date, date)
    bundle = load_model_bundle(date, use_online=False)
    matches = list((PROJECT_ROOT / "mlruns").glob(
        f"*/{CHAMPION_RECORDER_ID}/artifacts/params.pkl"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one Champion model, found {len(matches)}")
    # stored=None：forward 日期恒晚于 pred.pkl 末位（2026-07-02），UNREFERENCED_FORWARD 分支
    positions, cash, row = run_paper_day(
        HistoricalReplaySource(), date, positions, cash, ROOT, bundle, matches[0],
        qlib.__version__, snapshots, stored=None, require_stored_parity=False)
    append_nav(NAV, row)
    print(json.dumps(row, ensure_ascii=False, indent=2, default=str))
    return row


def report(root: Path = ROOT) -> dict:
    from scripts.replay_intraday_shadow import _metrics

    nav = pd.read_csv(root / "nav.csv")
    result = {"days": len(nav), **_metrics(nav.set_index("date")["nav"])}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cmd = sub.add_parser("day"); cmd.add_argument("--date", required=True)
    sub.add_parser("report")
    args = parser.parse_args()
    if args.command == "day":
        day(args.date)
    else:
        report()


if __name__ == "__main__":
    main()
```

注意：把文件顶部 import 中已存在的 `DAY_CAL, MIN_CAL, OVERLAY_ROOT, PROJECT_ROOT`
保持不变；`monkeypatch.setattr(paper_shadow, "DAY_CAL", ...)` 要求模块内以
`paper_shadow.DAY_CAL` 名字引用（函数内读模块全局即可，上码已满足）。
`_read_bin` 以模块属性导入（`from ... import read_bin as _read_bin`）供 monkeypatch。

- [ ] **Step 4: 测试通过**

Run: `conda run -n qlib_ifind_beta python -m pytest tests/test_paper_shadow.py -q`
Expected: PASS（3 个用例）

- [ ] **Step 5: CLI 冒烟**

```bash
conda run -n qlib_ifind_beta python scripts/paper_shadow.py --help
conda run -n qlib_ifind_beta python scripts/paper_shadow.py day --help
```
Expected: 两个 help 正常输出（不触发数据/网络）

- [ ] **Step 6: 全量测试 + 提交**

```bash
conda run -n qlib_ifind_beta python -m pytest -q
git add scripts/paper_shadow.py tests/test_paper_shadow.py
git commit -m "feat(paper): day/report 子命令 — 门禁(节假日跳过+同步轮询)+账本续跑+nav 追加"
```

---

### Task 4: cron 包装脚本 + 文档同步

**Files:**
- Create: `scripts/cron_paper_shadow.sh`
- Modify: `docs/paper-trading.md`
- Modify: `AGENTS.md`

**Interfaces:**
- Consumes: Task 3 的 `paper_shadow.py day`
- Produces: crontab 一行 + 文档现役表述

- [ ] **Step 1: 创建 `scripts/cron_paper_shadow.sh`**

```bash
#!/usr/bin/env bash
# Paper shadow account daily runner (16:30 Mon-Fri, after the 15:30 data sync).
set -euo pipefail
cd /home/zxh/projects/3.qlib_ifind_beta

exec 9>/tmp/qlib_paper_shadow.lock
if ! flock -n 9; then
    echo "$(date '+%F %T') another paper_shadow instance holds the lock — skip"
    exit 0
fi

PY=/home/zxh/miniconda3/envs/qlib_ifind_beta/bin/python
echo "$(date '+%F %T') === paper_shadow day start ==="
"$PY" scripts/paper_shadow.py day --date "$(date +%F)"
echo "$(date '+%F %T') === paper_shadow day done ==="
```

`chmod +x scripts/cron_paper_shadow.sh`

- [ ] **Step 2: 安装 crontab（一行）**

```bash
( crontab -l 2>/dev/null; echo '30 16 * * 1-5 /bin/bash /home/zxh/projects/3.qlib_ifind_beta/scripts/cron_paper_shadow.sh >> /home/zxh/projects/3.qlib_ifind_beta/logs/paper_shadow.log 2>&1' ) | crontab -
crontab -l | tail -3   # 确认写入
```

- [ ] **Step 3: 更新 `docs/paper-trading.md`**

把「盘后五步（纸面跟踪…）」至「数据文件与现状」两节替换为：

```markdown
## 影子模拟盘（2026-09-13 起，bin 同源自动撮合）

旧 `live_forward.py` 纸面线已删（2026-09-12）。新模拟盘复用回放状态机：

```bash
# 回放补齐（一次性，已完成见下）
conda run -n qlib_ifind_beta --no-capture-output python -W ignore \
  scripts/replay_intraday_shadow.py --start 2026-07-21 --end 2026-09-11 \
  --allow-unreferenced-scores --output data/paper_shadow

# 每交易日自动：cron 16:30 → scripts/cron_paper_shadow.sh
# 手动补跑 / 查看净值：
conda run -n qlib_ifind_beta python scripts/paper_shadow.py day --date 2026-09-14
conda run -n qlib_ifind_beta python scripts/paper_shadow.py report
```

- 撮合：09:41 价模拟成交、涨停/无 bar 拦截、先卖后买两阶段现金、费率 0.05%/0.15%（min 5 元）
- 账本：`data/paper_shadow/<date>/`（可审计全套产物）+ `nav.csv`（累计净值）
- 门禁：节假日跳过（无成分快照）；同步超时 60 分钟 exit 1；失败日不推进账本，可 `day --date` 补跑
- 评分：冻结 Champion `93d435e0`；不验证盘中实时路径（六步链职责）

| 文件 | 现状 |
| --- | --- |
| `data/paper_shadow/` | 2026-07-21 → 回放补齐起点 |
| `data/paper_shadow/nav.csv` | 回放补齐后 seed，cron 每日追加 |
```

同步更新页首"一图速览"表中盘后两行、以及「数据文件与现状」旧表
（旧 `data/live_*.csv` 行改为"已归档，7/03→7/14 旧纸面线，净值 0.812"一行）。

- [ ] **Step 4: 更新 `AGENTS.md` 当前状态**

在「当前状态」段追加一行：

```markdown
2026-09-13 上线全自动影子模拟盘：回放引擎日增量化（`paper_shadow.py` + 16:30 cron），
先补齐 7/21→9/11 再前向；评分恒用冻结 Champion。
```

- [ ] **Step 5: 全量测试 + 提交**

```bash
conda run -n qlib_ifind_beta python -m pytest -q
git add scripts/cron_paper_shadow.sh docs/paper-trading.md AGENTS.md
git commit -m "feat(paper): cron 包装 + 文档同步 — 影子模拟盘上线"
```

---

### Task 5: 执行回放补齐 + seed nav + 验证

**Files:**
- 无代码改动；运行资产落 `data/paper_shadow/`
- Modify: `AGENTS.md`（回填结果一行）

- [ ] **Step 1: 数据预检**

```bash
conda run -n qlib_ifind_beta python - << 'EOF'
import pandas as pd
snap = pd.read_csv("data/universe_snapshots.csv", usecols=["date"])
dates = sorted(set(snap["date"].astype(str)))
window = [d for d in dates if "2026-07-21" <= d <= "2026-09-11"]
print(f"universe 快照 7/21→9/11: {len(window)} 天")
print(f"首: {window[0]}  末: {window[-1]}")
EOF
tail -1 /home/zxh/.qlib/qlib_data/cn_data/calendars/day.txt
tail -1 /home/zxh/.qlib/qlib_data/cn_data_1min/calendars/1min.txt
```
Expected: 快照约 38 个交易日、两日历末 ≥ 2026-09-11。缺则先跑 15:30 同步链再回来。

- [ ] **Step 2: 回放补齐（预计 30-60 分钟）**

```bash
conda run -n qlib_ifind_beta --no-capture-output python -W ignore \
  scripts/replay_intraday_shadow.py \
  --start 2026-07-21 --end 2026-09-11 \
  --allow-unreferenced-scores \
  --output data/paper_shadow 2>&1 | tail -30
```
Expected: 逐日 `[NN/38] ... PASS ... forward-unreferenced ... nav=...`，
`replay_report.json` 的 `status: PASS`、`all_reconciled: true`。
失败日（若有）：查该日 1min 源，补数后单独重跑区间或该日。

- [ ] **Step 3: seed nav.csv**

```bash
conda run -n qlib_ifind_beta python - << 'EOF'
import pandas as pd
from pathlib import Path
daily = pd.read_csv("data/paper_shadow/daily_summary.csv")
nav = daily[["date", "cash", "market_value", "nav"]]
nav.to_csv("data/paper_shadow/nav.csv", index=False)
print(nav.tail(3).to_string(index=False))
EOF
```
Expected: 末行 date=2026-09-11、nav 为期末净值。

- [ ] **Step 4: report 验证**

```bash
conda run -n qlib_ifind_beta python scripts/paper_shadow.py report
```
Expected: `days ≈ 38` + total_return/max_drawdown 等指标。

- [ ] **Step 5: 账本链验证（day 能找到上日）**

```bash
conda run -n qlib_ifind_beta python - << 'EOF'
from scripts.paper_shadow import ROOT, load_latest_state
print(load_latest_state(ROOT, "2026-09-14")[0])  # 期望 2026-09-11
EOF
```

- [ ] **Step 6: AGENTS.md 回填结果 + 提交**

在「当前状态」段把 Task 4 那行改为含实际数字（例如：
"回放补齐 38 日完成（nav=…，回撤 …%），9/14 起 cron 前向"）：

```bash
git add AGENTS.md
git commit -m "docs: 模拟盘回放补齐结果落档"
```

- [ ] **Step 7: 下一个交易日（周一 9/14）收盘后检查**

```bash
tail -5 logs/paper_shadow.log
conda run -n qlib_ifind_beta python scripts/paper_shadow.py report
```
Expected: nav.csv 多出 2026-09-14 一行；若缺，手动 `day --date 2026-09-14` 复现报错。

---

## Self-Review 记录

- Spec 覆盖：§3 回放命令=Task 5、§4 三项改动=Task 1/2+3/4、§5 门禁与数据流=Task 3、
  §6 fail-closed=Task 3（SKIPPED_HOLIDAY/timeout/raise）+Task 5 补跑、§7 测试=Task 2/3 + Task 1/5 实跑回归
- 占位符：无 TBD/示意代码
- 类型一致：`run_paper_day` 返回三元组在 Task 1 定义、Task 3 消费；`load_latest_state`
  返回 `(str, DataFrame, float)` 与 Task 2 测试、Task 3 用法一致；`_bars_ready(date, probe)`
  的 `probe` 参数仅测试注入用
- ponytail-review 修订（2026-09-13，-22 行）：删 age_positions 独立函数+测试（守卫改为
  Task 1 的 2 日回放回归）；day() 删 stored 加载（forward 恒晚于 pred.pkl 末位）与
  root/timeout 参数；_DATE_DIR glob→iterdir；_snapshot_has→子串判定
