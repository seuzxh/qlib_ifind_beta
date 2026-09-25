"""PortAna 死锁诊断 — faulthandler 定时转储全部线程栈（无需 ptrace 权限）。

重现口径：BASE(18 因子) 默认切分，task_train 带 PortAnaRecord；
600s 未结束则 dump 全部线程栈到 stderr 并退出。
Run: conda run -n qlib_ifind_beta python scripts/diag_portana_deadlock.py
"""
from __future__ import annotations

import faulthandler
import os
import sys
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_DUMP = open("/tmp/portana_dump.txt", "w")
faulthandler.enable(file=_DUMP, all_threads=True)
faulthandler.dump_traceback_later(600, repeat=True, exit=False)

import qlib
from ruamel.yaml import YAML

from qlib_ifind_beta.config import OVERLAY_ROOT

qlib.init(provider_uri=str(OVERLAY_ROOT), region="cn")
# 死锁根因：PortAna 阶段 Exchange.get_quote_from_qlib 复用 loky 进程池时
# 重入死锁（task_train 前段 SignalRecord 已用过同池）。threading 后端无进程
# 池，彻底绕开（day 频数据加载为 IO/数值型，GIL 影响可接受）。
from qlib.config import C
C["joblib_backend"] = "threading"

from qlib.model.trainer import task_train

cfg = YAML(typ="safe").load(open(ROOT / "qrun" / "workflow_minute_enhanced_tk10_nd8.yaml"))
task = cfg["task"]
# PortAna 成交环境需要日线源 bin（change/limit/deal close）；99 只"仅分钟源"
# 股票会令涨跌停比较崩。用过滤版 universe（对 BASE/C1/C5 一视同仁）。
task["dataset"]["kwargs"]["handler"]["kwargs"]["instruments"] = "highbeta883926_dayok"
for rec in task.get("record", []):
    if rec.get("class") == "PortAnaRecord":
        exk = ((rec.get("kwargs", {}).get("config") or {}).get("backtest") or {}).get("exchange_kwargs") or {}
        if isinstance(exk.get("limit_threshold"), list):
            exk["limit_threshold"] = tuple(exk["limit_threshold"])
        # 默认 codes="all" 会载入全市场（含 99 只仅分钟源股票，日线 bin 缺失
        # 令涨跌停比较崩），显式限定为过滤 universe。
        exk["codes"] = "highbeta883926_dayok"
print("▶ task_train(PortAna) 开始，600s 保护计时", flush=True)
rec = task_train(task, experiment_name="factor_zoo_diag_portana")
print("▶ 完成", rec.id, flush=True)
