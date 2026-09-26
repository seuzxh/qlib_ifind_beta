"""qrun 等价 runner — 解决两个本机坑：

1. **LT_TP_EXP 要 tuple**：`Exchange._get_limit_type`（exchange.py:264）用
   `isinstance(limit_threshold, tuple)` 判分支，YAML 的 `[a,b]` 加载成 `list`
   会落到 `NotImplementedError`。本 runner 加载 YAML 后把
   `task.record[PortAnaRecord].kwargs.config.backtest.exchange_kwargs.limit_threshold`
   从 list 转 tuple，从而启用原生表达式型板块分级涨跌停拦截。

2. **mlflow 3.12.0 file-store maintenance**：预置 `MLFLOW_ALLOW_FILE_STORE=true`，
   否则 `task_train` 落 mlflow 时抛 `MlflowException`。

其余逻辑等同 `qlib.cli.run.workflow`（qlib_init → task_train）。

用法：
    conda run -n qlib_ifind_beta python qrun/run.py \
        qrun/workflow_minute_enhanced_tk10_nd8.yaml
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# 必须在 import qlib / mlflow 前设置
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

# 让 handler 的 module_path: qlib_ifind_beta.* 可 import。
# `python qrun/run.py` 把 qrun/（脚本目录）放 sys.path[0]，项目根不在路径上。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ruamel.yaml import YAML

import qlib
from qlib.config import C
from qlib.model.trainer import task_train


def _coerce_limit_threshold(task: dict) -> str:
    """把 PortAnaRecord 的 limit_threshold(list) → tuple。返回诊断串。"""
    hits = []
    for rec in task.get("record", []):
        if rec.get("class") != "PortAnaRecord":
            continue
        kwargs = rec.get("kwargs", {})
        cfg = kwargs.get("config") or {}
        exk = (cfg.get("backtest") or {}).get("exchange_kwargs") or {}
        lt = exk.get("limit_threshold")
        if isinstance(lt, list):
            exk["limit_threshold"] = tuple(lt)
            hits.append(f"{rec['class']}: list({len(lt)}) → tuple")
    return "; ".join(hits) if hits else "(no PortAnaRecord limit_threshold found)"


def run(config_path: str, experiment_name: str = "workflow") -> None:
    config_path = Path(config_path).resolve()
    yaml = YAML(typ="safe", pure=True)
    with open(config_path) as fp:
        config = yaml.load(fp)

    # tuple 修正（LT_TP_EXP）
    diag = _coerce_limit_threshold(config.get("task", {}))
    print(f"[run.py] limit_threshold coerce: {diag}")

    qlib_init = config.get("qlib_init", {})
    if "exp_manager" not in qlib_init:
        exp_manager = C["exp_manager"]
        exp_manager["kwargs"]["uri"] = "file:" + str(Path.cwd().resolve() / "mlruns")
        qlib.init(**qlib_init, exp_manager=exp_manager)
    else:
        qlib.init(**qlib_init)

    if "experiment_name" in config:
        experiment_name = config["experiment_name"]
    print(f"[run.py] experiment={experiment_name}  provider_uri={qlib_init.get('provider_uri')}")
    recorder = task_train(config.get("task"), experiment_name=experiment_name)
    recorder.save_objects(config=config)
    rid = getattr(recorder, "recorder_id", None) or getattr(recorder, "id", None)
    print(f"[run.py] done. recorder_id={rid}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python qrun/run.py <workflow.yaml> [experiment_name]")
    run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "workflow")
