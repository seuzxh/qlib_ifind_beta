"""冻结协议实验运行器（run_frozen）— 把 challenge1-4 四代脚本沉淀的
全部环境坑修复藏在一次调用后面。下一条研究线从这里出发，不再复制脚本。

吸收的坑（均有会话实证，详见 backtest-log/2026-09-24-factor-zoo-screen.md §5/§7.4）：
  ① loky 池重入死锁 → threading 后端；
  ② 仅分钟源股票缺日线 bin → handler/exchange 双用 dayok universe；
  ③ exchange 默认 codes="all" 载全市场 → 显式 codes；
  ④ task_train 按 module_path 二次导入脚本模块（≠__main__）→ handler 用
     类工厂注入，不用模块全局；
  ⑤ 字段名必须全小写 + 单下划线（engine 按小写定位 bin、双下划线被解析吞）；
  ⑥ PortAnaRecord 回测段可选（portana=False 时以 SignalRecord+手工重放替代，
     重放口径只排序不终判——excluded.json 方法论规则 2）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from ruamel.yaml import YAML

from qlib_ifind_beta.config import OVERLAY_ROOT, UNIVERSE_MARKET
from qlib_ifind_beta.factor.minute_enhanced_handler import MinuteEnhancedHandler

ROOT = Path(__file__).resolve().parent.parent.parent
CHAMPION_YAML = ROOT / "examples" / "champion" / "workflow_minute_enhanced_tk10_nd8.yaml"
DAYOK_MARKET = "highbeta883926_dayok"
OPEN_COST, CLOSE_COST = 0.0005, 0.0015


def safe_name(name: str) -> str:
    """研究 bin 字段名：全小写 + 单下划线（坑⑤），幂等（重复调用不叠加 fz_ 前缀）。"""
    s = re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9_]", "_", name).lower())
    return "fz_" + re.sub(r"^fz_", "", s)


def make_handler(features: list[str], drop: list[str]) -> type:
    """类工厂生成变体 Handler（坑④：不依赖可导入模块的全局变量）。

    get_feature_config 不走 super() 绑定、直接读 ENHANCED_FIELDS——
    传 None 也能调（测试与 task_train 双场景可用）。
    """
    class FrozenVariantHandler(MinuteEnhancedHandler):
        def get_feature_config(self):
            keep = [n for n in MinuteEnhancedHandler.ENHANCED_FIELDS if n not in drop]
            names = keep + list(features)
            return [f"${n}" for n in names], names
    return FrozenVariantHandler


def replay_topn(pred: pd.Series, label: pd.Series, topn: int = 10) -> pd.Series:
    """手工重放：逐日 topn 等权、label v2 毛收益 - 双边成本。只排序不终判。"""
    daily = {}
    for d, g in pred.groupby(level="datetime"):
        sel = g.sort_values(ascending=False).head(topn)
        y = label.xs(d, level="datetime").reindex(sel.index.get_level_values("instrument"))
        daily[d] = float(y.mean()) - OPEN_COST - CLOSE_COST
    return pd.Series(daily).sort_index()


@dataclass
class FrozenResult:
    tag: str
    ic: float
    rank_ic: float
    pred: pd.Series
    replay: pd.Series

    def replay_stats(self) -> str:
        r = self.replay
        cum = (1 + r).prod() - 1
        dd = ((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min()
        ann = (1 + cum) ** (252 / max(len(r), 1)) - 1
        return f"累计 {cum:+.1%} 年化 {ann:+.1%} 回撤 {dd:.1%}"


def run_frozen(tag: str,
               handler_cls: type,
               module_path: str = __name__,
               segments: dict | None = None,
               portana: bool = False,
               backtest_window: tuple[str, str] | None = None) -> FrozenResult:
    """按冻结协议训练一个变体并返回 IC/pred/重放。

    handler_cls 需可从 module_path 导入（类工厂产物请挂在模块属性上）。
    segments=None 用 Champion 冻结切分；backtest_window 覆盖 PortAna 回测窗。
    """
    from qlib.model.trainer import task_train
    task = YAML(typ="safe").load(open(CHAMPION_YAML))["task"]
    h = task["dataset"]["kwargs"]["handler"]
    h["class"] = handler_cls.__name__
    h["module_path"] = module_path
    h["kwargs"]["instruments"] = DAYOK_MARKET          # 坑②
    if segments:
        task["dataset"]["kwargs"]["segments"] = segments
    for rec in task.get("record", []):
        if rec.get("class") == "PortAnaRecord":
            bk = (rec.get("kwargs", {}).get("config") or {}).get("backtest") or {}
            exk = bk.get("exchange_kwargs") or {}
            if isinstance(exk.get("limit_threshold"), list):
                exk["limit_threshold"] = tuple(exk["limit_threshold"])
            exk["codes"] = DAYOK_MARKET                 # 坑③
            if backtest_window:
                bk["start_time"], bk["end_time"] = backtest_window
            if not portana:                             # 坑⑥
                task["record"] = [r for r in task["record"]
                                  if r.get("class") != "PortAnaRecord"]
    recorder = task_train(task, experiment_name="factor_zoo_augment")
    m = recorder.list_metrics()
    pred = recorder.load_object("pred.pkl")
    pred = (pred.iloc[:, 0] if isinstance(pred, pd.DataFrame) else pred)
    pred = pred.reorder_levels(["datetime", "instrument"]).sort_index()
    return FrozenResult(tag=tag, ic=float(m.get("IC", np.nan)),
                        rank_ic=float(m.get("Rank IC", np.nan)), pred=pred,
                        replay=pd.Series(dtype=float))
