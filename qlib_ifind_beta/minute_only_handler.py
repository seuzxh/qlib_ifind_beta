"""MinuteOnlyHandler — 实验性：仅用 14 个 T 日分钟因子（9:30-9:40），不用 Alpha158 日频。

2026-07-06 一次性实验（用户：尝试仅使用 1min 相关因子做训练回测）：验证「纯短周期分钟因子
单独是否够预测 9:41 label」，与 v2(172)/v3(85) 对比。

设计：继承 HighBetaAlpha158（复用 L1 shared DropnaProcessor(feature) 前视护栏 + __init__），
仅 override get_feature_config 丢弃 71 日频因子，只返回 14 个 $minute_field。T 日当天因子、
不 lag（与 HighBetaAlpha158 的分钟部分契约一致）。label / deal_price / 涨跌停拦截与
workflow.yaml 全一致（见 workflow_minute_only.yaml）——唯一变量是 feature 从 85 → 14。
"""
from __future__ import annotations

from .config import MINUTE_FACTOR_FIELDS
from .highbeta_handler import HighBetaAlpha158


class MinuteOnlyHandler(HighBetaAlpha158):
    def get_feature_config(self):
        fields = [f"${n}" for n in MINUTE_FACTOR_FIELDS]   # 14 分钟因子，T 日当天不 lag
        names = list(MINUTE_FACTOR_FIELDS)
        return fields, names
