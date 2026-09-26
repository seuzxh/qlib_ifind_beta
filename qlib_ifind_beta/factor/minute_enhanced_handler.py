"""MinuteEnhancedHandler — 当前生产使用的 18 因子增强集。

特征由 14 个 09:31–09:40 分钟因子和 4 个 extra 因子组成。Handler 继承
HighBetaAlpha158，以复用 shared DropnaProcessor(feature) 前视护栏和 Qlib
初始化流程；仅覆盖特征配置。T 日分钟特征不做 lag，label、成交价与涨跌停
约束由现役 Champion 工作流固定。
"""
from __future__ import annotations

from .config import MINUTE_FACTOR_EXTRA_FIELDS, MINUTE_FACTOR_FIELDS
from .highbeta_handler import HighBetaAlpha158


class MinuteEnhancedHandler(HighBetaAlpha158):
    """18 因子：14 个开盘分钟因子 + 4 个额外因子。"""

    ENHANCED_FIELDS = (
        tuple(MINUTE_FACTOR_FIELDS) + tuple(MINUTE_FACTOR_EXTRA_FIELDS)
    )

    def get_feature_config(self):
        fields = [f"${name}" for name in self.ENHANCED_FIELDS]
        return fields, list(self.ENHANCED_FIELDS)
