"""ShadowAmtHandler — champion(18) + 2 amount 因子（item 7 A/B 测试）。

20 = 18 champion（14 baseline + 4 extra） + 2 amt（amt_ratio_5m + amt_vs_yest）。
amt = volume × vwap，价格加权的量能信号。
仅 override get_feature_config，label/model/exchange 全 FROZEN vs champion。
"""
from __future__ import annotations

from .config import MINUTE_FACTOR_AMT_FIELDS
from .minute_enhanced_handler import MinuteEnhancedHandler


class ShadowAmtHandler(MinuteEnhancedHandler):
    """20 因子：champion 18 + 2 amount（amt_ratio_5m + amt_vs_yest）。"""

    SHADOW_FIELDS = MinuteEnhancedHandler.ENHANCED_FIELDS + tuple(MINUTE_FACTOR_AMT_FIELDS)

    def get_feature_config(self):
        fields = [f"${n}" for n in self.SHADOW_FIELDS]
        names = list(self.SHADOW_FIELDS)
        return fields, names
