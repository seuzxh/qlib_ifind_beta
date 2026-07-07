"""EnhancedWithDailyIndex — 30 因子合体（Step B，2026-07-07）。

两步走 Step B：在 champion=enhanced(18)@n_drop=15（18 分钟因子）基础上，叠加 Step A 验证有效的
12 个日频情绪 + 上证指数共振因子，共 30 因子，验证跨尺度互补能否提升 champion。

设计依据（第一性原理，不联想）：
  - Step A 已证 12 日频/指数因子单独 test 段 IC=0.0366（> m14 纯分钟 baseline 0.0341），非噪声；
    ICIR 字面卡线（0.2393<0.3）是日频尺度错配固有特性，Rank ICIR 0.3663 达标。Gate 实质通过。
  - 18 分钟因子（T 日 9:41 微观结构）+ 12 日频/指数（T-1 情绪/共振）= 跨时间尺度互补。
  - 下行风险：§19.1 教训——full(85=14min+71daily) IC 最低，日频 Alpha158 稀释 9:41 信号。但本任务
    12 日频非 Alpha158 全集（无 60 日均线等慢尺度），是精选短周期（5/9/20 日）+ 指数共振，
    尺度更贴近 ~1.5 天 label，理论上稀释风险低于 full(85)。结果不确定 → 本 handler 即为验证。

设计：继承 HighBetaAlpha158（复用 L1 shared DropnaProcessor(feature) 前视护栏 + __init__），
仅 override get_feature_config 返回 30 = 18 分钟 + 12 日频/指数。
  - 18 分钟：MinuteEnhancedHandler.ENHANCED_FIELDS 的 $field（T 日当天不 lag，与 champion 同序）
  - 12 日频/指数：IndexDailyHandler.FACTOR_FIELDS 的 Ref(...,1)（T-1 lag）

⚠️ L1 护栏等价性（关键，与 Step A 单独 handler 的核心区别）：
  Step A 的 IndexDailyHandler 因无分钟因子，p941→分钟因子→drop 联动断链（见 index_daily_handler.py
  docstring）。本 handler 含完整 18 分钟因子 → 联动恢复，DropnaProcessor 对 30 个 feature 任一 NaN
  即 drop，护栏与 champion 等价 → 回测数字可与 champion(+158.86%/−5.44%) 直接对比。

label / deal_price / 涨跌停拦截 / n_drop=15 全冻结与 champion 一致（见 workflow_enhanced_daily_index.yaml），
唯一变量是 feature 18 → 30，与 champion 严格可比。
"""
from __future__ import annotations

from .highbeta_handler import HighBetaAlpha158
from .index_daily_handler import IndexDailyHandler
from .minute_enhanced_handler import MinuteEnhancedHandler


class EnhancedWithDailyIndex(HighBetaAlpha158):
    """30 因子合体：18 分钟（champion）+ 12 日频情绪/上证指数共振（Step A）。

    18 分钟因子 T 日当天不 lag（与 champion 契约一致）；12 日频/指数因子 Ref(...,1) lag（T-1）。
    L1 护栏因含完整分钟因子，与 champion 等价。详见模块 docstring。
    """

    def get_feature_config(self):
        # 18 分钟（$field，T 日当天）+ 12 日频/指数（Ref expr，T-1 lag）
        fields = (
            [f"${n}" for n in MinuteEnhancedHandler.ENHANCED_FIELDS]      # 18 分钟，与 champion 同序
            + [expr for _, expr in IndexDailyHandler.FACTOR_FIELDS]       # 12 日频/指数
        )
        names = (
            list(MinuteEnhancedHandler.ENHANCED_FIELDS)
            + [name for name, _ in IndexDailyHandler.FACTOR_FIELDS]
        )
        return fields, names
