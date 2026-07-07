"""MinuteEnhancedHandler — 18 因子增强集（iteration-9，2026-07-07）。

背景：iteration-7 发现 n_drop=15 是策略甜点（m14 超额 +28%→+116%）。iteration-9 在该甜点
策略下验证因子维度是否仍有杠杆：14 分钟全集 + 4 extra = 18。

设计依据（第一性原理，不联想）：
  - §19.1 full(85=14min+71daily) IC 最低 → 日频 Alpha158 稀释 9:41 信号；但 4 extra 都是
    分钟族（vol_vs_yest_t2/t3/t5）或日频跳空（overnight_gap），与 ~1.5 天 label 同周期，
    不属于「5/10 日均线」那种尺度错配 → 加 4 extra 不会重蹈 full 覆辙。
  - 早期 8 因子消融实验（surgery，2026-07-07，已弃用移除）IC +0.0661 最高但 n_drop=5 实现
    alpha +14.34%；它砍掉了 startup_mom/accel/close_pos 族。n_drop=15 跟踪到位后，这些被砍的
    族可能重新有效 → 14 全集可能优于 8（历史对照论据，surgery 代码已删）。
  - enhanced(18) = 14 全集 + 4 extra：既保留 14 的全部动量/加速度/位置信号，又加 4 反转/跳空
    信号，在 n_drop=15 好策略下测上限。

设计：继承 HighBetaAlpha158（复用 L1 shared DropnaProcessor(feature) 前视护栏 + __init__），
仅 override get_feature_config 返回 18 个 $field。T 日当天因子、不 lag（与 HighBetaAlpha158
的分钟部分契约一致）。label / deal_price / 涨跌停拦截与 workflow_minute_only.yaml 全一致
（见 workflow_minute_enhanced.yaml）——唯一变量是 feature 从 14 → 18 + 策略 n_drop 5 → 15。
"""
from __future__ import annotations

from .config import MINUTE_FACTOR_EXTRA_FIELDS, MINUTE_FACTOR_FIELDS
from .highbeta_handler import HighBetaAlpha158


class MinuteEnhancedHandler(HighBetaAlpha158):
    """18 因子增强集：14 分钟全集 + 4 extra（vol_vs_yest_t2/t3/t5 + overnight_gap）。"""

    # 14 全集 + 4 extra = 18。顺序：先 14 分钟族（与 m14 同序，保留可复现性），再 4 extra。
    ENHANCED_FIELDS = tuple(MINUTE_FACTOR_FIELDS) + tuple(MINUTE_FACTOR_EXTRA_FIELDS)  # 14 + 4 = 18

    def get_feature_config(self):
        fields = [f"${n}" for n in self.ENHANCED_FIELDS]   # 18 因子，T 日当天不 lag
        names = list(self.ENHANCED_FIELDS)
        return fields, names
