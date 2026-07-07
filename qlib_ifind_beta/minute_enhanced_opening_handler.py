"""MinuteEnhancedOpening handlers — champion(18) + T-1/T-2 开盘因子（goal 2026-07-08）。

背景（goal「换个思路扩展不同区间的 1min 因子，提高 IC 和超额」）：
  §23/§24/§25 三重 IC→超额墙定论：champion(18) 的超额来自 topk=20 边界放得极锐（Δ≈1.03%/日），
  18 因子已在边界近最优——任何方向都重排并退化该边界。§25 尾盘失败的关键：T-1 尾盘是不同
  regime（与 T 日开盘正交 |corr|<0.07）→ 正交因子重排边界。本族测「同 regime 滞后」假设：
  T-1/T-2 **开盘**（slot 1-10，与 T 日开盘同 regime、仅时间平移）→ 若高贝塔开盘动量跨日持续
  则**共线强化**（锐化边界而非重排），可能破墙；若反转则正交 → 复刻 §25 失败。

IC 基线校准（第一性原理，2026-07-08，全段 2024-2026，5038 股 × 600 天，15 因子单变量 Spearman
rank IC vs label `Ref($close,-1)/$price_941-1`）：
  - vol_ratio 族**跨日持续**（共线信号，假设直接验证）：
      vol_ratio_5m_t2 +0.0297（全部 15 因子 #1，ICIR 0.224）/ vol_ratio_5m（baseline）+0.0252（#2）/
      vol_ratio_5m_t1 +0.0187（#4，ICIR 0.138）→ 全正、单调 → 开盘量比是真实跨日持续边际。
  - accel/startup_mom/startup_total 族**不持续**：T-1/T-2 lag 全段 |IC|<0.003（噪声）。
  - 单因子全段 |IC|<0.03 是 baseline 常态（champion signal IC 0.0545 来自 LGBM 非线性组合 18 因子）。

两个变体（用户「寻找多种可能性」）：
  - V1（§25 头对头，设计忠实）：18 + [vol_ratio_5m_t1, accel_5m_t1] = 20。同 count、同 shift、
    仅 regime 不同（开盘 vs 尾盘）→ 隔离「同 regime 滞后 vs 不同 regime」。accel_5m_t1 是 T-1 族
    |IC| #2（虽弱 -0.003），保留 top-2 T-1 选择忠实。
  - V1b（IC 驱动，最强共线信号）：18 + [vol_ratio_5m_t1, vol_ratio_5m_t2] = 20。两条持续量比 lag
    → 给模型 regime 检测原料（用户心法：T-1 K 线判共振/启动/惯性）。直接测「持续开盘量比是否
    共线强化边界」——假设的最强检验。

设计：继承 MinuteEnhancedHandler（复用 champion 18 的 ENHANCED_FIELDS + L1 前视护栏 + __init__），
仅 override get_feature_config 追加 opening $field。opening 因子已物化为 day.bin、shift-1/2
（min-cal 空间，T 行=T-1/T-2 开盘 9:31-9:40，无前视；T-1/T-2 9:40 远早于 T 日 9:41 决策），
handler $field 直接消费（与 14 分钟因子零 Ref 模式统一）。label / deal_price / 涨跌停拦截 /
策略 / 模型超参 / 切分全冻结 → 唯一变量是 feature 18→20。详见
docs/superpowers/specs/2026-07-08-t1-opening-factors-design.md + backtest-log §26。
"""
from __future__ import annotations

from .minute_enhanced_handler import MinuteEnhancedHandler


class MinuteEnhancedOpeningT1Handler(MinuteEnhancedHandler):
    """V1：20 因子 = champion(18) + vol_ratio_5m_t1 + accel_5m_t1（T-1 开盘 top-2 by |IC|）。

    §25 头对头：同 20 因子 count、同 shift-1、仅 regime 不同（T-1 开盘 vs T-1 尾盘）。
    """

    # IC 校准 T-1 族 |mean_ic| top-2：vol_ratio_5m_t1(+0.0187) / accel_5m_t1(-0.0029)。
    OPENING_T1_PICK = ("vol_ratio_5m_t1", "accel_5m_t1")

    def get_feature_config(self):
        fields = [f"${n}" for n in self.ENHANCED_FIELDS] + [f"${n}" for n in self.OPENING_T1_PICK]
        names = list(self.ENHANCED_FIELDS) + list(self.OPENING_T1_PICK)
        return fields, names


class MinuteEnhancedOpeningVolHandler(MinuteEnhancedHandler):
    """V1b：20 因子 = champion(18) + vol_ratio_5m_t1 + vol_ratio_5m_t2（跨日持续量比，最强共线）。

    IC 驱动：vol_ratio 族全段单调持续（t2#1 / baseline#2 / t1#4 全正）→ 两条 lag 直接测
    「持续开盘量比共线强化 topk=20 边界」假设（破墙的最强检验）。
    """

    # IC 校准：vol_ratio_5m_t1(+0.0187, T-1) + vol_ratio_5m_t2(+0.0297, T-2，全 15 因子 #1)。
    OPENING_VOL_PICK = ("vol_ratio_5m_t1", "vol_ratio_5m_t2")

    def get_feature_config(self):
        fields = [f"${n}" for n in self.ENHANCED_FIELDS] + [f"${n}" for n in self.OPENING_VOL_PICK]
        names = list(self.ENHANCED_FIELDS) + list(self.OPENING_VOL_PICK)
        return fields, names
