"""MinuteResonanceHandler — 21 因子共振集（goal 2026-07-08「分钟尺度 regime」实做）。

背景：§23-28 八重墙（factor + model + strategy 三层穷尽）后，唯一原则性未测的杠杆
= 分钟尺度指数开盘共振因子。用户领域模型原话「分钟频只能观测 T 日的开盘阶段，所以
我需要结合指数（大势）和 T-1 之前的 K 线来判断个股是否是共振、启动、高潮或者惯性
冲高」—— §28 测的是 **日频** regime（T-1 close RSV/bias）做择时，scale 错配 → OOS
失败；本 handler 测的是 **T 日 9:30-9:40 开盘分钟内个股 vs 指数的横截面相对强度**，
尺度恰与 ~1.5 天 label 对齐。champion 18 因子全 per-stock、无指数维度 → 3 idx 因子是
真增量；LGBModel 自动学「个股开盘动量 × 大势开盘强度」交互 = 用户「共振」语义。

设计依据（第一性原理，不联想）：
  - 3 idx 因子（idx_open_ret_10/idx_open_mom_5m/idx_open_accel_5m）与 champion 3 个代表
    早盘因子（startup_total/startup_mom_5m/accel_5m）公式同构（仅输入从个股 1min 换成
    SH000001 1min），刻画的都是「大势开盘态势」单一维度 → 给模型 champion 没有的大势
    调节维度，无尺度错配（都 T 日开盘分钟，与 label 同尺度）。
  - 物化 broadcast（Plan A）：每只股 overlay bin 存同日同值 idx 因子，复用 scatter 的
    valid/rel_v（仅依赖 code 的 si_dc）。SH000001 缺失日（train ~6%，test/valid 0%，
    probe-verified）→ 全池同日 NaN → DropnaProcessor drop（§L1 护栏自动生效）。
  - 单变量视角下 idx 因子大概率相关≈0（§28.B 已证日频 regime 单变量相关≈0），但交互
    视角（LGBModel 多变量）下有非零概率。是唯一原则性未测方向：要么破墙、要么第 9 重
    墙铁证关闭 regime 全尺度方向。

设计：继承 MinuteEnhancedHandler（复用 18 champion 因子 + L1 shared DropnaProcessor(feature)
前视护栏 + __init__），仅 override get_feature_config 返回 21 个 $field（18 + 3 idx）。
label / deal_price / 涨跌停拦截 / 策略与 workflow_minute_enhanced.yaml 全一致——唯一变量
是 feature 从 18 → 21。
"""
from __future__ import annotations

from .config import INDEX_OPENING_FIELDS
from .minute_enhanced_handler import MinuteEnhancedHandler


class MinuteResonanceHandler(MinuteEnhancedHandler):
    """21 因子共振集：18 champion + 3 上证指数开盘共振因子（broadcast 同值）。"""

    # 18 champion + 3 idx = 21。顺序：先 18 champion（与 enhanced 同序，保留可复现性），
    # 再 3 idx 开盘共振因子（大势开盘态势维度，供树模型学个股×大势交互）。
    RESONANCE_FIELDS = MinuteEnhancedHandler.ENHANCED_FIELDS + tuple(INDEX_OPENING_FIELDS)  # 18 + 3 = 21

    def get_feature_config(self):
        fields = [f"${n}" for n in self.RESONANCE_FIELDS]   # 21 因子，T 日当天不 lag
        names = list(self.RESONANCE_FIELDS)
        return fields, names
