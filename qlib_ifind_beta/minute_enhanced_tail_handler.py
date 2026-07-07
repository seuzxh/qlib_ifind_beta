"""MinuteEnhancedTailHandler — 20 因子（champion 18 + 2 个最强正交 T-1 尾盘因子，goal 因子优化 Step C2）。

背景（goal 2026-07-07「优化因子提升超额与 IC」）：
  T 日 9:30-9:40 分钟因子（champion 18）已用尽当日早盘信息；T-1 全天分钟结构此前完全未作因子
  （vol_vs_yest 分母仅把 T-1 全天量压成聚合标量）。新增 T-1 尾盘（14:50-15:00，slot 232-241）
  分钟族，刻画用户语义「惯性冲高/高潮/出货」次日领先信号。

IC 基线校准（第一性原理，2026-07-07，test 段 2026-04→07，23 因子全量单变量 IC）：
  - tail_vol_ratio_t1：|RankICIR|=0.398，全部 23 因子 #3（仅次于 vol_vs_yest_t5 0.497 /
    overnight_gap 0.458）。反转信号（T-1 尾盘放量出货 → 次日跌），与 vol_vs_yest 族同向但测
    不同时点（T-1 尾盘 vs T 日早盘）。
  - tail_accel_t1：|RankICIR|=0.263，#7。动量/惯性信号（尾盘加速度 → 次日惯性冲高）。
  - 正交性：5 tail 因子与 champion 18（overnight_gap/startup/accel/close_pos/vol_vs_yest）
    |corr| 全部 <0.07 → 完全正交，纯增量信息（无冗余）。
  - 全段（2024-2026）tail 单变量 IC 弱（|RankICIR|<0.07），但 baseline 14 因子里 startup_mom /
    close_pos / vol_ratio_1m 全段亦弱（仅 overnight_gap + vol_vs_yest 族全段强）→ 全段弱是
    baseline 常态，非 tail 独有缺陷；test 段涌现信号才是关键。
  - 仅取 2 个最强 tail（非全 5）：tail_close_pos/mom/last5 test 段 |RankICIR|<0.15 且与已选 2 个
    相关，加入徒增过拟合风险（Step B 教训：30 因子 IC +12% 但超额平、回撤恶化）。

设计：继承 MinuteEnhancedHandler（复用 champion 18 因子的 ENHANCED_FIELDS + L1 shared
DropnaProcessor(feature) 前视护栏 + __init__），仅 override get_feature_config 追加 2 tail
$field。tail 因子已物化为 day.bin、shift-1（T 行=T-1 尾盘，无前视；T-1 15:00 收盘 T 日 9:41
决策已知），handler $field 直接消费（与 14 分钟因子零 Ref 模式统一）。label / deal_price /
涨跌停拦截 / 策略 / 模型超参 / 切分全冻结合 workflow_minute_enhanced.yaml → 唯一变量是
feature 18 → 20。详见 backtest-log（goal Step C2）。
"""
from __future__ import annotations

from .minute_enhanced_handler import MinuteEnhancedHandler


class MinuteEnhancedTailHandler(MinuteEnhancedHandler):
    """20 因子：champion(18) + tail_vol_ratio_t1 + tail_accel_t1（T-1 尾盘最强 2 因子）。"""

    # IC 校准 test 段 |RankICIR| 排名 #3 / #7，且与 champion 18 因子全部 |corr|<0.07（正交）。
    # 顺序：先 champion 18（继承 ENHANCED_FIELDS，保留可复现性），再 2 tail。
    TAIL_PICK = ("tail_vol_ratio_t1", "tail_accel_t1")

    def get_feature_config(self):
        fields = [f"${n}" for n in self.ENHANCED_FIELDS] + [f"${n}" for n in self.TAIL_PICK]
        names = list(self.ENHANCED_FIELDS) + list(self.TAIL_PICK)
        return fields, names
