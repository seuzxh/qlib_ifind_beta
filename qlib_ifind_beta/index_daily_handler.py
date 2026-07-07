"""IndexDailyHandler — 12 个日频情绪 + 上证指数共振因子（Step A，2026-07-07）。

任务（用户 /loop 2026-07-07）：在 champion=enhanced(18)@n_drop=15 基础上加因子提升预测准确性：
  - 7 个日频情绪因子（个股启动/发酵/高潮阶段，避免拥挤度过高接盘）
  - 5 个上证指数因子（个股↔指数共振，根据指数情绪冰点/沸点判断高 beta 风险）

设计：继承 HighBetaAlpha158（复用 L1 shared DropnaProcessor(feature) 前视护栏 + __init__），
仅 override get_feature_config 返回 12 个 qlib expression。12 expression 已在
tests/test_index_factors.py 用手算 pandas rolling 对照 + 截断不变性 gate 全部验证（19/19 PASS）。

前视对齐（硬约束，与 champion 一致）：champion 在 T 日 9:41 成交（TopkDropoutStrategyTD0 shift=0），
T 日日频/指数数据 9:41 尚未收盘 → 所有 12 个 expression 必须 Ref(expr,1)（T 行 = T-1 收盘数据算的值）。
分钟因子不 lag（9:30-9:40 当天已知），但本 handler 不含分钟因子。

⚠️ corr_20 已移除（原第 13 因子，个股↔指数 20 日滚动相关）：qlib Corr._load_internal（ops.py:1488-1498）
override 用 `np.isclose(left_std, right_std)` 置零方差位为 NaN，但 numpy 广播不按 index 对齐——个股
收益序列（按个股 calendar，621 行）与 ChangeInstrument 上证收益序列（按指数 calendar，624 行）长度
不等时直接抛 `operands could not be broadcast together with shapes (621,) (624,)`，对 highbeta883926
池 296/5114 股票崩。Cov/Var（beta_20）无此 override → 零崩。改写 Cov/(Std·Std) 可消崩但缺口窗口
（停牌）三方有效样本不同 → 发散 max 0.3~0.8，是「corr 代理」非真 corr。综合：corr_20 与 beta_20 高度
共线（都测个股-指数联动），高贝塔池里 beta>1 的敏感度比 corr∈[-1,1] 更贴切 → 删 corr_20，共振由
beta_20 + idx_run_5/idx_bias_20 覆盖。决策详见 backtest-log §23 + Get笔记 2026-07-07。

⚠️ Step A 单独 handler 的已知局限（重要，结果解读须据此）：
  HighBetaAlpha158 的 L1 护栏依赖「p941 NaN ⟹ 14 分钟因子 NaN ⟹ 被 feature 组 drop」联动——
  本 handler 无分钟因子，这条联动断链 → p941（exchange deal_price）单独缺失时不会触发 feature drop，
  Exchange 仍可能回退到 $close[T] 前视路径。因此：
    - **主看 IC / ICIR（SigAnaRecord 不走 Exchange，干净）**
    - 若跑回测，数字仅供参考（p941 护栏弱于 champion），不能直接与 champion 回测对比
  Step B 合体 EnhancedWithDailyIndex(30=18min+12daily) 会恢复完整分钟因子联动，护栏与 champion 等价，
  那时回测才可与 champion 直接对比。

label / deal_price / 涨跌停拦截 / 策略 n_drop=15 全冻结与 champion 一致（见 workflow_daily_index.yaml）。
"""
from __future__ import annotations

from .highbeta_handler import HighBetaAlpha158

# ChangeInstrument 引用源：上证综指 SH000001（用户语义「上证指数」，共振/情绪因子引用源）。
# 区别于 benchmark=SH000300（回测基准冻结）。SH000001 的 7 base bin 在 build_overlay step6 被 link
# 进 overlay（仅 link 不 materialize——指数不交易）。
_SH = "SH000001"

# 指数字段表达式（拼字符串保持可读，与 test_index_factors.py EXPRS 逐字一致）
IDX_CLOSE = f"ChangeInstrument('{_SH}', $close)"
IDX_LOW = f"ChangeInstrument('{_SH}', $low)"
IDX_HIGH = f"ChangeInstrument('{_SH}', $high)"
IDX_VOLUME = f"ChangeInstrument('{_SH}', $volume)"

# 个股/指数日收益率（Cov/Var/Corr 的左右操作数）
STOCK_RET = "$close/Ref($close,1)-1"
INDEX_RET = f"{IDX_CLOSE}/Ref({IDX_CLOSE},1)-1"


class IndexDailyHandler(HighBetaAlpha158):
    """12 个日频情绪 + 上证指数共振因子（Step A 单独验证用）。

    全部 Ref(expr,1) lag1（T 行 = T-1 数据），无前视。详见模块 docstring 的 Step A 局限说明。
    """

    # name → qlib expression。顺序：先 7 日频情绪，后 5 指数共振（corr_20 已移除，见模块 docstring）。
    FACTOR_FIELDS = (
        # ---- 任务2：个股日频情绪族 7 个 ----
        # bias_5/20：收盘偏离均线（启动=低位负偏修复，高潮=高位正偏发散）
        ("bias_5", "Ref(($close-Mean($close,5))/Mean($close,5),1)"),
        ("bias_20", "Ref(($close-Mean($close,20))/Mean($close,20),1)"),
        # vol_ratio_20：近 20 日量能（发酵阶段放量）
        ("vol_ratio_20", "Ref($volume/Mean($volume,20),1)"),
        # run_up_5：5 日涨幅（启动动能）
        ("run_up_5", "Ref($close/Ref($close,5)-1,1)"),
        # rsv_9：9 日随机指标（超买超卖）
        ("rsv_9", "Ref(($close-Min($low,9))/(Max($high,9)-Min($low,9)),1)"),
        # dist_to_limit：距涨停距离（拥挤度——越接近涨停越拥挤，避免接盘）
        ("dist_to_limit", "Ref($change/$limit_up,1)"),
        # accel_mom：动量加速度（发酵→高潮的拐点）
        ("accel_mom", "Ref(($close/Ref($close,3)-1)-(Ref($close,3)/Ref($close,6)-1),1)"),
        # ---- 任务3：上证指数共振/情绪族 5 个（corr_20 已移除，见模块 docstring）----
        # idx_bias_20：指数 20 日偏离（指数沸点=高位正偏，冰点=低位负偏）
        ("idx_bias_20", f"Ref(({IDX_CLOSE}-Mean({IDX_CLOSE},20))/Mean({IDX_CLOSE},20),1)"),
        # idx_run_5：指数 5 日涨幅（指数趋势）
        ("idx_run_5", f"Ref({IDX_CLOSE}/Ref({IDX_CLOSE},5)-1,1)"),
        # idx_rsv_9：指数 9 日随机（指数超买超卖）
        ("idx_rsv_9", f"Ref(({IDX_CLOSE}-Min({IDX_LOW},9))/(Max({IDX_HIGH},9)-Min({IDX_LOW},9)),1)"),
        # idx_vol_ratio_20：指数量能（指数资金面）
        ("idx_vol_ratio_20", f"Ref({IDX_VOLUME}/Mean({IDX_VOLUME},20),1)"),
        # beta_20：个股对指数的 20 日 beta（高 beta 风险——沸点时高 beta 股风险放大）
        ("beta_20", f"Ref(Cov({STOCK_RET}, {INDEX_RET}, 20)/Var({INDEX_RET}, 20),1)"),
    )

    def get_feature_config(self):
        fields = [expr for _, expr in self.FACTOR_FIELDS]
        names = [name for name, _ in self.FACTOR_FIELDS]
        return fields, names
