"""HighBetaAlpha158 — Alpha158(rolling windows [5,10]，全 lag T-1) + 14 minute factors (T-day).

v2 (2026-07-06): wraps all Alpha158 daily fields in Ref(...,1) to lag them to
T-1 — T 日 9:41 撮合只用 T-1 及更早的日频数据，无前视。The 14 minute fields stay
T-day (they are 9:30-9:40, before the 9:41 buy). qlib reads all as daily overlay;
no frequency mixing at the Handler layer.

v3 (2026-07-06): rolling windows 砍到 [5,10]（剔除 20/30/60，−87 慢因子）。依据：
逐因子 IC 显示 87 慢因子无一 |ICIR|>0.3（最强 VMA60 仅 0.21），剔除几乎不损失信号；
高贝塔短周期 + 9:41 撮合主线偏快周期，20/30/60 日均线/动量与主线无关；降维 172→85
减 lightgbm 过拟合面（best iter=3 诊断见 backtest-log §14）。走 Alpha158DL config 原生
扩展点（rolling.windows），不偏离 qlib 机制。因子数 9 kbar + 4 price + 29×2 rolling
= 71 日频 + 14 分钟 = 85。

label / deal_price via qrun YAML (handler.kwargs.label, exchange_kwargs.deal_price).
pred[T]→T-day execution needs TopkDropoutStrategyTD0 — see td0_strategy.py.

L1 前视护栏使用 shared DropnaProcessor(feature)，详见类内
_DEFAULT_SHARED_PROCESSORS 注释。
"""
from __future__ import annotations

from qlib.contrib.data.handler import Alpha158

from .config import MINUTE_FACTOR_FIELDS


class HighBetaAlpha158(Alpha158):
    # L1 前视护栏（用户 2026-07-06 选定）：shared DropnaProcessor(feature) 让训练 + 推理
    # 都 drop feature NaN。PTYPE_A 流程 _data -[shared]- _shared_df -[infer]- _infer_df -[learn]-
    # _learn_df，shared 同时喂给 infer 与 learn。
    #
    # 为什么能防 9:41 前视：$price_941 不在 feature 组（它是 exchange 的 deal_price），但当
    # 某股 9:41 数据缺失（如 2026-07-01 cn_data_1min 全市场仅 6/5521 只有 1min 数据的生产事故）
    # → p941 NaN ⟹ 同源 1min slots 1-10 也 NaN ⟹ 14 个分钟因子全 NaN ⟹ 被 feature 组 drop
    # → 该 stock-day 被排除出 _infer_df → 不进 SignalRecord 的预测表 → 不进 TopkDropoutStrategy
    # 候选池 → 不进交易列表 → Exchange 永不触发 exchange.py:510-513 把 NaN deal_price 回退到
    # $close[T]（全天收盘 = 买入时刻的未来函数）的前视路径。训练侧同理 drop，样本更干净。
    #
    # 已知边界（可接受）：当 slots 1-10 完整但 slot 11（9:41，p941）单独缺失时，14 分钟
    # 因子非 NaN、p941 NaN，不会被 feature 组 drop —— 但实测 slot 11 单独缺失为 0 例
    # （test 段 p941 NaN 的 5,780 stock-day，9 个 slots1-10 同源因子同步全 NaN，无孤立
    # case），兜底由 exchange_kwargs.limit_threshold 的 $change_941 表达式承担。
    #
    # 误杀权衡（test 段实测）：整组 drop 会顺带丢"p941 有值但某 feature 边界 NaN"的样本，
    # 误杀 ~4.4%（防前视正确 drop ~1.9%）；主因是分钟因子部分缺失（$close_pos_1m/3m/5m、
    # $vol_vs_yest，后者在 2026-07-02 因 07-01 cn_data_1min 事故连锁全局 NaN）+ 日频
    # ROC/STD/CORR 除零。用户 2026-07-06 定方案 A（接受现状）；精化版（只 drop
    # $price_941 NaN，需自定义 Processor）不属于当前生产基线。
    _DEFAULT_SHARED_PROCESSORS = [{"class": "DropnaProcessor", "kwargs": {"fields_group": "feature"}}]

    def __init__(self, *args, **kwargs):
        # 默认挂上 L1 前视护栏；工作流 YAML 可显式传 shared_processors 覆盖。
        kwargs.setdefault("shared_processors", self._DEFAULT_SHARED_PROCESSORS)
        super().__init__(*args, **kwargs)

    def get_feature_config(self):
        # v3 (2026-07-06)：rolling windows 砍到 [5,10]，剔除 20/30/60 共 87 个慢因子。
        # 走 Alpha158DL 的 config 原生扩展点（rolling.windows）从源头不生成，而非 super()
        # 的默认 [5,10,20,30,60] 再事后过滤。conf 其余键与 Alpha158.get_feature_config 同
        # （kbar + price windows=[0] OPEN/HIGH/LOW/VWAP），仅 rolling.windows 不同 → 9 kbar
        # + 4 price + 29 算子×2 窗口 = 71 日频因子。
        from qlib.contrib.data.loader import Alpha158DL

        conf = {
            "kbar": {},
            "price": {"windows": [0], "feature": ["OPEN", "HIGH", "LOW", "VWAP"]},
            "rolling": {"windows": [5, 10]},
        }
        fields, names = Alpha158DL.get_feature_config(conf)
        lag_fields = [f"Ref({f}, 1)" for f in fields]         # v2: lag 日频 → T-1
        min_fields = [f"${n}" for n in MINUTE_FACTOR_FIELDS]  # 14 分钟因子，T 日当天不 lag
        return lag_fields + min_fields, names + min_fields
