"""MinuteEnhancedCSRankHandler — 18 因子 + 特征组横截面 Rank 归一化（2026-07-07）。

背景：champion enhanced(18) 的特征走 Alpha158 默认 infer_processors = RobustZScoreNorm
（**时间序列** robust z-score，fit on train）。highbeta883926 池每日换手 ~90%，绝对开盘
幅度混淆「个股相对强度」与「日级 regime」。本 handler 把特征归一化换成 **CSRankNorm**
（每日横截面 rank，无 fit → 零跨段泄漏），强制模型只用当日相对位置 → 提纯相对强度。

第一性原理：Spearman RankIC 对日内单调变换不变 → 单因子 RankIC 无法区分横截面 rank 与
原始值（每日相同）。故横截面假设是多元组合问题，只能训练验。本 handler = 最廉价判定实验。

机制（零泄漏）：DataHandlerLP 流 _learn_df = learn_proc(infer_proc(shared_proc(raw)))，
infer_processors 对训练+推理都生效 → CSRankNorm 在 train/predict 两端都施加。CSRankNorm
无 fit，每日独立 rank。shared DropnaProcessor(feature)（L1 前视护栏，从 HighBetaAlpha158
继承）先于 CSRankNorm 跑 → feature-NaN 行先 drop → CSRankNorm 在完整横截面 rank。label 侧
CSZScoreNorm（Alpha158 默认 learn_processor）不变 → rank 特征 → 横截面 label，口径自洽。

设计：继承 MinuteEnhancedHandler（复用 18 因子 get_feature_config + L1 shared 护栏 +
__init__），仅 override __init__ 注入 infer_processors = [CSRankNorm(feature), Fillna]，
替换 Alpha158 默认的 [RobustZScoreNorm(feature, clip), Fillna]。label / deal_price /
涨跌停拦截 / 模型超参 / 切分全部与 workflow_minute_enhanced.yaml 一致（见
workflow_minute_enhanced_csrank.yaml）——唯一变量是特征归一化时间序列→横截面。

详见 docs/superpowers/specs/2026-07-07-csrank-feature-normalization-design.md。
"""
from __future__ import annotations

from .minute_enhanced_handler import MinuteEnhancedHandler


class MinuteEnhancedCSRankHandler(MinuteEnhancedHandler):
    """18 因子 + 特征组 CSRankNorm（每日横截面 rank，替换时间序列 RobustZScoreNorm）。"""

    # 替换 Alpha158 默认 infer_processors：RobustZScoreNorm（时间序列）→ CSRankNorm（横截面 rank）。
    # Fillna 保留（CSRankNorm 在已 dropna 的特征上无 NaN，Fillna 为兜底安全网，与 Alpha158 默认结构对齐）。
    _CS_INFER_PROCESSORS = [
        {"class": "CSRankNorm", "kwargs": {"fields_group": "feature"}},
        {"class": "Fillna", "kwargs": {}},
    ]

    def __init__(self, *args, **kwargs):
        # 默认挂上横截面 rank 归一化；workflow.yaml 可显式传 infer_processors 覆盖。
        kwargs.setdefault("infer_processors", self._CS_INFER_PROCESSORS)
        super().__init__(*args, **kwargs)
