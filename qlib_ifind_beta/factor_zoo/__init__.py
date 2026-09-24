"""qlib-factor-zoo vendored 因子库（研究分支 research/factor-zoo-20260924）。

上游 https://github.com/JustinF8/qlib-factor-zoo（MIT），快照 ea21f315。
六大库（Alpha360/158/101、GTJA191、TDXGS、JQ110，约 1013 个日频表达式）
的表达式与自定义算子，用于：
  1. 日频 T-1 lag 对照筛选（路线 A）；
  2. 分钟域移植（B1 当日盘初 / B2 多日全天）；
  3. 加速状态变量构造（风险轨）。
详见 docs/research/2026-09-24-factor-zoo-minute-research-plan.md。
"""
from .registry import register_zoo_ops
from .zoo_libs import LIBRARIES, iter_all_factors

__all__ = ["register_zoo_ops", "LIBRARIES", "iter_all_factors"]
