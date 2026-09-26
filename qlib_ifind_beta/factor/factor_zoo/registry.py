"""把 vendored zoo 自定义算子注册进 pyqlib 0.9.7 表达式引擎。

pyqlib 0.9.7 的注册入口是 qlib.data.ops.register_all_ops（qlib.init 内部
同款）：reset 内置算子后统一注册 C.custom_ops。本模块幂等；兼容
qlib.init 前后调用。冒烟对照见 backtest-log 筛选报告 §0.2。
"""
from __future__ import annotations

from qlib.config import C

from .ops_compat import SHIMS, SHIMmed_NAMES

# 需要注册的算子类名（上游 handler.py 注册清单；实际注册 ops_compat 生成的
# 0.9.7 兼容同名子类，见 ops_compat.build_shims）。
_OP_NAMES = SHIMmed_NAMES


def register_zoo_ops() -> list[type]:
    """幂等注册全部 zoo 算子（0.9.7 兼容 shim）；返回注册的算子类列表。"""
    missing = [n for n in _OP_NAMES if n not in SHIMS]
    if missing:  # pragma: no cover - vendored 文件被改动时才触发
        raise AttributeError(f"custom_ops 缺少算子: {missing}")
    from qlib.data.ops import register_all_ops
    existing = list(getattr(C, "custom_ops", None) or [])
    C.custom_ops = existing + [op for op in SHIMS.values() if op not in existing]
    register_all_ops(C)  # reset + 内置 + custom_ops，与 qlib.init 同款
    return list(SHIMS.values())
