"""zoo 自定义算子 → pyqlib 0.9.7 的兼容 shim。

上游 custom_ops.py 的算子类覆写 `__init__` 且不调用 `super().__init__`，
因此 0.9.7 协议要求的 `self.feature` / 窗口递归不存在（fork 的 ops.py
内部协议不同）。本模块不改动 vendored 文件，而是为每个算子类生成同名
子类，补齐 0.9.7 需要的两个方法：

- `get_extended_window_size`：对实例属性里的全部子表达式取窗口并集，
  再加上整数窗口参数（N/M1/... 取最大值）的前视扩展；
- `get_longest_back_rolling`：同上口径。

`_load_internal` 全部沿用上游实现（其内部用 `self.close.load(...)` 等，
Feature.load 在 0.9.7 可用）。窗口偏保守（宁可多加载历史，不影响正确性）。
"""
from __future__ import annotations

from qlib.data.ops import Expression

from . import custom_ops as _co

#: 需要 shim 的算子类名（与 registry._OP_NAMES 中除原生兼容者外的集合相同；
#: 对已兼容的类生成 shim 也无害——并集口径与原生 Rolling 等价）。
SHIMmed_NAMES = (
    "TsArgmax", "TsArgmin", "SMA", "Amount",
    "ATR", "RSV", "RSI", "BIAS", "BBI", "WR", "CCI", "PDI", "MDI", "ADX", "ADXR",
    "BOLL_UP", "BOLL_DN", "BOLL_MID", "PSY", "PSYMA", "ROC", "MAROC", "MTM",
    "MTMMA", "TRIX", "TRMA", "VR", "CR", "AR", "BR", "OBV", "MFI", "DPO", "MADPO",
    "TAQ_UP", "TAQ_DN", "TAQ_MID", "KTN_UP", "KTN_DN", "KTN_MID", "EMV", "MAEMV",
    "MASS", "MA_MASS", "DFMA_DIF", "DFMA_DIFMA", "MA", "STD_TDX",
    "AroonUp", "AroonDown", "BullPower", "BearPower", "VPT", "WVAD", "VOSC",
    "Variance", "Skewness", "Kurtosis", "SharpeRatio", "PriceRank",
    "CumulativeRange", "MoneyFlow",
)


def _sub_expressions(obj: object) -> list[Expression]:
    return [v for v in vars(obj).values() if isinstance(v, Expression)]


def _max_window(obj: object) -> int:
    ints = [v for v in vars(obj).values()
            if isinstance(v, int) and not isinstance(v, bool)]
    return max(ints, default=1)


def _get_extended_window_size(self):  # noqa: ANN001 - qlib 协议方法
    left = right = 0
    for f in _sub_expressions(self):
        lf, rf = f.get_extended_window_size()
        left, right = max(left, lf), max(right, rf)
    return left + _max_window(self) - 1, right


def _get_longest_back_rolling(self):  # noqa: ANN001
    base = max((f.get_longest_back_rolling() for f in _sub_expressions(self)),
               default=0)
    return max(base, _max_window(self))


def build_shims() -> dict[str, type]:
    """为 SHIMmed_NAMES 里的每个上游算子类生成同名兼容子类。

    生成后注入本模块 globals（pickle 按 module+qualname 查找，joblib
    并行求值需要）；重复调用幂等。
    """
    shims: dict[str, type] = {}
    for name in SHIMmed_NAMES:
        base = getattr(_co, name, None)
        if base is None:  # pragma: no cover
            continue
        if name in globals():  # 幂等：已生成
            shims[name] = globals()[name]
            continue
        shim = type(name, (base,), {
            "get_extended_window_size": _get_extended_window_size,
            "get_longest_back_rolling": _get_longest_back_rolling,
            "__module__": __name__,
            "__qualname__": name,
        })
        globals()[name] = shim  # 供 pickle 按 (module, qualname) 反查
        shims[name] = shim
    return shims


#: 模块导入时即生成（保证 pickle 可用），registry 直接取用。
SHIMS: dict[str, type] = build_shims()
