"""六库表达式清单的统一入口（fields/names 均为字符串）。

- Alpha158/Alpha360：pyqlib 0.9.7 原生 loader（qlib.contrib.data.loader）。
- Alpha101/GTJA191：vendored loader（本包）。
- TDXGS/JQ110：expressions_tdxgs_jq110.json（ast 从上游 handler.py 提取）。

已知上游缺陷（不改写、如实使用）：Alpha101 表达式用 `Rank(X, 1)` 近似
WorldQuant 截面 rank（窗口 1 的时序 rank，退化为常数 1）；ALPHA002 用了
1 参 Rank(X)，fork 与 0.9.7 均无法求值——筛选时按不可用剔除并记录。
"""
from __future__ import annotations

import json
from pathlib import Path

_JSON = Path(__file__).resolve().parent / "expressions_tdxgs_jq110.json"


def _alpha158() -> tuple[list[str], list[str]]:
    from qlib.contrib.data.loader import Alpha158DL
    return Alpha158DL.get_feature_config()


def _alpha360() -> tuple[list[str], list[str]]:
    from qlib.contrib.data.loader import Alpha360DL
    return Alpha360DL.get_feature_config()


def _alpha101() -> tuple[list[str], list[str]]:
    from .loader_alpha101 import Alpha101DL
    return Alpha101DL.get_feature_config()


def _gtja191() -> tuple[list[str], list[str]]:
    from .loader_gtja191 import GTJA191DL
    return GTJA191DL.get_feature_config()


def _from_json(key: str) -> tuple[list[str], list[str]]:
    data = json.loads(_JSON.read_text())
    return list(data[key]["fields"]), list(data[key]["names"])


_BUILDERS = {
    "alpha360": _alpha360,
    "alpha158": _alpha158,
    "alpha101": _alpha101,
    "gtja191": _gtja191,
    "tdxgs": lambda: _from_json("TDXGS"),
    "jq110": lambda: _from_json("JQ110DataHandler"),
}


def load_library(name: str) -> tuple[list[str], list[str]]:
    """返回上游原样 (fields, names)；重复调用返回新列表。"""
    return _BUILDERS[name.lower()]()


#: 库名 → 因子数（惰性加载后填充）
LIBRARIES: dict[str, tuple[list[str], list[str]]] = {}


def _ensure_loaded() -> None:
    if not LIBRARIES:
        for name in _BUILDERS:
            LIBRARIES[name] = load_library(name)


def iter_all_factors() -> list[tuple[str, str, str]]:
    """全部因子清单：[(lib, name, expression), ...]（含跨库重名保持原样）。"""
    _ensure_loaded()
    rows = []
    for lib, (fields, names) in LIBRARIES.items():
        rows.extend((lib, n, f) for n, f in zip(names, fields))
    return rows


if __name__ == "__main__":  # pragma: no cover - 手工核查用
    _ensure_loaded()
    total = 0
    for lib, (fields, names) in LIBRARIES.items():
        assert len(fields) == len(names)
        total += len(fields)
        print(f"{lib:9s} {len(fields):4d}  e.g. {names[0]}")
    print(f"TOTAL    {total:4d}")
