"""表达式静态解析：窗口参数提取与 B1/B2 可实例化分类。

扫描表达式里的函数调用（括号配对），取"纯整数参数"为窗口参数（系数整数
不进函数参数，不会误报；分位数的 0.8 是浮点不算）。口径保守：窗口取全库
出现的最大整数。

分类（结合 Phase 0.2 冒烟的 999 个可用表达式）：
- b1：max_window ≤ 8（当日盘初 10 根 bar 内可算）
- b2：max_window ≤ 120（T-k 全天 240 根内可算，b1 ⊆ b2）
- 无窗口表达式（纯逐 bar 代数）单列 nowindow，两域都可算但语义弱

注意：EMA/SMA 类无限记忆算子按窗口参数计（分钟域切片口径的近似，见筛选
报告口径说明）。
"""
from __future__ import annotations

import re

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def iter_calls(expr: str):
    """yield (op_name, [arg_source, ...])，递归由调用方对 arg 再扫描。"""
    i, n = 0, len(expr)
    while i < n:
        m = _IDENT.search(expr, i)
        if not m:
            return
        name = m.group(0)
        j = m.end()
        # 跳过名字后空白
        while j < n and expr[j] in " \t":
            j += 1
        if j < n and expr[j] == "(" and not name.startswith("$"):
            # 收集参数
            depth = 1
            j += 1
            args, cur = [], []
            k = j
            while k < n and depth:
                ch = expr[k]
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        args.append("".join(cur))
                        break
                if ch == "," and depth == 1:
                    args.append("".join(cur))
                    cur = []
                else:
                    cur.append(ch)
                k += 1
            yield name, [a.strip() for a in args]
            i = k + 1
        else:
            i = m.end()


def max_window(expr: str, _seen: int = 0) -> int:
    """表达式内函数调用纯整数参数的最大值（无则 0）。防循环递归上限 64 层。"""
    if _seen > 64:
        return 0
    best = 0
    for _, args in iter_calls(expr):
        for a in args:
            if re.fullmatch(r"-?\d+", a):
                best = max(best, abs(int(a)))
            else:
                best = max(best, max_window(a, _seen + 1))
    return best


def ops_used(expr: str, _seen: int = 0) -> set[str]:
    """表达式引用的全部算子名（含嵌套）。"""
    if _seen > 64:
        return set()
    out: set[str] = set()
    for name, args in iter_calls(expr):
        out.add(name)
        for a in args:
            if not re.fullmatch(r"-?\d+(\.\d+)?", a):
                out |= ops_used(a, _seen + 1)
    return out


def classify(maxw: int, usable: bool) -> str:
    if not usable:
        return "unusable"
    if maxw <= 8:
        return "b1"      # ⊆ b2
    if maxw <= 120:
        return "b2_only"
    return "too_long"


if __name__ == "__main__":  # pragma: no cover - 手工核查
    for e in ["Mean($close, 5)/$close - 1",
              "-1*Corr(Rank(Delta(Log($volume+1),2)), Rank(($close-$open)/$open), 6)",
              "RSI($close, 14)"]:
        print(max_window(e), sorted(ops_used(e)), "|", e[:50])
