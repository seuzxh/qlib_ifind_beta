"""Phase 0.2 算子兼容冒烟 — vendored zoo 表达式在 pyqlib 0.9.7 的可行性。

两步：
  ① 数值对照：抽 5 个 zoo 自定义算子表达式在 day 数据上求值，与手工
     numpy 实现逐位对比（SMA 递归/TsArgmax/RSI/ATR/BOLL_UP）。
  ② 全量解析率：六库 1007 个表达式逐个在 3 只股票 × 60 日上求值，
     统计 OK / 报错清单（报错=路线 A 不可用因子，需在筛选中剔除）。

结论写入 docs/backtest-log/2026-09-24-factor-zoo-screen.md §0.2。
Run: conda run -n qlib_ifind_beta python scripts/validate_factor_zoo_smoke.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import qlib

from qlib_ifind_beta.config import OVERLAY_ROOT
from qlib_ifind_beta.factor.factor_zoo import register_zoo_ops, iter_all_factors

OUT = ROOT / "reports" / "factor_zoo" / "factor_zoo_smoke.json"


def _pick_universe_stocks(k: int) -> list[str]:
    """从 universe 文件里取跨度最长的 k 只（冒烟要在真实成员上跑）。"""
    lines = (OVERLAY_ROOT / "instruments" / "highbeta883926.txt").read_text().splitlines()
    spans: dict[str, int] = {}
    for ln in lines:
        parts = ln.split("\t")
        if len(parts) >= 3:
            spans[parts[0]] = spans.get(parts[0], 0) + 1
    top = sorted(spans.items(), key=lambda kv: -kv[1])[:k]
    print(f"universe 冒烟股票: {top}")
    return [s for s, _ in top]


def manual_sma(x: pd.Series, n: int, m: int) -> pd.Series:
    """上游 SMA 口径：每个滚动窗口内递归（seed=窗口首值，跳过 NaN）。"""
    def _sma(arr: np.ndarray) -> float:
        if np.isnan(arr).all():
            return np.nan
        result = arr[0]
        for i in range(1, len(arr)):
            if not np.isnan(arr[i]):
                result = (m * arr[i] + (n - m) * result) / n
        return result
    return x.rolling(n, min_periods=1).apply(_sma, raw=True)


def manual_rsi(x: pd.Series, n: int) -> pd.Series:
    """上游 RSI 口径：ewm(alpha=1/N) 的 up/dn 比 ×100。"""
    diff = x.diff(1)
    up = diff.where(diff > 0, 0.0)
    dn = (-diff).where(diff < 0, 0.0)
    su = up.ewm(alpha=1.0 / n, adjust=False).mean()
    sd = dn.ewm(alpha=1.0 / n, adjust=False).mean()
    return su / (sd + 1e-12) * 100.0


def main() -> None:
    qlib.init(provider_uri=str(OVERLAY_ROOT), region="cn")
    register_zoo_ops()
    from qlib.data import D

    inst = _pick_universe_stocks(3)
    start, end = "2025-01-01", "2025-06-30"
    warm_start = "2024-06-01"  # 与 ② 相同预热；参照必须含 warmup 才与引擎同口径
    raw = D.features(inst, ["$close", "$high", "$low"], start_time=warm_start, end_time=end)

    # ① 数值对照（逐股 .xs 对齐，避免多索引拼接问题）
    cases = [
        ("SMA($close, 12, 2)", lambda d: manual_sma(d["$close"], 12, 2)),
        ("TsArgmax($close, 10)", lambda d: d["$close"].rolling(10, min_periods=1).max()),
        ("RSI($close, 14)", lambda d: manual_rsi(d["$close"], 14)),
        ("ATR($close, $high, $low, 14)", None),  # 仅核对可求值+有限值
        ("BOLL_UP($close, 26, 2)", None),
    ]
    print("▶ ① 数值对照")
    numeric_ok = {}
    for expr, ref_fn in cases:
        try:
            got_all = D.features(inst, [expr], start_time=start, end_time=end)
            if ref_fn is not None:
                diffs, n_cmp = [], 0
                for s in inst:
                    g = got_all.xs(s, level="instrument").iloc[:, 0].dropna()
                    e_full = ref_fn(raw.xs(s, level="instrument"))
                    e = e_full.reindex(g.index).dropna()
                    common = g.index.intersection(e.index)
                    if len(common):
                        diffs.append((g.loc[common] - e.loc[common]).abs().max())
                        n_cmp += len(common)
                diff = max(diffs)
                numeric_ok[expr] = {"max_abs_diff": float(diff), "n": n_cmp}
                print(f"  {expr:28s} n={n_cmp:4d} max|diff|={diff:.3e}")
            else:
                finite = int(got_all.iloc[:, 0].dropna().shape[0])
                numeric_ok[expr] = {"finite_n": finite}
                print(f"  {expr:28s} 可求值 finite_n={finite}")
        except Exception as e:  # noqa: BLE001
            numeric_ok[expr] = {"error": f"{type(e).__name__}: {e}"}
            print(f"  {expr:28s} ERROR {type(e).__name__}: {str(e)[:80]}")

    # ② 全量解析率
    print("\n▶ ② 六库全量求值（3 股 × 60 日窗口 + 60 日预热）")
    allf = iter_all_factors()
    ok, failed = [], {}
    warm_start = "2024-06-01"
    for lib, name, expr in allf:
        try:
            df = D.features(inst, [expr], start_time=warm_start, end_time=end)
            if df.shape[1] != 1:
                raise ValueError(f"expected 1 col, got {df.shape[1]}")
            ok.append((lib, name))
        except Exception as e:  # noqa: BLE001
            failed[f"{lib}.{name}"] = f"{type(e).__name__}: {str(e)[:100]}"
    by_lib = pd.Series([l for l, _ in ok]).value_counts().to_dict()
    print(f"  可求值 {len(ok)}/{len(allf)}")
    for lib, cnt in sorted(by_lib.items()):
        print(f"    {lib:9s} {cnt}")
    fails_by_type = pd.Series([v.split(':')[0] for v in failed.values()]).value_counts()
    print("  报错类型分布:", fails_by_type.to_dict())
    for k, v in list(failed.items())[:10]:
        print(f"    例: {k} → {v[:90]}")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({
        "numeric_cases": numeric_ok, "ok_count": len(ok), "total": len(allf),
        "ok_by_lib": by_lib, "failed": failed,
    }, ensure_ascii=False, indent=1))
    print(f"\nsaved → {OUT}")


if __name__ == "__main__":
    main()
