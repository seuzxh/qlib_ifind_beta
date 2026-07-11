"""回撤优化方案对比测试。

5 个方案 + 基线，全部用 90 天滚动重训（361 天 OOS）回测，统一对比表。

方案：
  baseline: champion label + topk10/nd8
  A: label = 收益 - 0.5 × 近5日个股日收益std（波动率惩罚）
  B: label = 收益 / 下行标准差（Sortino）
  D: topk10/nd3（降换手）
  F: topk20/nd8（降集中度）

Run:
  conda run -n qlib_ifind_beta --no-capture-output python -W ignore \
      scripts/compare_drawdown_solutions.py
"""
from __future__ import annotations

import os, sys, pickle
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from qlib_ifind_beta.config import CHAMPION_LABEL_EXPR, OVERLAY_ROOT, UNIVERSE_MARKET

TRAIN_DAYS = 90
VALID_DAYS = 20
STEP = 20
ROLL_START = "2025-01-02"
TEST_END = "2026-07-02"

# --- label 表达式 ---
LABELS = {
    "baseline": "Ref($close, -1) / $price_941 - 1",
    "A_vol_penalty": "Ref($close, -1) / $price_941 - 1 - 0.5 * Std(Ref($close, -1) / $price_941 - 1, 5)",
    # B Sortino 需要自定义表达式，qlib 原生不支持下行标准差 → 用近似
    "B_sortino": "Ref($close, -1) / $price_941 - 1",  # 先占位，单独处理
}

# --- 策略参数 ---
STRATEGIES = {
    "baseline":   {"topk": 10, "n_drop": 8},
    "D_nd3":      {"topk": 10, "n_drop": 3},
    "F_topk20":   {"topk": 20, "n_drop": 8},
}


def _build_task(label_expr: str) -> dict:
    return {
        "model": {"class": "LGBModel", "module_path": "qlib.contrib.model.gbdt",
                  "kwargs": {"loss": "mse", "learning_rate": 0.05, "max_depth": 6,
                             "num_leaves": 64, "num_threads": 20,
                             "lambda_l1": 5.0, "lambda_l2": 10.0,
                             "num_boost_round": 200, "early_stopping_rounds": 20}},
        "dataset": {"class": "DatasetH", "module_path": "qlib.data.dataset",
                    "kwargs": {"handler": {
                        "class": "MinuteEnhancedHandler",
                        "module_path": "qlib_ifind_beta.minute_enhanced_handler",
                        "kwargs": {"instruments": UNIVERSE_MARKET,
                                   "start_time": ROLL_START, "end_time": TEST_END,
                                   "fit_start_time": ROLL_START, "fit_end_time": TEST_END,
                                   "label": [label_expr]}},
                        "segments": {"train": [ROLL_START, ROLL_START],
                                     "valid": [ROLL_START, ROLL_START],
                                     "test": [ROLL_START, ROLL_START]}}},
        "record": [{"class": "SignalRecord", "module_path": "qlib.workflow.record_temp"},
                   {"class": "SigAnaRecord", "module_path": "qlib.workflow.record_temp"}],
    }


def rolling_train(label_expr, exp_name):
    """跑一轮滚动重训，返回拼接的 pred + label。"""
    import qlib
    qlib.init(provider_uri=str(OVERLAY_ROOT), region="cn")
    from qlib.model.trainer import task_train
    from qlib.data import D

    cal = D.calendar(start_time="2024-06-01", end_time=TEST_END, freq="day")
    cal_dates = [pd.Timestamp(d) for d in cal]
    roll_start_idx = cal_dates.index(pd.Timestamp(ROLL_START))
    test_end_idx = cal_dates.index(pd.Timestamp(TEST_END))

    tasks = []
    pos = roll_start_idx
    while pos <= test_end_idx:
        tasks.append({
            "train": (cal_dates[pos - TRAIN_DAYS - VALID_DAYS], cal_dates[pos - VALID_DAYS - 1]),
            "valid": (cal_dates[pos - VALID_DAYS], cal_dates[pos - 1]),
            "test":  (cal_dates[pos], cal_dates[min(pos + STEP - 1, test_end_idx)]),
        })
        pos += STEP

    all_preds, all_labels = [], []
    for i, t in enumerate(tasks):
        task = _build_task(label_expr)
        hk = task["dataset"]["kwargs"]["handler"]["kwargs"]
        hk["start_time"] = t["train"][0].strftime("%Y-%m-%d")
        hk["end_time"] = t["test"][1].strftime("%Y-%m-%d")
        hk["fit_start_time"] = t["train"][0].strftime("%Y-%m-%d")
        hk["fit_end_time"] = t["train"][1].strftime("%Y-%m-%d")
        segs = task["dataset"]["kwargs"]["segments"]
        for key in ("train", "valid", "test"):
            segs[key] = [t[key][0].strftime("%Y-%m-%d"), t[key][1].strftime("%Y-%m-%d")]

        rec = task_train(task, experiment_name=exp_name)
        pred = rec.load_object("pred.pkl")
        label = rec.load_object("label.pkl")
        p = pred.iloc[:, 0] if isinstance(pred, pd.DataFrame) else pred
        l = label.iloc[:, 0] if isinstance(label, pd.DataFrame) else label
        ts, te = t["test"]
        mask = (p.index.get_level_values(0) >= ts) & (p.index.get_level_values(0) <= te)
        all_preds.append(p[mask])
        all_labels.append(l.reindex(p[mask].index))
        print(f"  [{i+1}/{len(tasks)}] done")

    return pd.concat(all_preds), pd.concat(all_labels).reindex(pd.concat(all_preds).index)


def backtest(pred_all, label_all, topk=10):
    """topk equal-weight 回测，返回 metrics dict。"""
    dates = sorted(pred_all.index.get_level_values(0).unique())
    OPEN_COST, CLOSE_COST = 0.0005, 0.0015
    rets = []
    for d in dates:
        dp = pred_all.xs(d, level=0).dropna()
        dl = label_all.xs(d, level=0).dropna()
        common = dp.index.intersection(dl.index)
        if len(common) < topk:
            continue
        top = dp.reindex(common).sort_values(ascending=False).head(topk)
        r = dl.reindex(top.index).values
        rets.append(np.mean((1 + r) * (1 - CLOSE_COST) / (1 + OPEN_COST) - 1))

    s = pd.Series(rets)
    if len(s) == 0:
        return {}
    nav = (1 + s).cumprod()
    dd = nav / nav.cummax() - 1
    # IC
    ics = []
    for d in dates:
        dp = pred_all.xs(d, level=0).dropna()
        dl = label_all.xs(d, level=0).dropna()
        common = dp.index.intersection(dl.index)
        if len(common) < 5:
            continue
        c = dp.reindex(common).corr(dl.reindex(common), method="spearman")
        if pd.notna(c):
            ics.append(c)
    ics = np.array(ics)

    return {
        "IC": f"{ics.mean():.4f}",
        "IC>0%": f"{(ics > 0).mean() * 100:.1f}%",
        "累计超额(net)": f"{(nav.iloc[-1] - 1) * 100:.1f}%",
        "IR(net)": f"{s.mean() / s.std() * np.sqrt(250):.2f}",
        "最大回撤": f"{dd.min() * 100:.1f}%",
        "天数": len(s),
    }


def main():
    results = {}

    # === 需要（重新）训练的方案 ===
    train_schemes = {
        "baseline (收益label)": ("baseline", "Ref($close, -1) / $price_941 - 1", 10),
        "A (收益-0.1×vol5)":    ("A_vol_penalty", "Ref($close, -1) / $price_941 - 1 - 0.1 * Std($close / Ref($close, 1) - 1, 5)", 10),
    }

    trained = {}
    for name, (exp, label_expr, topk) in train_schemes.items():
        print(f"\n{'='*60}")
        print(f"训练: {name}")
        print(f"{'='*60}")
        pred, label = rolling_train(label_expr, f"compare_{exp}")
        trained[name] = (pred, label, topk)
        results[name] = backtest(pred, label, topk)
        print(f"  → IC={results[name]['IC']}, 回撤={results[name]['最大回撤']}, 超额={results[name]['累计超额(net)']}")

    # === 不需重新训练的方案（用 baseline pred 改策略参数）===
    base_pred, base_label, _ = trained["baseline (收益label)"]
    for name, topk in [("D (topk15 降集中)", 15),
                        ("F (topk20 降集中)", 20),
                        ("F2 (topk30 降集中)", 30)]:
        print(f"\n回测: {name} (复用 baseline pred, topk={topk})")
        results[name] = backtest(base_pred, base_label, topk)
        print(f"  → 回撤={results[name]['最大回撤']}, 超额={results[name]['累计超额(net)']}")

    # === 对比表 ===
    print(f"\n\n{'='*80}")
    print("回撤优化方案对比（90 天滚动重训，361 天 OOS）")
    print(f"{'='*80}")
    cols = ["IC", "IC>0%", "累计超额(net)", "IR(net)", "最大回撤", "天数"]
    print(f"{'方案':<25} " + " ".join(f"{c:>14}" for c in cols))
    print("-" * 110)
    for name, m in results.items():
        print(f"{name:<25} " + " ".join(f"{m.get(c, '-'):>14}" for c in cols))

    # 保存
    with open("data/compare_drawdown.pkl", "wb") as f:
        pickle.dump({"results": results, "trained": {k: (p, l) for k, (p, l, _) in trained.items()}}, f)
    print(f"\n结果已保存: data/compare_drawdown.pkl")


if __name__ == "__main__":
    main()
