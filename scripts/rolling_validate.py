"""90 交易日滚动重训验证。

用 qlib RollingGen(step=20, ROLL_SD) 生成滚动任务，train=90 天/valid=20 天/test=20 天，
覆盖 test 2026-04-01~07-02。逐任务训练+预测，拼接所有 test 段 pred → 汇总 IC + 回测，
与 champion 单次训练对比。

Run:
  conda run -n qlib_ifind_beta --no-capture-output python -W ignore \
      scripts/rolling_validate.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pickle
import numpy as np
import pandas as pd

from qlib_ifind_beta.config import (
    CHAMPION_LABEL_EXPR, OVERLAY_ROOT, UNIVERSE_MARKET,
)

# --- 滚动窗口参数 ---
TRAIN_DAYS = 90      # 训练窗口（交易日）
VALID_DAYS = 20      # 验证窗口
STEP = 20            # 滚动步长（每 20 天重训一次）
TEST_DAYS = STEP     # 每个 test 段 = step 天
ROLL_START = "2025-01-02"  # 滚动起始日（2025 第一个交易日）
TEST_END = "2026-07-02"    # 滚动结束日


def _build_task_template() -> dict:
    """构建 champion FROZEN task_template（窗口由 RollingGen 覆盖）。"""
    return {
        "model": {
            "class": "HFLGBModel",
            "module_path": "qlib.contrib.model.highfreq_gdbt_model",
            "kwargs": {
                "loss": "binary", "learning_rate": 0.05, "max_depth": 6,
                "num_leaves": 64, "num_threads": 20,
                "lambda_l1": 5.0, "lambda_l2": 10.0,
            },
        },
        "dataset": {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": {
                    "class": "MinuteEnhancedHandler",
                    "module_path": "qlib_ifind_beta.minute_enhanced_handler",
                    "kwargs": {
                        "instruments": UNIVERSE_MARKET,
                        "start_time": ROLL_START,   # 会被逐任务覆盖
                        "end_time": TEST_END,
                        "fit_start_time": ROLL_START,
                        "fit_end_time": TEST_END,
                        "label": [CHAMPION_LABEL_EXPR],
                    },
                },
                "segments": {
                    "train": [ROLL_START, ROLL_START],
                    "valid": [ROLL_START, ROLL_START],
                    "test":  [ROLL_START, ROLL_START],
                },
            },
        },
        "record": [
            {"class": "SignalRecord", "module_path": "qlib.workflow.record_temp"},
            {"class": "SigAnaRecord", "module_path": "qlib.workflow.record_temp"},
        ],
    }


def main():
    import qlib
    qlib.init(provider_uri=str(OVERLAY_ROOT), region="cn")
    from qlib.workflow.task.gen import RollingGen
    from qlib.model.trainer import task_train
    from qlib.data import D

    EXP = "rolling_90d_validate"

    # 获取交易日历（往前多取半年，确保 train 窗口有足够前置数据）
    cal = D.calendar(start_time="2024-06-01", end_time=TEST_END, freq="day")
    cal_dates = [pd.Timestamp(d) for d in cal]

    # 滚动起点 = ROLL_START（2025 第一个交易日）
    roll_start_idx = cal_dates.index(pd.Timestamp(ROLL_START))
    test_end_idx = cal_dates.index(pd.Timestamp(TEST_END))

    # 手动生成滚动任务
    tasks = []
    pos = roll_start_idx
    while pos <= test_end_idx:
        train_start = cal_dates[pos - TRAIN_DAYS - VALID_DAYS]
        train_end = cal_dates[pos - VALID_DAYS - 1]
        valid_start = cal_dates[pos - VALID_DAYS]
        valid_end = cal_dates[pos - 1]
        test_start = cal_dates[pos]
        test_end_actual = cal_dates[min(pos + TEST_DAYS - 1, test_end_idx)]

        tasks.append({
            "train": (train_start, train_end),
            "valid": (valid_start, valid_end),
            "test": (test_start, test_end_actual),
        })
        pos += STEP

    print(f"▶ 滚动重训验证：{len(tasks)} 个任务，train={TRAIN_DAYS}d/valid={VALID_DAYS}d/step={STEP}")
    for i, t in enumerate(tasks):
        print(f"  [{i+1}] train={t['train'][0].strftime('%Y-%m-%d')}~{t['train'][1].strftime('%Y-%m-%d')} "
              f"valid={t['valid'][0].strftime('%Y-%m-%d')}~{t['valid'][1].strftime('%Y-%m-%d')} "
              f"test={t['test'][0].strftime('%Y-%m-%d')}~{t['test'][1].strftime('%Y-%m-%d')}")

    # 逐任务训练
    template = _build_task_template()
    all_preds = []
    all_labels = []

    for i, t in enumerate(tasks):
        task = _build_task_template()
        hk = task["dataset"]["kwargs"]["handler"]["kwargs"]
        hk["start_time"] = t["train"][0].strftime("%Y-%m-%d")
        hk["end_time"] = t["test"][1].strftime("%Y-%m-%d")
        hk["fit_start_time"] = t["train"][0].strftime("%Y-%m-%d")
        hk["fit_end_time"] = t["train"][1].strftime("%Y-%m-%d")
        segs = task["dataset"]["kwargs"]["segments"]
        segs["train"] = [t["train"][0].strftime("%Y-%m-%d"), t["train"][1].strftime("%Y-%m-%d")]
        segs["valid"] = [t["valid"][0].strftime("%Y-%m-%d"), t["valid"][1].strftime("%Y-%m-%d")]
        segs["test"] = [t["test"][0].strftime("%Y-%m-%d"), t["test"][1].strftime("%Y-%m-%d")]

        print(f"\n[{i+1}/{len(tasks)}] 训练中...")
        rec = task_train(task, experiment_name=EXP)

        pred = rec.load_object("pred.pkl")
        label = rec.load_object("label.pkl")
        p = pred.iloc[:, 0] if isinstance(pred, pd.DataFrame) else pred
        l = label.iloc[:, 0] if isinstance(label, pd.DataFrame) else label

        # 只取 test 段
        level = 0
        ts = t["test"][0]
        te = t["test"][1]
        mask = (p.index.get_level_values(level) >= ts) & (p.index.get_level_values(level) <= te)
        all_preds.append(p[mask])
        all_labels.append(l.reindex(p[mask].index))
        print(f"  test pred: {mask.sum()} 行")

    # 拼接
    pred_all = pd.concat(all_preds)
    label_all = pd.concat(all_labels).reindex(pred_all.index)
    print(f"\n✅ 拼接完成：{len(pred_all)} 行，{pred_all.index.get_level_values(0).nunique()} 天")

    # 汇总 IC
    dates = sorted(pred_all.index.get_level_values(0).unique())
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

    print(f"\n=== 滚动重训汇总 IC ===")
    print(f"IC:        {ics.mean():.4f}")
    print(f"ICIR:      {ics.mean()/ics.std():.2f}")
    print(f"IC>0 rate: {(ics>0).mean()*100:.1f}%")
    print(f"天数:      {len(ics)}")

    # 回测（top10 equal-weight）
    OPEN_COST, CLOSE_COST, TOPK = 0.0005, 0.0015, 10
    rets_net, rets_gross = [], []
    for d in dates:
        dp = pred_all.xs(d, level=0).dropna()
        dl = label_all.xs(d, level=0).dropna()
        common = dp.index.intersection(dl.index)
        if len(common) < TOPK:
            continue
        top = dp.reindex(common).sort_values(ascending=False).head(TOPK)
        r = dl.reindex(top.index).values
        rets_gross.append(np.mean(r))
        rets_net.append(np.mean((1+r)*(1-CLOSE_COST)/(1+OPEN_COST)-1))
    s_net = pd.Series(rets_net)
    s_gross = pd.Series(rets_gross)

    # 基准
    bench = D.features(["SH000300"], ["$close"], start_time=ROLL_START, end_time=TEST_END)
    bench_ret = bench["$close"].pct_change().groupby(level="datetime").first()

    print(f"\n=== 回测（top10 equal-weight, {len(s_net)} 天）===")
    print(f"累计超额(net):   {((1+s_net).prod()-1)*100:.1f}%")
    print(f"累计超额(gross): {((1+s_gross).prod()-1)*100:.1f}%")
    print(f"IR(net):         {s_net.mean()/s_net.std()*np.sqrt(250):.2f}")
    print(f"最大回撤(net):   {((1+s_net).cumprod()/(1+s_net).cumprod().cummax()-1).min()*100:.2f}%")

    # 保存结果
    out = Path("data/rolling_90d_result.pkl")
    with open(out, "wb") as f:
        pickle.dump({"pred": pred_all, "label": label_all, "ics": ics}, f)
    print(f"\n结果已保存: {out}")


if __name__ == "__main__":
    main()
