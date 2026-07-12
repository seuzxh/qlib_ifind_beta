"""模型对比：4 模型 × 90 天滚动重训（361 天 OOS）。

LGBModel / HFLGBModel(binary) / LinearModel / DEnsembleModel，
从 2025-01-02 起滚动 19 个任务，统一 top10 equal-weight 回测对比。

Run:
  conda run -n qlib_ifind_beta --no-capture-output python -W ignore \
      scripts/compare_models.py
"""
from __future__ import annotations

import os, sys, pickle, traceback
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

MODELS = {
    "LGBModel": {
        "class": "LGBModel", "module_path": "qlib.contrib.model.gbdt",
        "kwargs": {"loss": "mse", "learning_rate": 0.05, "max_depth": 6,
                   "num_leaves": 64, "num_threads": 20,
                   "lambda_l1": 5.0, "lambda_l2": 10.0,
                   "num_boost_round": 200, "early_stopping_rounds": 20},
    },
    "HFLGBModel": {
        "class": "HFLGBModel", "module_path": "qlib.contrib.model.highfreq_gdbt_model",
        "kwargs": {"loss": "binary", "learning_rate": 0.05, "max_depth": 6,
                   "num_leaves": 64, "num_threads": 20,
                   "lambda_l1": 5.0, "lambda_l2": 10.0},
    },
    "XGBModel": {
        "class": "XGBModel", "module_path": "qlib.contrib.model.xgboost",
        "kwargs": {"eta": 0.05, "max_depth": 6, "n_estimators": 200,
                   "reg_alpha": 5.0, "reg_lambda": 10.0,
                   "n_jobs": 20, "early_stopping_rounds": 20},
    },
    "CatBoostModel": {
        "class": "CatBoostModel", "module_path": "qlib.contrib.model.catboost_model",
        "kwargs": {"iterations": 200, "learning_rate": 0.05, "depth": 6,
                   "l2_leaf_reg": 10.0, "silent": True,
                   "early_stopping_rounds": 20},
    },
    "LinearModel": {
        "class": "LinearModel", "module_path": "qlib.contrib.model.linear",
        "kwargs": {},
    },
}


def _build_task(model_cfg, t):
    """构建 task，窗口由 t 覆盖。"""
    return {
        "model": model_cfg,
        "dataset": {"class": "DatasetH", "module_path": "qlib.data.dataset",
                    "kwargs": {"handler": {
                        "class": "MinuteEnhancedHandler",
                        "module_path": "qlib_ifind_beta.minute_enhanced_handler",
                        "kwargs": {"instruments": UNIVERSE_MARKET,
                                   "start_time": t["train"][0].strftime("%Y-%m-%d"),
                                   "end_time": t["test"][1].strftime("%Y-%m-%d"),
                                   "fit_start_time": t["train"][0].strftime("%Y-%m-%d"),
                                   "fit_end_time": t["train"][1].strftime("%Y-%m-%d"),
                                   "label": [CHAMPION_LABEL_EXPR]}},
                        "segments": {
                            "train": [t["train"][0].strftime("%Y-%m-%d"), t["train"][1].strftime("%Y-%m-%d")],
                            "valid": [t["valid"][0].strftime("%Y-%m-%d"), t["valid"][1].strftime("%Y-%m-%d")],
                            "test":  [t["test"][0].strftime("%Y-%m-%d"), t["test"][1].strftime("%Y-%m-%d")]}}},
        "record": [{"class": "SignalRecord", "module_path": "qlib.workflow.record_temp"},
                   {"class": "SigAnaRecord", "module_path": "qlib.workflow.record_temp"}],
    }


def rolling_train(model_name, model_cfg):
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
        try:
            task = _build_task(model_cfg, t)
            rec = task_train(task, experiment_name=f"model_cmp_{model_name.lower()}")
            pred = rec.load_object("pred.pkl")
            label = rec.load_object("label.pkl")
            p = pred.iloc[:, 0] if isinstance(pred, pd.DataFrame) else pred
            l = label.iloc[:, 0] if isinstance(label, pd.DataFrame) else label
            ts, te = t["test"]
            mask = (p.index.get_level_values(0) >= ts) & (p.index.get_level_values(0) <= te)
            all_preds.append(p[mask])
            all_labels.append(l.reindex(p[mask].index))
        except Exception as e:
            print(f"  [{i+1}/{len(tasks)}] ❌ {e}")
            continue
    if not all_preds:
        return None, None
    pred_all = pd.concat(all_preds)
    label_all = pd.concat(all_labels).reindex(pred_all.index)
    return pred_all, label_all


def backtest(pred_all, label_all, topk=10):
    """top10 equal-weight 回测。"""
    OPEN_COST, CLOSE_COST = 0.0005, 0.0015
    dates = sorted(pred_all.index.get_level_values(0).unique())
    rets, ics = [], []
    for d in dates:
        dp = pred_all.xs(d, level=0).dropna()
        dl = label_all.xs(d, level=0).dropna()
        common = dp.index.intersection(dl.index)
        if len(common) < topk:
            continue
        top = dp.reindex(common).sort_values(ascending=False).head(topk)
        r = dl.reindex(top.index).values
        rets.append(np.mean((1 + r) * (1 - CLOSE_COST) / (1 + OPEN_COST) - 1))
        c = dp.reindex(common).corr(dl.reindex(common), method="spearman")
        if pd.notna(c):
            ics.append(c)

    s = pd.Series(rets)
    nav = (1 + s).cumprod()
    dd = nav / nav.cummax() - 1
    ics = np.array(ics)
    return {
        "IC": f"{ics.mean():.4f}",
        "ICIR": f"{ics.mean()/ics.std():.2f}" if ics.std() > 0 else "—",
        "IC>0%": f"{(ics>0).mean()*100:.1f}%",
        "超额(net)": f"{(nav.iloc[-1]-1)*100:.1f}%",
        "IR(net)": f"{s.mean()/s.std()*np.sqrt(250):.2f}" if s.std() > 0 else "—",
        "最大回撤": f"{dd.min()*100:.1f}%",
        "Calmar": f"{(nav.iloc[-1]-1)/abs(dd.min()):.2f}" if dd.min() != 0 else "—",
        "天数": len(s),
    }


def main():
    results = {}

    for name, cfg in MODELS.items():
        print(f"\n{'='*60}")
        print(f"模型: {name}")
        print(f"{'='*60}")
        try:
            pred, label = rolling_train(name, cfg)
            if pred is None:
                results[name] = {"error": "所有任务失败"}
                print(f"  ❌ 所有任务失败")
                continue
            results[name] = backtest(pred, label)
            m = results[name]
            print(f"  → IC={m['IC']}, 超额={m['超额(net)']}, 回撤={m['最大回撤']}, Calmar={m['Calmar']}")
        except Exception as e:
            results[name] = {"error": str(e)[:80]}
            print(f"  ❌ {e}")
            traceback.print_exc()

    # 对比表
    print(f"\n\n{'='*90}")
    print("模型对比（90 天滚动重训，361 天 OOS，top10 equal-weight）")
    print(f"{'='*90}")
    cols = ["IC", "ICIR", "IC>0%", "超额(net)", "IR(net)", "最大回撤", "Calmar", "天数"]
    print(f"{'模型':<20} " + " ".join(f"{c:>12}" for c in cols))
    print("-" * 120)
    for name, m in results.items():
        if "error" in m:
            print(f"{name:<20} ❌ {m['error']}")
        else:
            print(f"{name:<20} " + " ".join(f"{m.get(c, '-'):>12}" for c in cols))

    with open("data/model_comparison_rolling.pkl", "wb") as f:
        pickle.dump(results, f)
    print("\n结果已保存: data/model_comparison_rolling.pkl")


if __name__ == "__main__":
    main()
