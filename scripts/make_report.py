#!/usr/bin/env python
"""从 mlruns 里最新 recorder 的 artifacts 生成 qlib 标准 report 图表（HTML）。

用法：
    conda run -n qlib_ifind_beta python scripts/make_report.py
    conda run -n qlib_ifind_beta python scripts/make_report.py --recorder-id <id>

图表（plotly 交互式 HTML，浏览器打开）：
    01_model_performance_*  模型层：IC 时序 / 累积 IC / 月度 IC 热力图 / 分组累计收益 / QQ / 自相关 / 换手
    02_report               组合层：净值 / 回撤 / 换手 / 成本 / 超额（多 subplot 合一）
    03_risk_analysis_*      风险分析：月度收益 / 年度对比等
    04_score_ic             score IC 时序

依据：qlib examples/workflow_by_code.ipynb + docs/component/report.md
（analysis_model.model_performance_graph / analysis_position.{report,risk_analysis,score_ic}_graph）。
"""
import argparse
import glob
import os
import pickle
import sys
import traceback

import pandas as pd
from qlib.contrib.report import analysis_model, analysis_position

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MLRUNS = os.path.join(PROJECT_ROOT, "mlruns")
OUT = os.path.join(PROJECT_ROOT, "reports")


def find_latest_recorder() -> str:
    """扫 mlruns 找最近一个带 pred.pkl 的 recorder artifacts 目录。"""
    cands = []
    for exp in glob.glob(os.path.join(MLRUNS, "*")):
        if not os.path.isdir(exp) or os.path.basename(exp) == ".trash":
            continue
        for rec in glob.glob(os.path.join(exp, "*")):
            if not os.path.isdir(rec) or os.path.basename(rec) == ".trash":
                continue
            if os.path.exists(os.path.join(rec, "artifacts", "pred.pkl")):
                cands.append((os.path.getmtime(rec), rec))
    if not cands:
        sys.exit(f"❌ 在 {MLRUNS} 下没找到带 pred.pkl 的 recorder")
    return os.path.join(max(cands)[1], "artifacts")


def save_figs(obj, name: str) -> list:
    """obj 可能是单个 Figure 或 list/tuple；统一存 HTML。"""
    if obj is None:
        return []
    figs = list(obj) if isinstance(obj, (list, tuple)) else [obj]
    out = []
    for i, f in enumerate(figs):
        if f is None:
            continue
        path = os.path.join(OUT, f"{name}{'_' + str(i) if len(figs) > 1 else ''}.html")
        # include_plotlyjs=True → plotly.js 内嵌进每个 HTML（~3.5MB/份，离线自包含）。
        # 不用 "cdn"：浏览器访问不到 cdn.plot.ly 会白屏（HTML 框架在但图表不渲染）。
        f.write_html(path, include_plotlyjs=True)
        out.append(path)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recorder-id", default=None, help="指定 recorder_id；缺省自动取最新")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)

    if args.recorder_id:
        # 传 id 时在 mlruns 下定位 artifacts 目录
        rec_dirs = glob.glob(os.path.join(MLRUNS, "*", args.recorder_id, "artifacts"))
        if not rec_dirs:
            sys.exit(f"❌ 找不到 recorder {args.recorder_id}")
        art = rec_dirs[0]
    else:
        art = find_latest_recorder()
    print(f"[make_report] artifacts = {art}")

    pred_df     = pickle.load(open(os.path.join(art, "pred.pkl"), "rb"))
    label_df    = pickle.load(open(os.path.join(art, "label.pkl"), "rb"))
    report_df   = pickle.load(open(os.path.join(art, "portfolio_analysis/report_normal_1day.pkl"), "rb"))
    analysis_df = pickle.load(open(os.path.join(art, "portfolio_analysis/port_analysis_1day.pkl"), "rb"))

    # model_performance_graph / score_ic_graph 要求列名 = ['label','score']
    pred_df.columns = ["score"]
    label_df.columns = ["label"]
    pred_label = pd.concat([label_df, pred_df], axis=1, sort=True).reindex(label_df.index)
    print(f"[make_report] pred={pred_df.shape} label={label_df.shape} "
          f"pred_label(dropna)={pred_label.dropna().shape} report={report_df.shape}")

    jobs = [
        ("01_model_performance", analysis_model.model_performance_graph, (pred_label,)),
        ("02_report",            analysis_position.report_graph,         (report_df,)),
        ("03_risk_analysis",     analysis_position.risk_analysis_graph,  (analysis_df, report_df)),
        ("04_score_ic",          analysis_position.score_ic_graph,       (pred_label,)),
    ]
    all_out = {}
    for name, fn, fn_args in jobs:
        try:
            all_out[name] = save_figs(fn(*fn_args, show_notebook=False), name)
            print(f"✅ {name}: {len(all_out[name])} 张")
        except Exception:
            traceback.print_exc()
            all_out[name] = ["FAILED"]

    print(f"\n=== DONE → {OUT} ===")
    for name, files in all_out.items():
        for f in files:
            print("  ", f)


if __name__ == "__main__":
    main()
