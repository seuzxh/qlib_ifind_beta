"""Research A/B: overnight_gap 口径对照 — 名义(现役) vs 后复权(变体)。

单变量实验：唯一差异是 overnight_gap 的计算口径。现役 Champion 用物化的
名义口径 bin（open[T]/f[T] / (close[T-1]/f[T-1]) - 1，每天各除各的 factor）；
变体直接用后复权表达式 $open/Ref($close,1)-1。

⚠️ 2026-09-11 发现（backtest-log/2026-09-11-gap-caliber-ab-and-factor-drift.md）：
本数据源 factor 序列逐日微漂 ±0.1%~0.3%（非阶梯函数），因此两口径在多数样本
上存在小幅差异（test 段 |diff|>1e-6 占 ~95%，中位 7e-4），并非"仅除权日不同"；
后复权序列才是内部自洽口径（一字板日 gap 精确为 0 可证）。

其余全部冻结：17 因子、label、HFLGB 超参、切分、TD0 策略、Exchange 成本，
与 qrun/workflow_minute_enhanced_tk10_nd8.yaml 逐字段相同。结论仅供研究，
不晋升、不改动 Champion。

Run:
  conda run -n qlib_ifind_beta python scripts/validate_gap_adjusted.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import qlib
from ruamel.yaml import YAML

from qlib_ifind_beta.config import CHAMPION_RECORDER_ID, OVERLAY_ROOT
from qlib_ifind_beta.minute_enhanced_handler import MinuteEnhancedHandler

ADJ_EXPR = "($open / Ref($close, 1)) - 1"
EXPERIMENT = "gap_adjusted_variant"


class MinuteEnhancedAdjGapHandler(MinuteEnhancedHandler):
    """Champion handler，仅把 overnight_gap 字段换成后复权表达式。"""

    def get_feature_config(self):
        fields, names = super().get_feature_config()
        return [ADJ_EXPR if n == "overnight_gap" else f for f, n in zip(fields, names)], names


def _coerce_limit_threshold(task: dict) -> None:
    for rec in task.get("record", []):
        if rec.get("class") != "PortAnaRecord":
            continue
        exk = ((rec.get("kwargs", {}).get("config") or {}).get("backtest") or {}).get("exchange_kwargs") or {}
        if isinstance(exk.get("limit_threshold"), list):
            exk["limit_threshold"] = tuple(exk["limit_threshold"])


def _daily_ic(score: pd.Series, label: pd.Series) -> float:
    def norm(s):
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        if set(s.index.names) == {"instrument", "datetime"}:
            s = s.reorder_levels(["datetime", "instrument"])
        return s.sort_index()
    s, y = norm(score), norm(label)
    common = s.dropna().index.intersection(y.dropna().index)
    s, y = s.loc[common], y.loc[common]
    ics = [s.xs(d).corr(y.xs(d)) for d in s.index.get_level_values("datetime").unique()]
    ics = [v for v in ics if pd.notna(v)]
    return float(pd.Series(ics).mean())


def main() -> None:
    qlib.init(provider_uri=str(OVERLAY_ROOT), region="cn")
    from qlib.data import D
    from qlib.model.trainer import task_train
    from qlib.workflow import R

    # ① 单因子层：两种口径的 overnight_gap 各自对 label 的日度 IC（test 段）
    print("▶ ① 单因子 IC 对照（test 2026-04-01→07-02）")
    uni = D.instruments(market="highbeta883926")
    df = D.features(uni, ["$overnight_gap", ADJ_EXPR, "Ref($close, -1) / $price_941 - 1"],
                    start_time="2026-04-01", end_time="2026-07-02")
    df.columns = ["gap_raw", "gap_adj", "label"]
    for col in ["gap_raw", "gap_adj"]:
        print(f"  {col}: 日均IC = {_daily_ic(df[col], df['label']):+.4f}")
    diff = (df["gap_raw"] - df["gap_adj"]).abs()
    print(f"  |diff|>1e-6 实质差异: {(diff > 1e-6).sum()} / {len(df)}"
          f"（max={diff.max():.4f}, p99={diff.quantile(0.99):.4f}；源数据 factor 逐日漂移所致）")

    # ② 全流程 A/B：变体 handler 完整训练 + 回测
    print("\n▶ ② 全流程变体训练（唯一变量 overnight_gap 口径）")
    cfg = YAML(typ="safe").load(open(ROOT / "qrun" / "workflow_minute_enhanced_tk10_nd8.yaml"))
    task = cfg["task"]
    handler = task["dataset"]["kwargs"]["handler"]
    handler["class"] = "MinuteEnhancedAdjGapHandler"
    handler["module_path"] = "scripts.validate_gap_adjusted"
    _coerce_limit_threshold(task)
    rec = task_train(task, experiment_name=EXPERIMENT)
    rid = getattr(rec, "recorder_id", None) or rec.id
    print(f"  variant recorder: {rid}")

    # ③ 对照报告
    champ = R.get_recorder(recorder_id=CHAMPION_RECORDER_ID, experiment_name="minute_enhanced_tk10_nd8")
    pred_v, pred_c = rec.load_object("pred.pkl"), champ.load_object("pred.pkl")
    norm = lambda s: (s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s).reorder_levels(
        ["datetime", "instrument"]).sort_index()
    pv, pc = norm(pred_v), norm(pred_c)
    common = pv.index.intersection(pc.index)
    maxdiff = float((pv.loc[common] - pc.loc[common]).abs().max())
    top_eq = sum(set(pv.xs(d).sort_values(ascending=False).head(10).index) ==
                 set(pc.xs(d).sort_values(ascending=False).head(10).index)
                 for d in set(pv.index.get_level_values("datetime")))

    def port_metrics(recorder):
        rep = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
        s, b = rep["return"].fillna(0), rep["bench"].fillna(0)
        cum = (1 + s).prod() - 1
        ann = (1 + cum) ** (252 / len(s)) - 1
        dd = ((1 + s).cumprod() / (1 + s).cumprod().cummax() - 1).min()
        exc = (1 + (s - b)).prod() - 1
        ir = (s - b).mean() / (s - b).std() * (252 ** 0.5)
        return {"策略累计": f"{cum:+.2%}", "策略年化": f"{ann:+.1%}", "超额累计": f"{exc:+.2%}",
                "IR(日频年化)": f"{ir:.2f}", "最大回撤": f"{dd:.2%}"}

    print("\n▶ ③ 结果对照（同窗 test 2026-04→07，含成本）")
    print(f"{'指标':<14}{ 'Champion(不复权gap)':>20}{ 'Variant(后复权gap)':>20}")
    mv, mc = rec.list_metrics(), champ.list_metrics()
    for m in ["IC", "Rank IC"]:
        print(f"{m:<14}{mc.get(m, float('nan')):>20.4f}{mv.get(m, float('nan')):>20.4f}")
    pm_c, pm_v = port_metrics(champ), port_metrics(rec)
    for k in pm_c:
        print(f"{k:<14}{pm_c[k]:>20}{pm_v[k]:>20}")
    print(f"\npred 对照: common={len(common)} max_abs_diff={maxdiff:.3e} "
          f"Top10 完全一致天数={top_eq}/{pv.index.get_level_values('datetime').nunique()}")
    print("结论: 研究口径仅供对照，Champion 不变；如需采纳须走候选验证与人工晋升。")


if __name__ == "__main__":
    main()
