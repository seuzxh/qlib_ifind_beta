"""Phase 2-alpha — 18+k augment 冻结协议验证（timebox）。

流程：
  ① 从筛选产物加载 shortlist 特征（b1__/b2_t1__/day__ 前缀自动路由）；
  ② 物化为 overlay 研究 bin `fz_<name>.day.bin`（additive，不动现役 bin；
     沿用 xd*/e94* 研究 bin 先例）；
  ③ 克隆 Champion yaml，仅替换 handler 类（18 因子 + k 个 fz 字段），
     task_train 全流程跑通；
  ④ 对照：冻结 Champion recorder + 同批未改动 handler 重训基线（excluded.json
     方法论规则 1：单次运行 ±10pp 级噪声，同批基线必跑）。
  结论只作排序参考；purged 19 段终审不在本 timebox 内（另行安排）。

Run:
  conda run -n qlib_ifind_beta python scripts/validate_factor_zoo_augment.py \
      --features b1__alpha360__CLOSE05 --baseline
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import qlib
from ruamel.yaml import YAML

from qlib_ifind_beta import binio
from qlib_ifind_beta.config import CHAMPION_RECORDER_ID, OVERLAY_ROOT
from qlib_ifind_beta.minute_enhanced_handler import MinuteEnhancedHandler

OUTDIR = ROOT / "reports" / "factor_zoo"
EXPERIMENT = "factor_zoo_augment"

#: 由 --features 填充；handler 类在 task_train 进程内实例化，全局即可。
FZ_FIELDS: list[str] = []


def safe_name(name: str) -> str:
    return "fz_" + re.sub(r"[^A-Za-z0-9_]", "_", name)


def load_feature(spec: str) -> pd.Series:
    """按前缀路由：b1__ / b2_t1__ / day__（路线A chunk）。"""
    if spec.startswith("b1__"):
        df: pd.DataFrame = pd.read_pickle(OUTDIR / "minute_b1.pkl")
        return df[spec[4:]]
    if spec.startswith("b2_t1__"):
        eod: pd.DataFrame = pd.read_pickle(OUTDIR / "minute_eod.pkl")
        meta = pd.read_pickle(OUTDIR / "dayf_meta.pkl")
        day_cal = sorted(meta.index.get_level_values("datetime").unique())
        col = spec[7:]
        parts = []
        for s, g in eod[[col]].groupby(level="instrument"):
            parts.append(g.droplevel("instrument").reindex(day_cal).shift(1))
        f = pd.concat(parts).rename_axis(["instrument", "datetime"]).swaplevel()
        f.index = f.index.set_levels(pd.to_datetime(f.index.levels[0]), level="datetime")
        return f
    if spec.startswith("day__"):
        target = spec[5:]
        for cf in sorted(OUTDIR.glob("dayf_chunk_*.pkl")):
            df = pd.read_pickle(cf)
            if target in df.columns:
                return df[target]
        raise KeyError(target)
    if spec.startswith("accel__"):
        df = pd.read_pickle(OUTDIR / "accel_state.pkl")
        return df[spec[7:]]
    raise ValueError(spec)


def materialize(spec: str) -> str:
    """特征 Series → overlay fz_<name>.day.bin（additive）。返回字段名。"""
    f = load_feature(spec)
    fname = safe_name(spec)
    day_cal = [d.strip() for d in (OVERLAY_ROOT / "calendars" / "day.txt").read_text().splitlines() if d.strip()]
    cal_pos = {d: i for i, d in enumerate(day_cal)}
    for s, g in f.groupby(level="instrument"):
        gg = g.droplevel("instrument")
        gg.index = pd.to_datetime(gg.index).strftime("%Y-%m-%d")
        vals = np.full(len(day_cal), np.nan, dtype=np.float32)
        for d, v in gg.items():
            i = cal_pos.get(d)
            if i is not None and pd.notna(v):
                vals[i] = np.float32(v)
        out = OVERLAY_ROOT / "features" / s / f"{fname}.day.bin"
        out.parent.mkdir(parents=True, exist_ok=True)
        start_idx = int(np.argmax(~np.isnan(vals))) if (~np.isnan(vals)).any() else 0
        binio.write_bin(out, start_idx, vals[start_idx:])
    print(f"  物化 {fname} 完成（{f.groupby(level='instrument').ngroups} 股）")
    return fname


class MinuteEnhancedFZHandler(MinuteEnhancedHandler):
    """Champion handler + fz 研究字段（FZ_FIELDS 注入）。"""

    def get_feature_config(self):
        fields, names = super().get_feature_config()
        return fields + [f"${n}" for n in FZ_FIELDS], names + list(FZ_FIELDS)


def port_metrics(recorder) -> dict:
    rep = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
    s, b = rep["return"].fillna(0), rep["bench"].fillna(0)
    cum = (1 + s).prod() - 1
    ann = (1 + cum) ** (252 / len(s)) - 1
    dd = ((1 + s).cumprod() / (1 + s).cumprod().cummax() - 1).min()
    exc = (1 + (s - b)).prod() - 1
    ir = (s - b).mean() / (s - b).std() * 252 ** 0.5
    return {"累计": f"{cum:+.1%}", "年化": f"{ann:+.1%}", "超额": f"{exc:+.1%}",
            "IR": f"{ir:.2f}", "回撤": f"{dd:.1%}",
            "Calmar": f"{ann/abs(dd):.2f}" if dd else "inf"}


def run_task(handler_class: str, module_path: str = "scripts.validate_factor_zoo_augment") -> object:
    from qlib.model.trainer import task_train
    cfg = YAML(typ="safe").load(open(ROOT / "qrun" / "workflow_minute_enhanced_tk10_nd8.yaml"))
    task = cfg["task"]
    h = task["dataset"]["kwargs"]["handler"]
    h["class"] = handler_class
    h["module_path"] = module_path
    for rec in task.get("record", []):
        if rec.get("class") == "PortAnaRecord":
            exk = ((rec.get("kwargs", {}).get("config") or {}).get("backtest") or {}).get("exchange_kwargs") or {}
            if isinstance(exk.get("limit_threshold"), list):
                exk["limit_threshold"] = tuple(exk["limit_threshold"])
    return task_train(task, experiment_name=EXPERIMENT)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="逗号分隔的筛选特征名")
    ap.add_argument("--baseline", action="store_true", help="附带同批未改动 handler 基线")
    a = ap.parse_args()
    specs = a.features.split(",")

    qlib.init(provider_uri=str(OVERLAY_ROOT), region="cn")
    from qlib.workflow import R

    print("▶ ① 物化研究 bin")
    FZ_FIELDS.clear()
    FZ_FIELDS.extend(materialize(sp) for sp in specs)
    print(f"   handler = 18 + {len(FZ_FIELDS)}: {FZ_FIELDS}")

    print("▶ ② 同批基线（未改动 18 因子，同配置重训）" if a.baseline else "▶ ② 跳过同批基线")
    if a.baseline:
        base_rec = run_task("MinuteEnhancedHandler",
                            "qlib_ifind_beta.minute_enhanced_handler")
        print("  baseline:", port_metrics(base_rec))

    print("▶ ③ augment 训练")
    rec = run_task("MinuteEnhancedFZHandler")
    rid = getattr(rec, "recorder_id", None) or rec.id
    print("  augment:", port_metrics(rec))
    m = rec.list_metrics()
    print(f"  IC={m.get('IC'):.4f} RankIC={m.get('Rank IC'):.4f} recorder={rid}")

    champ = R.get_recorder(recorder_id=CHAMPION_RECORDER_ID, experiment_name="minute_enhanced_tk10_nd8")
    print("▶ ④ 冻结 Champion 同窗:", port_metrics(champ))


if __name__ == "__main__":
    main()
