"""Phase 1A 路线 A — 六库日频表达式 T-1 lag 广谱筛选。

  stage=feat   分块计算 Ref(<expr>,1) 特征（999 个可用表达式），存 reports/
               factor_zoo/dayf_chunk_*.pkl + meta（label/18 因子）。
  stage=metrics 汇总指标：日度 Rank IC 均值/IR/t/覆盖率 + 残差 IC（对 18 因子
               逐日横截面残差化）+ 两半窗方向一致性 + graveyard 语义标记。

口径说明：
- universe 按日成分（market=highbeta883926），特征从 2024-01-02 起算——
  长窗因子在 2024 年初有 NaN 预热，IC 只计有效样本，覆盖率如实报告。
- Ref(·,1) 防 T 日前视；label = Champion v2（Ref($close1500,-1)/$close0941-1）。
- 运行噪声教训（excluded.json 规则 1）：本脚本只做排序与淘汰，不做终判。

Run:
  conda run -n qlib_ifind_beta python scripts/validate_factor_zoo_screen.py feat
  conda run -n qlib_ifind_beta python scripts/validate_factor_zoo_screen.py metrics
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import qlib

from qlib_ifind_beta.config import CHAMPION_LABEL_EXPR, OVERLAY_ROOT, UNIVERSE_MARKET
from qlib_ifind_beta.factor.factor_zoo import iter_all_factors, register_zoo_ops
from qlib_ifind_beta.experiment.screen_lib import residual_ic, summarize

OUTDIR = ROOT / "reports" / "factor_zoo"
CHUNK = 50
START, END = "2024-01-02", "2026-09-24"

#: graveyard 语义标记（名称规则近似，只标记不剔除）
_GRAVE_PATTERNS = [
    ("alpha158_y", re.compile(r"^(KMID|KLEN|KUP|KLOW|KSFT|OPEN|HIGH|LOW|VWAP|CLOSE|VOLUME|ROC|MA|STD|BETA|RSQR|RESI|MAX|MIN|QTLU|QTLD|RANK|RSV|IMAX|IMIN|IMXD|CORR|CORD|CNTP|CNTN|CNTD|SUMP|SUMN|SUMD|VMA|VSTD|WVMA|VSUMP|VSUMN|VSUMD)\d+$")),
    ("cross_period_daily", re.compile(r"(ROC|MOM|Price1M|Price3M|Price1Y|STD|VSTD|Variance|hist_sigma)", re.I)),
]


def _universe_features(fields: list[str], names: list[str]) -> pd.DataFrame:
    from qlib.data import D
    uni = D.instruments(market=UNIVERSE_MARKET)
    df = D.features(uni, fields, start_time=START, end_time=END)
    df.columns = names[: df.shape[1]]
    return df


def stage_feat() -> None:
    qlib.init(provider_uri=str(OVERLAY_ROOT), region="cn")
    register_zoo_ops()
    smoke = json.loads((OUTDIR / "factor_zoo_smoke.json").read_text())
    failed = set(smoke["failed"])
    rows = [(lib, name, expr) for lib, name, expr in iter_all_factors()
            if f"{lib}.{name}" not in failed]
    print(f"可用表达式 {len(rows)}，分 {int(np.ceil(len(rows)/CHUNK))} 块")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    names_all: list[str] = []
    t0 = time.time()
    for i in range(0, len(rows), CHUNK):
        part = rows[i:i + CHUNK]
        fields = [f"Ref({expr}, 1)" for _, _, expr in part]
        names = [f"{lib}__{name}" for lib, name, _ in part]
        try:
            df = _universe_features(fields, names)
        except Exception as e:  # noqa: BLE001 - 单块失败降级为逐表达式
            print(f"  chunk {i//CHUNK}: 整块失败({type(e).__name__})，逐表达式重试")
            pieces = []
            for (lib, name, expr), f, n in zip(part, fields, names):
                try:
                    pieces.append(_universe_features([f], [n]))
                except Exception as e2:  # noqa: BLE001
                    print(f"    弃用 {n}: {type(e2).__name__} {str(e2)[:60]}")
            df = pd.concat(pieces, axis=1) if pieces else pd.DataFrame()
        if df.shape[1]:
            df.to_pickle(OUTDIR / f"dayf_chunk_{i//CHUNK:02d}.pkl")
            names_all.extend(df.columns.tolist())
        print(f"  chunk {i//CHUNK:02d}: {df.shape} 累计 {time.time()-t0:.0f}s")
    (OUTDIR / "dayf_names.json").write_text(json.dumps(names_all))

    # meta：label + 18 champion 因子（不经 handler 实例化——其默认 instruments
    # 是 csi500，overlay 里不存在；直接用类属性拼表达式）
    from qlib_ifind_beta.factor.minute_enhanced_handler import MinuteEnhancedHandler
    names = list(MinuteEnhancedHandler.ENHANCED_FIELDS)
    fields = [f"${n}" for n in names]
    meta = _universe_features([CHAMPION_LABEL_EXPR] + fields, ["LABEL"] + names)
    meta.to_pickle(OUTDIR / "dayf_meta.pkl")
    print(f"meta {meta.shape} → dayf_meta.pkl；总用时 {time.time()-t0:.0f}s")


def _daily_ic(factor: pd.Series, label: pd.Series) -> pd.Series:
    """兼容别名 → screen_lib.daily_rank_ic。"""
    from qlib_ifind_beta.experiment.screen_lib import daily_rank_ic
    return daily_rank_ic(factor, label)


def _chunk_metrics(cf: Path, label: pd.Series, champs: pd.DataFrame) -> list[dict]:
    df: pd.DataFrame = pd.read_pickle(cf)
    out = []
    for col in df.columns:
        m = summarize(df[col], label, champs)
        # 库名前缀去掉后再匹配（alpha158__MIN5 → MIN5）
        bare = col.split("__")[-1]
        grave = next((g for g, pat in _GRAVE_PATTERNS if pat.search(bare)), "")
        out.append({"name": col, "grave_tag": grave, **m})
    return out


def stage_metrics() -> None:
    meta: pd.DataFrame = pd.read_pickle(OUTDIR / "dayf_meta.pkl")
    label = meta["LABEL"]
    champs = meta.drop(columns=["LABEL"])
    chunk_files = sorted(OUTDIR.glob("dayf_chunk_*.pkl"))

    from joblib import Parallel, delayed
    results = Parallel(n_jobs=min(20, len(chunk_files)))(
        delayed(_chunk_metrics)(cf, label, champs) for cf in chunk_files)
    recs = [r for part in results for r in part]
    print(f"  {len(recs)} 因子指标完成")
    res = pd.DataFrame(recs)
    res.to_csv(OUTDIR / "day_screen.csv", index=False)
    ok = res[res["n_days"] >= 100].copy()
    top = ok.reindex(ok["ic_ir"].abs().sort_values(ascending=False).index)
    print("\n=== 路线A top30（|IC IR| 排序）===")
    print(top.head(30).to_string(index=False,
          float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "feat"
    if stage == "feat":
        stage_feat()
    elif stage == "metrics":
        stage_metrics()
    else:
        raise SystemExit(f"unknown stage: {stage}")
