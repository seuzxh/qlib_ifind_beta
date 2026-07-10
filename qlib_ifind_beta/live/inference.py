"""P1 实战对接 — 推理核心。

T 日收盘后用冻结 champion 模型对 T 日因子推理，产出 top10 信号 + 成交辅助字段。
口径与 champion 回测 test 段逐位零偏离（spec §3.1 证明 + spike 2026-07-09 铁证）。

零前视：
- 18 因子 = T 日 9:30-9:40 分钟数据（9:40 ≪ 9:41 决策时刻）。
- aux 字段（price_941/change_941/limit_up/limit_down）全部 ≤ T。
- label 仅 fetch handler 结构（learned processor fit 用），P1 不读 label 值。
"""
from __future__ import annotations

import pandas as pd

from qlib_ifind_beta.config import (
    CHAMPION_DATA_START, CHAMPION_EXPERIMENT, CHAMPION_FIT_END, CHAMPION_FIT_START,
    CHAMPION_LABEL_EXPR, CHAMPION_RECORDER_ID, CHAMPION_TOPK, OVERLAY_ROOT,
    ROLLING_EXPERIMENT, UNIVERSE_MARKET,
)

_QLIB_INITED = False


def _ensure_qlib() -> None:
    """进程级单次 qlib.init（多次 init 会 warn + 浪费）。"""
    global _QLIB_INITED
    if not _QLIB_INITED:
        import qlib
        qlib.init(provider_uri=str(OVERLAY_ROOT), region="cn")
        _QLIB_INITED = True


def _squeeze_day(df_or_ser: pd.DataFrame | pd.Series, date: str) -> pd.Series:
    """从 MultiIndex ['datetime','instrument'] 取单日 → 一维 Series。

    pred / aux 都可能是单列 DataFrame（pred.pkl 即是），统一 squeeze 成 Series。
    """
    level = "datetime" if "datetime" in df_or_ser.index.names else 1
    row = df_or_ser.xs(date, level=level)
    if isinstance(row, pd.DataFrame):
        row = row.iloc[:, 0]
    return row.dropna() if row.ndim == 1 else row


def predict_day(date: str,
                recorder_id: str = CHAMPION_RECORDER_ID,
                experiment_name: str = CHAMPION_EXPERIMENT,
                market: str = UNIVERSE_MARKET,
                topk: int = CHAMPION_TOPK,
                use_online: bool = False) -> dict:
    """T 日收盘后推理：复刻 champion handler（fit 段 FROZEN）→ 冻结 model.predict → top10。

    Args:
        use_online: True 时从 ROLLING_EXPERIMENT 最新 online recorder 加载模型（每日滚动
            重训产出的新模型），替代 FROZEN CHAMPION_RECORDER_ID。需要先跑
            scripts/retrain.py 产出 online 模型。默认 False（向后兼容 P1 FROZEN）。

    Returns:
        {date, n_candidates, candidates:[{code,score,price_941,change_941,limit_up,limit_down}],
         topk:[...剔除封涨停后的前 topk]}
    """
    _ensure_qlib()
    from qlib.workflow import R
    from qlib.data import D
    from qlib.data.dataset import DatasetH
    from qlib_ifind_beta.minute_enhanced_handler import MinuteEnhancedHandler

    # 1. load model
    if use_online:
        # 从 ROLLING_EXPERIMENT 最新 online recorder 加载（每日滚动重训产出）
        from qlib.workflow.online.utils import OnlineToolR
        tool = OnlineToolR(ROLLING_EXPERIMENT)
        online_recs = tool.online_models(exp_name=ROLLING_EXPERIMENT)
        if not online_recs:
            raise RuntimeError(
                f"use_online=True 但 {ROLLING_EXPERIMENT} 无 online 模型。"
                "请先跑 scripts/retrain.py 产出滚动重训模型，或用 use_online=False（FROZEN champion）。")
        rec = online_recs[0]
    else:
        # FROZEN champion（默认，P1 向后兼容）
        rec = R.get_recorder(recorder_id=recorder_id, experiment_name=experiment_name)
    model = rec.load_object("params.pkl")

    # 2. 复刻 champion handler（fit 段 FROZEN，仅 end_time 扩到 date）
    handler = MinuteEnhancedHandler(
        instruments=market, start_time=CHAMPION_DATA_START, end_time=date,
        fit_start_time=CHAMPION_FIT_START, fit_end_time=CHAMPION_FIT_END,
        label=[CHAMPION_LABEL_EXPR],
    )
    dataset = DatasetH(handler=handler, segments={"inference": (date, date)})

    # 3. predict → 单日 Series（score per instrument）
    pred = model.predict(dataset, segment="inference")
    scores = _squeeze_day(pred, date)
    if scores.empty:
        return {"date": date, "n_candidates": 0, "candidates": [], "topk": []}

    # 4. aux 字段（≤T 无前视）：成交价 + 涨跌停判定
    aux = D.features(D.instruments(market=market),
                     ["$price_941", "$change_941", "$limit_up", "$limit_down"],
                     start_time=date, end_time=date)
    aux_day = {}   # code → (price_941, change_941, limit_up, limit_down)
    if not aux.empty:
        level = "datetime" if "datetime" in aux.index.names else 1
        a = aux.xs(date, level=level)
        for code, r in a.iterrows():
            aux_day[code] = (float(r.iloc[0]), float(r.iloc[1]), float(r.iloc[2]), float(r.iloc[3]))

    # 5. 组装候选（按 score 降序；aux 缺失则跳过）
    cands = []
    for code, sc in scores.sort_values(ascending=False).items():
        if code not in aux_day:
            continue
        price_941, change_941, limit_up, limit_down = aux_day[code]
        cands.append({"code": code, "score": float(sc),
                      "price_941": price_941, "change_941": change_941,
                      "limit_up": limit_up, "limit_down": limit_down})

    # 6. 买入拦截：change_941 >= limit_up → 封涨停剔出 topk（与回测 exchange 同源判定）
    tradable = [c for c in cands if not (c["change_941"] >= c["limit_up"])]
    return {"date": date, "n_candidates": len(cands),
            "candidates": cands, "topk": tradable[:topk]}
