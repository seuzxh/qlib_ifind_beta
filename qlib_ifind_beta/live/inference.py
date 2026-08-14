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
    UNIVERSE_MARKET,
)
from qlib_ifind_beta.model_ensemble import blend_scores, load_model_bundle

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
                use_online: bool = True) -> dict:
    """T 日收盘后推理：复刻 champion handler（fit 段 FROZEN）→ 冻结 model.predict → top10。

    Args:
        use_online: True 时优先加载覆盖目标日的滚动模型及其冻结门控 artifact；不存在
            匹配模型时自动回退 FROZEN champion。False 用于复现冻结冠军。

    Returns:
        {date, n_candidates, candidates:[{code,score,price_941,change_941,limit_up,limit_down}],
         topk:[...剔除封涨停后的前 topk]}
    """
    _ensure_qlib()
    from qlib.data import D
    from qlib.data.dataset import DatasetH
    from qlib_ifind_beta.minute_enhanced_handler import MinuteEnhancedHandler

    # 1. Load a date-matched rolling bundle. Explicit recorder arguments remain
    # available only for the frozen compatibility path.
    if use_online:
        bundle = load_model_bundle(date, use_online=True)
    else:
        from qlib.workflow import R
        rec = R.get_recorder(recorder_id=recorder_id, experiment_name=experiment_name)
        from qlib_ifind_beta.model_ensemble import OnlineModelBundle
        bundle = OnlineModelBundle(rec.load_object("params.pkl"), None, 0.0, "frozen")

    # 2. 复刻 champion handler（fit 段 FROZEN，仅 end_time 扩到 date）
    handler = MinuteEnhancedHandler(
        instruments=market, start_time=CHAMPION_DATA_START, end_time=date,
        fit_start_time=CHAMPION_FIT_START, fit_end_time=CHAMPION_FIT_END,
        label=[CHAMPION_LABEL_EXPR],
    )
    # HFLGBModel.predict 硬编码读 "test" segment（不接受 segment 参数），故 key 用 "test"。
    # LGBModel.predict 接受 segment= 参数但默认也是 "test" → 向后兼容。
    dataset = DatasetH(handler=handler, segments={"test": (date, date)})

    # 3. predict → 单日 Series（score per instrument）
    hflgb_scores = _squeeze_day(bundle.hflgb.predict(dataset), date)
    if bundle.weight > 0 and bundle.xgb is not None:
        xgb_scores = _squeeze_day(bundle.xgb.predict(dataset), date)
        scores = blend_scores(hflgb_scores, xgb_scores, bundle.weight)
    else:
        scores = hflgb_scores
    if scores.empty:
        return {"date": date, "n_candidates": 0, "candidates": [], "topk": [],
                "model_source": bundle.source, "ensemble_weight": bundle.weight}

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
            "candidates": cands, "topk": tradable[:topk],
            "model_source": bundle.source, "ensemble_weight": bundle.weight}
