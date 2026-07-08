"""P1 实战对接 inference 测试。CLAUDE.md 7 项全覆盖。

零偏离(P1 正确性根基) + NAV + 涨跌停 + 前视 + settle + 幂等 + IC 边界。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest


@pytest.fixture(scope="module")
def qlib_init():
    import qlib
    qlib.init(provider_uri="data/qlib_root", region="cn")
    yield


# ---------------------------------------------------------------------------
# Task 2: 零偏离（P1 正确性根基）
# ---------------------------------------------------------------------------
def test_predict_day_zero_drift(qlib_init):
    """零偏离：predict_day('2026-07-02') vs champion pred.pkl 逐位 max|diff|<1e-6，top10 集合一致。

    spike 2026-07-09 铁证：max|diff|=0.000e+00, top10 10/10。此测试锚定该口径不退化。
    """
    from qlib.workflow import R
    from qlib_ifind_beta.live.inference import predict_day
    from qlib_ifind_beta.config import CHAMPION_RECORDER_ID, CHAMPION_EXPERIMENT

    DAY = "2026-07-02"   # test 末日，pred.pkl 含此日
    result = predict_day(DAY)
    # 结构校验
    assert set(result.keys()) >= {"date", "n_candidates", "candidates", "topk"}
    assert result["date"] == DAY
    assert result["n_candidates"] == len(result["candidates"])

    # champion pred.pkl ground truth
    rec = R.get_recorder(recorder_id=CHAMPION_RECORDER_ID, experiment_name=CHAMPION_EXPERIMENT)
    ref = rec.load_object("pred.pkl")
    ref_day = ref.xs(DAY, level="datetime")
    if isinstance(ref_day, pd.DataFrame):
        ref_day = ref_day.iloc[:, 0]

    # P1 candidates（未剔除集，与 ref 同口径）逐位比对
    cands = {c["code"]: c["score"] for c in result["candidates"]}
    common = set(cands) & set(ref_day.index)
    assert len(common) > 0, "P1 candidates 与 champion pred 无交集（口径错）"
    diffs = [abs(cands[c] - ref_day.loc[c]) for c in common]
    assert max(diffs) < 1e-6, f"max|diff|={max(diffs):.3e} 超阈值，口径偏离"


# ---------------------------------------------------------------------------
# Task 3: NAV 累积（数值正确性手算对照）
# ---------------------------------------------------------------------------
def test_compute_nav_compound(tmp_path):
    """equal-weight compound NAV 手算对照。无成本，2 日。

    day1: A buy=10 sell=11 (ret=+0.10), B buy=10 sell=9 (ret=-0.10) → 组合 ret=0 → nav=1.0
    day2: A buy=10 sell=12 (ret=+0.20), B buy=10 sell=11 (ret=+0.10) → 组合 ret=+0.15 → nav=1.15
    """
    from qlib_ifind_beta.live.track import compute_nav
    settle = pd.DataFrame([
        {"signal_date": "2026-04-01", "code": "A", "buy_price": 10.0, "sell_date": "2026-04-02", "sell_price": 11.0},
        {"signal_date": "2026-04-01", "code": "B", "buy_price": 10.0, "sell_date": "2026-04-02", "sell_price": 9.0},
        {"signal_date": "2026-04-02", "code": "A", "buy_price": 10.0, "sell_date": "2026-04-03", "sell_price": 12.0},
        {"signal_date": "2026-04-02", "code": "B", "buy_price": 10.0, "sell_date": "2026-04-03", "sell_price": 11.0},
    ])
    out = tmp_path / "nav.csv"
    nav = compute_nav(settle, out, open_cost=0.0, close_cost=0.0)
    by_sell = nav.set_index("sell_date")["daily_ret_net"]
    assert abs(by_sell["2026-04-02"] - 0.0) < 1e-9
    assert abs(by_sell["2026-04-03"] - 0.15) < 1e-9
    assert abs(nav["net_nav"].iloc[-1] - 1.15) < 1e-9


def test_compute_nav_with_cost(tmp_path):
    """含成本：单笔 ret 受 open_cost/close_cost 双向侵蚀。"""
    from qlib_ifind_beta.live.track import compute_nav
    settle = pd.DataFrame([
        {"signal_date": "2026-04-01", "code": "A", "buy_price": 10.0, "sell_date": "2026-04-02", "sell_price": 11.0},
    ])
    out = tmp_path / "nav.csv"
    nav = compute_nav(settle, out, open_cost=0.001, close_cost=0.001)
    # ret_net = 11*0.999/(10*1.001) - 1 = 10.989/10.01 - 1 = 0.097902...
    expected = 11 * 0.999 / (10 * 1.001) - 1
    assert abs(nav["daily_ret_net"].iloc[0] - expected) < 1e-9
    assert abs(nav["gross_nav"].iloc[0] - 1.10) < 1e-9   # gross = 11/10-1 = 0.10


# ---------------------------------------------------------------------------
# Task 6a: 涨停买入拦截（predict_day topk 不含封涨停）
# ---------------------------------------------------------------------------
def test_predict_day_limit_up_filtered(qlib_init):
    """topk 中不得含封涨停（change_941 >= limit_up），且 |topk| <= 10。"""
    from qlib_ifind_beta.live.inference import predict_day
    result = predict_day("2026-07-02")
    for c in result["topk"]:
        assert c["change_941"] < c["limit_up"], f"{c['code']} 封涨停却进 topk（买入拦截失效）"
    assert len(result["topk"]) <= 10


# ---------------------------------------------------------------------------
# Task 6b: 无前视（handler end_time == T，不扩展到 T+1）
# ---------------------------------------------------------------------------
def test_predict_day_no_lookahead(qlib_init, monkeypatch):
    """handler end_time 必须 == 推理日 T（不 fetch T+1 label 数据 → 无前视）。"""
    from qlib_ifind_beta.minute_enhanced_handler import MinuteEnhancedHandler
    from qlib_ifind_beta.live import inference

    captured = {}
    orig_init = MinuteEnhancedHandler.__init__

    def spy(self, *a, **kw):
        captured["end_time"] = kw.get("end_time")
        orig_init(self, *a, **kw)

    monkeypatch.setattr(MinuteEnhancedHandler, "__init__", spy)
    inference.predict_day("2026-07-02")
    assert captured.get("end_time") == "2026-07-02", \
        f"handler end_time={captured.get('end_time')} 超 T（前视泄漏）"


# ---------------------------------------------------------------------------
# Task 6c: 跌停卖出拦截（settle_prev: change <= limit_down → blocked, sell_price=NaN）
# ---------------------------------------------------------------------------
def test_settle_prev_limit_down_blocked(tmp_path):
    """T+1 封跌停的持仓 → blocked=True, sell_price=NaN（递延，该笔 ret=0 持平）。"""
    from qlib_ifind_beta.live.track import settle_prev
    sig_p, set_p = tmp_path / "sig.csv", tmp_path / "set.csv"
    pd.DataFrame([
        {"date": "2026-07-01", "code": "A", "score": 0.9, "price_941": 10.0,
         "change_941": 0.0, "limit_up": 0.095, "limit_down": -0.095, "in_topk": True},
        {"date": "2026-07-01", "code": "B", "score": 0.8, "price_941": 20.0,
         "change_941": 0.0, "limit_up": 0.095, "limit_down": -0.095, "in_topk": True},
    ]).to_csv(sig_p, index=False)
    n = settle_prev("2026-07-01", "2026-07-02",
                    close_lookup={"A": 11.0, "B": 18.0},
                    change_lookup={"A": 0.05, "B": -0.10},      # B 封跌停
                    limit_down_lookup={"A": -0.095, "B": -0.095},
                    signals_path=sig_p, settle_path=set_p)
    assert n == 2
    df = pd.read_csv(set_p).set_index("code")
    assert bool(df.loc["A", "blocked"]) is False
    assert df.loc["A", "sell_price"] == 11.0
    assert bool(df.loc["B", "blocked"]) is True
    assert pd.isna(df.loc["B", "sell_price"])


# ---------------------------------------------------------------------------
# Task 6d: record_signal 幂等 + in_topk 标记
# ---------------------------------------------------------------------------
def test_record_signal_idempotent(tmp_path):
    """同 date 二次写入幂等（不重复）；in_topk 标记封涨停剔除（A 入选，B 封涨停）。"""
    from qlib_ifind_beta.live.track import record_signal
    cand_a = {"code": "A", "score": 0.9, "price_941": 10.0, "change_941": 0.02,
              "limit_up": 0.095, "limit_down": -0.095}
    cand_b = {"code": "B", "score": 0.8, "price_941": 20.0, "change_941": 0.10,
              "limit_up": 0.095, "limit_down": -0.095}   # 封涨停
    result = {"date": "2026-07-02", "candidates": [cand_a, cand_b], "topk": [cand_a]}
    p = tmp_path / "sig.csv"
    record_signal(result, p)
    record_signal(result, p)                # 二次（幂等）
    df = pd.read_csv(p)
    assert len(df) == 2                      # 不重复
    assert bool(df[df.code == "A"].iloc[0]["in_topk"]) is True
    assert bool(df[df.code == "B"].iloc[0]["in_topk"]) is False


# ---------------------------------------------------------------------------
# Task 6e: daily IC 边界（< 5 对 → None）
# ---------------------------------------------------------------------------
def test_daily_ic_too_few_returns_none(tmp_path):
    """< 5 对 score/label → Spearman 无意义，返回 None。"""
    from qlib_ifind_beta.live.track import daily_ic
    p = tmp_path / "sig.csv"
    pd.DataFrame([
        {"date": "2026-07-02", "code": "A", "score": 0.9},
        {"date": "2026-07-02", "code": "B", "score": 0.8},
    ]).to_csv(p, index=False)
    assert daily_ic(p, {"A": 0.01, "B": 0.02}, "2026-07-02") is None
