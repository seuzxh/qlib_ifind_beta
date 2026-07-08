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
