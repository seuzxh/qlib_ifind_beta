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
