import json

import pandas as pd


def _mk_day(root, date, cash, code="SH600000", qty=100):
    d = root / date
    d.mkdir(parents=True)
    pd.DataFrame([{"code": code, "quantity": qty,
                   "sellable_quantity": 0, "hold_days": 0}]
                 ).to_csv(d / "positions_after_close.csv", index=False)
    (d / "cash_after_buy.json").write_text(json.dumps({"cash_after_buy": cash}))


def test_load_latest_state_picks_latest_before(tmp_path):
    from scripts.paper_shadow import load_latest_state

    _mk_day(tmp_path, "2026-09-10", 100.0)
    _mk_day(tmp_path, "2026-09-11", 200.0)
    day, positions, cash = load_latest_state(tmp_path, "2026-09-14")
    assert day == "2026-09-11"
    assert cash == 200.0
    assert positions.iloc[0]["code"] == "SH600000"


def test_append_nav_appends_and_is_idempotent_per_date(tmp_path):
    from scripts.paper_shadow import append_nav

    nav = tmp_path / "nav.csv"
    append_nav(nav, {"date": "2026-09-10", "cash": 1.0, "market_value": 2.0, "nav": 3.0})
    append_nav(nav, {"date": "2026-09-11", "cash": 1.0, "market_value": 2.0, "nav": 4.0})
    append_nav(nav, {"date": "2026-09-11", "cash": 1.0, "market_value": 2.0, "nav": 5.0})
    frame = pd.read_csv(nav)
    assert frame["date"].tolist() == ["2026-09-10", "2026-09-11"]
    assert frame["nav"].tolist() == [3.0, 5.0]
