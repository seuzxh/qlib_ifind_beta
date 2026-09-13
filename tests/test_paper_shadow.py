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


def test_load_latest_state_skips_partial_day_dirs(tmp_path):
    from scripts.paper_shadow import load_latest_state

    _mk_day(tmp_path, "2026-09-10", 100.0)
    partial = tmp_path / "2026-09-11"
    partial.mkdir()
    pd.DataFrame([{"code": "SH600000", "quantity": 100,
                   "sellable_quantity": 0, "hold_days": 0}]
                 ).to_csv(partial / "positions_after_close.csv", index=False)
    day, positions, cash = load_latest_state(tmp_path, "2026-09-14")
    assert day == "2026-09-10"
    assert cash == 100.0


def test_append_nav_appends_and_is_idempotent_per_date(tmp_path):
    from scripts.paper_shadow import append_nav

    nav = tmp_path / "nav.csv"
    append_nav(nav, {"date": "2026-09-10", "cash": 1.0, "market_value": 2.0, "nav": 3.0})
    append_nav(nav, {"date": "2026-09-11", "cash": 1.0, "market_value": 2.0, "nav": 4.0})
    append_nav(nav, {"date": "2026-09-11", "cash": 1.0, "market_value": 2.0, "nav": 5.0})
    frame = pd.read_csv(nav)
    assert frame["date"].tolist() == ["2026-09-10", "2026-09-11"]
    assert frame["nav"].tolist() == [3.0, 5.0]


def test_bars_ready_requires_calendar_and_bar(tmp_path, monkeypatch):
    from scripts import paper_shadow

    day_cal = tmp_path / "day.txt"
    day_cal.write_text("2026-09-11\n")
    min_cal = tmp_path / "1min.txt"
    min_cal.write_text("2026-09-11 09:41:00\n")
    monkeypatch.setattr(paper_shadow, "DAY_CAL", str(day_cal))
    monkeypatch.setattr(paper_shadow, "MIN_CAL", str(min_cal))

    def fake_read_bin(path):
        return 0, pd.array([100.0], dtype="float32")  # si=0, 有限值

    monkeypatch.setattr(paper_shadow, "_read_bin", fake_read_bin)
    assert paper_shadow._bars_ready("2026-09-11", probe=str(tmp_path / "v.bin"))
    assert not paper_shadow._bars_ready("2026-09-14", probe=str(tmp_path / "v.bin"))
