"""Unattended bin-sourced paper shadow account (spec 2026-09-13).

回放补齐（replay_intraday_shadow.py --output data/paper_shadow）之后，
每个交易日 16:30 由 cron 调 `day` 续跑一日并追加净值；账本 = 最新日期目录。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from qlib_ifind_beta.config import DAY_CAL, MIN_CAL, OVERLAY_ROOT, PROJECT_ROOT

ROOT = PROJECT_ROOT / "data" / "paper_shadow"
NAV = ROOT / "nav.csv"


def load_latest_state(root: Path, before: str) -> tuple[str, pd.DataFrame, float]:
    """最新 <before 日期目录 即账本；返回 (date, positions, cash)。"""
    days = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name < before)
    if not days:
        raise RuntimeError(f"no ledger day before {before} under {root}")
    last = root / days[-1]
    positions = pd.read_csv(last / "positions_after_close.csv")
    cash = float(json.loads((last / "cash_after_buy.json").read_text())["cash_after_buy"])
    return days[-1], positions, cash


def append_nav(nav_path: Path, row: dict) -> None:
    """追加一行 (date, cash, market_value, nav)；同日覆盖（幂等）。"""
    frame = pd.DataFrame([{k: row[k] for k in ("date", "cash", "market_value", "nav")}])
    if nav_path.exists():
        old = pd.read_csv(nav_path)
        frame = pd.concat([old.loc[old["date"] != row["date"]], frame], ignore_index=True)
    frame.to_csv(nav_path, index=False)


def main() -> None:  # Task 3 填充子命令
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()


if __name__ == "__main__":
    main()
