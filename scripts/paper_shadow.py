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

from qlib_ifind_beta.data.binio import read_bin as _read_bin
from qlib_ifind_beta.config import CHAMPION_EXPERIMENT, CHAMPION_RECORDER_ID, FEATURES_1MIN_SRC
from qlib_ifind_beta.config import DAY_CAL, MIN_CAL, OVERLAY_ROOT, PROJECT_ROOT

ROOT = PROJECT_ROOT / "data" / "paper_shadow"
NAV = ROOT / "nav.csv"


def load_latest_state(root: Path, before: str) -> tuple[str, pd.DataFrame, float]:
    """最新完整 <before 日期目录 即账本；返回 (date, positions, cash)。

    完整日 = 同时含 positions_after_close.csv 与 cash_after_buy.json；
    失败/残缺日（spec §6）不推进账本，自动跨过取更早的最新完整日。
    """
    days = sorted(
        p.name for p in root.iterdir()
        if p.is_dir() and p.name < before
        and (p / "positions_after_close.csv").exists()
        and (p / "cash_after_buy.json").exists()
    )
    if not days:
        raise RuntimeError(f"no complete ledger day before {before} under {root}")
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


def _snapshot_has(date: str) -> bool:
    cache = PROJECT_ROOT / "data" / "universe_snapshots.csv"
    return cache.exists() and f"{date}," in cache.read_text()


def _bars_ready(date: str, probe: str | None = None) -> bool:
    """T 已进 day 日历，且样本股 1min bin 的 09:41 行有有限值（防 daily 先于 1min 完成）。"""
    cal = {line.strip() for line in Path(DAY_CAL).read_text().splitlines() if line.strip()}
    if date not in cal:
        return False
    lines = [line.strip() for line in Path(MIN_CAL).read_text().splitlines() if line.strip()]
    wanted = f"{date} 09:41:00"
    if wanted not in lines:
        return False
    path = Path(probe) if probe else (
        FEATURES_1MIN_SRC / "sh600004" / "volume.1min.bin")
    si, vol = _read_bin(path)
    idx = lines.index(wanted) - si
    return 0 <= idx < vol.size and pd.notna(vol[idx])


def day(date: str) -> dict:
    ROOT.mkdir(parents=True, exist_ok=True)
    # 交易日门禁：幂等补拉当日快照（update_universe 内部对节假日静默跳过）
    proc = subprocess.run([sys.executable, "-m", "scripts.update_universe", "--end", date],
                          cwd=PROJECT_ROOT, check=False)
    if proc.returncode != 0 and not _snapshot_has(date):
        # 工作日快照仍缺失但拉取已失败（网络/token 事故）——不是节假日，fail-closed
        raise RuntimeError(
            f"universe fetch failed for {date} (rc={proc.returncode}), snapshot missing")
    if not _snapshot_has(date):
        print(f"{date}: no constituent snapshot (holiday?) — skip")
        return {"date": date, "status": "SKIPPED_HOLIDAY"}
    # 数据门禁：轮询等 15:30 同步落 bin（超时=同步事故，exit 非 0）
    deadline = time.time() + 60 * 60
    while not _bars_ready(date):
        if time.time() >= deadline:
            raise RuntimeError(f"sync gate timeout for {date}")
        time.sleep(60)
    import qlib
    qlib.init(provider_uri=str(OVERLAY_ROOT), region="cn")
    from qlib_ifind_beta.live.historical_replay import HistoricalReplaySource
    from qlib_ifind_beta.experiment.model_ensemble import load_model_bundle
    from scripts.replay_intraday_shadow import _eligible_snapshots, run_paper_day

    _, positions, cash = load_latest_state(ROOT, date)
    snapshots, _ = _eligible_snapshots(date, date)
    bundle = load_model_bundle(date, use_online=False)
    matches = list((PROJECT_ROOT / "mlruns").glob(
        f"*/{CHAMPION_RECORDER_ID}/artifacts/params.pkl"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one Champion model, found {len(matches)}")
    # stored=None：forward 日期恒晚于 pred.pkl 末位（2026-07-02），UNREFERENCED_FORWARD 分支
    positions, cash, row = run_paper_day(
        HistoricalReplaySource(), date, positions, cash, ROOT, bundle, matches[0],
        qlib.__version__, snapshots, stored=None, require_stored_parity=False)
    append_nav(NAV, row)
    print(json.dumps(row, ensure_ascii=False, indent=2, default=str))
    return row


def report(root: Path = ROOT) -> dict:
    from scripts.replay_intraday_shadow import _metrics

    nav = pd.read_csv(root / "nav.csv")
    result = {"days": len(nav), **_metrics(nav.set_index("date")["nav"])}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cmd = sub.add_parser("day"); cmd.add_argument("--date", required=True)
    sub.add_parser("report")
    args = parser.parse_args()
    if args.command == "day":
        day(args.date)
    else:
        report()


if __name__ == "__main__":
    main()
