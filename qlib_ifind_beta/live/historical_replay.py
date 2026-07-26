"""Offline historical adapters for the intraday shadow workflow.

Only local Qlib bins and cached constituent snapshots are used.  Binary values
are read through memory maps so replaying rotating universes does not load the
entire minute history of thousands of stocks into RAM.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from qlib_ifind_beta.binio import DTYPE
from qlib_ifind_beta.config import DAY_CAL, FEATURES_1MIN_SRC, FEATURES_SRC, MIN_CAL
from qlib_ifind_beta.live.intraday import EXECUTION_TIME, FACTOR_TIMES


class HistoricalReplaySource:
    minute_fields = ("open", "high", "low", "close", "volume", "vwap")

    def __init__(self) -> None:
        self.day_dates = [line.strip() for line in Path(DAY_CAL).read_text().splitlines()
                          if line.strip()]
        self.day_row = {date: idx for idx, date in enumerate(self.day_dates)}
        minute_text = [line.strip() for line in Path(MIN_CAL).read_text().splitlines()
                       if line.strip()]
        self.minute_dates = np.array([value[:10] for value in minute_text])
        self.minute_times = np.array([value[11:16] for value in minute_text])
        self.minute_rows: dict[str, np.ndarray] = {}
        for date in np.unique(self.minute_dates):
            self.minute_rows[str(date)] = np.flatnonzero(self.minute_dates == date)

    @staticmethod
    def _take(path: Path, calendar_rows: np.ndarray) -> np.ndarray:
        result = np.full(len(calendar_rows), np.nan, dtype=np.float64)
        if not path.exists() or path.stat().st_size < 8:
            return result
        raw = np.memmap(path, dtype=DTYPE, mode="r")
        start = int(raw[0])
        relative = calendar_rows.astype(np.int64) - start + 1
        valid = (relative >= 1) & (relative < len(raw))
        result[valid] = np.asarray(raw[relative[valid]], dtype=np.float64)
        del raw
        return result

    def bars(self, date: str, codes: list[str]) -> pd.DataFrame:
        rows = self.minute_rows.get(date, np.array([], dtype=int))
        wanted = np.isin(self.minute_times[rows], [*FACTOR_TIMES, EXECUTION_TIME])
        rows = rows[wanted]
        times = self.minute_times[rows]
        if len(rows) != 11:
            raise RuntimeError(f"{date}: expected 11 historical morning rows, got {len(rows)}")
        output = []
        for code in codes:
            root = FEATURES_1MIN_SRC / code.lower()
            values = {field: self._take(root / f"{field}.1min.bin", rows)
                      for field in self.minute_fields}
            required = np.column_stack([values[f] for f in self.minute_fields])
            valid = np.isfinite(required).all(axis=1)
            for idx in np.flatnonzero(valid):
                volume = float(values["volume"][idx])
                output.append({
                    "date": date, "code": code, "bar_time": str(times[idx]),
                    "open": values["open"][idx], "high": values["high"][idx],
                    "low": values["low"][idx], "close": values["close"][idx],
                    "volume": volume, "amount": volume * float(values["vwap"][idx]),
                    "fetch_time": f"{date}T{times[idx]}:02+08:00", "source": "historical_1min",
                })
        return pd.DataFrame(output)

    def previous_volumes(self, date: str, codes: list[str]) -> dict[str, list[float]]:
        target = self.day_row[date]
        previous = [self.day_dates[target - k] if target >= k else None for k in (1, 2, 3, 5)]
        output = {}
        for code in codes:
            path = FEATURES_1MIN_SRC / code.lower() / "volume.1min.bin"
            totals = []
            for prev in previous:
                rows = self.minute_rows.get(prev, np.array([], dtype=int)) if prev else np.array([], dtype=int)
                values = self._take(path, rows)
                totals.append(float(np.nansum(values)) if np.isfinite(values).any() else 0.0)
            output[code] = totals
        return output

    def daily_info(self, date: str, codes: list[str]) -> dict[str, dict]:
        target = self.day_row[date]
        if target == 0:
            return {}
        rows = np.array([target - 1, target], dtype=int)
        output = {}
        for code in codes:
            root = FEATURES_SRC / code.lower()
            close = self._take(root / "close.day.bin", rows)
            factor = self._take(root / "factor.day.bin", rows)
            opened = self._take(root / "open.day.bin", rows)
            values = [close[0], factor[0], factor[1], opened[1]]
            if np.isfinite(values).all() and factor[0] != 0 and factor[1] != 0:
                output[code] = {
                    "prev_close": float(close[0]), "prev_factor": float(factor[0]),
                    "open": float(opened[1]), "factor": float(factor[1]),
                    "factor_confirmed": True,
                }
        # Some historical daily bins contain calendar holes even though the
        # corresponding minute session is complete.  For offline replay only,
        # reconstruct the same decision-time fields from T-1 15:00 and T 09:31.
        # Both bars are known before the 09:41 decision, so this does not relax
        # the live production fail-closed rule or introduce future data.
        missing = [code for code in codes if code not in output]
        if missing:
            previous_date = self.day_dates[target - 1]
            prev_rows = self.minute_rows.get(previous_date, np.array([], dtype=int))
            today_rows = self.minute_rows.get(date, np.array([], dtype=int))
            prev_close_rows = prev_rows[self.minute_times[prev_rows] == "15:00"]
            today_open_rows = today_rows[self.minute_times[today_rows] == "09:31"]
            if len(prev_close_rows) == 1 and len(today_open_rows) == 1:
                minute_rows = np.array(
                    [int(prev_close_rows[0]), int(today_open_rows[0])], dtype=int
                )
                for code in missing:
                    root = FEATURES_1MIN_SRC / code.lower()
                    close = self._take(root / "close.1min.bin", minute_rows)
                    opened = self._take(root / "open.1min.bin", minute_rows)
                    factor = self._take(root / "factor.1min.bin", minute_rows)
                    values = [close[0], factor[0], factor[1], opened[1]]
                    if (
                        np.isfinite(values).all()
                        and factor[0] != 0
                        and factor[1] != 0
                    ):
                        output[code] = {
                            "prev_close": float(close[0]),
                            "prev_factor": float(factor[0]),
                            "open": float(opened[1]),
                            "factor": float(factor[1]),
                            "factor_confirmed": True,
                            "historical_source": "minute_fallback",
                        }
        return output

    def valuation_close(self, date: str, codes: list[str]) -> dict[str, float]:
        """Return Qlib-adjusted close, matching the adjusted 09:41 minute price."""
        row = np.array([self.day_row[date]], dtype=int)
        output = {}
        for code in codes:
            root = FEATURES_SRC / code.lower()
            close = self._take(root / "close.day.bin", row)[0]
            factor = self._take(root / "factor.day.bin", row)[0]
            if np.isfinite(close) and np.isfinite(factor) and factor != 0:
                output[code] = float(close)
        missing = [code for code in codes if code not in output]
        if missing:
            rows = self.minute_rows.get(date, np.array([], dtype=int))
            close_rows = rows[self.minute_times[rows] == "15:00"]
            if len(close_rows) == 1:
                minute_row = np.array([int(close_rows[0])], dtype=int)
                for code in missing:
                    root = FEATURES_1MIN_SRC / code.lower()
                    close = self._take(root / "close.1min.bin", minute_row)[0]
                    factor = self._take(root / "factor.1min.bin", minute_row)[0]
                    if np.isfinite(close) and np.isfinite(factor) and factor != 0:
                        output[code] = float(close)
        return output
