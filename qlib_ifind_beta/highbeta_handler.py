"""HighBetaAlpha158 — Alpha158(全 lag T-1) + 14 materialized minute factors (T-day).

v2 (2026-07-06): wraps all 158 Alpha158 daily fields in Ref(...,1) to lag them to
T-1 — T 日 9:41 撮合只用 T-1 及更早的日频数据，无前视。The 14 minute fields stay
T-day (they are 9:30-9:40, before the 9:41 buy). qlib reads all as daily overlay;
no frequency mixing at the Handler layer.

label / deal_price via qrun YAML (handler.kwargs.label, exchange_kwargs.deal_price).
pred[T]→T-day execution needs TopkDropoutStrategyTD0 — see td0_strategy.py.
"""
from __future__ import annotations

from qlib.contrib.data.handler import Alpha158

from .config import MINUTE_FACTOR_FIELDS


class HighBetaAlpha158(Alpha158):
    def get_feature_config(self):
        fields, names = super().get_feature_config()          # native 158
        lag_fields = [f"Ref({f}, 1)" for f in fields]         # v2: lag 158 日频 → T-1
        min_fields = [f"${n}" for n in MINUTE_FACTOR_FIELDS]  # 14 分钟因子，T 日当天不 lag
        return lag_fields + min_fields, names + min_fields
