"""HighBetaAlpha158 — Alpha158 + 14 materialized minute factors.

Overrides get_feature_config to append 14 $-prefixed minute fields (materialized
as day.bin by materialize_minute.py) to Alpha158's native 158. qlib reads them
like any other field; no frequency mixing at the Handler layer.

label / deal_price are passed via qrun YAML (handler.kwargs.label,
exchange_kwargs.deal_price), not here — see qrun/workflow.yaml.
"""
from __future__ import annotations

from qlib.contrib.data.handler import Alpha158

from .config import MINUTE_FACTOR_FIELDS


class HighBetaAlpha158(Alpha158):
    def get_feature_config(self):
        fields, names = super().get_feature_config()   # native 158
        min_fields = [f"${n}" for n in MINUTE_FACTOR_FIELDS]
        return fields + min_fields, names + min_fields
