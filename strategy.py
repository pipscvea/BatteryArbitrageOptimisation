"""Stage 3 glue — turn a fitted forecasting model into per-period trade requests.

(This file previously ran the entire pipeline at import time and mixed feature building,
labelling and a hand-written rule. That has been decomposed into config/data_pipeline/
features/labels/forecasting/decision/simulate. This module now does one thing.)
"""
from __future__ import annotations

import pandas as pd

from config import BatteryConfig, TradingConfig
from decision import decide


def model_requests(model, X: pd.DataFrame, df: pd.DataFrame,
                   batt: BatteryConfig, trade: TradingConfig) -> pd.Series:
    """Predict forward price change from ``X`` and convert each to a signed request.

    ``X`` and ``df`` share the same (test) index. Returns battery-side kWh requests.
    """
    forecast = pd.Series(model.predict(X), index=X.index)
    reqs = {
        t: decide(float(forecast.loc[t]),
                  float(df["SystemSellPrice"].loc[t]),
                  float(df["SystemBuyPrice"].loc[t]),
                  batt, trade).request_kwh
        for t in X.index
    }
    return pd.Series(reqs).reindex(X.index).fillna(0.0)
