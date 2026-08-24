"""Stage 4 — benchmark strategies to answer "does the ML complexity create value?".

Each returns a signed request Series (battery-side kWh) to feed ``simulate.simulate``:
  * perfect_foresight_myopic — the SAME myopic 1-step decision rule, but fed the realised
    forward change instead of a forecast. It isolates "value lost to forecast error" for
    this policy. NOTE: it is NOT a true optimal-dispatch upper bound — a myopic 1-step
    rule can be beaten over a full path, so a better forecast+policy can exceed it. The
    genuine upper bound is a perfect-foresight LP/DP over SoC (Stage 3, see issue).
  * naive_time_of_day — charge overnight, discharge evening peak; a zero-forecast baseline.
"""
from __future__ import annotations

import pandas as pd

from config import BatteryConfig, TradingConfig
from decision import decide
from labels import forward_price_change


def perfect_foresight_myopic(df: pd.DataFrame, batt: BatteryConfig, trade: TradingConfig,
                             horizon: int = 1) -> pd.Series:
    """Run the SAME myopic decision rule but with the true realised forward change.

    Reference for "value lost to forecast error", NOT a global optimum (see module note).
    """
    realised = forward_price_change(df, horizon)
    reqs = {t: decide(float(realised.loc[t]), float(df["SystemSellPrice"].loc[t]),
                      float(df["SystemBuyPrice"].loc[t]), batt, trade).request_kwh
            for t in df.index if pd.notna(realised.loc[t])}
    return pd.Series(reqs).reindex(df.index).fillna(0.0)


def naive_time_of_day(df: pd.DataFrame, batt: BatteryConfig,
                      charge_hours=range(0, 5), discharge_hours=range(16, 20)) -> pd.Series:
    """Charge overnight, discharge evening peak. No forecast at all."""
    vol = batt.max_energy_per_period_kwh
    hours = df.index.hour
    req = pd.Series(0.0, index=df.index)
    req[pd.Index(hours).isin(list(charge_hours))] = +vol
    req[pd.Index(hours).isin(list(discharge_hours))] = -vol
    return req
