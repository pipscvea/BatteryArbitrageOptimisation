"""Stage 2 — feature engineering.

Pure functions, no import-time side effects (the old ``strategy.py`` ran the whole
pipeline at import). Every feature here uses ONLY information available at or before
the decision time ``t`` — no forward-looking columns. The forward-looking part of the
problem lives entirely in ``labels.py`` (the target), which keeps the train/test
evaluation honest.
"""
from __future__ import annotations

import pandas as pd

# Features consumed by the forecasting model. All are known at decision time t.
FEATURE_COLUMNS = [
    "SystemSellPrice", "SystemBuyPrice", "spread",
    "ssp_lag1", "ssp_lag2", "ssp_ma6", "sbp_ma6",
    "ssp_rolling_mean_48", "ssp_rolling_std_3", "price_vol_48",
    "Demand", "Demand_lag1", "Demand_lag2", "Demand_ma6", "Demand_rolling_std_3",
    "hour", "day_of_week", "month", "is_weekend",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with engineered, past-only feature columns added.

    Assumes a DatetimeIndex and columns ``SystemSellPrice``, ``SystemBuyPrice``, ``Demand``.
    """
    out = df.copy()

    # Price features (all trailing / current — no shift(-n) anywhere).
    out["ssp_lag1"] = out["SystemSellPrice"].shift(1)
    out["ssp_lag2"] = out["SystemSellPrice"].shift(2)
    out["ssp_ma6"] = out["SystemSellPrice"].rolling(6).mean()
    out["sbp_ma6"] = out["SystemBuyPrice"].rolling(6).mean()
    out["ssp_rolling_mean_48"] = out["SystemSellPrice"].rolling(48).mean()
    out["ssp_rolling_std_3"] = out["SystemSellPrice"].rolling(3).std()
    out["spread"] = out["SystemBuyPrice"] - out["SystemSellPrice"]

    # Volatility as the trailing std of absolute price changes (£/MWh). Uses first
    # differences, not log returns, because UK imbalance prices go negative and zero
    # (log returns are undefined there).
    mid = (out["SystemSellPrice"] + out["SystemBuyPrice"]) / 2
    out["price_vol_48"] = mid.diff().rolling(48).std()

    # Demand features.
    out["Demand_lag1"] = out["Demand"].shift(1)
    out["Demand_lag2"] = out["Demand"].shift(2)
    out["Demand_ma6"] = out["Demand"].rolling(6).mean()
    out["Demand_rolling_std_3"] = out["Demand"].rolling(3).std()

    # Calendar features (from the index).
    out["hour"] = out.index.hour
    out["day_of_week"] = out.index.dayofweek  # 0 = Monday
    out["month"] = out.index.month
    out["is_weekend"] = out["day_of_week"].isin([5, 6]).astype(int)

    return out


def add_driver_features(df: pd.DataFrame) -> pd.DataFrame:
    """Stage 2 extension point — engineer features from fundamental drivers.

    Once wind/solar/interconnector/gas/carbon/weather columns are merged in via
    ``data_pipeline.merge_driver``, derive lags/ratios/rolling stats here and append
    the new names to ``FEATURE_COLUMNS``. Left as a documented no-op until that data
    is sourced. This is what turns the project from generic ML into market analysis.
    """
    return df


def feature_matrix(df: pd.DataFrame):
    """Return ``(X, index)`` of finite feature rows only, preserving the datetime index."""
    feats = build_features(df)
    X = feats[FEATURE_COLUMNS]
    mask = X.notna().all(axis=1)
    return X[mask], feats.index[mask]
