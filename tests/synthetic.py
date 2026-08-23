"""Deterministic synthetic UK-imbalance-like market data for tests.

A daily sinusoidal price (48 periods/day) with enough amplitude to beat round-trip costs,
plus mild noise and a correlated demand series. No real CSVs needed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_market(days: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = days * 48
    idx = pd.date_range("2017-01-01", periods=n, freq="30min", name="StartTime")
    phase = np.arange(n) % 48
    daily = np.sin(2 * np.pi * phase / 48)
    ssp = 50 + 40 * daily + rng.normal(0, 2, n)          # £/MWh, clear daily swing
    sbp = ssp + 1.0 + rng.normal(0, 0.2, n).clip(0)       # buy slightly above sell
    demand = 30000 + 8000 * (-daily) + rng.normal(0, 300, n)
    return pd.DataFrame(
        {"SystemSellPrice": ssp, "SystemBuyPrice": sbp, "Demand": demand}, index=idx
    )
