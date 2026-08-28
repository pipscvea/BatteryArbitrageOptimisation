"""Stage 3 — perfect-foresight optimal dispatch via linear programming.

Given the realised prices over a window, this computes the *globally* optimal
charge/discharge schedule subject to the real battery constraints (capacity, min/max
SoC, per-period power, one-way efficiency, transaction + degradation costs). Unlike the
myopic ``perfect_foresight_myopic`` benchmark, this is a genuine **upper bound**: no
causal strategy can beat it, so it is the right yardstick for "what fraction of the
achievable value is the forecast capturing?".

It cheats (uses future prices), so it is a benchmark only — never a tradeable strategy.

Battery-side decision variables per period t:
    a_t >= 0  energy added to the battery  (buy a_t/eff from grid)
    r_t >= 0  energy removed from the battery (sell r_t*eff to grid)
SoC_t = SoC_0 + sum_{i<=t} (a_i - r_i),   lo <= SoC_t <= hi,   0 <= a_t,r_t <= max_step.

Maximise realised cash + terminal liquidation of leftover SoC (matching simulate.py's
mark-to-market), which is linear in a, r.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

from config import BatteryConfig, TradingConfig


def optimal_dispatch(df: pd.DataFrame, batt: BatteryConfig, trade: TradingConfig) -> pd.Series:
    """Return the optimal signed request Series (battery-side kWh; + charge / - discharge)
    to feed ``simulate.simulate``."""
    idx = df.index
    T = len(idx)
    if T == 0:
        return pd.Series(dtype=float)

    p_sell = df["SystemSellPrice"].to_numpy(dtype=float)
    p_buy = df["SystemBuyPrice"].to_numpy(dtype=float)
    eff = batt.efficiency_one_way
    costs = trade.transaction_cost_per_mwh + batt.degradation_cost_per_mwh
    max_step = batt.max_energy_per_period_kwh
    soc0 = batt.capacity_kwh * batt.soc_init
    lo, hi = batt.capacity_kwh * batt.soc_min, batt.capacity_kwh * batt.soc_max
    p_final = p_sell[-1]

    # Per-unit profit (£ per kWh battery-side). /1000 converts kWh -> MWh for pricing.
    # charge a_t: pay (p_buy+costs)/eff per MWh grid; terminal value +eff*p_final.
    prof_a = (-(p_buy + costs) / eff + eff * p_final) / 1000.0
    # discharge r_t: receive eff*(p_sell-costs) per MWh; terminal value -eff*p_final.
    prof_r = (eff * (p_sell - costs) - eff * p_final) / 1000.0
    c = -np.concatenate([prof_a, prof_r])  # linprog minimises

    # Cumulative SoC bounds: L is lower-triangular ones (prefix sums).
    L = sparse.tril(np.ones((T, T)))
    # upper:  L a - L r <=  hi - soc0
    # lower: -L a + L r <=  soc0 - lo
    A_ub = sparse.vstack([
        sparse.hstack([L, -L]),
        sparse.hstack([-L, L]),
    ]).tocsr()
    b_ub = np.concatenate([np.full(T, hi - soc0), np.full(T, soc0 - lo)])

    bounds = [(0.0, max_step)] * (2 * T)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"LP dispatch failed: {res.message}")

    a = res.x[:T]
    r = res.x[T:]
    return pd.Series(a - r, index=idx)
