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


def solve_dispatch(p_sell, p_buy, soc0: float, batt: BatteryConfig, trade: TradingConfig):
    """Core LP. Given price arrays and an initial SoC (kWh), return battery-side
    ``(a, r)`` arrays (charge / discharge energy per period). Reused by the full-window
    optimum (real prices) and by MPC (forecast prices over a rolling window)."""
    p_sell = np.asarray(p_sell, dtype=float)
    p_buy = np.asarray(p_buy, dtype=float)
    T = len(p_sell)
    if T == 0:
        return np.array([]), np.array([])

    eff = batt.efficiency_one_way
    costs = trade.transaction_cost_per_mwh + batt.degradation_cost_per_mwh
    max_step = batt.max_energy_per_period_kwh
    lo, hi = batt.capacity_kwh * batt.soc_min, batt.capacity_kwh * batt.soc_max
    p_final = p_sell[-1]

    # Per-unit profit (£ per kWh battery-side). /1000 converts kWh -> MWh for pricing.
    prof_a = (-(p_buy + costs) / eff + eff * p_final) / 1000.0
    prof_r = (eff * (p_sell - costs) - eff * p_final) / 1000.0
    c = -np.concatenate([prof_a, prof_r])  # linprog minimises

    L = sparse.tril(np.ones((T, T)))
    A_ub = sparse.vstack([sparse.hstack([L, -L]), sparse.hstack([-L, L])]).tocsr()
    b_ub = np.concatenate([np.full(T, hi - soc0), np.full(T, soc0 - lo)])

    bounds = [(0.0, max_step)] * (2 * T)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"LP dispatch failed: {res.message}")
    return res.x[:T], res.x[T:]


def optimal_dispatch(df: pd.DataFrame, batt: BatteryConfig, trade: TradingConfig) -> pd.Series:
    """Full-window perfect-foresight optimum: the true upper bound. Returns a signed
    request Series (battery-side kWh; + charge / - discharge) for ``simulate.simulate``."""
    if df.empty:
        return pd.Series(dtype=float)
    a, r = solve_dispatch(df["SystemSellPrice"], df["SystemBuyPrice"],
                          batt.capacity_kwh * batt.soc_init, batt, trade)
    return pd.Series(a - r, index=df.index)
