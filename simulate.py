"""Stage 3/4 — battery dispatch simulator.

Replays a signed per-period trade request through the battery, enforcing every physical
and commercial constraint: capacity, min/max SoC, per-period power limit, one-way
efficiency (applied on both charge and discharge => round-trip ~eff^2), transaction
cost and degradation cost. Produces a mark-to-market equity curve so downstream risk
metrics (Sharpe, drawdown, VaR) are computed on true period-by-period P&L.

Energy convention: ``request_kwh`` is battery-side energy.
    charge  a: buy a/eff from grid (losses on the way in)
    discharge r: sell r*eff to grid (losses on the way out)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import BatteryConfig, TradingConfig


def _mwh(kwh: float) -> float:
    return kwh / 1000.0


@dataclass
class SimResult:
    index: pd.Index
    starting_balance: float = 0.0
    soc_kwh: list = field(default_factory=list)
    cash: list = field(default_factory=list)      # cumulative realised cash (£)
    equity: list = field(default_factory=list)    # starting_balance + cash + liquidation
    n_trades: int = 0
    throughput_kwh: float = 0.0

    def equity_series(self) -> pd.Series:
        return pd.Series(self.equity, index=self.index)

    def period_pnl(self) -> pd.Series:
        """Per-period change in equity (£) — realised cash flow PLUS the change in
        mark-to-market value of stored energy. Used for risk metrics."""
        eq = self.equity_series()
        if eq.empty:
            return eq
        return eq.diff().fillna(eq.iloc[0] - self.starting_balance)

    def realized_pnl(self) -> pd.Series:
        """Per-period REALISED cash flow (£), excluding mark-to-market revaluation of
        held inventory. This is the honest 'what did trading earn this period' series —
        e.g. charging at a negative price is a positive realised cash flow even though the
        stored energy's MtM value is negative."""
        cash = pd.Series(self.cash, index=self.index)
        if cash.empty:
            return cash
        return cash.diff().fillna(cash.iloc[0])


def simulate(requests_kwh, prices_sell, prices_buy, batt: BatteryConfig,
             trade: TradingConfig) -> SimResult:
    """Run the dispatch. ``requests_kwh`` / ``prices_*`` are aligned pandas Series."""
    idx = requests_kwh.index
    soc = batt.capacity_kwh * batt.soc_init
    cash = 0.0
    costs_per_mwh = trade.transaction_cost_per_mwh + batt.degradation_cost_per_mwh
    max_step = batt.max_energy_per_period_kwh
    lo, hi = batt.capacity_kwh * batt.soc_min, batt.capacity_kwh * batt.soc_max
    eff = batt.efficiency_one_way

    res = SimResult(index=idx, starting_balance=trade.starting_balance)
    for t in idx:
        req = float(requests_kwh.loc[t])
        p_sell, p_buy = float(prices_sell.loc[t]), float(prices_buy.loc[t])

        if req > 0:  # charge
            a = min(req, max_step, hi - soc)
            if a > 0:
                grid = a / eff
                cash -= _mwh(grid) * p_buy + _mwh(grid) * costs_per_mwh
                soc += a
                res.n_trades += 1
                res.throughput_kwh += a
        elif req < 0:  # discharge
            r = min(-req, max_step, soc - lo)
            if r > 0:
                grid = r * eff
                cash += _mwh(grid) * p_sell - _mwh(grid) * costs_per_mwh
                soc -= r
                res.n_trades += 1
                res.throughput_kwh += r

        # Liquidation value of stored energy: what discharging it now would fetch.
        liquidation = _mwh(soc * eff) * p_sell
        res.soc_kwh.append(soc)
        res.cash.append(cash)
        res.equity.append(trade.starting_balance + cash + liquidation)

    return res
