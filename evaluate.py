"""Stage 4 & 5 — commercial outcome and risk analysis.

Turns a simulation into the numbers that matter for a trading desk: total P&L, gross
value per MWh cycled, annualised Sharpe, maximum drawdown, historical VaR, hit rate.
No numbers are hardcoded — everything is derived from the fed simulation, so figures are
only as real as the data behind them (synthetic today; UK market data once sourced).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import BatteryConfig
from simulate import SimResult


@dataclass
class Metrics:
    total_pnl: float
    gross_value_per_mwh: float
    sharpe_annualised: float
    max_drawdown: float          # fraction of peak equity (0..1)
    var_95_per_period: float     # historical 5% VaR of per-period P&L (£, positive = loss)
    hit_rate: float              # fraction of trading periods with positive P&L
    n_trades: int
    throughput_mwh: float

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    running_peak = equity.cummax()
    # Guard against zero/negative peaks when expressing as a fraction.
    denom = running_peak.replace(0, np.nan).abs()
    dd = (running_peak - equity) / denom
    return float(dd.max(skipna=True) or 0.0)


def sharpe(period_pnl: pd.Series, periods_per_year: int) -> float:
    std = period_pnl.std(ddof=1)
    if not np.isfinite(std) or std == 0:
        return 0.0
    return float(period_pnl.mean() / std * np.sqrt(periods_per_year))


def historical_var(period_pnl: pd.Series, alpha: float = 0.05) -> float:
    """5% historical VaR expressed as a positive loss magnitude (£)."""
    if period_pnl.empty:
        return 0.0
    return float(-np.quantile(period_pnl, alpha))


def evaluate(res: SimResult, batt: BatteryConfig) -> Metrics:
    equity = res.equity_series()
    pnl = res.period_pnl()
    total = float(equity.iloc[-1] - res.starting_balance) if len(equity) else 0.0
    throughput_mwh = res.throughput_kwh / 1000.0
    traded = pnl[pnl != 0]
    return Metrics(
        total_pnl=total,
        gross_value_per_mwh=(total / throughput_mwh) if throughput_mwh > 0 else 0.0,
        sharpe_annualised=sharpe(pnl, batt.periods_per_year),
        max_drawdown=max_drawdown(equity),
        var_95_per_period=historical_var(pnl),
        hit_rate=float((traded > 0).mean()) if len(traded) else 0.0,
        n_trades=res.n_trades,
        throughput_mwh=throughput_mwh,
    )
