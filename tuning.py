"""Stage 3 tuning — choose the forecast horizon and trade-sizing on a VALIDATION window.

A battery arbitrages the daily price swing, so a 1-period-ahead forecast is the wrong
lever — the horizon has to reach far enough to see a trough-to-peak move. We select the
horizon (and the confidence-sizing scale) by commercial P&L on a held-out *validation*
slice that sits between train and test. The test window is never touched here, so the
final backtest stays honest.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from config import BatteryConfig, TradingConfig
from evaluate import evaluate
from features import active_feature_columns
from forecasting import fit_regressor
from labels import forward_price_change
from simulate import simulate
from strategy import model_requests

DEFAULT_HORIZONS = [1, 2, 4, 8, 12, 16, 24]      # 30 min … 12 h ahead
DEFAULT_SIZE_SCALES = [0.0, 10.0, 30.0]           # 0 = all-or-nothing; else £/MWh ramp


@dataclass
class SweepRow:
    horizon: int
    size_scale: float
    total_pnl: float
    sharpe: float
    n_trades: int


def sweep(df: pd.DataFrame, feats: pd.DataFrame, batt: BatteryConfig, trade: TradingConfig,
          train_idx: pd.Index, val_idx: pd.Index,
          horizons=DEFAULT_HORIZONS, size_scales=DEFAULT_SIZE_SCALES):
    """Return ``(best_row, all_rows)`` ranked by validation P&L. One quick RF fit per
    horizon (sizing is applied at decision time, so it needs no refit)."""
    X_all = feats[active_feature_columns(feats)]
    rows: list[SweepRow] = []

    for h in horizons:
        y = forward_price_change(df, h)
        valid = X_all.notna().all(axis=1) & y.notna()
        tr = train_idx[valid.loc[train_idx]]
        model = fit_regressor(X_all.loc[tr], y.loc[tr])

        vi = val_idx[X_all.loc[val_idx].notna().all(axis=1)]
        Xval, dval = X_all.loc[vi], df.loc[vi]
        ssp, sbp = dval["SystemSellPrice"], dval["SystemBuyPrice"]
        for s in size_scales:
            req = model_requests(model, Xval, dval, batt, trade, size_scale=s)
            m = evaluate(simulate(req, ssp, sbp, batt, trade), batt)
            rows.append(SweepRow(h, s, m.total_pnl, m.sharpe_annualised, m.n_trades))

    rows.sort(key=lambda r: r.total_pnl, reverse=True)
    return rows[0], rows
