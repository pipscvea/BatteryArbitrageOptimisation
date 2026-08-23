"""Stage 1 — forward-looking forecast targets.

These are the ONLY place forward information (``shift(-h)``) is allowed. Because the
target looks forward and every feature looks backward, a chronological train/test split
gives an honest, leakage-free estimate of out-of-sample performance.

We deliberately forecast *commercially meaningful* quantities rather than "the next
price": the forward arbitrage edge, its direction, and — the trading-oriented one — the
probability that the next period offers a spread large enough to justify a trade.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def forward_price_change(df: pd.DataFrame, horizon: int = 1, price_col: str = "SystemSellPrice"):
    """Regression target: price change over the next ``horizon`` periods (£/MWh)."""
    return df[price_col].shift(-horizon) - df[price_col]


def forward_arbitrage_edge(df: pd.DataFrame, efficiency: float, horizon: int = 1):
    """Regression target: gross £/MWh from buying now and selling ``horizon`` ahead.

    Buy at the current System Buy Price, sell at the future System Sell Price, losing the
    one-way efficiency on the way in. Positive => a charge-now/sell-later opportunity.
    (Costs are applied later in the decision layer, not baked into the label.)
    """
    future_sell = df["SystemSellPrice"].shift(-horizon)
    return future_sell * efficiency - df["SystemBuyPrice"]


def forward_direction(df: pd.DataFrame, horizon: int = 1, price_col: str = "SystemSellPrice"):
    """Classification target: sign of the forward price change (-1 / 0 / +1)."""
    return np.sign(forward_price_change(df, horizon, price_col)).astype("Int64")


def tradeable_move(df: pd.DataFrame, efficiency: float, edge_threshold: float,
                   horizon: int = 1) -> pd.Series:
    """Probabilistic target (Stage 1 headline): 1 if the next period's arbitrage edge
    exceeds ``edge_threshold`` £/MWh, else 0.

    Training a *classifier* on this and using ``predict_proba`` yields:
        "P(next period offers a spread large enough to justify a battery trade)".
    """
    edge = forward_arbitrage_edge(df, efficiency, horizon)
    return (edge > edge_threshold).astype(int)
