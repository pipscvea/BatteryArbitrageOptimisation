"""Stage 3 — receding-horizon (Model Predictive Control) dispatch.

This is the causal counterpart of the perfect-foresight LP: at each replan point it
forecasts the price path over the next ``window`` periods (from information available
*now*), solves the same battery LP over that window, executes the first ``replan_every``
actions, then rolls forward with the updated state of charge and re-plans.

Unlike ``optimize.optimal_dispatch`` it never sees future prices — it only sees the
model's *forecast* of them — so it IS a tradeable strategy, and the gap between it and
the LP optimum is pure forecast error. It reuses ``optimize.solve_dispatch`` verbatim,
turning the forecast into a decision rather than using the myopic edge heuristic.

EMPIRICAL FINDING (real Q-window backtest): with an oracle price path MPC reaches ~LP
(see tests), but with the *actual* forecast it UNDERPERFORMS the myopic edge heuristic and
barely beats naive. Feeding a point forecast into an LP makes the optimiser over-confident
- it commits to forecast paths that are wrong and the errors compound - whereas the
edge-gate only acts on high-confidence single-step signals. This is the project's central
lesson in miniature: forecast accuracy != P&L, and added optimisation complexity only pays
once the forecast is good and its *uncertainty* is respected. Principled fixes
(robust/stochastic MPC, shrinking far-horizon forecasts, chance constraints) are future work.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import BatteryConfig, TradingConfig
from optimize import solve_dispatch


def mpc_requests(path_model, X: pd.DataFrame, df: pd.DataFrame,
                 batt: BatteryConfig, trade: TradingConfig,
                 window: int = 48, replan_every: int = 12,
                 mean_spread: float | None = None) -> pd.Series:
    """Return the signed request Series (battery-side kWh) produced by MPC.

    ``path_model`` predicts the (window,) vector of forward price *changes* from features.
    ``mean_spread`` (SBP-SSP) approximates the forecast buy price as sell + spread; if
    None it is estimated from ``df``.
    """
    idx = X.index
    T = len(idx)
    if T == 0:
        return pd.Series(dtype=float)
    if mean_spread is None:
        mean_spread = float((df["SystemBuyPrice"] - df["SystemSellPrice"]).mean())

    ssp_now = df["SystemSellPrice"].to_numpy(dtype=float)
    preds = np.asarray(path_model.predict(X), dtype=float)  # (T, window); DataFrame keeps names

    soc = batt.capacity_kwh * batt.soc_init
    requests = np.zeros(T)
    s = 0
    while s < T:
        w = min(window, T - s)                       # shrink horizon near the end
        sell_hat = ssp_now[s] + preds[s, :w]         # forecast sell-price path
        buy_hat = sell_hat + mean_spread             # approx forecast buy-price path
        a, r = solve_dispatch(sell_hat, buy_hat, soc, batt, trade)

        k = min(replan_every, w)                      # execute only the first k actions
        act = a[:k] - r[:k]
        # Clip to the exact feasible SoC/power the simulator will enforce, and roll SoC.
        lo, hi = batt.capacity_kwh * batt.soc_min, batt.capacity_kwh * batt.soc_max
        step = batt.max_energy_per_period_kwh
        for j in range(k):
            move = float(np.clip(act[j], -step, step))
            move = float(np.clip(move, lo - soc, hi - soc))
            requests[s + j] = move
            soc += move
        s += k

    return pd.Series(requests, index=idx)
