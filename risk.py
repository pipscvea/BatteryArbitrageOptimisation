"""Stage 5 — risk analysis.

Beyond headline P&L: distributional/tail risk (VaR, CVaR, downside deviation, Sortino),
drawdown depth AND duration, exposure, and two trading-desk staples:

  * forecast-error sensitivity — inject increasing noise into the forecast and watch P&L
    degrade. This quantifies how much the strategy leans on forecast quality (the same
    fragility that sank the MPC policy) and is the natural bridge to VaR/hedging work.
  * regime stress — decompose realised P&L by market regime (high vs low volatility,
    price spikes, negative prices) to see where the money is made and where it bleeds.

Builds on the per-period P&L from ``simulate.SimResult``; nothing is hardcoded.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import BatteryConfig, TradingConfig
from decision import decide
from evaluate import max_drawdown, sharpe
from simulate import SimResult, simulate

HORIZON = 4
TRAIN_FRAC = 0.8
NOISE_SIGMAS = [0, 5, 10, 20, 40, 80]  # £/MWh added to the forecast change


@dataclass
class RiskMetrics:
    vol_annualised: float
    sharpe: float
    sortino: float
    downside_dev: float
    var_95: float          # historical 5% VaR (£, positive = loss)
    cvar_95: float         # expected shortfall beyond the 5% VaR (£)
    max_drawdown: float
    max_dd_duration: int   # longest run (periods) below the prior equity peak
    skew: float
    kurtosis: float        # excess kurtosis
    exposure: float        # fraction of periods actively trading

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def _drawdown_duration(equity: pd.Series) -> int:
    peak = equity.cummax()
    below = equity < peak
    longest = run = 0
    for b in below.to_numpy():
        run = run + 1 if b else 0
        longest = max(longest, run)
    return longest


def extended_risk(res: SimResult, batt: BatteryConfig) -> RiskMetrics:
    pnl = res.period_pnl()
    equity = res.equity_series()
    ppy = batt.periods_per_year
    downside = pnl[pnl < 0]
    dd = np.sqrt((downside ** 2).mean()) if len(downside) else 0.0
    var = float(-np.quantile(pnl, 0.05)) if len(pnl) else 0.0
    tail = pnl[pnl <= -var]
    cvar = float(-tail.mean()) if len(tail) else 0.0
    traded = int((pnl != 0).sum())
    return RiskMetrics(
        vol_annualised=float(pnl.std(ddof=1) * np.sqrt(ppy)) if len(pnl) > 1 else 0.0,
        sharpe=sharpe(pnl, ppy),
        sortino=float(pnl.mean() / dd * np.sqrt(ppy)) if dd > 0 else 0.0,
        downside_dev=dd,
        var_95=var,
        cvar_95=cvar,
        max_drawdown=max_drawdown(equity),
        max_dd_duration=_drawdown_duration(equity),
        skew=float(pnl.skew()),
        kurtosis=float(pnl.kurt()),
        exposure=traded / len(pnl) if len(pnl) else 0.0,
    )


def forecast_error_sensitivity(forecast: pd.Series, df: pd.DataFrame, batt: BatteryConfig,
                               trade: TradingConfig, sigmas, size_scale: float = 0.0,
                               seed: int = 0):
    """Add Gaussian noise (std ``sigma`` £/MWh) to the forecast change, re-run the decision
    policy, and record P&L. Returns list of ``(sigma, pnl, sharpe)``. sigma=0 is baseline."""
    rng = np.random.default_rng(seed)
    ssp, sbp = df["SystemSellPrice"], df["SystemBuyPrice"]
    out = []
    for sigma in sigmas:
        noise = pd.Series(rng.normal(0, sigma, len(forecast)), index=forecast.index) if sigma > 0 else 0.0
        fc = forecast + noise
        req = pd.Series(
            {t: decide(float(fc.loc[t]), float(ssp.loc[t]), float(sbp.loc[t]),
                       batt, trade, size_scale).request_kwh for t in forecast.index},
        ).reindex(forecast.index).fillna(0.0)
        m = simulate(req, ssp, sbp, batt, trade)
        pnl = m.equity_series().iloc[-1] - m.starting_balance
        out.append((float(sigma), float(pnl), sharpe(m.period_pnl(), batt.periods_per_year)))
    return out


def stress_by_regime(res: SimResult, df: pd.DataFrame) -> pd.DataFrame:
    """Attribute realised per-period P&L to market regimes. Uses the ML sim's period P&L
    (SoC-consistent) grouped by regime masks, so it shows where value is made/lost."""
    pnl = res.period_pnl()
    mid = (df["SystemSellPrice"] + df["SystemBuyPrice"]) / 2
    vol = mid.diff().abs().rolling(48).std()
    hi_vol = vol > vol.median()
    regimes = {
        "high volatility": hi_vol,
        "low volatility": ~hi_vol,
        "price spike (>1.5x 48-mean)": df["SystemSellPrice"] > 1.5 * df["SystemSellPrice"].rolling(48).mean(),
        "negative price": df["SystemSellPrice"] < 0,
        "all periods": pd.Series(True, index=df.index),
    }
    rows = []
    for name, mask in regimes.items():
        m = mask.reindex(pnl.index).fillna(False)
        seg = pnl[m]
        rows.append({
            "regime": name, "periods": int(m.sum()),
            "total_pnl": float(seg.sum()),
            "pnl_per_period": float(seg.mean()) if len(seg) else 0.0,
            "share_of_pnl": float(seg.sum() / pnl.sum()) if pnl.sum() else float("nan"),
        })
    return pd.DataFrame(rows)


def run():
    """Full Stage 5 report on the tuned ML strategy over the held-out test window."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    from config import load_battery_config, load_trading_config
    from data_pipeline import assemble_market_data
    from features import active_feature_columns, build_features
    from forecasting import fit_regressor
    from labels import forward_price_change
    from strategy import model_requests

    batt, trade = load_battery_config(), load_trading_config()
    df = assemble_market_data()
    feats = build_features(df)
    cols = active_feature_columns(feats)
    X = feats[cols]
    valid_idx = X.index[X.notna().all(axis=1)]
    cut = int(len(valid_idx) * TRAIN_FRAC)
    train_idx, test_idx = valid_idx[:cut], valid_idx[cut:]
    y = forward_price_change(df, HORIZON)
    tr = train_idx[y.loc[train_idx].notna()]

    model = fit_regressor(X.loc[tr], y.loc[tr])
    df_test = df.loc[test_idx]
    req = model_requests(model, X.loc[test_idx], df_test, batt, trade)
    res = simulate(req, df_test["SystemSellPrice"], df_test["SystemBuyPrice"], batt, trade)

    rm = extended_risk(res, batt)
    print(f"\nRisk metrics (tuned ML, {len(test_idx)} test periods):")
    for k, v in rm.as_dict().items():
        print(f"  {k:<18} {v:,.3f}" if abs(v) < 1000 else f"  {k:<18} {v:,.0f}")

    forecast = pd.Series(model.predict(X.loc[test_idx]), index=test_idx)
    sens = forecast_error_sensitivity(forecast, df_test, batt, trade, NOISE_SIGMAS)
    print("\nForecast-error sensitivity (noise std -> P&L):")
    for sigma, pnl, sh in sens:
        print(f"  sigma={sigma:>4.0f} £/MWh   P&L £{pnl:>8,.0f}   Sharpe {sh:>6.2f}")

    regimes = stress_by_regime(res, df_test)
    print("\nRegime stress (P&L attribution):")
    print(regimes.to_string(index=False))

    _plot(res, sens, plt, Path(__file__).resolve().parent / "figs")
    return rm, sens, regimes


def _plot(res, sens, plt, figdir):
    figdir.mkdir(exist_ok=True)
    pnl = res.period_pnl()
    equity = res.equity_series() - res.starting_balance
    var = float(-np.quantile(pnl, 0.05))

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].hist(pnl, bins=60, color="steelblue")
    ax[0].axvline(-var, color="red", ls="--", label=f"VaR95 £{var:,.0f}")
    ax[0].set_title("Per-period P&L distribution")
    ax[0].set_xlabel("£ / period")
    ax[0].legend()

    ax[1].fill_between(equity.index, (equity.cummax() - equity), color="crimson", alpha=0.6)
    ax[1].set_title("Drawdown from peak (£)")
    ax[1].tick_params(axis="x", rotation=45)

    sig = [s for s, _, _ in sens]
    pn = [p for _, p, _ in sens]
    ax[2].plot(sig, pn, "o-", color="darkgreen")
    ax[2].axhline(0, color="black", lw=0.8)
    ax[2].set_title("Forecast-error sensitivity")
    ax[2].set_xlabel("forecast noise std (£/MWh)")
    ax[2].set_ylabel("P&L (£)")
    ax[2].grid(True, alpha=0.3)

    fig.tight_layout()
    out = figdir / "risk_analysis.png"
    fig.savefig(out, dpi=110)
    print(f"\nSaved plot -> {out}")


if __name__ == "__main__":
    run()
