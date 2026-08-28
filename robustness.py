"""Stage 4/5 — multi-regime robustness via expanding-window walk-forward.

One 3-month backtest proves nothing. Here we walk forward across every calendar quarter
in the dataset: for each test quarter, train on ALL data before it (expanding window),
then evaluate out-of-sample on the quarter. This exposes how the strategy holds up across
seasons and price regimes — the credibility upgrade over a single window.

The forecast horizon is fixed (tuning.sweep selects h=4 consistently); re-running the
full GridSearch sweep per quarter would be prohibitive, so we use the fast single-fit
regressor. Each quarter reports commercial + risk metrics for the ML strategy, the
perfect-foresight myopic reference and the naive baseline, plus aggregate stats and plots.
"""
from __future__ import annotations

from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import benchmarks
from config import load_battery_config, load_trading_config
from data_pipeline import assemble_market_data
from evaluate import evaluate
from features import active_feature_columns, build_features
from forecasting import fit_regressor
from labels import forward_price_change
from simulate import simulate
from strategy import model_requests

HORIZON = 4
MIN_TRAIN_PERIODS = 48 * 60  # need ~2 months of history before the first test quarter


@dataclass
class QuarterResult:
    quarter: str
    periods: int
    ml_pnl: float
    ml_sharpe: float
    ml_maxdd: float
    ml_var95: float
    perfect_pnl: float
    naive_pnl: float
    capture: float   # ml_pnl / perfect_pnl (share of achievable value)


def run():
    batt, trade = load_battery_config(), load_trading_config()
    df = assemble_market_data()
    feats = build_features(df)
    cols = active_feature_columns(feats)
    X = feats[cols]
    valid = X.notna().all(axis=1)
    X, dfv = X[valid], df[valid]
    y = forward_price_change(df, HORIZON)

    quarters = dfv.index.tz_localize(None).to_period("Q")
    results: list[QuarterResult] = []
    ml_equity_pieces = []

    for q in quarters.unique():
        test_mask = quarters == q
        test_idx = dfv.index[test_mask]
        train_idx = dfv.index[dfv.index < test_idx[0]]
        train_idx = train_idx[y.loc[train_idx].notna()]
        if len(train_idx) < MIN_TRAIN_PERIODS:
            continue  # not enough history yet

        model = fit_regressor(X.loc[train_idx], y.loc[train_idx])
        dft = dfv.loc[test_idx]
        ssp, sbp = dft["SystemSellPrice"], dft["SystemBuyPrice"]

        ml_req = model_requests(model, X.loc[test_idx], dft, batt, trade)
        ml_sim = simulate(ml_req, ssp, sbp, batt, trade)
        ml = evaluate(ml_sim, batt)
        perfect = evaluate(simulate(
            benchmarks.perfect_foresight_myopic(dft, batt, trade, HORIZON), ssp, sbp, batt, trade), batt)
        naive = evaluate(simulate(
            benchmarks.naive_time_of_day(dft, batt), ssp, sbp, batt, trade), batt)

        results.append(QuarterResult(
            quarter=str(q), periods=len(test_idx),
            ml_pnl=ml.total_pnl, ml_sharpe=ml.sharpe_annualised, ml_maxdd=ml.max_drawdown,
            ml_var95=ml.var_95_per_period, perfect_pnl=perfect.total_pnl,
            naive_pnl=naive.total_pnl,
            capture=(ml.total_pnl / perfect.total_pnl) if perfect.total_pnl > 0 else float("nan"),
        ))
        # Collect per-period P&L increments; cumsum across quarters gives a continuous
        # curve. Zero the first period of each quarter (it only revalues the identical
        # initial inventory, not a trade result) to avoid a stitching sawtooth.
        piece = ml_sim.period_pnl().copy()
        if len(piece):
            piece.iloc[0] = 0.0
        ml_equity_pieces.append(piece)

    _report(results)
    _plot(results, ml_equity_pieces)
    return results


def _report(results):
    print(f"\nWalk-forward by quarter (expanding window, horizon={HORIZON}):\n")
    hdr = f"{'quarter':>8}{'periods':>8}{'ML P&L':>10}{'Sharpe':>8}{'MaxDD':>8}" \
          f"{'perfect':>10}{'naive':>9}{'capture':>9}"
    print(hdr)
    for r in results:
        print(f"{r.quarter:>8}{r.periods:>8}{r.ml_pnl:>10,.0f}{r.ml_sharpe:>8.2f}"
              f"{r.ml_maxdd:>8.2%}{r.perfect_pnl:>10,.0f}{r.naive_pnl:>9,.0f}{r.capture:>9.0%}")

    if not results:
        print("(no quarters with enough history)")
        return
    ml_total = sum(r.ml_pnl for r in results)
    perfect_total = sum(r.perfect_pnl for r in results)
    beats_naive = sum(r.ml_pnl > r.naive_pnl for r in results)
    profitable = sum(r.ml_pnl > 0 for r in results)
    print(f"\nAggregate over {len(results)} quarters:")
    print(f"  ML total P&L         £{ml_total:,.0f}")
    print(f"  Capture vs perfect   {ml_total / perfect_total:.0%}" if perfect_total else "")
    print(f"  Profitable quarters  {profitable}/{len(results)}")
    print(f"  Beats naive          {beats_naive}/{len(results)}")
    print(f"  Mean quarter Sharpe  {sum(r.ml_sharpe for r in results) / len(results):.2f}")
    print(f"  Worst quarter P&L    £{min(r.ml_pnl for r in results):,.0f}")


def _plot(results, ml_equity_pieces):
    if not results:
        return
    from pathlib import Path
    figdir = Path(__file__).resolve().parent / "figs"
    figdir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    quarters = [r.quarter for r in results]
    ax[0].bar(quarters, [r.ml_pnl for r in results], color="steelblue", label="ML")
    ax[0].plot(quarters, [r.perfect_pnl for r in results], "o--", color="green", label="perfect (myopic)")
    ax[0].plot(quarters, [r.naive_pnl for r in results], "s--", color="grey", label="naive")
    ax[0].axhline(0, color="black", lw=0.8)
    ax[0].set_title(f"Out-of-sample P&L by quarter (walk-forward, h={HORIZON})")
    ax[0].set_ylabel("P&L (£)")
    ax[0].tick_params(axis="x", rotation=45)
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    pnl = pd.concat(ml_equity_pieces)
    equity = pnl.groupby(pnl.index).last().sort_index().cumsum()
    ax[1].plot(equity.index, equity.values, color="steelblue")
    ax[1].set_title("Cumulative ML P&L across regimes")
    ax[1].set_ylabel("Cumulative P&L (£)")
    ax[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out = figdir / "robustness_walkforward.png"
    fig.savefig(out, dpi=110)
    print(f"\nSaved plot -> {out}")


if __name__ == "__main__":
    run()
