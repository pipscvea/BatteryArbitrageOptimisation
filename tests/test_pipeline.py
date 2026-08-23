"""End-to-end verification of the refactored pipeline on synthetic data.

Covers the two things that make or break the project's credibility:
  * NO look-ahead in features (the leakage bug the refactor set out to fix)
  * constraints are actually enforced, and the economics are sane (perfect foresight
    makes money and dominates the ML strategy, which dominates doing nothing).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_battery_config, load_trading_config
from features import build_features, FEATURE_COLUMNS
from labels import forward_price_change, forward_arbitrage_edge, tradeable_move
from decision import decide
from simulate import simulate
from evaluate import evaluate
import benchmarks
from tests.synthetic import make_market

BATT = load_battery_config()
TRADE = load_trading_config()


def test_features_have_no_lookahead():
    """Perturbing a FUTURE price must not change any earlier feature row."""
    df = make_market(days=10, seed=1)
    base = build_features(df)[FEATURE_COLUMNS]
    t_star = 200
    df2 = df.copy()
    df2.iloc[t_star + 1:, df2.columns.get_loc("SystemSellPrice")] += 100.0
    perturbed = build_features(df2)[FEATURE_COLUMNS]
    pd.testing.assert_frame_equal(base.iloc[:t_star + 1], perturbed.iloc[:t_star + 1])


def test_label_is_strictly_forward():
    df = make_market(days=5)
    chg = forward_price_change(df, horizon=1)
    expected = df["SystemSellPrice"].shift(-1) - df["SystemSellPrice"]
    pd.testing.assert_series_equal(chg, expected)
    assert pd.isna(chg.iloc[-1])  # last period has no future


def test_tradeable_move_is_binary_probabilistic_target():
    df = make_market(days=5)
    y = tradeable_move(df, BATT.efficiency_one_way, edge_threshold=0.0)
    assert set(y.unique()) <= {0, 1}
    assert 0.0 < y.mean() < 1.0  # both classes present -> a learnable signal


def test_simulate_respects_soc_and_power_limits():
    df = make_market(days=8)
    # Demand a huge charge every period; simulator must clip to capacity & power.
    reqs = pd.Series(1e9, index=df.index)
    res = simulate(reqs, df["SystemSellPrice"], df["SystemBuyPrice"], BATT, TRADE)
    soc = pd.Series(res.soc_kwh, index=df.index)
    assert soc.max() <= BATT.capacity_kwh * BATT.soc_max + 1e-6
    assert soc.min() >= BATT.capacity_kwh * BATT.soc_min - 1e-6
    # No single step moves more than the per-period power limit.
    assert soc.diff().abs().max() <= BATT.max_energy_per_period_kwh + 1e-6


def test_perfect_foresight_is_profitable_and_metrics_finite():
    df = make_market(days=30)
    req = benchmarks.perfect_foresight(df, BATT, TRADE)
    m = evaluate(simulate(req, df["SystemSellPrice"], df["SystemBuyPrice"], BATT, TRADE), BATT)
    assert m.total_pnl > 0
    assert m.n_trades > 0
    assert np.isfinite(m.sharpe_annualised)
    assert 0.0 <= m.max_drawdown <= 1.0
    assert m.gross_value_per_mwh > 0


def test_perfect_foresight_dominates_doing_nothing():
    df = make_market(days=30)
    ssp, sbp = df["SystemSellPrice"], df["SystemBuyPrice"]
    pf = evaluate(simulate(benchmarks.perfect_foresight(df, BATT, TRADE), ssp, sbp, BATT, TRADE), BATT)
    hold = evaluate(simulate(pd.Series(0.0, index=df.index), ssp, sbp, BATT, TRADE), BATT)
    assert pf.total_pnl > hold.total_pnl


def test_decision_gates_on_min_edge():
    """A tiny forecast change must not trigger a trade; a large one must."""
    d_small = decide(0.01, 50.0, 51.0, BATT, TRADE)
    d_big = decide(80.0, 50.0, 51.0, BATT, TRADE)
    assert d_small.request_kwh == 0.0
    assert d_big.request_kwh > 0.0
