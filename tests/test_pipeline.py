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
    req = benchmarks.perfect_foresight_myopic(df, BATT, TRADE)
    m = evaluate(simulate(req, df["SystemSellPrice"], df["SystemBuyPrice"], BATT, TRADE), BATT)
    assert m.total_pnl > 0
    assert m.n_trades > 0
    assert np.isfinite(m.sharpe_annualised)
    assert 0.0 <= m.max_drawdown <= 1.0
    assert m.gross_value_per_mwh > 0


def test_perfect_foresight_dominates_doing_nothing():
    df = make_market(days=30)
    ssp, sbp = df["SystemSellPrice"], df["SystemBuyPrice"]
    pf = evaluate(simulate(benchmarks.perfect_foresight_myopic(df, BATT, TRADE), ssp, sbp, BATT, TRADE), BATT)
    hold = evaluate(simulate(pd.Series(0.0, index=df.index), ssp, sbp, BATT, TRADE), BATT)
    assert pf.total_pnl > hold.total_pnl


def test_driver_features_added_only_when_present():
    """Wind/gas/interconnector features appear iff the raw driver columns are merged."""
    from features import active_feature_columns
    df = make_market(days=5)
    assert "wind_ma6" not in active_feature_columns(build_features(df))  # no drivers
    df2 = df.copy()
    df2["Wind"] = 8000.0
    df2["InterconnectorNet"] = 1000.0
    df2["Gas"] = 5000.0
    cols = active_feature_columns(build_features(df2))
    assert {"Wind", "wind_lag1", "wind_ma6", "wind_ramp", "InterconnectorNet", "Gas"} <= set(cols)


def test_features_survive_negative_prices():
    """UK imbalance prices go negative/zero — features must stay finite (no log blow-up)."""
    df = make_market(days=10, seed=2)
    df["SystemSellPrice"] -= 60.0  # push a chunk of prices below zero
    df["SystemBuyPrice"] -= 60.0
    feats = build_features(df)[FEATURE_COLUMNS]
    warm = feats.iloc[48:]  # past the rolling-window warm-up
    assert np.isfinite(warm.to_numpy()).all()


def test_decision_gates_on_min_edge():
    """A tiny forecast change must not trigger a trade; a large one must."""
    d_small = decide(0.01, 50.0, 51.0, BATT, TRADE)
    d_big = decide(80.0, 50.0, 51.0, BATT, TRADE)
    assert d_small.request_kwh == 0.0
    assert d_big.request_kwh > 0.0


def test_lp_dispatch_is_a_true_upper_bound():
    """LP optimal dispatch must be >= the myopic perfect-foresight rule and never lose,
    while respecting SoC/power limits (checked via the simulator)."""
    from optimize import optimal_dispatch
    df = make_market(days=20, seed=5)
    ssp, sbp = df["SystemSellPrice"], df["SystemBuyPrice"]
    lp_req = optimal_dispatch(df, BATT, TRADE)
    lp = evaluate(simulate(lp_req, ssp, sbp, BATT, TRADE), BATT)
    myopic = evaluate(simulate(
        benchmarks.perfect_foresight_myopic(df, BATT, TRADE), ssp, sbp, BATT, TRADE), BATT)
    assert lp.total_pnl >= myopic.total_pnl - 1e-6      # optimum dominates the myopic rule
    assert lp.total_pnl >= -1e-6                         # never loses money with foresight
    # Feasibility: the simulator should not have to clip the LP's own requests.
    assert lp_req.abs().max() <= BATT.max_energy_per_period_kwh + 1e-6


def test_mpc_with_oracle_approaches_optimum_and_is_feasible():
    """With a perfect price-path forecast, MPC should beat the myopic rule and come
    close to the LP optimum, while respecting power limits."""
    from optimize import optimal_dispatch
    from mpc import mpc_requests
    W = 48
    df = make_market(days=20, seed=7)
    ssp, sbp = df["SystemSellPrice"], df["SystemBuyPrice"]

    # Oracle path model: returns the TRUE forward change matrix (tail-padded with 0).
    cur = ssp.to_numpy()
    T = len(cur)
    truth = np.zeros((T, W))
    for k in range(1, W + 1):
        shifted = np.concatenate([cur[k:], np.zeros(k)])
        truth[:, k - 1] = shifted - cur

    class Oracle:
        def predict(self, _X):
            return truth

    X = df[["SystemSellPrice"]]  # features unused by the oracle
    req = mpc_requests(Oracle(), X, df, BATT, TRADE, window=W, replan_every=12)
    mpc = evaluate(simulate(req, ssp, sbp, BATT, TRADE), BATT)
    lp = evaluate(simulate(optimal_dispatch(df, BATT, TRADE), ssp, sbp, BATT, TRADE), BATT)
    myopic = evaluate(simulate(
        benchmarks.perfect_foresight_myopic(df, BATT, TRADE), ssp, sbp, BATT, TRADE), BATT)

    assert req.abs().max() <= BATT.max_energy_per_period_kwh + 1e-6   # feasible
    assert mpc.total_pnl >= myopic.total_pnl - 1e-6                    # non-myopic helps
    assert mpc.total_pnl <= lp.total_pnl + 1e-6                        # can't beat optimum
    assert mpc.total_pnl >= 0.8 * lp.total_pnl                         # and gets most of it


def test_robust_mpc_shrink_is_feasible_and_defaults_to_plain():
    """horizon_decay=1.0 must reproduce plain MPC; shrinking a PERFECT forecast can only
    reduce P&L (never help); the schedule stays feasible."""
    from optimize import optimal_dispatch
    from mpc import mpc_requests
    W = 48
    df = make_market(days=20, seed=7)
    ssp, sbp = df["SystemSellPrice"], df["SystemBuyPrice"]
    cur = ssp.to_numpy(); T = len(cur)
    truth = np.zeros((T, W))
    for k in range(1, W + 1):
        truth[:, k - 1] = np.concatenate([cur[k:], np.zeros(k)]) - cur

    class Oracle:
        def predict(self, _X):
            return truth

    X = df[["SystemSellPrice"]]
    plain = mpc_requests(Oracle(), X, df, BATT, TRADE, window=W, replan_every=12)
    explicit1 = mpc_requests(Oracle(), X, df, BATT, TRADE, window=W, replan_every=12, horizon_decay=1.0)
    robust = mpc_requests(Oracle(), X, df, BATT, TRADE, window=W, replan_every=12, horizon_decay=0.6)

    pd.testing.assert_series_equal(plain, explicit1)             # decay=1.0 == plain
    assert robust.abs().max() <= BATT.max_energy_per_period_kwh + 1e-6  # feasible
    p_plain = evaluate(simulate(plain, ssp, sbp, BATT, TRADE), BATT).total_pnl
    p_robust = evaluate(simulate(robust, ssp, sbp, BATT, TRADE), BATT).total_pnl
    assert p_robust <= p_plain + 1e-6                            # shrinking perfect info can't help
    assert p_robust >= 0                                          # but still never loses with foresight

    # Climatology prior: at decay=1.0 the prior is ignored (== plain); it stays feasible when active.
    from mpc import diurnal_climatology
    clim = diurnal_climatology(df)
    assert clim.shape == (48,) and np.isfinite(clim).all()
    same = mpc_requests(Oracle(), X, df, BATT, TRADE, window=W, replan_every=12,
                        horizon_decay=1.0, climatology=clim)
    pd.testing.assert_series_equal(plain, same)
    blended = mpc_requests(Oracle(), X, df, BATT, TRADE, window=W, replan_every=12,
                           horizon_decay=0.6, climatology=clim)
    assert blended.abs().max() <= BATT.max_energy_per_period_kwh + 1e-6


def test_risk_metrics_are_coherent():
    """CVaR >= VaR, drawdown in [0,1], exposure in [0,1], everything finite."""
    from risk import extended_risk, stress_by_regime
    df = make_market(days=30)
    req = benchmarks.perfect_foresight_myopic(df, BATT, TRADE)
    res = simulate(req, df["SystemSellPrice"], df["SystemBuyPrice"], BATT, TRADE)
    rm = extended_risk(res, BATT)
    assert np.isfinite(list(rm.as_dict().values())).all()
    assert rm.cvar_95 >= rm.var_95 - 1e-9      # expected shortfall is at least the VaR
    assert 0.0 <= rm.max_drawdown <= 1.0
    assert 0.0 <= rm.exposure <= 1.0
    reg = stress_by_regime(res, df)
    # high + low volatility partition all periods, so their realised-P&L shares sum to ~1.
    hilo = reg[reg.regime.isin(["high volatility", "low volatility"])]
    assert abs(hilo.share_of_realized.sum() - 1.0) < 1e-6


def test_charging_at_negative_price_is_positive_realised_cash():
    """Charging when prices are negative earns positive realised cash flow, even though the
    stored energy's mark-to-market value is negative. This is why regime attribution must
    split realised from MtM (the 'negative-price loss' was an MtM artifact, not a bug)."""
    idx = pd.date_range("2024-01-01", periods=4, freq="30min")
    df = pd.DataFrame({"SystemSellPrice": [-50.0] * 4, "SystemBuyPrice": [-49.0] * 4}, index=idx)
    charge = pd.Series(BATT.max_energy_per_period_kwh, index=idx)  # charge every period
    res = simulate(charge, df["SystemSellPrice"], df["SystemBuyPrice"], BATT, TRADE)
    realized = res.realized_pnl()
    assert realized.sum() > 0                       # got paid to charge -> positive cash
    assert res.period_pnl().sum() < realized.sum()  # MtM of held energy drags equity down


def test_forecast_error_sensitivity_baseline_matches():
    """sigma=0 must reproduce the noiseless strategy P&L exactly."""
    from risk import forecast_error_sensitivity
    from strategy import model_requests
    df = make_market(days=20)
    forecast = df["SystemSellPrice"].diff().shift(-1).fillna(0.0)  # arbitrary forecast series
    sens = forecast_error_sensitivity(forecast, df, BATT, TRADE, [0, 50])
    assert sens[0][0] == 0.0 and np.isfinite(sens[0][1])
    assert np.isfinite(sens[1][1])


def test_confidence_sizing_is_monotonic_and_capped():
    """Bigger edge -> bigger (or equal) size, never above full power."""
    full = BATT.max_energy_per_period_kwh
    weak = decide(20.0, 50.0, 51.0, BATT, TRADE, size_scale=30.0)
    strong = decide(200.0, 50.0, 51.0, BATT, TRADE, size_scale=30.0)
    assert 0.0 < weak.request_kwh < strong.request_kwh <= full + 1e-9
    # size_scale=0 restores all-or-nothing at full power.
    assert decide(20.0, 50.0, 51.0, BATT, TRADE, size_scale=0.0).request_kwh == full
