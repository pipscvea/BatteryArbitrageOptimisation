"""Stage 1-5 orchestrator — the honest, out-of-sample backtest.

Fixes the core leakage bug in the old ``apply_model.py`` (which predicted over the WHOLE
dataset, train + test). Here:
  * features look backward, the label looks forward (see features.py / labels.py)
  * the split is strictly chronological — train on the earlier window, evaluate ONLY on
    the later, unseen window
  * the ML strategy is reported alongside a perfect-foresight *myopic* reference and a
    naive baseline, all on the SAME test window, so the headline is commercial value, not
    forecast accuracy. (A true optimal-dispatch upper bound via LP/DP is Stage 3 work.)

Requires the market CSVs (see README) — will not run on the cloned repo alone. The
plumbing is verified independently on synthetic data in tests/.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config import load_battery_config, load_trading_config
from data_pipeline import assemble_market_data
from features import FEATURE_COLUMNS, build_features
from labels import forward_price_change, tradeable_move
from forecasting import train_regressor, train_classifier, feature_importances
from strategy import model_requests
from simulate import simulate
from evaluate import evaluate
import benchmarks

HORIZON = 1
TEST_FRAC = 0.2


def chronological_split(index: pd.Index, test_frac: float = TEST_FRAC):
    cut = int(len(index) * (1 - test_frac))
    return index[:cut], index[cut:]


def prepare(df: pd.DataFrame, batt):
    feats = build_features(df)
    y_change = forward_price_change(df, HORIZON)
    y_move = tradeable_move(df, batt.efficiency_one_way, edge_threshold=0.0, horizon=HORIZON)
    X = feats[FEATURE_COLUMNS]
    valid = X.notna().all(axis=1) & y_change.notna()
    return X[valid], y_change[valid], y_move[valid]


def run():
    batt, trade = load_battery_config(), load_trading_config()
    df = assemble_market_data()
    X, y_change, y_move = prepare(df, batt)
    train_idx, test_idx = chronological_split(X.index)

    # Stage 1: forecast (regressor drives decisions; classifier gives P(tradeable move)).
    reg, _, _ = train_regressor(X.loc[train_idx], y_change.loc[train_idx])
    clf, _, _ = train_classifier(X.loc[train_idx], y_move.loc[train_idx])
    auc = roc_auc_score(y_move.loc[test_idx], clf.predict_proba(X.loc[test_idx])[:, 1])

    # Stages 3-5, TEST WINDOW ONLY.
    df_test = df.loc[test_idx]
    ssp, sbp = df_test["SystemSellPrice"], df_test["SystemBuyPrice"]

    strategies = {
        "ML forecast": model_requests(reg, X.loc[test_idx], df_test, batt, trade),
        "Perfect-foresight myopic rule": benchmarks.perfect_foresight_myopic(df_test, batt, trade, HORIZON),
        "Naive time-of-day": benchmarks.naive_time_of_day(df_test, batt),
    }

    print(f"Test periods: {len(test_idx)}   P(tradeable move) AUC: {auc:.3f}\n")
    print(f"{'strategy':<34}{'P&L £':>12}{'£/MWh':>10}{'Sharpe':>9}{'MaxDD':>8}{'VaR95':>10}")
    for name, req in strategies.items():
        m = evaluate(simulate(req, ssp, sbp, batt, trade), batt)
        print(f"{name:<34}{m.total_pnl:>12,.0f}{m.gross_value_per_mwh:>10.2f}"
              f"{m.sharpe_annualised:>9.2f}{m.max_drawdown:>8.2%}{m.var_95_per_period:>10,.0f}")

    print("\nTop forecast drivers (Stage 2 attribution):")
    print(feature_importances(reg, FEATURE_COLUMNS).head(6).to_string())


if __name__ == "__main__":
    run()
