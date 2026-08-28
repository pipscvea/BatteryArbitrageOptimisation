"""Stage 1-5 orchestrator — the honest, out-of-sample backtest.

Three-way chronological split: train | validation | test.
  * Stage 1 forecast + Stage 3 decision are TUNED on validation (horizon & trade sizing),
    never on test (see tuning.py).
  * the final model is refit on train+validation, then evaluated ONLY on the unseen test
    window, alongside a perfect-foresight *myopic* reference and a naive baseline.
The headline is commercial value after costs and risk, not forecast accuracy.

Requires the market CSVs — fetch them with ``python fetch_bmrs.py`` (see README). The
plumbing is verified independently on synthetic data in tests/.
"""
from __future__ import annotations

import pandas as pd
from sklearn.metrics import roc_auc_score

from config import load_battery_config, load_trading_config
from data_pipeline import assemble_market_data
from features import active_feature_columns, build_features
from labels import forward_price_change, tradeable_move
from forecasting import train_regressor, train_classifier, feature_importances
from strategy import model_requests
from simulate import simulate
from evaluate import evaluate
from tuning import sweep
import benchmarks

TRAIN_FRAC, VAL_FRAC = 0.6, 0.2  # remainder is test


def three_way_split(index: pd.Index):
    n = len(index)
    a, b = int(n * TRAIN_FRAC), int(n * (TRAIN_FRAC + VAL_FRAC))
    return index[:a], index[a:b], index[b:]


def run():
    batt, trade = load_battery_config(), load_trading_config()
    df = assemble_market_data()
    feats = build_features(df)
    cols = active_feature_columns(feats)
    valid_idx = feats.index[feats[cols].notna().all(axis=1)]
    train_idx, val_idx, test_idx = three_way_split(valid_idx)

    # Stage 3 tuning on validation (never on test).
    best, rows = sweep(df, feats, batt, trade, train_idx, val_idx)
    print("Horizon x sizing sweep (validation P&L):")
    print(f"{'horizon':>8}{'size_scale':>12}{'val P&L £':>12}{'Sharpe':>9}{'trades':>8}")
    for r in rows[:8]:
        print(f"{r.horizon:>8}{r.size_scale:>12.0f}{r.total_pnl:>12,.0f}{r.sharpe:>9.2f}{r.n_trades:>8}")
    print(f"-> chosen: horizon={best.horizon}, size_scale={best.size_scale}\n")

    # Final fit on train+validation at the chosen horizon; evaluate on test only.
    fit_idx = train_idx.union(val_idx)
    X = feats[cols]
    y_change = forward_price_change(df, best.horizon)
    y_move = tradeable_move(df, batt.efficiency_one_way, 0.0, best.horizon)
    fit_valid = fit_idx[y_change.loc[fit_idx].notna()]
    reg, _, _ = train_regressor(X.loc[fit_valid], y_change.loc[fit_valid])
    clf, _, _ = train_classifier(X.loc[fit_valid], y_move.loc[fit_valid])
    auc = roc_auc_score(y_move.loc[test_idx], clf.predict_proba(X.loc[test_idx])[:, 1])

    df_test = df.loc[test_idx]
    ssp, sbp = df_test["SystemSellPrice"], df_test["SystemBuyPrice"]
    strategies = {
        "ML forecast (tuned)": model_requests(reg, X.loc[test_idx], df_test, batt, trade,
                                              size_scale=best.size_scale),
        "Perfect-foresight myopic rule": benchmarks.perfect_foresight_myopic(
            df_test, batt, trade, best.horizon),
        "Naive time-of-day": benchmarks.naive_time_of_day(df_test, batt),
    }

    print(f"Test periods: {len(test_idx)}   P(tradeable move) AUC: {auc:.3f}\n")
    print(f"{'strategy':<34}{'P&L £':>12}{'£/MWh':>10}{'Sharpe':>9}{'MaxDD':>8}{'VaR95':>10}")
    for name, req in strategies.items():
        m = evaluate(simulate(req, ssp, sbp, batt, trade), batt)
        print(f"{name:<34}{m.total_pnl:>12,.0f}{m.gross_value_per_mwh:>10.2f}"
              f"{m.sharpe_annualised:>9.2f}{m.max_drawdown:>8.2%}{m.var_95_per_period:>10,.0f}")

    print("\nTop forecast drivers (Stage 2 attribution):")
    print(feature_importances(reg, cols).head(8).to_string())


if __name__ == "__main__":
    run()
