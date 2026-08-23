"""Train and persist the forecasting models on the training window only.

Thin orchestrator. The full evaluation lives in ``backtest.py``. Requires market CSVs.
"""
from __future__ import annotations

from config import load_battery_config
from data_pipeline import assemble_market_data
from forecasting import train_regressor, train_classifier
from labels import forward_price_change, tradeable_move
from backtest import prepare, chronological_split


def main():
    batt = load_battery_config()
    df = assemble_market_data()
    X, y_change, y_move = prepare(df, batt)
    train_idx, _ = chronological_split(X.index)

    _, p_reg, s_reg = train_regressor(
        X.loc[train_idx], y_change.loc[train_idx], save_path="forecast_regressor.joblib")
    _, p_clf, s_clf = train_classifier(
        X.loc[train_idx], y_move.loc[train_idx], save_path="forecast_classifier.joblib")

    print("Regressor:", p_reg, "CV score:", round(s_reg, 4))
    print("Classifier:", p_clf, "CV AUC:", round(s_clf, 4))


if __name__ == "__main__":
    main()
