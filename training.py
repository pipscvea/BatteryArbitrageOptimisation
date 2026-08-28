"""Train and persist the forecasting models on the training window only.

Thin orchestrator. The full tuned evaluation lives in ``backtest.py``. Requires market
CSVs (fetch with ``python fetch_bmrs.py``).
"""
from __future__ import annotations

from config import load_battery_config
from data_pipeline import assemble_market_data
from features import active_feature_columns, build_features
from forecasting import train_regressor, train_classifier
from labels import forward_price_change, tradeable_move

HORIZON = 8       # matches the arbitrage horizon selected by tuning.sweep; retune there
TRAIN_FRAC = 0.8


def main():
    batt = load_battery_config()
    df = assemble_market_data()
    feats = build_features(df)
    X = feats[active_feature_columns(feats)]
    valid_idx = X.index[X.notna().all(axis=1)]
    train_idx = valid_idx[: int(len(valid_idx) * TRAIN_FRAC)]

    y_change = forward_price_change(df, HORIZON)
    y_move = tradeable_move(df, batt.efficiency_one_way, 0.0, HORIZON)
    tr = train_idx[y_change.loc[train_idx].notna()]

    _, p_reg, s_reg = train_regressor(
        X.loc[tr], y_change.loc[tr], save_path="forecast_regressor.joblib")
    _, p_clf, s_clf = train_classifier(
        X.loc[tr], y_move.loc[tr], save_path="forecast_classifier.joblib")

    print("Regressor:", p_reg, "CV score:", round(s_reg, 4))
    print("Classifier:", p_clf, "CV AUC:", round(s_clf, 4))


if __name__ == "__main__":
    main()
