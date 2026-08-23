"""Stage 1 — train the forecasting model.

Uses ``TimeSeriesSplit`` (NOT plain k-fold) so cross-validation never trains on data
that comes after the validation fold — the same discipline that keeps the final
train/test split honest.

The default model is a probabilistic classifier for the ``tradeable_move`` target, whose
``predict_proba`` gives P(tradeable move) for the decision layer. A regressor variant is
provided for the forward-edge target.
"""
from __future__ import annotations

from joblib import dump
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

_DEFAULT_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_leaf": [1, 4],
    "max_features": ["sqrt"],
}


def _search(estimator, X_train, y_train, param_grid, n_splits, scoring):
    grid = GridSearchCV(
        estimator,
        param_grid=param_grid or _DEFAULT_GRID,
        cv=TimeSeriesSplit(n_splits=n_splits),
        n_jobs=-1,
        verbose=0,
        scoring=scoring,
    )
    grid.fit(X_train, y_train)
    return grid


def train_classifier(X_train, y_train, param_grid=None, n_splits=4,
                     scoring="roc_auc", save_path: str | None = None):
    """Train a probabilistic classifier (for ``tradeable_move``). Returns the best model."""
    grid = _search(RandomForestClassifier(random_state=42, class_weight="balanced"),
                   X_train, y_train, param_grid, n_splits, scoring)
    best = grid.best_estimator_
    if save_path:
        dump(best, save_path)
    return best, grid.best_params_, grid.best_score_


def train_regressor(X_train, y_train, param_grid=None, n_splits=4,
                    scoring="neg_root_mean_squared_error", save_path: str | None = None):
    """Train a regressor (for the forward-edge target). Returns the best model."""
    grid = _search(RandomForestRegressor(random_state=42),
                   X_train, y_train, param_grid, n_splits, scoring)
    best = grid.best_estimator_
    if save_path:
        dump(best, save_path)
    return best, grid.best_params_, grid.best_score_


def feature_importances(model, feature_names):
    """Stage 2 attribution hook: rank drivers by importance to answer *why* the model
    predicted a move. (For rigorous attribution, swap in SHAP later.)"""
    import pandas as pd
    return pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
