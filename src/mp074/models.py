"""Model factory: one sklearn pipeline per candidate, same preprocessing for all."""

from __future__ import annotations

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mp074.cleaning import FEATURES_CATEGORICAL, FEATURES_NUMERIC

_ESTIMATORS = {
    "ridge": Ridge,
    "random_forest": RandomForestRegressor,
    "gradient_boosting": GradientBoostingRegressor,
}


def make_estimator(kind: str, params: dict[str, Any], random_state: int):
    """Build the regressor named in params.yaml."""
    if kind not in _ESTIMATORS:
        raise ValueError(f"unknown model kind: {kind}")
    if kind == "ridge":
        return Ridge(**params)
    return _ESTIMATORS[kind](random_state=random_state, **params)


def build_pipeline(estimator) -> Pipeline:
    """Impute and scale numbers, one-hot the categories, then regress.

    Keeping preprocessing inside the pipeline means the saved model.joblib
    can be called on cleaned raw columns and gives the right answer.
    """
    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), FEATURES_NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             FEATURES_CATEGORICAL),
        ]
    )
    return Pipeline([("prep", preprocess), ("model", estimator)])
