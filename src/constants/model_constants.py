"""Central place for model names and hyperparameters."""

from __future__ import annotations

from typing import Any


TARGET_COLUMN = "sentiment"

# Change this value to switch the training model.
# 1 -> RandomForest
# 2 -> LogisticRegression
# 3 -> GradientBoosting
# 4 -> MultinomialNB
# 5 -> XGBoost
SELECTED_MODEL_KEY = 1

MODEL_OPTIONS: dict[int, dict[str, Any]] = {
    1: {
        "model_name": "RandomForest",
        "model_params": {
            "n_estimators": 200,
            "max_depth": 10,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "random_state": 42,
        },
    },
    2: {
        "model_name": "LogisticRegression",
        "model_params": {
            "C": 1.0,
            "penalty": "l1",
            "solver": "liblinear",
            "max_iter": 800,
        },
    },
    3: {
        "model_name": "GradientBoosting",
        "model_params": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3,
            "random_state": 42,
        },
    },
    4: {
        "model_name": "MultinomialNB",
        "model_params": {
            "alpha": 1.0,
        },
    },
    5: {
        "model_name": "XGBoost",
        "model_params": {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "random_state": 42,
            "eval_metric": "logloss",
        },
    },
}

MLFLOW_MODEL_CONFIG: dict[str, Any] = {
    "model_name": "sentiment-classifier",
    "candidate_alias": "candidate",
    "production_alias": "champion",
}


def get_model_building_config() -> dict[str, Any]:
    """Return a copy of the configured training model settings."""
    if SELECTED_MODEL_KEY == 1:
        selected_model = MODEL_OPTIONS[1]
    elif SELECTED_MODEL_KEY == 2:
        selected_model = MODEL_OPTIONS[2]
    elif SELECTED_MODEL_KEY == 3:
        selected_model = MODEL_OPTIONS[3]
    elif SELECTED_MODEL_KEY == 4:
        selected_model = MODEL_OPTIONS[4]
    elif SELECTED_MODEL_KEY == 5:
        selected_model = MODEL_OPTIONS[5]
    else:
        raise ValueError(
            f"Unsupported SELECTED_MODEL_KEY '{SELECTED_MODEL_KEY}'. "
            f"Choose one of {sorted(MODEL_OPTIONS)}."
        )

    return {
        "target_column": TARGET_COLUMN,
        "model_name": selected_model["model_name"],
        "model_params": dict(selected_model["model_params"]),
    }


def get_mlflow_model_config() -> dict[str, Any]:
    """Return a copy of the configured MLflow model settings."""
    return dict(MLFLOW_MODEL_CONFIG)
