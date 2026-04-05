import json
import os
import pickle
from typing import Any

import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from src.constants.model_constants import get_mlflow_model_config, get_model_building_config
from src.logger.logging_file import logger

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None


DEFAULT_CONFIG = {
    "data_paths": {
        "train_features": "./data/processed/train_bow.csv",
        "test_features": "./data/processed/test_bow.csv",
    },
    "model_building": get_model_building_config(),
    "artifacts": {
        "model_dir": "./models",
        "model_path": "./models/model.pkl",
        "metadata_path": "./models/model_metadata.json",
    },
    "mlflow": get_mlflow_model_config(),
}


def load_params(params_path: str = "params.yaml") -> dict[str, Any]:
    """Load pipeline configuration from YAML or fall back to defaults."""
    config = json.loads(json.dumps(DEFAULT_CONFIG))

    if not os.path.exists(params_path):
        logger.warning("Parameter file not found at %s. Using default configuration.", params_path)
        return config

    try:
        with open(params_path, "r", encoding="utf-8") as file:
            params = yaml.safe_load(file) or {}

        for section in ("data_paths", "artifacts"):
            config[section].update(params.get(section, {}))
        logger.info("Loaded pipeline parameters from %s", params_path)
        return config
    except yaml.YAMLError as e:
        logger.error("YAML error while loading %s: %s", params_path, e)
        raise
    except Exception as e:
        logger.error("Unexpected error while loading params from %s: %s", params_path, e)
        raise


def load_data(file_path: str, target_column: str) -> pd.DataFrame:
    """Load feature-engineered data from CSV."""
    try:
        df = pd.read_csv(file_path)
        if target_column not in df.columns:
            raise KeyError(f"Target column '{target_column}' not found in {file_path}")
        logger.info("Loaded data from %s with shape %s", file_path, df.shape)
        return df
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the CSV file %s: %s", file_path, e)
        raise
    except Exception as e:
        logger.error("Unexpected error while loading data from %s: %s", file_path, e)
        raise


def get_model_registry() -> dict[str, Any]:
    """Return supported production models."""
    registry = {
        "LogisticRegression": LogisticRegression,
        "RandomForest": RandomForestClassifier,
        "GradientBoosting": GradientBoostingClassifier,
        "MultinomialNB": MultinomialNB,
    }

    if XGBClassifier is not None:
        registry["XGBoost"] = XGBClassifier

    return registry


def split_features_and_target(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features and target."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def build_model(model_name: str, model_params: dict[str, Any]) -> Any:
    """Create the configured model instance."""
    registry = get_model_registry()
    if model_name not in registry:
        raise ValueError(
            f"Unsupported model '{model_name}'. Supported models: {sorted(registry.keys())}"
        )

    if model_name == "XGBoost" and XGBClassifier is None:
        raise ImportError("xgboost is not installed, so XGBoost cannot be used.")

    logger.info("Building model %s with params %s", model_name, model_params)
    return registry[model_name](**model_params)


def train_model(model: Any, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    """Train the configured model."""
    model.fit(X_train, y_train)
    logger.info("Model training completed successfully")
    return model


def save_model(model: Any, file_path: str) -> None:
    """Save the trained serving model."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as file:
        pickle.dump(model, file)
    logger.info("Saved serving model to %s", file_path)


def save_metadata(metadata: dict[str, Any], file_path: str) -> None:
    """Save training metadata for traceability."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)
    logger.info("Saved model metadata to %s", file_path)


def main() -> None:
    try:
        logger.info("Model building pipeline started")
        config = load_params("params.yaml")

        target_column = config["model_building"]["target_column"]
        train_data = load_data(config["data_paths"]["train_features"], target_column)
        X_train, y_train = split_features_and_target(train_data, target_column)

        model_name = config["model_building"]["model_name"]
        model_params = config["model_building"]["model_params"]

        model = build_model(model_name, model_params)
        trained_model = train_model(model, X_train, y_train)

        metadata = {
            "model_name": model_name,
            "registry_model_name": config["mlflow"]["model_name"],
            "model_params": model_params,
            "target_column": target_column,
            "train_shape": list(X_train.shape),
            "artifacts": {
                "model_path": config["artifacts"]["model_path"],
                "metadata_path": config["artifacts"]["metadata_path"],
            },
        }

        save_model(trained_model, config["artifacts"]["model_path"])
        save_metadata(metadata, config["artifacts"]["metadata_path"])

        logger.info("Model building completed successfully with serving model %s", model_name)
    except Exception as e:
        logger.exception("Failed to complete the model building process: %s", e)
        raise


if __name__ == "__main__":
    main()
