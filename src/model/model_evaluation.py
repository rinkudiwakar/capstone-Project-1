import json
import os
import pickle
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from src.logger.logging_file import logger
from src.model.mlflow_config import configure_mlflow


DEFAULT_CONFIG = {
    "data_paths": {
        "test_features": "./data/processed/test_bow.csv",
    },
    "artifacts": {
        "model_path": "./models/model.pkl",
        "metadata_path": "./models/model_metadata.json",
        "metrics_path": "./reports/metrics.json",
        "experiment_info_path": "./reports/experiment_info.json",
    },
    "model_building": {
        "target_column": "sentiment",
    },
    "mlflow": {
        "experiment_name": "my-dvc-pipeline",
        "model_name": "sentiment-classifier",
    },
    "model_registry": {
        "description": "Sentiment analysis model for classifying reviews as positive or negative.",
        "created_by": "Prateek",
        "tags": {
            "project": "capstone-project-1",
            "task": "sentiment-classification",
            "framework": "scikit-learn",
        },
    },
}


def load_params(params_path: str = "params.yaml") -> dict[str, Any]:
    config = json.loads(json.dumps(DEFAULT_CONFIG))

    if not os.path.exists(params_path):
        logger.warning("Parameter file not found at %s. Using default configuration.", params_path)
        return config

    with open(params_path, "r", encoding="utf-8") as file:
        params = yaml.safe_load(file) or {}

    for section in ("data_paths", "artifacts", "mlflow"):
        config[section].update(params.get(section, {}))

    config["model_building"].update(params.get("model_building", {}))
    config["model_registry"].update(params.get("model_registry", {}))
    config["model_registry"]["tags"].update(params.get("model_registry", {}).get("tags", {}))
    logger.info("Loaded evaluation parameters from %s", params_path)
    return config


def load_model(file_path: str):
    with open(file_path, "rb") as file:
        model = pickle.load(file)
    logger.info("Model loaded from %s", file_path)
    return model


def load_metadata(file_path: str) -> dict[str, Any]:
    with open(file_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)
    logger.info("Model metadata loaded from %s", file_path)
    return metadata


def load_data(file_path: str, target_column: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found in {file_path}")
    logger.info("Evaluation data loaded from %s with shape %s", file_path, df.shape)
    return df


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float | None]:
    y_pred = model.predict(X_test)
    metrics: dict[str, float | None] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": None,
    }

    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y_test, y_pred_proba)

    logger.info("Model evaluation metrics calculated successfully")
    return metrics


def save_json(payload: dict[str, Any], file_path: str) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=4)
    logger.info("Saved JSON artifact to %s", file_path)


def build_model_uri(run_id: str, model_name: str) -> str:
    """Build a run-scoped model URI for later registration."""
    model_uri = f"runs:/{run_id}/{model_name}"
    logger.info("Prepared model URI %s", model_uri)
    return model_uri


def build_run_description(
    metadata: dict[str, Any],
    config: dict[str, Any],
    test_shape: list[int],
) -> str:
    lines = [
        config["model_registry"]["description"],
        "",
        f"Created by: {config['model_registry']['created_by']}",
        f"Production model type: {metadata['model_name']}",
        f"Registry model name: {metadata['registry_model_name']}",
        f"Target column: {metadata['target_column']}",
        f"Train feature shape: {metadata['train_shape']}",
        f"Test feature shape: {test_shape}",
        f"Evaluation dataset: {config['data_paths']['test_features']}",
    ]
    return "\n".join(lines)


def log_run_context(
    metadata: dict[str, Any],
    config: dict[str, Any],
    test_shape: list[int],
) -> None:
    mlflow.log_param("target_column", metadata["target_column"])
    mlflow.log_param("train_rows", metadata["train_shape"][0])
    mlflow.log_param("train_feature_count", metadata["train_shape"][1])
    mlflow.log_param("test_rows", test_shape[0])
    mlflow.log_param("test_feature_count", test_shape[1])
    mlflow.log_param("model_artifact_path", config["artifacts"]["model_path"])
    mlflow.log_param("test_features_path", config["data_paths"]["test_features"])

    tags = dict(config["model_registry"].get("tags", {}))
    tags.update(
        {
            "created_by": config["model_registry"]["created_by"],
            "algorithm": metadata["model_name"],
            "registry_model_name": metadata["registry_model_name"],
            "pipeline_stage": "model_evaluation",
        }
    )
    mlflow.set_tags(tags)
    mlflow.set_tag("mlflow.note.content", build_run_description(metadata, config, test_shape))


def main() -> None:
    try:
        config = load_params("params.yaml")
        configure_mlflow(config["mlflow"]["experiment_name"])

        target_column = config["model_building"]["target_column"]
        model = load_model(config["artifacts"]["model_path"])
        metadata = load_metadata(config["artifacts"]["metadata_path"])
        test_data = load_data(config["data_paths"]["test_features"], target_column)

        X_test = test_data.drop(columns=[target_column])
        y_test = test_data[target_column]
        metrics = evaluate_model(model, X_test, y_test)

        save_json(metrics, config["artifacts"]["metrics_path"])

        with mlflow.start_run(run_name=metadata["model_name"]) as run:
            log_run_context(metadata, config, list(X_test.shape))
            for metric_name, metric_value in metrics.items():
                if metric_value is not None:
                    mlflow.log_metric(metric_name, metric_value)

            mlflow.log_param("model_name", metadata["model_name"])
            mlflow.log_param("registry_model_name", metadata["registry_model_name"])
            for param_name, param_value in metadata["model_params"].items():
                mlflow.log_param(param_name, param_value)

            logged_model = mlflow.sklearn.log_model(model, name="model")
            mlflow.log_artifact(config["artifacts"]["model_path"], artifact_path="model_pickle")

            experiment_info = {
                "run_id": run.info.run_id,
                "model_path": "model",
                "model_uri": getattr(logged_model, "model_uri", build_model_uri(run.info.run_id, "model")),
                "model_name": metadata["registry_model_name"],
            }
            save_json(experiment_info, config["artifacts"]["experiment_info_path"])
            mlflow.log_artifact(config["artifacts"]["metrics_path"])

        logger.info("Model evaluation pipeline completed successfully")
    except Exception as e:
        logger.exception("Failed to complete the model evaluation process: %s", e)
        raise


if __name__ == "__main__":
    main()
