import json
import os
from typing import Any

import dagshub
import mlflow
import yaml
from dotenv import load_dotenv

from src.logger.logging_file import logger


load_dotenv()


DEFAULT_CONFIG = {
    "artifacts": {
        "experiment_info_path": "./reports/experiment_info.json",
    },
    "mlflow": {
        "use_dagshub": False,
        "stage": "Staging",
        "model_name": "sentiment-classifier",
    },
}


def load_params(params_path: str = "params.yaml") -> dict[str, Any]:
    config = json.loads(json.dumps(DEFAULT_CONFIG))

    if not os.path.exists(params_path):
        logger.warning("Parameter file not found at %s. Using default configuration.", params_path)
        return config

    with open(params_path, "r", encoding="utf-8") as file:
        params = yaml.safe_load(file) or {}

    config["artifacts"].update(params.get("artifacts", {}))
    config["mlflow"].update(params.get("mlflow", {}))
    logger.info("Loaded registration parameters from %s", params_path)
    return config


def configure_mlflow(mlflow_config: dict[str, Any]) -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    dagshub_repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
    dagshub_repo_name = os.getenv("DAGSHUB_REPO_NAME")
    dagshub_token = os.getenv("CAPSTONE_TEST")

    if mlflow_config.get("use_dagshub"):
        if not dagshub_token:
            raise EnvironmentError("CAPSTONE_TEST environment variable is not set")
        if not dagshub_repo_owner or not dagshub_repo_name:
            raise EnvironmentError(
                "DAGSHUB_REPO_OWNER and DAGSHUB_REPO_NAME environment variables are required"
            )

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
        dagshub.init(
            repo_owner=dagshub_repo_owner,
            repo_name=dagshub_repo_name,
            mlflow=True,
        )

    if not tracking_uri:
        raise EnvironmentError("MLFLOW_TRACKING_URI environment variable is not set")

    mlflow.set_tracking_uri(tracking_uri)
    logger.info("MLflow tracking URI configured for model registration")


def load_model_info(file_path: str) -> dict[str, Any]:
    with open(file_path, "r", encoding="utf-8") as file:
        model_info = json.load(file)
    logger.info("Loaded model info from %s", file_path)
    return model_info


def validate_model_info(model_info: dict[str, Any]) -> None:
    """Ensure evaluation saved enough information for registration."""
    if "model_uri" not in model_info or not model_info["model_uri"]:
        raise ValueError(
            "model_uri is missing from experiment_info.json. "
            "Please rerun model evaluation before registration."
        )
    logger.info("Validated model URI %s", model_info["model_uri"])


def register_model(model_name: str, model_info: dict[str, Any], stage: str) -> None:
    registered_model = mlflow.register_model(model_info["model_uri"], model_name)

    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=registered_model.version,
        stage=stage,
    )
    logger.info(
        "Registered model %s version %s and transitioned it to %s",
        model_name,
        registered_model.version,
        stage,
    )


def main() -> None:
    try:
        config = load_params("params.yaml")
        configure_mlflow(config["mlflow"])

        model_info = load_model_info(config["artifacts"]["experiment_info_path"])
        model_name = model_info.get("model_name", config["mlflow"]["model_name"])
        validate_model_info(model_info)
        register_model(model_name, model_info, config["mlflow"]["stage"])
    except Exception as e:
        logger.exception("Failed to complete the model registration process: %s", e)
        raise


if __name__ == "__main__":
    main()
