import json
import os
from typing import Any

import mlflow
import yaml
from dotenv import load_dotenv

from src.logger.logging_file import logger


load_dotenv()


DEFAULT_CONFIG = {
    "artifacts": {
        "experiment_info_path": "./reports/experiment_info.json",
        "metadata_path": "./models/model_metadata.json",
    },
    "mlflow": {
        "use_dagshub": False,
        "stage": "Staging",
        "model_name": "sentiment-classifier",
    },
    "model_registry": {
        "description": "Sentiment analysis model for classifying reviews as positive or negative.",
        "created_by": "Rinku",
        "aliases": ["candidate"],
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

    config["artifacts"].update(params.get("artifacts", {}))
    config["mlflow"].update(params.get("mlflow", {}))
    config["model_registry"].update(params.get("model_registry", {}))
    config["model_registry"]["tags"].update(params.get("model_registry", {}).get("tags", {}))
    logger.info("Loaded registration parameters from %s", params_path)
    return config


def configure_mlflow(mlflow_config: dict[str, Any]) -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    dagshub_url = "https://dagshub.com"
    repo_owner = "rinkudiwakar"
    repo_name = "capstone-Project-1"

    if not tracking_uri:
        tracking_uri = f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow"

    if mlflow_config.get("use_dagshub"):
        dagshub_token = os.getenv("CAPSTONE_TEST")
        if not dagshub_token:
            raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    mlflow.set_tracking_uri(tracking_uri)
    logger.info("MLflow tracking URI configured for model registration")


def load_model_info(file_path: str) -> dict[str, Any]:
    with open(file_path, "r", encoding="utf-8") as file:
        model_info = json.load(file)
    logger.info("Loaded model info from %s", file_path)
    return model_info


def load_model_metadata(file_path: str) -> dict[str, Any]:
    if not os.path.exists(file_path):
        logger.warning("Model metadata file not found at %s. Continuing without it.", file_path)
        return {}

    with open(file_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)
    logger.info("Loaded model metadata from %s", file_path)
    return metadata


def validate_model_info(model_info: dict[str, Any]) -> None:
    """Ensure evaluation saved enough information for registration."""
    if "model_uri" not in model_info or not model_info["model_uri"]:
        raise ValueError(
            "model_uri is missing from experiment_info.json. "
            "Please rerun model evaluation before registration."
    )
    logger.info("Validated model URI %s", model_info["model_uri"])


def build_registered_model_description(
    registry_config: dict[str, Any],
    model_metadata: dict[str, Any],
) -> str:
    base_description = registry_config["description"]
    model_name = model_metadata.get("model_name")
    target_column = model_metadata.get("target_column")

    extra_lines = []
    if model_name:
        extra_lines.append(f"Production model type: {model_name}")
    if target_column:
        extra_lines.append(f"Target column: {target_column}")

    if extra_lines:
        return f"{base_description}\n\n" + "\n".join(extra_lines)
    return base_description


def build_model_version_description(
    stage: str,
    model_info: dict[str, Any],
    model_metadata: dict[str, Any],
) -> str:
    lines = [
        f"Registered from MLflow run: {model_info['run_id']}",
        f"Source model URI: {model_info['model_uri']}",
        f"Target stage: {stage}",
    ]

    if model_metadata.get("model_params"):
        lines.append(f"Model params: {json.dumps(model_metadata['model_params'])}")

    return "\n".join(lines)


def apply_registry_metadata(
    client: mlflow.tracking.MlflowClient,
    registered_model_name: str,
    model_version: str,
    stage: str,
    registry_config: dict[str, Any],
    model_info: dict[str, Any],
    model_metadata: dict[str, Any],
) -> None:
    client.update_registered_model(
        name=registered_model_name,
        description=build_registered_model_description(registry_config, model_metadata),
    )
    client.update_model_version(
        name=registered_model_name,
        version=model_version,
        description=build_model_version_description(stage, model_info, model_metadata),
    )

    combined_tags = dict(registry_config.get("tags", {}))
    combined_tags.update(
        {
            "created_by": registry_config.get("created_by", "unknown"),
            "run_id": model_info["run_id"],
            "source_model_uri": model_info["model_uri"],
            "current_stage": stage,
        }
    )

    if model_metadata.get("model_name"):
        combined_tags["algorithm"] = model_metadata["model_name"]

    for key, value in combined_tags.items():
        client.set_registered_model_tag(registered_model_name, key, str(value))
        client.set_model_version_tag(registered_model_name, model_version, key, str(value))

    aliases = registry_config.get("aliases", [])
    for alias in aliases:
        client.set_registered_model_alias(registered_model_name, alias, model_version)

    logger.info(
        "Applied description, tags, and aliases to registered model %s version %s",
        registered_model_name,
        model_version,
    )


def register_model(
    model_name: str,
    model_info: dict[str, Any],
    stage: str,
    registry_config: dict[str, Any],
    model_metadata: dict[str, Any],
) -> None:
    registered_model = mlflow.register_model(model_info["model_uri"], model_name)

    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=registered_model.version,
        stage=stage,
    )
    apply_registry_metadata(
        client=client,
        registered_model_name=model_name,
        model_version=registered_model.version,
        stage=stage,
        registry_config=registry_config,
        model_info=model_info,
        model_metadata=model_metadata,
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
        model_metadata = load_model_metadata(config["artifacts"]["metadata_path"])
        model_name = model_info.get("model_name", config["mlflow"]["model_name"])
        validate_model_info(model_info)
        register_model(
            model_name=model_name,
            model_info=model_info,
            stage=config["mlflow"]["stage"],
            registry_config=config["model_registry"],
            model_metadata=model_metadata,
        )
    except Exception as e:
        logger.exception("Failed to complete the model registration process: %s", e)
        raise


if __name__ == "__main__":
    main()
