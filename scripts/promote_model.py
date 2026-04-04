import yaml

import mlflow

from src.model.mlflow_config import configure_mlflow


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def promote_model() -> None:
    params = load_params("params.yaml")
    model_name = params["mlflow"]["model_name"]

    configure_mlflow()
    client = mlflow.MlflowClient()

    staging_versions = client.get_latest_versions(model_name, stages=["Staging"])
    if not staging_versions:
        raise ValueError(f"No staging model version found for '{model_name}'")

    latest_version_staging = staging_versions[0].version

    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    for version in prod_versions:
        client.transition_model_version_stage(
            name=model_name,
            version=version.version,
            stage="Archived",
        )

    client.transition_model_version_stage(
        name=model_name,
        version=latest_version_staging,
        stage="Production",
    )
    print(f"Model '{model_name}' version {latest_version_staging} promoted to Production")


if __name__ == "__main__":
    promote_model()
