import yaml

import mlflow

from src.model.mlflow_config import configure_mlflow


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def promote_model() -> None:
    params = load_params("params.yaml")
    model_name = params["mlflow"]["model_name"]
    candidate_alias = params["mlflow"].get("candidate_alias", "candidate")
    production_alias = params["mlflow"].get("production_alias", "champion")

    configure_mlflow()
    client = mlflow.MlflowClient()

    candidate_version = client.get_model_version_by_alias(model_name, candidate_alias)
    if not candidate_version:
        raise ValueError(f"No candidate alias '{candidate_alias}' found for '{model_name}'")

    client.set_registered_model_alias(model_name, production_alias, candidate_version.version)
    print(
        f"Model '{model_name}' version {candidate_version.version} promoted to alias '{production_alias}'"
    )


if __name__ == "__main__":
    promote_model()
