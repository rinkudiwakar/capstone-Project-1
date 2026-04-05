import mlflow

from src.constants.model_constants import get_mlflow_model_config
from src.model.mlflow_config import configure_mlflow


def promote_model() -> None:
    mlflow_model_config = get_mlflow_model_config()
    model_name = mlflow_model_config["model_name"]
    candidate_alias = mlflow_model_config.get("candidate_alias", "candidate")
    production_alias = mlflow_model_config.get("production_alias", "champion")

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
