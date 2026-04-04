import os

import mlflow


REQUIRED_MLFLOW_ENV_VARS = (
    "MLFLOW_TRACKING_URI",
    "MLFLOW_TRACKING_USERNAME",
    "MLFLOW_TRACKING_PASSWORD",
)


def configure_mlflow(experiment_name: str | None = None) -> None:
    """Configure MLflow from process environment only.

    Expected env vars:
    - MLFLOW_TRACKING_URI
    - MLFLOW_TRACKING_USERNAME
    - MLFLOW_TRACKING_PASSWORD
    """
    missing = [env_var for env_var in REQUIRED_MLFLOW_ENV_VARS if not os.getenv(env_var)]
    if missing:
        raise EnvironmentError(
            "Missing MLflow environment variables: "
            f"{', '.join(missing)}. "
            "Set them in your shell, CI job, or the command that invokes `dvc repro`."
        )

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

    if experiment_name:
        mlflow.set_experiment(experiment_name)
