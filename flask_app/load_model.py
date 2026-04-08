from pathlib import Path
import os
import pickle

import mlflow

from src.model.mlflow_config import configure_mlflow


BASE_DIR = Path(__file__).resolve().parent
MODEL_ARTIFACT_FILES = (
    "vectorizer.pkl",
    "variance_selector.pkl",
    "maxabs_scaler.pkl",
    "select_k_best.pkl",
)
_MLFLOW_ARTIFACT_CACHE: dict[tuple[str, str], Path | None] = {}
REGISTERED_MODEL_ALIASES = ("champion", "production", "candidate", "latest")


def print_startup_step(message: str) -> None:
    print(f"[startup] {message}")


def load_local_pickle(file_path: Path):
    with open(file_path, "rb") as file:
        return pickle.load(file)


def get_latest_model_version(model_name: str) -> tuple[str | None, str | None]:
    client = mlflow.MlflowClient()

    for alias in REGISTERED_MODEL_ALIASES:
        try:
            version = client.get_model_version_by_alias(model_name, alias)
            if version:
                print_startup_step(
                    f"Resolved registered model alias '{alias}' to version {version.version}"
                )
                return str(version.version), alias
        except Exception:
            continue
    return None, None


def get_registered_model_reference() -> dict[str, str | None]:
    """Return the currently promoted/registered model reference without loading artifacts."""
    model_name = os.getenv("MLFLOW_MODEL_NAME", "sentiment-classifier")
    configure_mlflow()

    model_version, resolved_alias = get_latest_model_version(model_name)
    if model_version is None:
        return {
            "model_name": model_name,
            "model_version": None,
            "resolved_alias": None,
            "run_id": None,
        }

    client = mlflow.MlflowClient()
    model_version_info = client.get_model_version(model_name, str(model_version))
    return {
        "model_name": model_name,
        "model_version": str(model_version),
        "resolved_alias": resolved_alias,
        "run_id": getattr(model_version_info, "run_id", None),
    }


def download_mlflow_artifact(run_id: str, artifact_path: str) -> Path | None:
    cache_key = (run_id, artifact_path)
    if cache_key in _MLFLOW_ARTIFACT_CACHE:
        return _MLFLOW_ARTIFACT_CACHE[cache_key]

    try:
        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=artifact_path,
        )
        resolved_path = Path(local_path)
        _MLFLOW_ARTIFACT_CACHE[cache_key] = resolved_path
        return resolved_path
    except Exception as exc:
        print(f"Warning: could not load MLflow artifact '{artifact_path}' from run {run_id}: {exc}")
        _MLFLOW_ARTIFACT_CACHE[cache_key] = None
        return None


def load_file_from_downloaded_directory(directory_path: Path, file_name: str):
    file_path = directory_path / file_name
    if file_path.exists():
        return load_local_pickle(file_path), str(file_path)
    return None, None


def download_candidate_directories(run_id: str) -> dict[str, Path]:
    """Download shared MLflow artifact directories for a registered model version."""
    downloaded_directories: dict[str, Path] = {}
    for directory_name in ("preprocessing", "model_pickle"):
        print_startup_step(f"Checking MLflow artifact directory '{directory_name}'")
        local_directory = download_mlflow_artifact(run_id, directory_name)
        if local_directory and local_directory.exists() and local_directory.is_dir():
            downloaded_directories[directory_name] = local_directory
            print_startup_step(f"Using MLflow artifact directory '{directory_name}'")
    return downloaded_directories


def load_pickle_from_mlflow(
    file_name: str,
    run_id: str,
    downloaded_directories: dict[str, Path] | None = None,
):
    print_startup_step(f"Resolving artifact {file_name} using run_id={run_id}")
    downloaded_directories = downloaded_directories or {}

    for directory_name, local_directory in downloaded_directories.items():
        artifact_obj, resolved_file = load_file_from_downloaded_directory(
            local_directory,
            file_name,
        )
        if artifact_obj is not None:
            print(f"Loaded {file_name} from MLflow artifact directory '{directory_name}'")
            return artifact_obj, {
                "source": "mlflow",
                "artifact_path": f"{directory_name}/{file_name}",
                "resolved_path": resolved_file,
            }

    local_artifact = download_mlflow_artifact(run_id, file_name)
    if local_artifact and local_artifact.exists():
        print(f"Loaded {file_name} from MLflow artifact '{file_name}'")
        return load_local_pickle(local_artifact), {
            "source": "mlflow",
            "artifact_path": file_name,
            "resolved_path": str(local_artifact),
        }

    raise FileNotFoundError(
        f"Required MLflow artifact '{file_name}' was not found for run {run_id}"
    )


def load_serving_model_from_mlflow():
    model_name = os.getenv("MLFLOW_MODEL_NAME", "sentiment-classifier")
    print_startup_step(f"Trying MLflow registry model '{model_name}'")
    configure_mlflow()
    model_version, resolved_alias = get_latest_model_version(model_name)
    if model_version is None:
        raise FileNotFoundError(
            f"No registered/promoted MLflow model version found for '{model_name}'"
        )

    client = mlflow.MlflowClient()
    model_version_info = client.get_model_version(model_name, str(model_version))
    model_uri = f"models:/{model_name}/{model_version}"
    print_startup_step(f"Loading serving model from MLflow registry: {model_uri}")
    return mlflow.pyfunc.load_model(model_uri), {
        "source": "mlflow",
        "model_uri": model_uri,
        "model_name": model_name,
        "model_version": str(model_version),
        "resolved_alias": resolved_alias,
    }, getattr(model_version_info, "run_id", None)


def load_inference_artifacts(
    run_id: str,
) -> tuple[dict[str, object | None], dict[str, dict[str, str | None]]]:
    artifacts = {}
    artifact_sources = {}
    downloaded_directories = download_candidate_directories(run_id)

    for file_name in MODEL_ARTIFACT_FILES:
        artifact_obj, artifact_source = load_pickle_from_mlflow(
            file_name=file_name,
            run_id=run_id,
            downloaded_directories=downloaded_directories,
        )
        artifacts[file_name] = artifact_obj
        artifact_sources[file_name] = artifact_source
    return artifacts, artifact_sources


def load_mlflow_bundle():
    model, model_source, run_id = load_serving_model_from_mlflow()
    if not run_id:
        raise ValueError("MLflow model version did not provide a run_id for artifact loading")

    artifacts, artifact_sources = load_inference_artifacts(run_id)
    missing_artifacts = [
        file_name for file_name, artifact_obj in artifacts.items() if artifact_obj is None
    ]
    if missing_artifacts:
        raise FileNotFoundError(
            f"Missing MLflow preprocessing artifacts for run {run_id}: {missing_artifacts}"
        )

    return {
        "run_id": run_id,
        "model": model,
        "model_source": model_source,
        "vectorizer": artifacts["vectorizer.pkl"],
        "variance_selector": artifacts["variance_selector.pkl"],
        "maxabs_scaler": artifacts["maxabs_scaler.pkl"],
        "select_k_best": artifacts["select_k_best.pkl"],
        "artifact_sources": artifact_sources,
    }


def bootstrap_inference_assets():
    print_startup_step("Bootstrapping inference assets")
    if not os.getenv("MLFLOW_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT"):
        print_startup_step(
            "Tip: set MLFLOW_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT to limit slow MLflow artifact downloads."
        )

    try:
        latest_registered_reference = get_registered_model_reference()
        print_startup_step(
            "Latest registered MLflow reference: "
            f"model={latest_registered_reference.get('model_name')}, "
            f"alias={latest_registered_reference.get('resolved_alias')}, "
            f"version={latest_registered_reference.get('model_version')}, "
            f"run_id={latest_registered_reference.get('run_id')}"
        )
    except Exception as exc:
        print_startup_step(f"Could not resolve latest registered MLflow reference: {exc}")
    bootstrap_payload = load_mlflow_bundle()

    if bootstrap_payload["vectorizer"] is None:
        raise FileNotFoundError(
            "vectorizer.pkl was not found in the selected inference bundle"
        )

    print_bootstrap_summary(bootstrap_payload)
    return bootstrap_payload


def print_bootstrap_summary(inference_assets: dict) -> None:
    model_source = inference_assets["model_source"]
    artifact_sources = inference_assets["artifact_sources"]

    print_startup_step(
        "Model source summary: "
        f"{model_source['source']} "
        f"(name={model_source.get('model_name')}, version={model_source.get('model_version')})"
    )
    for artifact_name, artifact_info in artifact_sources.items():
        print_startup_step(f"Artifact {artifact_name}: {artifact_info['source']}")
