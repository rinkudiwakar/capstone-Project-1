from pathlib import Path
import json
import os
import pickle

import mlflow

from src.model.mlflow_config import configure_mlflow


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
MODEL_ARTIFACT_FILES = (
    "vectorizer.pkl",
    "variance_selector.pkl",
    "maxabs_scaler.pkl",
    "select_k_best.pkl",
)
_MLFLOW_ARTIFACT_CACHE: dict[tuple[str, str], Path | None] = {}


def env_flag(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def print_startup_step(message: str) -> None:
    print(f"[startup] {message}")


def get_latest_model_version(model_name: str) -> str | None:
    client = mlflow.MlflowClient()

    try:
        for alias in ("champion", "production", "candidate", "latest"):
            try:
                version = client.get_model_version_by_alias(model_name, alias)
                if version:
                    return version.version
            except Exception:
                continue
    except Exception:
        pass

    try:
        for stage in ("Production", "Staging", "None"):
            versions = client.get_latest_versions(model_name, stages=[stage])
            if versions:
                return versions[0].version
    except Exception:
        return None

    return None


def get_latest_run_id() -> str | None:
    exp_info_path = REPO_ROOT / "reports" / "experiment_info.json"
    if not exp_info_path.exists():
        return None

    try:
        with open(exp_info_path, "r", encoding="utf-8") as file:
            payload = json.load(file)
        return payload.get("run_id")
    except Exception as exc:
        print(f"Warning: failed to parse {exp_info_path}: {exc}")
        return None


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
        with open(file_path, "rb") as file:
            return pickle.load(file), str(file_path)
    return None, None


def download_candidate_directories(run_id: str) -> dict[str, Path]:
    """Download shared artifact directories once so startup stays fast."""
    downloaded_directories: dict[str, Path] = {}
    for directory_name in ("preprocessing", "artifacts"):
        print_startup_step(f"Checking MLflow artifact directory '{directory_name}'")
        local_directory = download_mlflow_artifact(run_id, directory_name)
        if local_directory and local_directory.exists() and local_directory.is_dir():
            downloaded_directories[directory_name] = local_directory
            print_startup_step(f"Using MLflow artifact directory '{directory_name}'")
    return downloaded_directories


def load_pickle_from_mlflow_or_local(
    file_name: str,
    run_id: str | None,
    downloaded_directories: dict[str, Path] | None = None,
):
    prefer_local_artifacts = env_flag("FLASK_APP_PREFER_LOCAL_ARTIFACTS", True)
    try_remote_artifacts = env_flag("FLASK_APP_ENABLE_REMOTE_ARTIFACTS", True)

    local_path = REPO_ROOT / "models" / file_name
    if prefer_local_artifacts and local_path.exists():
        print_startup_step(f"Using local preprocessing artifact for {file_name}")
        with open(local_path, "rb") as file:
            return pickle.load(file), {
                "source": "local",
                "artifact_path": None,
                "resolved_path": str(local_path),
            }

    if try_remote_artifacts and run_id:
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
            with open(local_artifact, "rb") as file:
                print(f"Loaded {file_name} from MLflow artifact '{file_name}'")
                return pickle.load(file), {
                    "source": "mlflow",
                    "artifact_path": file_name,
                    "resolved_path": str(local_artifact),
                }

        print_startup_step(f"MLflow artifact unavailable for {file_name}. Falling back to local file.")
    elif not try_remote_artifacts:
        print_startup_step(f"Remote artifact loading disabled for {file_name}. Using local fallback.")

    if not local_path.exists():
        print(f"Warning: local fallback artifact not found at {local_path}")
        return None, {
            "source": "missing",
            "artifact_path": None,
            "resolved_path": str(local_path),
        }

    with open(local_path, "rb") as file:
        print(f"Loaded {file_name} from local path {local_path}")
        return pickle.load(file), {
            "source": "local",
            "artifact_path": None,
            "resolved_path": str(local_path),
        }


def load_serving_model():
    model_name = os.getenv("MLFLOW_MODEL_NAME", "sentiment-classifier")
    try_remote_model = env_flag("FLASK_APP_ENABLE_REMOTE_MODEL", True)

    if try_remote_model:
        try:
            print_startup_step(f"Trying MLflow registry model '{model_name}'")
            configure_mlflow()
            model_version = get_latest_model_version(model_name)
            if model_version is not None:
                model_uri = f"models:/{model_name}/{model_version}"
                print_startup_step(f"Loading serving model from MLflow registry: {model_uri}")
                return mlflow.pyfunc.load_model(model_uri), {
                    "source": "mlflow",
                    "model_uri": model_uri,
                    "model_name": model_name,
                    "model_version": str(model_version),
                }
            print_startup_step(f"No MLflow model version found for '{model_name}'. Falling back to local.")
        except Exception as exc:
            print(f"Warning: failed to load model from MLflow registry: {exc}")
            print_startup_step("Falling back to local model file.")
    else:
        print_startup_step("Remote model loading disabled. Using local fallback.")

    local_model_path = REPO_ROOT / "models" / "model.pkl"
    if not local_model_path.exists():
        raise FileNotFoundError(f"Local fallback model not found at {local_model_path}")

    with open(local_model_path, "rb") as file:
        print(f"Using local model fallback: {local_model_path}")
        return pickle.load(file), {
            "source": "local",
            "model_uri": None,
            "model_name": model_name,
            "model_version": None,
            "resolved_path": str(local_model_path),
        }


def load_inference_artifacts(run_id: str | None) -> tuple[dict[str, object | None], dict[str, dict[str, str | None]]]:
    artifacts = {}
    artifact_sources = {}
    downloaded_directories: dict[str, Path] | None = None

    for file_name in MODEL_ARTIFACT_FILES:
        local_path = REPO_ROOT / "models" / file_name
        if (
            downloaded_directories is None
            and run_id
            and env_flag("FLASK_APP_ENABLE_REMOTE_ARTIFACTS", True)
            and not local_path.exists()
        ):
            downloaded_directories = download_candidate_directories(run_id)

        artifact_obj, artifact_source = load_pickle_from_mlflow_or_local(
            file_name,
            run_id,
            downloaded_directories=downloaded_directories or {},
        )
        artifacts[file_name] = artifact_obj
        artifact_sources[file_name] = artifact_source
    return artifacts, artifact_sources


def bootstrap_inference_assets():
    print_startup_step("Bootstrapping inference assets")
    if not os.getenv("MLFLOW_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT"):
        print_startup_step(
            "Tip: set MLFLOW_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT to limit slow MLflow artifact downloads."
        )

    run_id = get_latest_run_id()
    print_startup_step(f"Latest run_id from reports: {run_id}")
    model, model_source = load_serving_model()
    artifacts, artifact_sources = load_inference_artifacts(run_id)

    vectorizer = artifacts["vectorizer.pkl"]
    if vectorizer is None:
        raise FileNotFoundError(
            "vectorizer.pkl was not found in MLflow artifacts or local models directory"
        )

    bootstrap_payload = {
        "run_id": run_id,
        "model": model,
        "model_source": model_source,
        "vectorizer": vectorizer,
        "variance_selector": artifacts["variance_selector.pkl"],
        "maxabs_scaler": artifacts["maxabs_scaler.pkl"],
        "select_k_best": artifacts["select_k_best.pkl"],
        "artifact_sources": artifact_sources,
    }
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
