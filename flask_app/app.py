from pathlib import Path
import os
import threading
import sys
import time
import warnings

from dotenv import load_dotenv
from flask import Flask, g, jsonify, render_template, request
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from .load_model import bootstrap_inference_assets, get_registered_model_reference, print_startup_step
    from .preprocessing_utility import (
        ensure_nltk_resources,
        predict_label,
        preprocess_input_text,
        transform_text_to_features,
    )
except ImportError:
    from load_model import bootstrap_inference_assets, get_registered_model_reference, print_startup_step
    from preprocessing_utility import (
        ensure_nltk_resources,
        predict_label,
        preprocess_input_text,
        transform_text_to_features,
    )


warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# Local developer convenience only. CI/production should provide env vars directly.
load_dotenv()
load_dotenv(REPO_ROOT / ".env")


ensure_nltk_resources()
_inference_assets_lock = threading.Lock()
_inference_assets = bootstrap_inference_assets()
_bundle_reference = {
    "model_source": _inference_assets["model_source"]["source"],
    "model_name": _inference_assets["model_source"].get("model_name"),
    "model_version": _inference_assets["model_source"].get("model_version"),
    "resolved_alias": _inference_assets["model_source"].get("resolved_alias"),
    "run_id": _inference_assets.get("run_id"),
}
_watcher_started = False


def get_inference_assets():
    with _inference_assets_lock:
        return _inference_assets


def set_inference_assets(new_assets):
    global _inference_assets, _bundle_reference
    with _inference_assets_lock:
        _inference_assets = new_assets
        _bundle_reference = {
            "model_source": new_assets["model_source"]["source"],
            "model_name": new_assets["model_source"].get("model_name"),
            "model_version": new_assets["model_source"].get("model_version"),
            "resolved_alias": new_assets["model_source"].get("resolved_alias"),
            "run_id": new_assets.get("run_id"),
        }
    update_model_metadata_metrics(new_assets)


def get_bundle_reference():
    with _inference_assets_lock:
        return dict(_bundle_reference)


def get_hot_reload_interval_seconds() -> int:
    raw_value = os.getenv("FLASK_APP_MODEL_REFRESH_INTERVAL_SECONDS", "30")
    try:
        return max(5, int(raw_value))
    except ValueError:
        return 30


def get_latest_registered_reference():
    try:
        return get_registered_model_reference()
    except Exception:
        return {
            "model_name": None,
            "model_version": None,
            "resolved_alias": None,
            "run_id": None,
        }


def is_reload_candidate_changed() -> bool:
    current_reference = get_bundle_reference()
    try:
        registered_reference = get_registered_model_reference()
    except Exception as exc:
        print_startup_step(f"Hot reload check skipped because MLflow lookup failed: {exc}")
        return False

    if registered_reference["model_version"] is None:
        return False

    if current_reference["model_source"] != "mlflow":
        return True

    return (
        registered_reference["model_name"] != current_reference["model_name"]
        or registered_reference["model_version"] != current_reference["model_version"]
        or registered_reference["resolved_alias"] != current_reference["resolved_alias"]
        or registered_reference["run_id"] != current_reference["run_id"]
    )


def refresh_inference_assets_if_needed() -> None:
    if not is_reload_candidate_changed():
        return

    print_startup_step("Detected a new promoted MLflow model version. Reloading inference bundle.")
    MODEL_RELOAD_ATTEMPTS_TOTAL.labels(trigger="background_watcher").inc()
    reload_started_at = time.time()
    try:
        new_assets = bootstrap_inference_assets()
        set_inference_assets(new_assets)
        MODEL_RELOAD_TOTAL.labels(result="success", trigger="background_watcher").inc()
        MODEL_RELOAD_DURATION_SECONDS.observe(time.time() - reload_started_at)
        MODEL_LAST_RELOAD_TIMESTAMP.set_to_current_time()
        print_startup_step("Inference bundle reloaded successfully.")
    except Exception:
        MODEL_RELOAD_TOTAL.labels(result="failure", trigger="background_watcher").inc()
        raise


def watch_for_promoted_model_updates() -> None:
    interval_seconds = get_hot_reload_interval_seconds()
    print_startup_step(
        f"Background model watcher started. Checking for promoted model updates every {interval_seconds} seconds."
    )
    while True:
        time.sleep(interval_seconds)
        try:
            refresh_inference_assets_if_needed()
        except Exception as exc:
            print_startup_step(f"Background model watcher failed to refresh inference bundle: {exc}")


def start_model_watcher_if_needed() -> None:
    global _watcher_started
    if _watcher_started:
        return

    if os.getenv("FLASK_APP_ENABLE_REMOTE_MODEL", "true").strip().lower() not in {"1", "true", "yes", "on"}:
        print_startup_step("Background model watcher disabled because remote model loading is turned off.")
        return

    if app.debug and os.getenv("WERKZEUG_RUN_MAIN") != "true":
        return

    watcher = threading.Thread(target=watch_for_promoted_model_updates, daemon=True)
    watcher.start()
    _watcher_started = True

# Initialize Flask app
app = Flask(__name__)

# Create a custom registry
registry = CollectorRegistry()

# HTTP request metrics
HTTP_REQUESTS_TOTAL = Counter(
    "flask_http_requests_total",
    "Total number of HTTP requests handled by the Flask app.",
    ["method", "endpoint", "status"],
    registry=registry,
)
HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "flask_http_request_duration_seconds",
    "Latency of HTTP requests in seconds.",
    ["method", "endpoint", "status"],
    registry=registry,
)

HTTP_IN_PROGRESS_REQUESTS = Gauge(
    "flask_http_in_progress_requests",
    "Number of in-flight HTTP requests currently being processed.",
    ["method", "endpoint"],
    registry=registry,
)

# Prediction metrics
PREDICTION_REQUESTS_TOTAL = Counter(
    "flask_prediction_requests_total",
    "Total number of prediction requests received.",
    ["status"],
    registry=registry,
)
PREDICTIONS_TOTAL = Counter(
    "flask_predictions_total",
    "Total predictions produced by the model grouped by sentiment.",
    ["prediction"],
    registry=registry,
)
PREDICTION_INPUT_WORDS = Histogram(
    "flask_prediction_input_words",
    "Number of words in raw user prediction input.",
    buckets=(1, 3, 5, 10, 20, 50, 100, float("inf")),
    registry=registry,
)
PREDICTION_PROCESSED_WORDS = Histogram(
    "flask_prediction_processed_words",
    "Number of words remaining after preprocessing prediction input.",
    buckets=(1, 3, 5, 10, 20, 50, 100, float("inf")),
    registry=registry,
)
PREDICTION_DURATION_SECONDS = Histogram(
    "flask_prediction_duration_seconds",
    "Latency for prediction requests in seconds.",
    ["result"],
    registry=registry,
)

# Error and validation metrics
APP_ERRORS_TOTAL = Counter(
    "flask_app_errors_total",
    "Total number of Flask application errors by endpoint and exception type.",
    ["endpoint", "error_type"],
    registry=registry,
)
INPUT_VALIDATION_FAILURES_TOTAL = Counter(
    "flask_input_validation_failures_total",
    "Total number of invalid prediction requests rejected by validation.",
    ["reason"],
    registry=registry,
)

# Model metadata and reload metrics
MODEL_RELOAD_ATTEMPTS_TOTAL = Counter(
    "flask_model_reload_attempts_total",
    "Total number of attempts to refresh the in-memory inference bundle.",
    ["trigger"],
    registry=registry,
)
MODEL_RELOAD_TOTAL = Counter(
    "flask_model_reload_total",
    "Total number of completed model bundle reload attempts grouped by outcome.",
    ["result", "trigger"],
    registry=registry,
)
MODEL_RELOAD_DURATION_SECONDS = Histogram(
    "flask_model_reload_duration_seconds",
    "Latency of inference bundle reloads in seconds.",
    registry=registry,
)
MODEL_LAST_RELOAD_TIMESTAMP = Gauge(
    "flask_model_last_reload_timestamp_seconds",
    "Unix timestamp of the last successful model bundle reload.",
    registry=registry,
)
ACTIVE_MODEL_METADATA = Info(
    "flask_active_model",
    "Metadata for the currently loaded model bundle.",
    registry=registry,
)
LATEST_REGISTERED_MODEL_METADATA = Info(
    "flask_latest_registered_model",
    "Metadata for the latest registered/promoted MLflow model reference.",
    registry=registry,
)


def normalize_metric_value(value):
    if value is None:
        return "none"
    return str(value)


def update_model_metadata_metrics(inference_assets):
    model_source = inference_assets["model_source"]
    ACTIVE_MODEL_METADATA.info(
        {
            "source": normalize_metric_value(model_source.get("source")),
            "model_name": normalize_metric_value(model_source.get("model_name")),
            "model_version": normalize_metric_value(model_source.get("model_version")),
            "resolved_alias": normalize_metric_value(model_source.get("resolved_alias")),
            "model_uri": normalize_metric_value(model_source.get("model_uri")),
            "serving_run_id": normalize_metric_value(inference_assets.get("run_id")),
        }
    )


def update_latest_registered_model_metrics():
    latest_registered_reference = get_latest_registered_reference()
    LATEST_REGISTERED_MODEL_METADATA.info(
        {
            "model_name": normalize_metric_value(latest_registered_reference.get("model_name")),
            "model_version": normalize_metric_value(latest_registered_reference.get("model_version")),
            "resolved_alias": normalize_metric_value(latest_registered_reference.get("resolved_alias")),
            "run_id": normalize_metric_value(latest_registered_reference.get("run_id")),
        }
    )


start_model_watcher_if_needed()
update_model_metadata_metrics(_inference_assets)
MODEL_LAST_RELOAD_TIMESTAMP.set_to_current_time()


@app.before_request
def before_request_metrics():
    g.request_started_at = time.time()
    g.request_endpoint = request.path
    g.request_method = request.method
    g.request_metrics_decremented = False
    HTTP_IN_PROGRESS_REQUESTS.labels(method=request.method, endpoint=request.path).inc()


@app.after_request
def after_request_metrics(response):
    endpoint = getattr(g, "request_endpoint", request.path)
    method = getattr(g, "request_method", request.method)
    status = str(response.status_code)
    started_at = getattr(g, "request_started_at", None)
    if started_at is not None:
        elapsed = time.time() - started_at
        HTTP_REQUEST_DURATION_SECONDS.labels(method=method, endpoint=endpoint, status=status).observe(elapsed)
    HTTP_REQUESTS_TOTAL.labels(method=method, endpoint=endpoint, status=status).inc()
    HTTP_IN_PROGRESS_REQUESTS.labels(method=method, endpoint=endpoint).dec()
    g.request_metrics_decremented = True
    return response


@app.teardown_request
def teardown_request_metrics(exception):
    if exception is None:
        return

    endpoint = getattr(g, "request_endpoint", request.path if request else "unknown")
    method = getattr(g, "request_method", request.method if request else "unknown")
    APP_ERRORS_TOTAL.labels(endpoint=endpoint, error_type=type(exception).__name__).inc()
    try:
        if getattr(g, "request_metrics_decremented", False):
            return
        HTTP_IN_PROGRESS_REQUESTS.labels(method=method, endpoint=endpoint).dec()
    except Exception:
        pass


@app.route("/")
def home():
    return render_template("index.html", result=None)


@app.route("/predict", methods=["POST"])
def predict():
    prediction_started_at = time.time()

    inference_assets = get_inference_assets()
    raw_text = request.form["text"]
    PREDICTION_INPUT_WORDS.observe(len(raw_text.split()))
    processed_text = preprocess_input_text(raw_text)
    processed_word_count = len(processed_text.split())
    PREDICTION_PROCESSED_WORDS.observe(processed_word_count)
    if len(processed_text.split()) < 3:
        INPUT_VALIDATION_FAILURES_TOTAL.labels(reason="too_short_after_preprocessing").inc()
        PREDICTION_REQUESTS_TOTAL.labels(status="validation_failed").inc()
        PREDICTION_DURATION_SECONDS.labels(result="validation_failed").observe(
            time.time() - prediction_started_at
        )
        return render_template(
            "index.html",
            result="Input is too short after preprocessing. Please enter a longer review.",
        )

    features_df = transform_text_to_features(
        processed_text,
        vectorizer=inference_assets["vectorizer"],
        variance_selector=inference_assets["variance_selector"],
        maxabs_scaler=inference_assets["maxabs_scaler"],
        select_k_best=inference_assets["select_k_best"],
        model=inference_assets["model"],
    )
    prediction = predict_label(inference_assets["model"], features_df)

    PREDICTION_REQUESTS_TOTAL.labels(status="success").inc()
    PREDICTIONS_TOTAL.labels(prediction=prediction).inc()
    PREDICTION_DURATION_SECONDS.labels(result="success").observe(time.time() - prediction_started_at)

    return render_template("index.html", result=prediction)


@app.route("/metrics", methods=["GET"])
def metrics():
    """Expose only custom Prometheus metrics."""
    update_latest_registered_model_metrics()
    return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}


@app.route("/health", methods=["GET"])
def health():
    """Show whether inference assets came from MLflow or local fallback."""
    inference_assets = get_inference_assets()
    model_source = inference_assets["model_source"]
    artifact_sources = inference_assets["artifact_sources"]
    latest_registered_reference = get_latest_registered_reference()
    artifact_status = {
        artifact_name: artifact_info["source"]
        for artifact_name, artifact_info in artifact_sources.items()
    }

    return jsonify(
        {
            "status": "ok",
            "model_strategy": "registered-mlflow-first with local-bundle-fallback",
            "preprocessing_strategy": "same-source-as-model",
            "model_source": model_source["source"],
            "model_name": model_source.get("model_name"),
            "model_version": model_source.get("model_version"),
            "resolved_alias": model_source.get("resolved_alias"),
            "model_uri": model_source.get("model_uri"),
            "serving_run_id": inference_assets["run_id"],
            "latest_registered_model_name": latest_registered_reference.get("model_name"),
            "latest_registered_model_version": latest_registered_reference.get("model_version"),
            "latest_registered_alias": latest_registered_reference.get("resolved_alias"),
            "latest_registered_run_id": latest_registered_reference.get("run_id"),
            "artifacts": artifact_status,
            "artifact_details": artifact_sources,
        }
    )


if __name__ == "__main__":
    print("[startup] Flask app initialization complete. Starting web server on port 5000.")
    app.run(debug=True, host="0.0.0.0", port=5000)
