from __future__ import annotations

import os
import sys
import threading
import time
import warnings
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, g, jsonify, render_template, request, send_from_directory
from flask_cors import CORS
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
    from .config import AppConfig
    from .load_model import bootstrap_inference_assets, get_registered_model_reference, print_startup_step
    from .movie_catalog import enrich_movies_with_stats, normalize_watchmode_movie
    from .movie_repository import SupabaseMovieRepository
    from .preprocessing_utility import ensure_nltk_resources, predict_label, preprocess_input_text, transform_text_to_features
    from .watchmode_service import WatchModeAPIError, get_watchmode_service
except ImportError:
    from config import AppConfig
    from load_model import bootstrap_inference_assets, get_registered_model_reference, print_startup_step
    from movie_catalog import enrich_movies_with_stats, normalize_watchmode_movie
    from movie_repository import SupabaseMovieRepository
    from preprocessing_utility import ensure_nltk_resources, predict_label, preprocess_input_text, transform_text_to_features
    from watchmode_service import WatchModeAPIError, get_watchmode_service


warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

config = AppConfig.from_env()
if config.load_dotenv:
    load_dotenv()
    load_dotenv(REPO_ROOT / ".env")
    config = AppConfig.from_env()

REACT_BUILD_DIR = BASE_DIR / "react_app" / "build"
if REACT_BUILD_DIR.exists():
    app = Flask(
        __name__,
        static_folder=str(REACT_BUILD_DIR / "static"),
        static_url_path="/static",
        template_folder=str(REACT_BUILD_DIR),
    )
else:
    app = Flask(__name__, static_folder="static", template_folder="templates")

CORS(app)

INFERENCE_LOCK = threading.Lock()
_inference_assets: dict | None = None
_runtime_error: str | None = None
_watcher_started = False
_repository: SupabaseMovieRepository | None = None
_repository_error: str | None = None
_startup_completed = False
_startup_error: str | None = None
_startup_lock = threading.Lock()

registry = CollectorRegistry()
HTTP_REQUESTS_TOTAL = Counter("flask_http_requests_total", "Total number of HTTP requests handled by the Flask app.", ["method", "endpoint", "status"], registry=registry)
HTTP_REQUEST_DURATION_SECONDS = Histogram("flask_http_request_duration_seconds", "Latency of HTTP requests in seconds.", ["method", "endpoint", "status"], registry=registry)
HTTP_IN_PROGRESS_REQUESTS = Gauge("flask_http_in_progress_requests", "Number of in-flight HTTP requests currently being processed.", ["method", "endpoint"], registry=registry)
PREDICTION_REQUESTS_TOTAL = Counter("flask_prediction_requests_total", "Total number of prediction requests received.", ["status"], registry=registry)
PREDICTIONS_TOTAL = Counter("flask_predictions_total", "Total predictions produced by the model grouped by sentiment.", ["prediction"], registry=registry)
PREDICTION_INPUT_WORDS = Histogram("flask_prediction_input_words", "Number of words in raw user prediction input.", buckets=(1, 3, 5, 10, 20, 50, 100, float("inf")), registry=registry)
PREDICTION_PROCESSED_WORDS = Histogram("flask_prediction_processed_words", "Number of words remaining after preprocessing prediction input.", buckets=(1, 3, 5, 10, 20, 50, 100, float("inf")), registry=registry)
PREDICTION_DURATION_SECONDS = Histogram("flask_prediction_duration_seconds", "Latency for prediction requests in seconds.", ["result"], registry=registry)
APP_ERRORS_TOTAL = Counter("flask_app_errors_total", "Total number of Flask application errors by endpoint and exception type.", ["endpoint", "error_type"], registry=registry)
MODEL_RELOAD_ATTEMPTS_TOTAL = Counter("flask_model_reload_attempts_total", "Total number of attempts to refresh the in-memory inference bundle.", ["trigger"], registry=registry)
MODEL_RELOAD_TOTAL = Counter("flask_model_reload_total", "Total number of completed model bundle reload attempts grouped by outcome.", ["result", "trigger"], registry=registry)
MODEL_RELOAD_DURATION_SECONDS = Histogram("flask_model_reload_duration_seconds", "Latency of inference bundle reloads in seconds.", registry=registry)
MODEL_LAST_RELOAD_TIMESTAMP = Gauge("flask_model_last_reload_timestamp_seconds", "Unix timestamp of the last successful model bundle reload.", registry=registry)
ACTIVE_MODEL_METADATA = Info("flask_active_model", "Metadata for the currently loaded model bundle.", registry=registry)
LATEST_REGISTERED_MODEL_METADATA = Info("flask_latest_registered_model", "Metadata for the latest registered/promoted MLflow model reference.", registry=registry)


def normalize_metric_value(value):
    return "none" if value is None else str(value)


def update_model_metadata_metrics(inference_assets: dict | None) -> None:
    model_source = (inference_assets or {}).get("model_source", {})
    ACTIVE_MODEL_METADATA.info(
        {
            "source": normalize_metric_value(model_source.get("source", "not_loaded")),
            "model_name": normalize_metric_value(model_source.get("model_name")),
            "model_version": normalize_metric_value(model_source.get("model_version")),
            "resolved_alias": normalize_metric_value(model_source.get("resolved_alias")),
            "model_uri": normalize_metric_value(model_source.get("model_uri")),
            "serving_run_id": normalize_metric_value((inference_assets or {}).get("run_id")),
        }
    )


def update_latest_registered_model_metrics():
    latest = get_latest_registered_reference()
    LATEST_REGISTERED_MODEL_METADATA.info(
        {
            "model_name": normalize_metric_value(latest.get("model_name")),
            "model_version": normalize_metric_value(latest.get("model_version")),
            "resolved_alias": normalize_metric_value(latest.get("resolved_alias")),
            "run_id": normalize_metric_value(latest.get("run_id")),
        }
    )


def get_latest_registered_reference():
    try:
        return get_registered_model_reference()
    except Exception:
        return {"model_name": None, "model_version": None, "resolved_alias": None, "run_id": None}


def get_repository() -> SupabaseMovieRepository:
    global _repository, _repository_error
    if _repository is not None:
        return _repository
    if not config.supabase_url or not config.supabase_service_role_key:
        _repository_error = "SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY is not configured"
        raise RuntimeError(_repository_error)
    try:
        repo = SupabaseMovieRepository(
            config.supabase_url,
            config.supabase_service_role_key,
        )
        repo.init_schema()
        _repository = repo
        _repository_error = None
        return repo
    except Exception as exc:
        _repository_error = str(exc)
        raise RuntimeError(f"Database initialization failed: {exc}") from exc


def get_watchmode():
    try:
        return get_watchmode_service(config.watchmode_api_key)
    except Exception as exc:
        raise RuntimeError(f"WatchMode service unavailable: {exc}") from exc


def set_inference_assets(new_assets):
    global _inference_assets, _runtime_error
    _inference_assets = new_assets
    _runtime_error = None
    update_model_metadata_metrics(new_assets)
    MODEL_LAST_RELOAD_TIMESTAMP.set_to_current_time()


def initialize_runtime(force_reload: bool = False):
    global _runtime_error
    with INFERENCE_LOCK:
        if _inference_assets is not None and not force_reload:
            return _inference_assets
        ensure_nltk_resources()
        try:
            set_inference_assets(bootstrap_inference_assets())
        except Exception as exc:
            _runtime_error = str(exc)
            raise RuntimeError(f"Failed to initialize inference assets: {exc}") from exc
        start_model_watcher_if_needed()
        return _inference_assets


def get_inference_assets():
    if _inference_assets is None:
        return initialize_runtime()
    return _inference_assets


def get_hot_reload_interval_seconds() -> int:
    raw_value = os.getenv("FLASK_APP_MODEL_REFRESH_INTERVAL_SECONDS", "30")
    try:
        return max(5, int(raw_value))
    except ValueError:
        return 30


def is_reload_candidate_changed() -> bool:
    if _inference_assets is None:
        return False
    current_source = _inference_assets["model_source"]
    registered_reference = get_latest_registered_reference()
    if registered_reference["model_version"] is None:
        return False
    if current_source.get("source") != "mlflow":
        return True
    return (
        registered_reference["model_name"] != current_source.get("model_name")
        or registered_reference["model_version"] != current_source.get("model_version")
        or registered_reference["resolved_alias"] != current_source.get("resolved_alias")
        or registered_reference["run_id"] != _inference_assets.get("run_id")
    )


def refresh_inference_assets_if_needed() -> None:
    if not is_reload_candidate_changed():
        return
    MODEL_RELOAD_ATTEMPTS_TOTAL.labels(trigger="background_watcher").inc()
    reload_started_at = time.time()
    try:
        set_inference_assets(bootstrap_inference_assets())
        MODEL_RELOAD_TOTAL.labels(result="success", trigger="background_watcher").inc()
        MODEL_RELOAD_DURATION_SECONDS.observe(time.time() - reload_started_at)
        print_startup_step("Inference bundle reloaded successfully.")
    except Exception:
        MODEL_RELOAD_TOTAL.labels(result="failure", trigger="background_watcher").inc()
        raise


def watch_for_promoted_model_updates() -> None:
    interval_seconds = get_hot_reload_interval_seconds()
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
    if app.debug and os.getenv("WERKZEUG_RUN_MAIN") != "true":
        return
    watcher = threading.Thread(target=watch_for_promoted_model_updates, daemon=True)
    watcher.start()
    _watcher_started = True


def warm_movie_catalog() -> None:
    print_startup_step(
        f"Warming movie catalog from WatchMode with limit={config.movie_catalog_limit}, region={config.watchmode_region}"
    )
    movies = get_catalog_movies(limit=config.movie_catalog_limit)
    print_startup_step(f"Movie catalog warm-up complete with {len(movies)} movies")


def bootstrap_application_state(force_reload: bool = False) -> None:
    global _startup_completed, _startup_error
    with _startup_lock:
        if _startup_completed and not force_reload:
            return

        print_startup_step("Starting application bootstrap")
        started_at = time.time()

        try:
            print_startup_step("Initializing database connection")
            get_repository()
            print_startup_step("Database connection ready")

            print_startup_step("Initializing WatchMode service")
            get_watchmode()
            print_startup_step("WatchMode service ready")

            print_startup_step("Initializing MLflow model and preprocessing artifacts")
            initialize_runtime(force_reload=force_reload)
            print_startup_step("MLflow model bundle ready")

            if config.preload_movies_on_startup:
                warm_movie_catalog()
            else:
                print_startup_step("Skipping movie catalog warm-up because FLASK_APP_PRELOAD_MOVIES is disabled")

            _startup_completed = True
            _startup_error = None
            print_startup_step(
                f"Application bootstrap completed in {time.time() - started_at:.2f}s"
            )
        except Exception as exc:
            _startup_completed = False
            _startup_error = str(exc)
            print_startup_step(f"Application bootstrap failed: {exc}")
            raise


def render_frontend(result=None):
    if REACT_BUILD_DIR.exists():
        return send_from_directory(REACT_BUILD_DIR, "index.html")
    return render_template("index.html", result=result)


def wants_json_response() -> bool:
    return request.is_json or request.path.startswith("/api/")


def build_prediction_payload(raw_text: str) -> int:
    inference_assets = get_inference_assets()
    PREDICTION_INPUT_WORDS.observe(len(raw_text.split()))
    processed_text = preprocess_input_text(raw_text)
    processed_word_count = len(processed_text.split())
    PREDICTION_PROCESSED_WORDS.observe(processed_word_count)
    if processed_word_count < 3:
        raise ValueError("Input is too short after preprocessing. Please enter a longer review.")
    features_df = transform_text_to_features(
        processed_text,
        vectorizer=inference_assets["vectorizer"],
        variance_selector=inference_assets["variance_selector"],
        maxabs_scaler=inference_assets["maxabs_scaler"],
        select_k_best=inference_assets["select_k_best"],
        model=inference_assets["model"],
    )
    return predict_label(inference_assets["model"], features_df)


def persist_movie_snapshot(movie: dict) -> dict:
    repo = get_repository()
    normalized = normalize_watchmode_movie(movie)
    repo.upsert_movie(normalized)
    return normalized


def get_catalog_movies(limit: int | None = None) -> list[dict]:
    watchmode = get_watchmode()
    repo = get_repository()
    movies = [
        normalize_watchmode_movie(movie)
        for movie in watchmode.get_popular_movies(
            limit=limit or config.movie_catalog_limit,
            region=config.watchmode_region,
        )
    ]
    for movie in movies:
        repo.upsert_movie(movie)
    stats = repo.list_movie_stats([movie["id"] for movie in movies])
    return enrich_movies_with_stats(movies, stats)


def get_movie_payload(movie_id: int) -> dict:
    watchmode = get_watchmode()
    repo = get_repository()
    details = normalize_watchmode_movie(watchmode.get_title_details(movie_id, include_sources=True))
    repo.upsert_movie(details)
    stats = repo.list_movie_stats([movie_id]).get(movie_id, {})
    return {
        **details,
        "review_count": stats.get("review_count", 0),
        "average_sentiment": stats.get("average_sentiment"),
        "average_rating": stats.get("average_rating"),
        "positive_reviews": stats.get("positive_reviews", 0),
        "negative_reviews": stats.get("negative_reviews", 0),
    }


def search_movie_payloads(query: str) -> list[dict]:
    watchmode = get_watchmode()
    repo = get_repository()
    movies = [
        normalize_watchmode_movie(movie)
        for movie in watchmode.search_titles(query, search_type="movie")
    ]
    for movie in movies:
        repo.upsert_movie(movie)
    stats = repo.list_movie_stats([movie["id"] for movie in movies])
    return enrich_movies_with_stats(movies, stats)


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
        HTTP_REQUEST_DURATION_SECONDS.labels(method=method, endpoint=endpoint, status=status).observe(time.time() - started_at)
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
        if not getattr(g, "request_metrics_decremented", False):
            HTTP_IN_PROGRESS_REQUESTS.labels(method=method, endpoint=endpoint).dec()
    except Exception:
        pass


@app.route("/")
def home():
    return render_frontend()


@app.route("/predict", methods=["POST"])
def predict():
    started_at = time.time()
    payload = request.get_json(silent=True) or {}
    raw_text = request.form.get("text", payload.get("text", "")).strip()
    if not raw_text:
        PREDICTION_REQUESTS_TOTAL.labels(status="validation_failed").inc()
        PREDICTION_DURATION_SECONDS.labels(result="validation_failed").observe(time.time() - started_at)
        message = "Review text is required."
        if wants_json_response():
            return jsonify({"error": message}), 400
        return render_template("index.html", result=message), 400
    try:
        prediction = build_prediction_payload(raw_text)
    except ValueError as exc:
        PREDICTION_REQUESTS_TOTAL.labels(status="validation_failed").inc()
        PREDICTION_DURATION_SECONDS.labels(result="validation_failed").observe(time.time() - started_at)
        if wants_json_response():
            return jsonify({"error": str(exc)}), 400
        return render_template("index.html", result=str(exc)), 400
    except Exception as exc:
        PREDICTION_REQUESTS_TOTAL.labels(status="error").inc()
        PREDICTION_DURATION_SECONDS.labels(result="error").observe(time.time() - started_at)
        if wants_json_response():
            return jsonify({"error": str(exc)}), 500
        return render_template("index.html", result="Prediction failed. Please try again later."), 500
    PREDICTION_REQUESTS_TOTAL.labels(status="success").inc()
    PREDICTIONS_TOTAL.labels(prediction=str(prediction)).inc()
    PREDICTION_DURATION_SECONDS.labels(result="success").observe(time.time() - started_at)
    if wants_json_response():
        return jsonify({"sentiment": int(prediction)})
    return render_template("index.html", result=prediction)


@app.route("/metrics", methods=["GET"])
def metrics():
    update_latest_registered_model_metrics()
    return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}


@app.route("/health", methods=["GET"])
def health():
    latest = get_latest_registered_reference()
    db_ready = False
    try:
        get_repository()
        db_ready = True
    except Exception:
        db_ready = False
    return jsonify(
        {
            "status": "ok" if db_ready else "degraded",
            "database": {"ready": db_ready, "error": _repository_error},
            "watchmode_api_configured": bool(config.watchmode_api_key),
            "runtime_initialized": _inference_assets is not None,
            "runtime_error": _runtime_error,
            "startup_completed": _startup_completed,
            "startup_error": _startup_error,
            "active_model_source": (_inference_assets or {}).get("model_source", {}).get("source", "not_loaded"),
            "latest_registered_model_name": latest.get("model_name"),
            "latest_registered_model_version": latest.get("model_version"),
            "latest_registered_alias": latest.get("resolved_alias"),
        }
    )


@app.route("/ready", methods=["GET"])
def ready():
    try:
        bootstrap_application_state()
    except Exception as exc:
        return jsonify({"status": "not_ready", "reason": str(exc)}), 503
    if not config.watchmode_api_key:
        return jsonify({"status": "not_ready", "reason": "WATCHMODE_API_KEY is not configured"}), 503
    return jsonify({"status": "ready"}), 200


@app.route("/api/movies", methods=["GET"])
def get_movies():
    try:
        limit = request.args.get("limit", config.movie_catalog_limit, type=int)
        return jsonify(get_catalog_movies(limit=limit))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/movies/random", methods=["GET"])
def get_random_movie():
    try:
        movies = get_catalog_movies(limit=max(6, config.movie_catalog_limit))
        if not movies:
            return jsonify({"error": "No movies available"}), 404
        return jsonify(movies[int(time.time()) % len(movies)])
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/movies/<int:movie_id>", methods=["GET"])
def get_movie(movie_id):
    try:
        return jsonify(get_movie_payload(movie_id))
    except WatchModeAPIError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/reviews/<int:movie_id>", methods=["GET"])
def get_reviews(movie_id):
    try:
        limit = request.args.get("limit", 10, type=int)
        return jsonify(get_repository().get_reviews_for_movie(movie_id, limit))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/movies/<int:movie_id>/sentiment", methods=["GET"])
def get_movie_sentiment_endpoint(movie_id):
    try:
        stats = get_repository().list_movie_stats([movie_id]).get(movie_id, {})
        average = stats.get("average_sentiment")
        return jsonify(
            {
                "movie_id": movie_id,
                "average_sentiment": average,
                "review_count": stats.get("review_count", 0),
                "positive_reviews": stats.get("positive_reviews", 0),
                "negative_reviews": stats.get("negative_reviews", 0),
                "sentiment_label": "Positive" if average is not None and average >= 0.5 else "Negative" if average is not None else "No Reviews",
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/stats", methods=["GET"])
def get_stats():
    try:
        return jsonify(get_repository().get_global_stats())
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/predict-sentiment", methods=["POST"])
def predict_sentiment_api():
    started_at = time.time()
    data = request.get_json(silent=True) or {}
    review_text = str(data.get("text", "")).strip()
    movie_id = data.get("movie_id")
    user_id = data.get("user_id")
    display_name = str(data.get("display_name", "")).strip() or None
    is_anonymous = bool(data.get("is_anonymous", True))
    rating = data.get("rating")
    if not movie_id:
        return jsonify({"error": "movie_id is required"}), 400
    if not review_text:
        PREDICTION_REQUESTS_TOTAL.labels(status="validation_failed").inc()
        PREDICTION_DURATION_SECONDS.labels(result="validation_failed").observe(time.time() - started_at)
        return jsonify({"error": "Review text is required"}), 400
    if not is_anonymous and not display_name:
        return jsonify({"error": "display_name is required when publishing with your name"}), 400
    if rating is not None:
        try:
            rating = int(rating)
        except (TypeError, ValueError):
            return jsonify({"error": "rating must be an integer between 1 and 5"}), 400
        if rating < 1 or rating > 5:
            return jsonify({"error": "rating must be an integer between 1 and 5"}), 400
    try:
        persist_movie_snapshot(get_watchmode().get_title_details(int(movie_id), include_sources=True))
        prediction = build_prediction_payload(review_text)
        review_id = get_repository().add_review(
            int(movie_id),
            review_text,
            int(prediction),
            user_id,
            display_name,
            is_anonymous,
            rating=rating,
        )
    except ValueError as exc:
        PREDICTION_REQUESTS_TOTAL.labels(status="validation_failed").inc()
        PREDICTION_DURATION_SECONDS.labels(result="validation_failed").observe(time.time() - started_at)
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        PREDICTION_REQUESTS_TOTAL.labels(status="error").inc()
        PREDICTION_DURATION_SECONDS.labels(result="error").observe(time.time() - started_at)
        return jsonify({"error": str(exc)}), 500
    PREDICTION_REQUESTS_TOTAL.labels(status="success").inc()
    PREDICTIONS_TOTAL.labels(prediction=str(prediction)).inc()
    PREDICTION_DURATION_SECONDS.labels(result="success").observe(time.time() - started_at)
    return jsonify(
        {
            "sentiment": int(prediction),
            "sentiment_label": "Positive" if int(prediction) == 1 else "Negative",
            "review_id": review_id,
            "rating": rating,
            "display_name": None if is_anonymous else display_name,
            "is_anonymous": is_anonymous,
        }
    )


@app.route("/api/search-movies", methods=["GET"])
def search_movies():
    query = request.args.get("q", "", type=str).strip()
    if len(query) < 2:
        return jsonify({"error": "Search query too short"}), 400
    try:
        return jsonify(search_movie_payloads(query))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/<path:path>")
def serve_react(path):
    if REACT_BUILD_DIR.exists():
        asset_path = REACT_BUILD_DIR / path
        if asset_path.exists() and asset_path.is_file():
            return send_from_directory(REACT_BUILD_DIR, path)
        return send_from_directory(REACT_BUILD_DIR, "index.html")
    return "Not Found", 404


update_model_metadata_metrics(None)
MODEL_LAST_RELOAD_TIMESTAMP.set(0)


if config.eager_startup:
    try:
        bootstrap_application_state()
    except Exception as exc:
        print_startup_step(f"Eager startup failed during import-time bootstrap: {exc}")


if __name__ == "__main__":
    if config.eager_startup and not _startup_completed:
        bootstrap_application_state()
    print("[startup] Flask app initialization complete. Starting web server.")
    app.run(debug=False, host="0.0.0.0", port=config.flask_port)
