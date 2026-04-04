from pathlib import Path
import sys
import time
import warnings

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from .load_model import bootstrap_inference_assets
    from .preprocessing_utility import (
        ensure_nltk_resources,
        predict_label,
        preprocess_input_text,
        transform_text_to_features,
    )
except ImportError:
    from load_model import bootstrap_inference_assets
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
inference_assets = bootstrap_inference_assets()
model = inference_assets["model"]
model_source = inference_assets["model_source"]
vectorizer = inference_assets["vectorizer"]
variance_selector = inference_assets["variance_selector"]
maxabs_scaler = inference_assets["maxabs_scaler"]
select_k_best = inference_assets["select_k_best"]
artifact_sources = inference_assets["artifact_sources"]

# Initialize Flask app
app = Flask(__name__)

# Create a custom registry
registry = CollectorRegistry()

# Define custom metrics
REQUEST_COUNT = Counter(
    "app_request_count", "Total number of requests to the app", ["method", "endpoint"], registry=registry
)
REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds", "Latency of requests in seconds", ["endpoint"], registry=registry
)
PREDICTION_COUNT = Counter(
    "model_prediction_count", "Count of predictions for each class", ["prediction"], registry=registry
)


@app.route("/")
def home():
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    start_time = time.time()
    response = render_template("index.html", result=None)
    REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start_time)
    return response


@app.route("/predict", methods=["POST"])
def predict():
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    start_time = time.time()

    raw_text = request.form["text"]
    processed_text = preprocess_input_text(raw_text)
    if len(processed_text.split()) < 3:
        return render_template(
            "index.html",
            result="Input is too short after preprocessing. Please enter a longer review.",
        )

    features_df = transform_text_to_features(
        processed_text,
        vectorizer=vectorizer,
        variance_selector=variance_selector,
        maxabs_scaler=maxabs_scaler,
        select_k_best=select_k_best,
        model=model,
    )
    prediction = predict_label(model, features_df)

    PREDICTION_COUNT.labels(prediction=prediction).inc()
    REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)

    return render_template("index.html", result=prediction)


@app.route("/metrics", methods=["GET"])
def metrics():
    """Expose only custom Prometheus metrics."""
    return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}


@app.route("/health", methods=["GET"])
def health():
    """Show whether inference assets came from MLflow or local fallback."""
    artifact_status = {
        artifact_name: artifact_info["source"]
        for artifact_name, artifact_info in artifact_sources.items()
    }

    return jsonify(
        {
            "status": "ok",
            "model_strategy": "mlflow-first with local-fallback",
            "preprocessing_strategy": "local-first with mlflow-backup",
            "model_source": model_source["source"],
            "model_name": model_source.get("model_name"),
            "model_version": model_source.get("model_version"),
            "model_uri": model_source.get("model_uri"),
            "run_id": inference_assets["run_id"],
            "artifacts": artifact_status,
            "artifact_details": artifact_sources,
        }
    )


if __name__ == "__main__":
    print("[startup] Flask app initialization complete. Starting web server on port 5000.")
    app.run(debug=True, host="0.0.0.0", port=5000)
