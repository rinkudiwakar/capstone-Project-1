# MovieSentiment

MovieSentiment is a full-stack movie review platform that combines public movie discovery with machine learning powered sentiment analysis. The application uses WatchMode for movie metadata, Supabase for user reviews and aggregate statistics, and an MLflow-managed sentiment model to classify each submitted review as positive or negative.

The project is built to support both local development and cloud deployment. In production, the React frontend is built into static assets and served by the Flask backend from a single container image, making it straightforward to deploy on Amazon EKS.

## Features

- Browse a curated set of movies on the home page
- Search for any movie available through the WatchMode API
- Submit ratings and written reviews for a movie
- Publish reviews anonymously or with a public display name
- Store reviews, ratings, and reviewer identity mode in Supabase
- Predict review sentiment using an MLflow-managed model
- Surface aggregate review counts, ratings, and sentiment metrics per movie
- Deploy the application to EKS through GitHub Actions

## Architecture

### Application

- **Frontend:** React
- **Backend:** Flask + Gunicorn
- **Movie metadata:** WatchMode API
- **Database:** Supabase
- **Model serving:** MLflow model registry + artifacts
- **Observability:** Health, readiness, and Prometheus-style metrics endpoints

### Production runtime

The deployed application runs as a single container:

1. React is compiled during the Docker build.
2. The compiled frontend is copied into the Flask application image.
3. Gunicorn serves the Flask API and the built frontend from the same container.

This keeps the production deployment simple because there is only one service to expose from Kubernetes.

## Repository Structure

```text
.
├── flask_app/                # Flask app, React app, config, runtime integration
├── src/                      # Model training, preprocessing, MLflow pipeline code
├── scripts/                  # Utility scripts such as model promotion
├── supabase/                 # Supabase schema
├── tests/                    # Backend and model tests
├── Dockerfile                # Multi-stage production image build
├── docker-compose.yml        # Local container run configuration
├── deployment.yaml           # Kubernetes deployment, service, and HPA
├── dvc.yaml                  # Data and training pipeline definition
├── params.yaml               # Pipeline and artifact configuration
└── .github/workflows/ci.yaml # CI/CD pipeline
```

## Core Workflow

### User flow

1. A user opens the application and browses or searches for a movie.
2. The backend fetches movie metadata from WatchMode.
3. The user submits a rating and review.
4. The backend preprocesses the review and runs sentiment inference.
5. The review, rating, sentiment, and anonymity/public-name choice are stored in Supabase.
6. The movie page and homepage update using stored aggregate statistics.

### Model flow

1. The training pipeline processes IMDB review data.
2. Feature engineering artifacts are generated and stored.
3. A model is trained and evaluated.
4. MLflow stores the model and artifacts.
5. A promoted alias in MLflow is used by the Flask app at startup.
6. The deployed service loads the latest promoted bundle and can refresh when a newer promoted model is detected.

## Environment Variables

The backend relies on environment variables for external services.

### Application runtime

- `WATCHMODE_API_KEY`
- `WATCHMODE_REGION`
- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `MOVIE_CATALOG_LIMIT`
- `FLASK_APP_LOAD_DOTENV`
- `FLASK_APP_EAGER_STARTUP`
- `FLASK_APP_PRELOAD_MOVIES`
- `PORT`

### MLflow and model access

- `MLFLOW_TRACKING_URI`
- `MLFLOW_TRACKING_USERNAME`
- `MLFLOW_TRACKING_PASSWORD`
- `MLFLOW_MODEL_NAME`

### AWS

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_DEFAULT_REGION`

See [flask_app/.env.example](/d:/Data%20Science%20Learning/MLOps/capstone-Project-1/flask_app/.env.example) for a local example.

## Local Development

### Backend

From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r flask_app\requirements.txt
python -m flask_app.app
```

### Frontend

From `flask_app/react_app`:

```powershell
cmd /c npm.cmd install
cmd /c npm.cmd start
```

### Local container run

```powershell
docker compose up --build
```

## Docker

The `Dockerfile` uses a multi-stage build:

- **Stage 1:** Build the React frontend
- **Stage 2:** Install Python dependencies and run Flask/Gunicorn

The final image serves:

- the React frontend
- the Flask API
- the MLflow-backed sentiment service

## Kubernetes and EKS

The Kubernetes manifest in [deployment.yaml](/d:/Data%20Science%20Learning/MLOps/capstone-Project-1/deployment.yaml) defines:

- a `Deployment`
- a `LoadBalancer` `Service`
- a `HorizontalPodAutoscaler`

Important production behavior:

- frontend and backend are served together from the same pod
- readiness depends on the application becoming fully available
- startup is tuned to tolerate MLflow model loading

## CI/CD

GitHub Actions is defined in [ci.yaml](/d:/Data%20Science%20Learning/MLOps/capstone-Project-1/.github/workflows/ci.yaml).

### Pull requests

- Flask app tests
- frontend build
- Docker build validation

### Pushes to main/master

- data pipeline execution
- model tests
- model promotion in MLflow
- Flask tests
- frontend build
- Docker image build
- image push to Amazon ECR
- deployment to Amazon EKS

## Supabase Schema

Supabase table creation and policies live in [supabase/schema.sql](/d:/Data%20Science%20Learning/MLOps/capstone-Project-1/supabase/schema.sql).

The schema stores:

- application users
- movie snapshots
- reviews
- ratings
- anonymity/public-name choices
- aggregate movie review statistics

## API Endpoints

Key backend endpoints include:

- `/health`
- `/ready`
- `/metrics`
- `/api/movies`
- `/api/movies/<movie_id>`
- `/api/search-movies`
- `/api/reviews/<movie_id>`
- `/api/predict-sentiment`
- `/api/stats`

## Notes

- WatchMode remains the source of truth for movie metadata.
- Supabase stores application-owned review and rating data.
- The application is optimized for deployment as a single container on EKS.
- For smaller clusters, startup cost is reduced by skipping eager movie catalog preloading.

## Security

- Do not commit real `.env` files or secrets.
- Keep Supabase service-role keys on the backend only.
- Use Kubernetes secrets or GitHub Actions secrets for cloud deployment.
- Rotate any credentials that have ever been committed or shared accidentally.

## License

This project is distributed under the terms of the [LICENSE](LICENSE).
