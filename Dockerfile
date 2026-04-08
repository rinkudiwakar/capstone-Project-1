FROM node:20-alpine AS frontend-builder
WORKDIR /build/react_app

COPY flask_app/react_app/package*.json ./
RUN npm ci

COPY flask_app/react_app/public ./public
COPY flask_app/react_app/src ./src
COPY flask_app/react_app/postcss.config.js ./
COPY flask_app/react_app/tailwind.config.js ./

RUN npm run build

FROM python:3.10-slim AS runtime
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000 \
    FLASK_APP_LOAD_DOTENV=false \
    FLASK_APP_EAGER_STARTUP=true \
    FLASK_APP_PRELOAD_MOVIES=true

COPY flask_app/requirements.txt /tmp/flask_app-requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/flask_app-requirements.txt && \
    python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger_eng'); nltk.download('omw-1.4')"

COPY flask_app /app/flask_app
COPY src /app/src
COPY --from=frontend-builder /build/react_app/build /app/flask_app/react_app/build

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=5)"

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "8", "--timeout", "180", "--graceful-timeout", "30", "--access-logfile", "-", "--error-logfile", "-", "--preload", "flask_app.app:app"]
