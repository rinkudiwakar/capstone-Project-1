FROM python:3.10-slim

WORKDIR /app

# Copy only the necessary directories for the Flask app
COPY flask_app/ /app/flask_app/
COPY models/ /app/models/
COPY src/ /app/src/

# Copy configuration files
COPY flask_app/requirements.txt /app/requirements.txt
COPY setup.py /app/setup.py

# Install dependencies
RUN pip install -r requirements.txt

# Download NLTK data


RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger_eng'); nltk.download('omw-1.4')"

EXPOSE 5000


# Production with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "flask_app.app:app"]

