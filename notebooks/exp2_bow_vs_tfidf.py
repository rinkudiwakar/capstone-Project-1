import re
import string
import copy
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
import nltk

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import scipy.sparse
import warnings

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")

# ========================== DOWNLOAD NLTK ==========================
nltk.download('stopwords')
nltk.download('wordnet')

# ========================== CONFIG ==========================
CONFIG = {
    "data_path": "notebooks/data.csv",
    "test_size": 0.2,
    "mlflow_tracking_uri": "https://dagshub.com/rinkudiwakar/capstone-Project-1.mlflow",
    "dagshub_repo_owner": "rinkudiwakar",
    "dagshub_repo_name": "capstone-Project-1",
    "experiment_name": "Bow vs TfIdf (fixed)"
}

# ========================== SETUP ==========================
mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])
dagshub.init(
    repo_owner=CONFIG["dagshub_repo_owner"],
    repo_name=CONFIG["dagshub_repo_name"],
    mlflow=True
)
mlflow.set_experiment(CONFIG["experiment_name"])

# ========================== GLOBALS ==========================
STOP_WORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ========================== TEXT PREPROCESSING ==========================
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", ' ', text)
    text = ''.join([char for char in text if not char.isdigit()])
    
    words = text.split()
    words = [word for word in words if word not in STOP_WORDS]
    words = [lemmatizer.lemmatize(word) for word in words]
    
    return " ".join(words)

def normalize_text(df):
    df['review'] = df['review'].astype(str).apply(preprocess_text)
    return df

# ========================== LOAD DATA ==========================
def load_data(file_path):
    df = pd.read_csv(file_path)
    df = normalize_text(df)
    
    df = df[df['sentiment'].isin(['positive', 'negative'])]
    df['sentiment'] = df['sentiment'].map({'negative': 0, 'positive': 1})
    
    return df

# ========================== VECTORIZERS ==========================
VECTORIZERS = {
    'BoW': CountVectorizer(max_features=5000),
    'TF-IDF': TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
}

# ========================== MODELS ==========================
ALGORITHMS = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'MultinomialNB': MultinomialNB(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    'RandomForest': RandomForestClassifier(random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42)
}

# ========================== LOG PARAMS ==========================
def log_model_params(algo_name, model):
    params = {}

    if algo_name == 'LogisticRegression':
        params["C"] = model.C
    elif algo_name == 'MultinomialNB':
        params["alpha"] = model.alpha
    elif algo_name == 'XGBoost':
        params["n_estimators"] = model.n_estimators
        params["learning_rate"] = model.learning_rate
    elif algo_name == 'RandomForest':
        params["n_estimators"] = model.n_estimators
        params["max_depth"] = model.max_depth
    elif algo_name == 'GradientBoosting':
        params["n_estimators"] = model.n_estimators
        params["learning_rate"] = model.learning_rate

    mlflow.log_params(params)

# ========================== TRAIN ==========================
def train_and_evaluate(df):

    X = df['review']
    y = df['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CONFIG["test_size"], random_state=42
    )

    with mlflow.start_run(run_name="All Experiments"):

        for algo_name, algorithm in ALGORITHMS.items():
            for vec_name, vectorizer in VECTORIZERS.items():

                with mlflow.start_run(run_name=f"{algo_name} with {vec_name}", nested=True):
                    try:
                        # Fresh model instance
                        model = type(algorithm)()

                        # Vectorization (NO DATA LEAKAGE)
                        X_train_vec = vectorizer.fit_transform(X_train)
                        X_test_vec = vectorizer.transform(X_test)

                        # Train
                        model.fit(X_train_vec, y_train)

                        # Predict
                        y_pred = model.predict(X_test_vec)

                        # Metrics
                        metrics = {
                            "accuracy": accuracy_score(y_test, y_pred),
                            "precision": precision_score(y_test, y_pred, zero_division=0),
                            "recall": recall_score(y_test, y_pred, zero_division=0),
                            "f1_score": f1_score(y_test, y_pred, zero_division=0)
                        }

                        mlflow.log_params({
                            "vectorizer": vec_name,
                            "algorithm": algo_name,
                            "test_size": CONFIG["test_size"]
                        })

                        mlflow.log_metrics(metrics)

                        log_model_params(algo_name, model)

                        # Log model
                        input_example = X_test_vec[:5].toarray() if scipy.sparse.issparse(X_test_vec) else X_test_vec[:5]

                        mlflow.sklearn.log_model(
                            model,
                            "model",
                            input_example=input_example
                        )

                        print(f"\n{algo_name} + {vec_name}")
                        print(metrics)

                    except Exception as e:
                        print(f"Error: {e}")
                        mlflow.set_tag("error", str(e))

# ========================== MAIN ==========================
if __name__ == "__main__":
    df = load_data(CONFIG["data_path"])
    train_and_evaluate(df)