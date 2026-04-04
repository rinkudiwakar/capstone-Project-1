import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from src.data.data_preprocessing import clean_text, ensure_nltk_resources


def preprocess_input_text(text: str) -> str:
    """Mirror the training-time text preprocessing for a single request."""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    return clean_text(text, lemmatizer, stop_words)


def transform_text_to_features(
    text: str,
    vectorizer,
    variance_selector=None,
    maxabs_scaler=None,
    select_k_best=None,
    model=None,
) -> pd.DataFrame:
    """Reproduce the training-time feature engineering sequence for inference."""
    matrix = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()

    if variance_selector is not None:
        matrix = variance_selector.transform(matrix)
        feature_names = feature_names[variance_selector.get_support()]

    if maxabs_scaler is not None:
        matrix = maxabs_scaler.transform(matrix)

    if select_k_best is not None:
        matrix = select_k_best.transform(matrix)
        feature_names = feature_names[select_k_best.get_support()]

    feature_values = matrix.toarray() if hasattr(matrix, "toarray") else matrix
    features_df = pd.DataFrame(feature_values, columns=[str(name) for name in feature_names])

    if model is not None and hasattr(model, "feature_names_in_"):
        expected_columns = list(model.feature_names_in_)
        if len(expected_columns) == features_df.shape[1]:
            features_df.columns = expected_columns

    return features_df


def predict_label(model, features_df: pd.DataFrame) -> str:
    """Predict using either an MLflow pyfunc model or a local sklearn estimator."""
    result = model.predict(features_df)
    prediction = result[0]

    if isinstance(prediction, str):
        return prediction

    return "positive" if int(prediction) == 1 else "negative"
