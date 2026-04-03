import os
import re
import string

import nltk
nltk.download('averaged_perceptron_tagger_eng')
import pandas as pd
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

from src.logger.logging_file import logger


TEXT_COLUMN = "review"
TRAIN_DATA_PATH = "./data/raw/train.csv"
TEST_DATA_PATH = "./data/raw/test.csv"
OUTPUT_DIR = "./data/interim"
MIN_WORDS = 3


def ensure_nltk_resources() -> None:
    """Download required NLTK resources only when missing."""
    resource_checks = {
        "stopwords": lambda: stopwords.words("english"),
        "wordnet": lambda: wordnet.ensure_loaded(),
        "averaged_perceptron_tagger": lambda: pos_tag(["sample"]),
    }

    for resource_name, checker in resource_checks.items():
        try:
            checker()
            logger.debug("NLTK resource already available: %s", resource_name)
        except LookupError:
            logger.info("Downloading missing NLTK resource: %s", resource_name)
            nltk.download(resource_name, quiet=True)
            checker()


def get_wordnet_pos(tag: str):
    """Map NLTK POS tags to WordNet POS tags for better lemmatization."""
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("N"):
        return wordnet.NOUN
    if tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def clean_text(text: str, lemmatizer: WordNetLemmatizer, stop_words: set[str]) -> str:
    """Normalize and clean a single review string."""
    text = str(text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = text.lower()
    text = re.sub(r"\d+", " ", text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return ""

    tokens = [token for token in text.split() if token not in stop_words and len(token) > 1]
    if not tokens:
        return ""

    tagged_tokens = pos_tag(tokens)
    lemmatized_tokens = [
        lemmatizer.lemmatize(token, get_wordnet_pos(tag))
        for token, tag in tagged_tokens
    ]
    return " ".join(lemmatized_tokens)


def preprocess_dataframe(df: pd.DataFrame, col: str = TEXT_COLUMN) -> pd.DataFrame:
    """
    Preprocess the given text column in a DataFrame and remove unusable rows.
    """
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in dataframe")

    logger.info("Starting preprocessing for dataframe with %s rows", len(df))

    processed_df = df.copy()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    initial_rows = len(processed_df)
    processed_df = processed_df.dropna(subset=[col])
    logger.info("Dropped %s rows with missing text", initial_rows - len(processed_df))

    processed_df[col] = processed_df[col].astype(str).str.strip()
    processed_df = processed_df[processed_df[col] != ""].copy()
    logger.info("Rows remaining after removing blank text: %s", len(processed_df))

    processed_df[col] = processed_df[col].apply(
        lambda text: clean_text(text, lemmatizer, stop_words)
    )

    processed_df = processed_df[processed_df[col].str.split().str.len() >= MIN_WORDS].copy()
    logger.info(
        "Rows remaining after cleaning and minimum word filtering: %s",
        len(processed_df),
    )

    processed_df = processed_df.drop_duplicates(subset=[col]).reset_index(drop=True)
    logger.info(
        "Preprocessing completed. Final row count: %s",
        len(processed_df),
    )
    return processed_df


def load_dataset(file_path: str) -> pd.DataFrame:
    """Load a CSV dataset from disk."""
    logger.info("Loading dataset from %s", file_path)
    df = pd.read_csv(file_path)
    logger.info("Loaded dataset with shape %s", df.shape)
    return df


def save_dataset(df: pd.DataFrame, file_path: str) -> None:
    """Persist a preprocessed dataset to disk."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
    logger.info("Saved preprocessed dataset to %s", file_path)


def main() -> None:
    try:
        logger.info("Data preprocessing pipeline started")
        ensure_nltk_resources()

        train_data = load_dataset(TRAIN_DATA_PATH)
        test_data = load_dataset(TEST_DATA_PATH)

        train_processed_data = preprocess_dataframe(train_data, TEXT_COLUMN)
        test_processed_data = preprocess_dataframe(test_data, TEXT_COLUMN)

        save_dataset(train_processed_data, os.path.join(OUTPUT_DIR, "train_processed.csv"))
        save_dataset(test_processed_data, os.path.join(OUTPUT_DIR, "test_processed.csv"))
        logger.info("Data preprocessing pipeline completed successfully")
    except Exception as e:
        logger.exception("Failed to complete the data preprocessing process: %s", e)
        raise


if __name__ == "__main__":
    main()
