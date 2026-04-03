import os
import re
import string

import nltk
import pandas as pd
import yaml
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

from src.logger.logging_file import logger


DEFAULT_CONFIG = {
    "train_data_path": "./data/raw/train.csv",
    "test_data_path": "./data/raw/test.csv",
    "output_dir": "./data/interim",
    "text_column": "review",
    "min_words": 3,
}


def load_params(params_path: str = "params.yaml") -> dict:
    """Load data preprocessing parameters from YAML."""
    config = DEFAULT_CONFIG.copy()

    try:
        with open(params_path, "r", encoding="utf-8") as file:
            params = yaml.safe_load(file) or {}
        config.update(params.get("data_preprocessing", {}))
        logger.info("Data preprocessing parameters loaded from %s", params_path)
        return config
    except FileNotFoundError:
        logger.warning(
            "Parameter file not found at %s. Using default preprocessing config.",
            params_path,
        )
        return config
    except yaml.YAMLError as e:
        logger.error("YAML error while loading %s: %s", params_path, e)
        raise
    except Exception as e:
        logger.error("Unexpected error while loading params from %s: %s", params_path, e)
        raise


def ensure_nltk_resources() -> None:
    """Download required NLTK resources only when missing."""
    resource_checks = {
        "stopwords": lambda: stopwords.words("english"),
        "wordnet": lambda: wordnet.ensure_loaded(),
        "averaged_perceptron_tagger_eng": lambda: pos_tag(["sample"]),
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


def preprocess_dataframe(df: pd.DataFrame, text_column: str, min_words: int) -> pd.DataFrame:
    """Preprocess the given text column in a DataFrame and remove unusable rows."""
    if text_column not in df.columns:
        raise KeyError(f"Column '{text_column}' not found in dataframe")

    logger.info("Starting preprocessing for dataframe with %s rows", len(df))

    processed_df = df.copy()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    initial_rows = len(processed_df)
    processed_df = processed_df.dropna(subset=[text_column])
    logger.info("Dropped %s rows with missing text", initial_rows - len(processed_df))

    processed_df[text_column] = processed_df[text_column].astype(str).str.strip()
    processed_df = processed_df[processed_df[text_column] != ""].copy()
    logger.info("Rows remaining after removing blank text: %s", len(processed_df))

    processed_df[text_column] = processed_df[text_column].apply(
        lambda text: clean_text(text, lemmatizer, stop_words)
    )

    processed_df = processed_df[
        processed_df[text_column].str.split().str.len() >= min_words
    ].copy()
    logger.info(
        "Rows remaining after cleaning and minimum word filtering: %s",
        len(processed_df),
    )

    processed_df = processed_df.drop_duplicates(subset=[text_column]).reset_index(drop=True)
    logger.info("Preprocessing completed. Final row count: %s", len(processed_df))
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
        config = load_params("params.yaml")
        ensure_nltk_resources()

        train_data = load_dataset(config["train_data_path"])
        test_data = load_dataset(config["test_data_path"])

        train_processed_data = preprocess_dataframe(
            train_data,
            text_column=config["text_column"],
            min_words=config["min_words"],
        )
        test_processed_data = preprocess_dataframe(
            test_data,
            text_column=config["text_column"],
            min_words=config["min_words"],
        )

        save_dataset(
            train_processed_data,
            os.path.join(config["output_dir"], "train_processed.csv"),
        )
        save_dataset(
            test_processed_data,
            os.path.join(config["output_dir"], "test_processed.csv"),
        )
        logger.info("Data preprocessing pipeline completed successfully")
    except Exception as e:
        logger.exception("Failed to complete the data preprocessing process: %s", e)
        raise


if __name__ == "__main__":
    main()
