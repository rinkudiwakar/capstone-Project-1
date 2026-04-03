import os

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.connections import s3_connection
from src.logger.logging_file import logger


DEFAULT_CONFIG = {
    "data_url": "https://raw.githubusercontent.com/vikashishere/Datasets/refs/heads/main/data.csv",
    "test_size": 0.2,
    "random_state": 42,
    "target_column": "sentiment",
    "text_column": "review",
    "raw_data_dir": "./data/raw",
}


def load_params(params_path: str = "params.yaml") -> dict:
    """Load data ingestion parameters from YAML."""
    config = DEFAULT_CONFIG.copy()

    try:
        with open(params_path, "r", encoding="utf-8") as file:
            params = yaml.safe_load(file) or {}
        config.update(params.get("data_ingestion", {}))
        logger.info("Data ingestion parameters loaded from %s", params_path)
        return config
    except FileNotFoundError:
        logger.warning("Parameter file not found at %s. Using default ingestion config.", params_path)
        return config
    except yaml.YAMLError as e:
        logger.error("YAML error while loading %s: %s", params_path, e)
        raise
    except Exception as e:
        logger.error("Unexpected error while loading params from %s: %s", params_path, e)
        raise


def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a remote or local CSV source."""
    try:
        df = pd.read_csv(data_url)
        logger.info("Data loaded from %s", data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the CSV file from %s: %s", data_url, e)
        raise
    except Exception as e:
        logger.error("Unexpected error occurred while loading the data from %s: %s", data_url, e)
        raise


def preprocess_data(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Filter to binary classes and encode sentiment labels."""
    try:
        logger.info("Data ingestion preprocessing started")
        final_df = df[df[target_column].isin(["positive", "negative"])].copy()
        final_df[target_column] = final_df[target_column].replace(
            {"positive": 1, "negative": 0}
        )
        logger.info("Label encoding completed for column %s", target_column)
        return final_df
    except KeyError as e:
        logger.error("Missing target column in dataframe: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error during ingestion preprocessing: %s", e)
        raise


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, raw_data_dir: str) -> None:
    """Save train and test data into the configured raw data directory."""
    try:
        os.makedirs(raw_data_dir, exist_ok=True)
        train_path = os.path.join(raw_data_dir, "train.csv")
        test_path = os.path.join(raw_data_dir, "test.csv")
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)
        logger.info("Train and test data saved to %s", raw_data_dir)
    except Exception as e:
        logger.error("Unexpected error occurred while saving the data: %s", e)
        raise


def main() -> None:
    try:
        config = load_params("params.yaml")

        data_url = config["data_url"]
        target_column = config["target_column"]
        test_size = config["test_size"]
        random_state = config["random_state"]
        raw_data_dir = config["raw_data_dir"]

        df = load_data(data_url=data_url)
        # s3 = s3_connection.s3_operations("bucket-name", "accesskey", "secretkey")
        # df = s3.fetch_file_from_s3("data.csv")

        final_df = preprocess_data(df, target_column=target_column)
        train_data, test_data = train_test_split(
            final_df,
            test_size=test_size,
            random_state=random_state,
        )
        save_data(train_data, test_data, raw_data_dir=raw_data_dir)
    except Exception as e:
        logger.exception("Failed to complete the data ingestion process: %s", e)
        raise


if __name__ == "__main__":
    main()
