import os
import pickle

import pandas as pd
import yaml
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, VarianceThreshold, chi2
from sklearn.preprocessing import MaxAbsScaler

from src.logger.logging_file import logger


DEFAULT_CONFIG = {
    "train_data_path": "./data/interim/train_processed.csv",
    "test_data_path": "./data/interim/test_processed.csv",
    "output_dir": "./data/processed",
    "model_dir": "./models",
    "text_column": "review",
    "target_column": "sentiment",
    "max_features": 80,
    "ngram_range": [1, 1],
    "min_df": 1,
    "apply_variance_threshold": True,
    "variance_threshold": 0.0,
    "apply_maxabs_scaler": True,
    "apply_select_k_best": True,
    "select_k_best": 30,
}


def load_params(params_path: str = "params.yaml") -> dict:
    """Load feature engineering parameters from YAML."""
    config = DEFAULT_CONFIG.copy()

    if not os.path.exists(params_path):
        logger.warning(
            "Parameter file not found at %s. Using default feature engineering config.",
            params_path,
        )
        return config

    try:
        with open(params_path, "r", encoding="utf-8") as file:
            params = yaml.safe_load(file) or {}

        feature_params = params.get("feature_engineering", {})
        config.update({key: value for key, value in feature_params.items() if value is not None})
        logger.info("Feature engineering parameters loaded from %s", params_path)
        return config
    except yaml.YAMLError as e:
        logger.error("YAML parsing error in %s: %s", params_path, e)
        raise
    except Exception as e:
        logger.error("Unexpected error while loading params from %s: %s", params_path, e)
        raise


def load_data(file_path: str, text_column: str, target_column: str) -> pd.DataFrame:
    """Load a dataset and validate required columns."""
    try:
        df = pd.read_csv(file_path)
        missing_columns = {text_column, target_column} - set(df.columns)
        if missing_columns:
            raise KeyError(f"Missing columns in {file_path}: {sorted(missing_columns)}")

        df = df[[text_column, target_column]].copy()
        df[text_column] = df[text_column].fillna("").astype(str)
        logger.info("Loaded dataset from %s with shape %s", file_path, df.shape)
        return df
    except pd.errors.ParserError as e:
        logger.error("Failed to parse CSV file %s: %s", file_path, e)
        raise
    except Exception as e:
        logger.error("Unexpected error while loading data from %s: %s", file_path, e)
        raise


def save_pickle_artifact(obj, file_path: str) -> None:
    """Persist a fitted sklearn artifact."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as file:
        pickle.dump(obj, file)
    logger.info("Saved artifact to %s", file_path)


def build_feature_dataframe(matrix, feature_names, target, target_column: str) -> pd.DataFrame:
    """Convert transformed features to a dataframe and append target labels."""
    feature_df = pd.DataFrame(matrix.toarray(), columns=feature_names)
    feature_df[target_column] = target
    return feature_df


def apply_feature_engineering(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply vectorization and optional feature selection/scaling."""
    try:
        logger.info("Starting feature engineering")
        text_column = config["text_column"]
        target_column = config["target_column"]
        model_dir = config["model_dir"]

        vectorizer = CountVectorizer(
            max_features=config["max_features"],
            ngram_range=tuple(config["ngram_range"]),
            min_df=config["min_df"],
        )

        X_train = train_data[text_column].values
        y_train = train_data[target_column].values
        X_test = test_data[text_column].values
        y_test = test_data[target_column].values

        logger.info(
            "Applying CountVectorizer with max_features=%s, ngram_range=%s, min_df=%s",
            config["max_features"],
            tuple(config["ngram_range"]),
            config["min_df"],
        )
        X_train_features = vectorizer.fit_transform(X_train)
        X_test_features = vectorizer.transform(X_test)
        feature_names = vectorizer.get_feature_names_out()
        logger.info("Vectorization complete. Generated %s features", len(feature_names))

        save_pickle_artifact(vectorizer, os.path.join(model_dir, "vectorizer.pkl"))

        if config["apply_variance_threshold"] and X_train_features.shape[1] > 0:
            selector = VarianceThreshold(threshold=config["variance_threshold"])
            X_train_features = selector.fit_transform(X_train_features)
            X_test_features = selector.transform(X_test_features)
            feature_names = feature_names[selector.get_support()]
            save_pickle_artifact(selector, os.path.join(model_dir, "variance_selector.pkl"))
            logger.info(
                "VarianceThreshold applied with threshold=%s. Remaining features: %s",
                config["variance_threshold"],
                len(feature_names),
            )

        if config["apply_maxabs_scaler"] and X_train_features.shape[1] > 0:
            scaler = MaxAbsScaler()
            X_train_features = scaler.fit_transform(X_train_features)
            X_test_features = scaler.transform(X_test_features)
            save_pickle_artifact(scaler, os.path.join(model_dir, "maxabs_scaler.pkl"))
            logger.info("MaxAbsScaler applied to feature matrix")

        if config["apply_select_k_best"] and X_train_features.shape[1] > 0:
            k = min(config["select_k_best"], X_train_features.shape[1])
            selector = SelectKBest(score_func=chi2, k=k)
            X_train_features = selector.fit_transform(X_train_features, y_train)
            X_test_features = selector.transform(X_test_features)
            feature_names = feature_names[selector.get_support()]
            save_pickle_artifact(selector, os.path.join(model_dir, "select_k_best.pkl"))
            logger.info("SelectKBest applied with k=%s. Remaining features: %s", k, len(feature_names))

        train_df = build_feature_dataframe(X_train_features, feature_names, y_train, target_column)
        test_df = build_feature_dataframe(X_test_features, feature_names, y_test, target_column)
        logger.info(
            "Feature engineering completed. Train shape=%s, Test shape=%s",
            train_df.shape,
            test_df.shape,
        )
        return train_df, test_df
    except Exception as e:
        logger.error("Error during feature engineering: %s", e)
        raise


def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Save a dataframe to CSV."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.info("Saved dataframe to %s", file_path)
    except Exception as e:
        logger.error("Unexpected error while saving data to %s: %s", file_path, e)
        raise


def main() -> None:
    try:
        logger.info("Feature engineering pipeline started")
        config = load_params("params.yaml")
        train_data = load_data(
            config["train_data_path"],
            config["text_column"],
            config["target_column"],
        )
        test_data = load_data(
            config["test_data_path"],
            config["text_column"],
            config["target_column"],
        )

        train_df, test_df = apply_feature_engineering(train_data, test_data, config)

        save_data(train_df, os.path.join(config["output_dir"], "train_bow.csv"))
        save_data(test_df, os.path.join(config["output_dir"], "test_bow.csv"))
        logger.info("Feature engineering pipeline completed successfully")
    except Exception as e:
        logger.exception("Failed to complete the feature engineering process: %s", e)
        raise


if __name__ == "__main__":
    main()
