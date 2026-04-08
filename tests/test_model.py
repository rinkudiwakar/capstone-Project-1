import json
import os
import unittest
from pathlib import Path

import mlflow
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.constants.model_constants import get_mlflow_model_config
from src.model.mlflow_config import REQUIRED_MLFLOW_ENV_VARS, configure_mlflow


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_params(params_path: Path) -> dict:
    with open(params_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


@unittest.skipUnless(all(os.getenv(name) for name in REQUIRED_MLFLOW_ENV_VARS), "MLflow credentials are not configured for integration tests.")
class TestModelLoading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        params = load_params(REPO_ROOT / "params.yaml")
        mlflow_model_config = get_mlflow_model_config()
        cls.model_name = mlflow_model_config["model_name"]
        cls.target_alias = mlflow_model_config.get("candidate_alias", "candidate")
        cls.test_data_path = REPO_ROOT / params["data_paths"]["test_features"]
        cls.experiment_info_path = REPO_ROOT / params["artifacts"]["experiment_info_path"]

        configure_mlflow()
        cls.client = mlflow.MlflowClient()
        cls.model_version = cls.get_model_version_by_alias(cls.model_name, cls.target_alias)
        cls.model_uri = f"models:/{cls.model_name}/{cls.model_version}"
        cls.model = mlflow.pyfunc.load_model(cls.model_uri)
        cls.holdout_data = pd.read_csv(cls.test_data_path)

        with open(cls.experiment_info_path, "r", encoding="utf-8") as file:
            cls.experiment_info = json.load(file)

    @classmethod
    def get_model_version_by_alias(cls, model_name: str, alias: str) -> str:
        model_version = cls.client.get_model_version_by_alias(model_name, alias)
        if not model_version:
            raise ValueError(f"No model version found for '{model_name}' with alias '{alias}'")
        return model_version.version

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.model)

    def test_model_alias_exists(self):
        self.assertIsNotNone(self.model_version)
        self.assertTrue(str(self.model_version).strip())

    def test_model_predicts_holdout_rows(self):
        x_holdout = self.holdout_data.iloc[:, 0:-1]
        predictions = self.model.predict(x_holdout)
        self.assertEqual(len(predictions), len(x_holdout))

    def test_model_performance(self):
        x_holdout = self.holdout_data.iloc[:, 0:-1]
        y_holdout = self.holdout_data.iloc[:, -1]

        y_pred = self.model.predict(x_holdout)
        accuracy = accuracy_score(y_holdout, y_pred)
        precision = precision_score(y_holdout, y_pred)
        recall = recall_score(y_holdout, y_pred)
        f1 = f1_score(y_holdout, y_pred)

        self.assertGreaterEqual(accuracy, 0.40, "Accuracy should be at least 0.40")
        self.assertGreaterEqual(precision, 0.40, "Precision should be at least 0.40")
        self.assertGreaterEqual(recall, 0.40, "Recall should be at least 0.40")
        self.assertGreaterEqual(f1, 0.40, "F1 score should be at least 0.40")

    def test_experiment_info_matches_registry_model(self):
        self.assertEqual(self.experiment_info["model_name"], self.model_name)


if __name__ == "__main__":
    unittest.main()
