import json
import unittest
from pathlib import Path

import mlflow
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.model.mlflow_config import configure_mlflow


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_params(params_path: Path) -> dict:
    with open(params_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


class TestModelLoading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        params = load_params(REPO_ROOT / "params.yaml")
        cls.model_name = params["mlflow"]["model_name"]
        cls.target_stage = params["mlflow"].get("stage", "Staging")
        cls.test_data_path = REPO_ROOT / params["data_paths"]["test_features"]
        cls.experiment_info_path = REPO_ROOT / params["artifacts"]["experiment_info_path"]

        configure_mlflow()
        cls.client = mlflow.MlflowClient()
        cls.model_version = cls.get_latest_model_version(cls.model_name, cls.target_stage)
        cls.model_uri = f"models:/{cls.model_name}/{cls.model_version}"
        cls.model = mlflow.pyfunc.load_model(cls.model_uri)
        cls.holdout_data = pd.read_csv(cls.test_data_path)

        with open(cls.experiment_info_path, "r", encoding="utf-8") as file:
            cls.experiment_info = json.load(file)

    @classmethod
    def get_latest_model_version(cls, model_name: str, stage: str) -> str:
        latest_versions = cls.client.get_latest_versions(model_name, stages=[stage])
        if not latest_versions:
            raise ValueError(f"No model versions found for '{model_name}' in stage '{stage}'")
        return latest_versions[0].version

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.model)

    def test_model_stage_exists(self):
        self.assertIsNotNone(self.model_version)
        self.assertTrue(str(self.model_version).strip())

    def test_model_predicts_holdout_rows(self):
        X_holdout = self.holdout_data.iloc[:, 0:-1]
        predictions = self.model.predict(X_holdout)
        self.assertEqual(len(predictions), len(X_holdout))

    def test_model_performance(self):
        X_holdout = self.holdout_data.iloc[:, 0:-1]
        y_holdout = self.holdout_data.iloc[:, -1]

        y_pred = self.model.predict(X_holdout)
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
