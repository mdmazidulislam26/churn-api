import json
import os
from pathlib import Path

import joblib
import pandas as pd

from app.schemas import CustomerFeatures, PredictionResponse


class ModelService:
    def __init__(self, model, meta: dict):
        self.model = model
        self.meta = meta
        self.threshold = float(meta["threshold"])
        self.version = str(meta["model_version"])

    @classmethod
    def from_local(cls, artifact_dir: str):
        path = Path(artifact_dir)
        with open(path / "metadata.json") as f:
            meta = json.load(f)
        return cls(joblib.load(path / "model.joblib"), meta)

    @classmethod
    def from_registry(cls, name: str, alias: str):
        import mlflow.sklearn
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        mv = client.get_model_version_by_alias(name, alias)
        model = mlflow.sklearn.load_model(f"models:/{name}@{alias}")
        run = client.get_run(mv.run_id)
        meta = {
            "model_name": name,
            "model_version": str(mv.version),
            "threshold": float(mv.tags["threshold"]),
            "metrics": {k: round(v, 4) for k, v in run.data.metrics.items()},
        }
        return cls(model, meta)

    @classmethod
    def from_env(cls, artifact_dir: str):
        # Registry mode is opt-in via env vars; default stays local (Week 1 tests)
        name = os.getenv("MODEL_REGISTRY_NAME")
        if name:
            alias = os.getenv("MODEL_ALIAS", "champion")
            return cls.from_registry(name, alias)
        return cls.from_local(artifact_dir)

    def _risk_level(self, p: float) -> str:
        if p >= self.threshold:
            return "high"
        if p >= 0.15:
            return "medium"
        return "low"

    def predict(self, customers: list[CustomerFeatures]) -> list[PredictionResponse]:
        df = pd.DataFrame([c.model_dump() for c in customers])
        # None becomes object dtype; force numeric so the imputer works
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        probs = self.model.predict_proba(df)[:, 1]

        return [
            PredictionResponse(
                churn_probability=round(float(p), 4),
                churn_prediction=bool(p >= self.threshold),
                risk_level=self._risk_level(float(p)),
                threshold=self.threshold,
                model_version=self.version,
            )
            for p in probs
        ]
