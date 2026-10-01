import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Request

from app.model_service import ModelService
from app.schemas import (
    BatchRequest, BatchResponse, CustomerFeatures, PredictionResponse,
)

ARTIFACT_DIR = os.getenv("MODEL_DIR", "artifacts")

logger = logging.getLogger("churn_api")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(handler)

# In-memory counters — resets on restart, fine for a teaching-grade dashboard
metrics_state = {"total_requests": 0, "high_risk_count": 0, "probability_sum": 0.0}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.service = ModelService.from_env(ARTIFACT_DIR)
    print(f"Model loaded: v{app.state.service.version}")
    yield
    print("Shutting down")


app = FastAPI(title="Churn Prediction API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def add_process_time(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-ms"] = f"{(time.perf_counter() - start) * 1000:.2f}"
    return response


def _log_prediction(pred: PredictionResponse):
    metrics_state["total_requests"] += 1
    metrics_state["probability_sum"] += pred.churn_probability
    if pred.risk_level == "high":
        metrics_state["high_risk_count"] += 1

    logger.info(json.dumps({
        "event": "prediction",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "churn_probability": pred.churn_probability,
        "risk_level": pred.risk_level,
        "model_version": pred.model_version,
    }))


@app.get("/health")
def health(request: Request):
    return {"status": "ok", "model_loaded": hasattr(request.app.state, "service")}


@app.get("/model-info")
def model_info(request: Request):
    meta = request.app.state.service.meta
    return {k: meta[k] for k in ("model_name", "model_version", "threshold", "metrics")}


@app.get("/metrics")
def metrics():
    total = metrics_state["total_requests"]
    avg_prob = (metrics_state["probability_sum"] / total) if total else 0.0
    return {
        "total_requests": total,
        "high_risk_count": metrics_state["high_risk_count"],
        "high_risk_rate": round(metrics_state["high_risk_count"] / total, 4) if total else 0.0,
        "avg_churn_probability": round(avg_prob, 4),
        "note": "In-memory counters; resets on container restart.",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures, request: Request):
    pred = request.app.state.service.predict([customer])[0]
    _log_prediction(pred)
    return pred


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(payload: BatchRequest, request: Request):
    preds = request.app.state.service.predict(payload.customers)
    for p in preds:
        _log_prediction(p)
    return BatchResponse(predictions=preds, count=len(preds))
