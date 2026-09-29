import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from app.model_service import ModelService
from app.schemas import (
    BatchRequest, BatchResponse, CustomerFeatures, PredictionResponse,
)

ARTIFACT_DIR = os.getenv("MODEL_DIR", "artifacts")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once at startup, not per request
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


# Plain `def` on purpose: model inference is blocking CPU work,
# so FastAPI runs it in a threadpool instead of freezing the event loop
@app.get("/health")
def health(request: Request):
    return {"status": "ok", "model_loaded": hasattr(request.app.state, "service")}


@app.get("/model-info")
def model_info(request: Request):
    meta = request.app.state.service.meta
    return {k: meta[k] for k in ("model_name", "model_version", "threshold", "metrics")}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures, request: Request):
    return request.app.state.service.predict([customer])[0]


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(payload: BatchRequest, request: Request):
    preds = request.app.state.service.predict(payload.customers)
    return BatchResponse(predictions=preds, count=len(preds))
