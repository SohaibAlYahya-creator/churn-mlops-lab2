"""FastAPI service that serves the churn model."""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from src.train.train import DEFAULT_MODEL, FEATURES

MODEL_VERSION = os.getenv("MODEL_VERSION", "dev")


class Customer(BaseModel):
    """One customer. Anything outside these rules is rejected with HTTP 422."""

    tenure_months: int = Field(ge=0, le=120)
    monthly_charges: float = Field(ge=0, le=1000)
    contract: Literal["month-to-month", "one-year", "two-year"]
    internet_service: Literal["dsl", "fiber", "none"]
    support_calls: int = Field(ge=0, le=100)


class Prediction(BaseModel):
    churn_probability: float
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup. If the file is missing, fail loudly."""
    model_path = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL)))
    app.state.model = joblib.load(model_path)
    yield


app = FastAPI(title="Churn model service", lifespan=lifespan)


@app.get("/health")
def health(request: Request) -> dict:
    return {
        "status": "ok",
        "model_loaded": request.app.state.model is not None,
        "model_version": MODEL_VERSION,
    }


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer, request: Request) -> Prediction:
    row = pd.DataFrame([customer.model_dump()])[FEATURES]
    proba = float(request.app.state.model.predict_proba(row)[0, 1])
    return Prediction(churn_probability=round(proba, 3), model_version=MODEL_VERSION)
