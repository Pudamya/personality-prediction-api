from pathlib import Path
from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Personality Prediction API",
    description="Predicts whether a person is Introvert or Extrovert from behavioral characteristics.",
    version="1.0.0",
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "personality_model.joblib"

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    model_load_error = str(e)


class PersonalityInput(BaseModel):
    Time_spent_Alone: Optional[float] = Field(None, ge=0, le=24, example=4)
    Stage_fear: Optional[Literal["Yes", "No"]] = Field(None, example="No")
    Social_event_attendance: Optional[float] = Field(None, ge=0, le=20, example=6)
    Going_outside: Optional[float] = Field(None, ge=0, le=20, example=5)
    Drained_after_socializing: Optional[Literal["Yes", "No"]] = Field(None, example="No")
    Friends_circle_size: Optional[float] = Field(None, ge=0, le=100, example=10)
    Post_frequency: Optional[float] = Field(None, ge=0, le=100, example=5)


@app.get("/")
def root():
    return {
        "message": "Personality Prediction API is running",
        "docs": "/docs",
        "predict_endpoint": "/predict"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH)
    }


@app.post("/predict")
def predict(payload: PersonalityInput):
    if model is None:
        raise HTTPException(
            status_code=500,
            detail=f"Model failed to load: {model_load_error}"
        )

    try:
        row = pd.DataFrame([payload.model_dump()])
        prediction = model.predict(row)[0]

        response = {
            "prediction": str(prediction)
        }

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(row)[0]
            confidence = float(max(probabilities))
            class_probabilities = {
                str(cls): round(float(prob), 4)
                for cls, prob in zip(model.classes_, probabilities)
            }

            response["confidence"] = round(confidence, 4)
            response["probabilities"] = class_probabilities

        return response

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {str(e)}"
        )