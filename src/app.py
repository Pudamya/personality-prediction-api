from pathlib import Path
from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "personality_model.joblib"
model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Personality Prediction API",
    description="Predicts whether a person is Introvert or Extrovert from behavioral characteristics.",
    version="1.0.0",
)

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
    return {"message": "Personality Prediction API is running", "docs": "/docs", "predict_endpoint": "/predict"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
def predict(payload: PersonalityInput):
    row = pd.DataFrame([payload.model_dump()])
    prediction = model.predict(row)[0]
    probabilities = model.predict_proba(row)[0]
    confidence = float(max(probabilities))
    class_probabilities = {cls: float(prob) for cls, prob in zip(model.classes_, probabilities)}
    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "probabilities": class_probabilities,
    }
