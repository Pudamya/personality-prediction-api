from pathlib import Path
from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# App setup 

app = FastAPI(
    title="Personality Prediction API",
    description=(
        "Predicts whether a person is an **Introvert** or **Extrovert** "
        "from seven behavioural characteristics.\n\n"
        "All fields are optional — missing values are handled internally by the model pipeline."
    ),
    version="1.0.0",
)

# Model loading 

BASE_DIR   = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "personality_model.joblib"

model            = None
model_load_error = None

try:
    model = joblib.load(MODEL_PATH)
except Exception as exc:
    model_load_error = str(exc)


# Schema 

class PersonalityInput(BaseModel):
    """
    All fields mirror the original dataset columns exactly.
    Every field is optional; the sklearn Pipeline imputes missing values
    using median (numeric) or most-frequent (categorical) strategies.
    """
    Time_spent_Alone:          Optional[float]            = Field(None, ge=0,  le=24,  example=4)
    Stage_fear:                Optional[Literal["Yes", "No"]] = Field(None,          example="No")
    Social_event_attendance:   Optional[float]            = Field(None, ge=0,  le=20,  example=6)
    Going_outside:             Optional[float]            = Field(None, ge=0,  le=20,  example=5)
    Drained_after_socializing: Optional[Literal["Yes", "No"]] = Field(None,          example="No")
    Friends_circle_size:       Optional[float]            = Field(None, ge=0,  le=100, example=10)
    Post_frequency:            Optional[float]            = Field(None, ge=0,  le=100, example=5)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "summary": "Typical extrovert",
                    "value": {
                        "Time_spent_Alone": 1,
                        "Stage_fear": "No",
                        "Social_event_attendance": 9,
                        "Going_outside": 7,
                        "Drained_after_socializing": "No",
                        "Friends_circle_size": 12,
                        "Post_frequency": 8,
                    },
                },
                {
                    "summary": "Typical introvert",
                    "value": {
                        "Time_spent_Alone": 9,
                        "Stage_fear": "Yes",
                        "Social_event_attendance": 2,
                        "Going_outside": 1,
                        "Drained_after_socializing": "Yes",
                        "Friends_circle_size": 3,
                        "Post_frequency": 2,
                    },
                },
            ]
        }
    }


# Routes 

@app.get("/", tags=["Info"])
def root():
    """API info and navigation links."""
    return {
        "message":          "Personality Prediction API is running ✅",
        "docs":             "/docs",
        "predict_endpoint": "/predict",
        "health_endpoint":  "/health",
    }


@app.get("/health", tags=["Info"])
def health():
    """Returns model load status — useful for uptime monitors."""
    return {
        "status":       "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "model_path":   str(MODEL_PATH),
        **({"error": model_load_error} if model_load_error else {}),
    }


@app.post("/predict", tags=["Prediction"])
def predict(payload: PersonalityInput):
    """
    Predict personality from behavioural features.

    Returns:
    - **prediction** – `"Introvert"` or `"Extrovert"`
    - **confidence** – probability of the predicted class (0 – 1)
    - **probabilities** – probability for each class
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model is not loaded. Reason: {model_load_error}",
        )

    try:
        # Build a single-row DataFrame; column names must match training features
        row = pd.DataFrame([payload.model_dump()])

        prediction    = model.predict(row)[0]
        probabilities = model.predict_proba(row)[0]
        confidence    = float(max(probabilities))

        class_probs = {
            str(cls): round(float(prob), 4)
            for cls, prob in zip(model.classes_, probabilities)
        }

        return {
            "prediction":    str(prediction),
            "confidence":    round(confidence, 4),
            "probabilities": class_probs,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {exc}",
        )
