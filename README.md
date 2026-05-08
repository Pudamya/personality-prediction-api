<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=32&duration=3000&pause=1000&color=6366F1&center=true&vCenter=true&width=600&lines=Personality+Prediction+API;Introvert+or+Extrovert%3F;ML-Powered+%7C+REST+API+%7C+Live" alt="Typing SVG" />

<br/>

<img src="https://img.shields.io/badge/Python-3.11.9-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/FastAPI-0.115.0-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
<img src="https://img.shields.io/badge/scikit--learn-1.5.1-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
<img src="https://img.shields.io/badge/Deployed-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white"/>
<img src="https://img.shields.io/badge/Accuracy-91.21%25-success?style=for-the-badge"/>

<br/><br/>

> **A production-grade machine learning API that predicts whether a person is an Introvert or Extrovert from 7 behavioural characteristics — trained, evaluated, and deployed end-to-end.**

<br/>

[**Live API**](https://personality-prediction-api-1.onrender.com) · [ **Swagger Docs**](https://personality-prediction-api-1.onrender.com/docs) · [**Health Check**](https://personality-prediction-api-1.onrender.com/health)

</div>

---

## Table of Contents

- [Overview](#-overview)
- [Model Performance](#-model-performance)
- [Project Structure](#-project-structure)
- [API Reference](#-api-reference)
- [Quick Start](#-quick-start)
- [How It Works](#-how-it-works)
- [Tech Stack](#-tech-stack)
- [Local Development](#-local-development)

---

## Overview

This project builds and deploys a **Gradient Boosting Classifier** that predicts personality type (Introvert / Extrovert) based on behavioural data. It was built as part of the **Decryptogen AI/ML Internship — Round 02 Technical Assignment**.

The solution covers the complete ML lifecycle:

```
Raw Data  →  EDA  →  Preprocessing  →  Model Comparison  →  Training  →  Evaluation  →  Deployment
```

All fields sent to the API are **optional** — the sklearn pipeline handles missing values internally using median/most-frequent imputation, making the API robust to partial inputs.

---

## Model Performance

Four models were compared using **5-fold Stratified Cross-Validation**:

| Model | CV Accuracy | CV F1 | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.8681 | 0.8679 | 0.9279 |
| Random Forest | 0.9055 | 0.9054 | 0.9507 |
| Extra Trees | 0.9072 | 0.9071 | 0.9491 |
| ** Gradient Boosting** | **0.9293** | **0.9293** | **0.9557** |

### Final Model — Test Set Results

```
Accuracy           : 91.21%
Weighted Precision : 91.33%
Weighted Recall    : 91.21%
Weighted F1        : 91.21%
ROC-AUC            : 95.37%
```

```
              precision    recall  f1-score   support
   Extrovert       0.94      0.89      0.91       298
   Introvert       0.89      0.94      0.91       282
    accuracy                           0.91       580
```

**Confusion Matrix:**
```
                 Predicted
                 Introvert   Extrovert
Actual Introvert    264          18
       Extrovert     33         265
```

---

##  Project Structure

```
personality-prediction-api/
│
├── data/
│   └── personality_dataset.csv          # Raw dataset (2,900 records × 8 columns)
│
├── docs/
│   ├── final_model_metrics.json         # Test-set metrics + confusion matrix
│   └── model_comparison.csv             # 4-model CV comparison table
│
├── models/
│   └── personality_model.joblib         # Trained sklearn Pipeline (643 KB)
│
├── src/
│   └── app.py                           # FastAPI application
│
├── personality_model_training.ipynb  # Full training notebook (EDA → Save)
├── Procfile                          # Render deployment config
├── requirements.txt                  # Pinned dependencies
└── runtime.txt                       # Python 3.11.9
```

---

## API Reference

### Base URL
```
https://personality-prediction-api-1.onrender.com
```

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | API info & navigation |
| `GET` | `/health` | Model load status |
| `POST` | `/predict` | Predict personality |
| `GET` | `/docs` | Interactive Swagger UI |

---

### `POST /predict`

**Request Body** — all fields are optional:

| Field | Type | Range | Example |
|---|---|---|---|
| `Time_spent_Alone` | `float` | 0 – 24 | `7` |
| `Stage_fear` | `"Yes"` / `"No"` | — | `"Yes"` |
| `Social_event_attendance` | `float` | 0 – 20 | `2` |
| `Going_outside` | `float` | 0 – 20 | `3` |
| `Drained_after_socializing` | `"Yes"` / `"No"` | — | `"Yes"` |
| `Friends_circle_size` | `float` | 0 – 100 | `4` |
| `Post_frequency` | `float` | 0 – 100 | `2` |

**Response:**
```json
{
  "prediction": "Introvert",
  "confidence": 0.9531,
  "probabilities": {
    "Extrovert": 0.0469,
    "Introvert": 0.9531
  }
}
```

---

## Quick Start

### Test in 30 seconds — cURL

**Introvert profile:**
```bash
curl -X POST https://personality-prediction-api-1.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Time_spent_Alone": 7,
    "Stage_fear": "Yes",
    "Social_event_attendance": 2,
    "Going_outside": 3,
    "Drained_after_socializing": "Yes",
    "Friends_circle_size": 4,
    "Post_frequency": 2
  }'
```

**Extrovert profile:**
```bash
curl -X POST https://personality-prediction-api-1.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Time_spent_Alone": 1,
    "Stage_fear": "No",
    "Social_event_attendance": 9,
    "Going_outside": 7,
    "Drained_after_socializing": "No",
    "Friends_circle_size": 12,
    "Post_frequency": 8
  }'
```

### Test with Python

```python
import requests

URL = "https://personality-prediction-api-1.onrender.com/predict"

profiles = {
    "Introvert": {
        "Time_spent_Alone": 7, "Stage_fear": "Yes",
        "Social_event_attendance": 2, "Going_outside": 3,
        "Drained_after_socializing": "Yes",
        "Friends_circle_size": 4, "Post_frequency": 2
    },
    "Extrovert": {
        "Time_spent_Alone": 1, "Stage_fear": "No",
        "Social_event_attendance": 9, "Going_outside": 7,
        "Drained_after_socializing": "No",
        "Friends_circle_size": 12, "Post_frequency": 8
    }
}

for expected, payload in profiles.items():
    res = requests.post(URL, json=payload).json()
    print(f"Expected: {expected} → Got: {res['prediction']} ({res['confidence']:.0%} confidence)")
```

> ⚠️ **Note:** Render's free tier spins down after 15 min of inactivity. First request may take **~60 seconds** to cold-start — this is normal.

---

## How It Works

### 1. Data
- **Dataset:** 2,900 records, 7 features + 1 target (`Personality`)
- **Class balance:** 51% Extrovert / 49% Introvert — well-balanced, no oversampling needed
- **Missing values:** Present in all columns — handled via imputation inside the pipeline

### 2. Preprocessing Pipeline (no data leakage)
```
Numeric features  →  Median Imputer  →  Standard Scaler
Categorical features  →  Most-Frequent Imputer  →  OneHotEncoder (drop if-binary)
```

### 3. Top Feature Importances
```
1. Time_spent_Alone           ████████████████  (strongest predictor)
2. Social_event_attendance    ██████████████
3. Drained_after_socializing  ████████████
4. Going_outside              ██████████
5. Stage_fear                 ████████
6. Friends_circle_size        ██████
7. Post_frequency             ████
```

### 4. Why Gradient Boosting?
- Highest CV accuracy (92.93%) and ROC-AUC (95.57%) among all 4 candidates
- Handles feature interactions naturally without polynomial expansion
- Robust to outliers compared to linear models
- Consistent performance: CV std of only ±0.006

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.11.9 |
| **ML Framework** | scikit-learn 1.5.1 |
| **API Framework** | FastAPI 0.115.0 |
| **Server** | Uvicorn 0.30.6 |
| **Data** | pandas 2.2.2 + numpy 1.26.4 |
| **Serialization** | joblib 1.4.2 |
| **Validation** | Pydantic 2.8.2 |
| **Deployment** | Render (free tier) |

---

## Local Development

```bash
# 1. Clone the repo
git clone https://github.com/Pudamya/personality-prediction-api.git
cd personality-prediction-api

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install exact dependencies
pip install -r requirements.txt

# 4. Run the API
uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000/docs** for the interactive Swagger UI.

### Re-train the model

```bash
# Open the notebook and run all cells
jupyter notebook personality_model_training.ipynb
```

The final cell automatically saves the trained model to `models/personality_model.joblib`.

---

<div align="center">

<img src="https://img.shields.io/badge/Status-Live-brightgreen?style=flat-square"/>
<img src="https://img.shields.io/badge/License-MIT-blue?style=flat-square"/>
<img src="https://img.shields.io/github/last-commit/Pudamya/personality-prediction-api?style=flat-square&color=6366F1"/>

</div>
