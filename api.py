from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Union
import pandas as pd
import joblib
import os
import sys
from sklearn.preprocessing import FunctionTransformer

# --- CUSTOM FUNCTIONS REQUIRED BY THE PIPELINE ---
def preprocessing_raw_data(X):
    df = X.copy()
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    internet_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    phone_cols = ["MultipleLines"]
    binary_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "MultipleLines", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]

    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
    for col in internet_cols:
        if col in df.columns:
            df[col] = df[col].replace("no internet service", "no")
    for col in phone_cols:
        if col in df.columns:
            df[col] = df[col].replace("no phone service", "no")

    df["Stability"] = df["Partner"].astype(str) + "_" + df["Dependents"].astype(str)
    return df

def binaryEncoder(X):
    df = X.copy()
    binary_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "MultipleLines", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    mapping = {"no": 0, "yes": 1}
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(df[col])
    return df

# Bulletproof namespace injection for Uvicorn
import __main__
setattr(sys.modules['__main__'], 'preprocessing_raw_data', preprocessing_raw_data)
setattr(sys.modules['__main__'], 'binaryEncoder', binaryEncoder)

# Initialize dummy transformers
new_feature_clean_transformer = FunctionTransformer(preprocessing_raw_data)
binary_encoder_transformer = FunctionTransformer(binaryEncoder)

# --- PYDANTIC SCHEMA ---
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: Union[str, float]

# --- API INITIALIZATION ---
app = FastAPI(title="Telecom Churn Production API")
model = None
PIPELINE_PATH = "production_pipeline.pkl"

@app.on_event("startup")
def load_model():
    global model
    try:
        print(f"Loading pipeline from: {PIPELINE_PATH}")
        if os.path.exists(PIPELINE_PATH):
            model = joblib.load(PIPELINE_PATH)
            print("SUCCESS: Pipeline loaded and ready for inference.")
        else:
            print("CRITICAL ERROR: production_pipeline.pkl not found in root directory.")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load model. Details: {e}")

@app.get("/info")
def model_info():
    return {"status": "online", "model_loaded": model is not None, "pipeline": PIPELINE_PATH}

@app.post("/predict")
async def predict(data: CustomerData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model pipeline not loaded.")
    try:
        input_df = pd.DataFrame([data.model_dump()])
        probs = model.predict_proba(input_df)
        probability = float(probs[0][1])
        prediction = 1 if probability >= 0.5 else 0

        return {
            "prediction": prediction,
            "probability": round(probability, 4),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Error: {str(e)}")

@app.post("/predict_batch")
async def predict_batch(data_list: list[CustomerData]):
    if model is None:
        raise HTTPException(status_code=503, detail="Model pipeline not loaded.")
    try:
        input_df = pd.DataFrame([data.model_dump() for data in data_list])
        probs = model.predict_proba(input_df)[:, 1]

        results = [{"row_index": i, "probability": round(float(prob), 4), "prediction": 1 if prob >= 0.5 else 0} for i, prob in enumerate(probs)]
        return {"batch_results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))