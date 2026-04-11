from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow.sklearn
from pathlib import Path
import os

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
    TotalCharges: str

RUN_ID = "64d602b049d4495a9eee58b189922a90"
MASTER_PATH = Path("C:/Users/priya/Desktop/PyCh_Pro/Churn_Analysis_and_Modelling")
DB_URI = f"sqlite:///{MASTER_PATH.as_posix()}/mlflow.db"

app = FastAPI(title="Telecom Churn Production API")
@app.on_event("startup")
def load_model():
    global model
    try:
        print(f"Connecting to database: {DB_URI}")
        mlflow.set_tracking_uri(DB_URI)

        model_uri = f"runs:/{RUN_ID}/model"
        print(f"Loading model from: {model_uri}")

        model = mlflow.sklearn.load_model(model_uri)
        print("SUCCESS: Model loaded and ready for inference.")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load model. Details: {e}")
        # This will keep the variable 'model' as None, causing /predict to fail safely


@app.get("/info")
def model_info():
    """Returns metadata about the active model for the UI."""
    return {"status": "online", "active_run_id": RUN_ID, "model_loaded": model is not None}


@app.post("/predict")
async def predict(data: CustomerData):

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
    """Handles bulk CSV processing instantly."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        input_df = pd.DataFrame([data.model_dump() for data in data_list])

        probs = model.predict_proba(input_df)[:, 1]

        results = []
        for i, prob in enumerate(probs):
            results.append({
                "row_index": i,
                "probability": round(float(prob), 4),
                "prediction": 1 if prob >= 0.5 else 0
            })
        return {"batch_results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))