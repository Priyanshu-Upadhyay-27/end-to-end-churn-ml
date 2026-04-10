from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow.sklearn
from pathlib import Path

# Define the input schema based on your 19 features
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
    TotalCharges: str  # Kept as string for pipeline compatibility

# --- CONFIGURATION ---
RUN_ID = "d33d7c73174d449781bb7e02bf8b8440"
MASTER_PATH = Path("C:/Users/priya/Desktop/PyCh_Pro/Churn_Analysis_and_Modelling")
DB_URI = f"sqlite:///{MASTER_PATH.as_posix()}/mlflow_v2.db"

mlflow.set_tracking_uri(DB_URI)
app = FastAPI(title="Churn Prediction API")

try:
    model_uri = f"runs:/{RUN_ID}/model"
    model = mlflow.sklearn.load_model(model_uri)
    print("XGBoost Pipeline Loaded from MLflow")
except Exception as e:
    print(f"Failed to load model: {e}")


@app.post("/predict")
async def predict(data: CustomerData):
    input_df = pd.DataFrame([data.model_dump()])

    try:
        # 2. Get the raw probability of churn (class 1)
        # result is [[prob_0, prob_1]], so we grab [0][1]
        probability = model.predict_proba(input_df)[0][1]

        # 3. Standard classification decision
        prediction = 1 if probability >= 0.5 else 0

        return {
            "probability": round(float(probability), 4),
            "prediction": int(prediction),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))