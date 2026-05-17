from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Union
import pandas as pd
import joblib
import os
import sys
import uuid
from sklearn.preprocessing import FunctionTransformer
from database import engine, Base, AsyncSessionLocal, ChurnLog


# --- CUSTOM PIPELINE FUNCTIONS ---
def preprocessing_raw_data(X):
    df = X.copy()
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    internet_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
                     "StreamingMovies"]
    phone_cols = ["MultipleLines"]
    binary_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
                   "StreamingMovies", "MultipleLines", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]

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
    binary_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
                   "StreamingMovies", "MultipleLines", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    mapping = {"no": 0, "yes": 1}
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(df[col])
    return df


import __main__

setattr(sys.modules['__main__'], 'preprocessing_raw_data', preprocessing_raw_data)
setattr(sys.modules['__main__'], 'binaryEncoder', binaryEncoder)


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


app = FastAPI(title="Telecom Churn Production API")
model = None
PIPELINE_PATH = "production_pipeline.pkl"

@app.get("/")
async def root():
    return {"status": "healthy", "message": "Telecom Churn Production API is fully operational"}
@app.on_event("startup")
async def startup_event():
    global model
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    except Exception as e:
        print(f"DB ERROR: {e}")

    try:
        if os.path.exists(PIPELINE_PATH):
            model = joblib.load(PIPELINE_PATH)
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")


async def save_predictions_to_db(predictions_list: list, mode: str, batch_id: str = None):
    async with AsyncSessionLocal() as session:
        db_logs = []
        for item in predictions_list:
            log = ChurnLog(
                id=str(uuid.uuid4()),
                prediction_mode=mode,
                batch_id=batch_id,
                input_features=item["features"],
                churn_probability=item["probability"],
                churn_prediction=item["prediction"]
            )
            db_logs.append(log)
        session.add_all(db_logs)
        await session.commit()


@app.post("/predict")
async def predict(data: CustomerData, background_tasks: BackgroundTasks):
    if model is None:
        raise HTTPException(status_code=503, detail="Model pipeline not loaded.")
    try:
        input_df = pd.DataFrame([data.model_dump()])
        probs = model.predict_proba(input_df)
        probability = float(probs[0][1])
        prediction = 1 if probability >= 0.5 else 0

        background_tasks.add_task(
            save_predictions_to_db,
            predictions_list=[{"features": data.model_dump(), "probability": probability, "prediction": prediction}],
            mode="single"
        )
        return {"prediction": prediction, "probability": round(probability, 4), "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_batch")
async def predict_batch(data_list: list[CustomerData], background_tasks: BackgroundTasks):
    if model is None:
        raise HTTPException(status_code=503, detail="Model pipeline not loaded.")
    try:
        batch_id = str(uuid.uuid4())
        input_df = pd.DataFrame([data.model_dump() for data in data_list])
        probs = model.predict_proba(input_df)[:, 1]

        results = [{"row_index": i, "probability": round(float(prob), 4), "prediction": 1 if prob >= 0.5 else 0} for
                   i, prob in enumerate(probs)]

        db_payload = [{"features": d.model_dump(), "probability": r["probability"], "prediction": r["prediction"]} for
                      d, r in zip(data_list, results)]

        background_tasks.add_task(
            save_predictions_to_db,
            predictions_list=db_payload,
            mode="batch",
            batch_id=batch_id
        )
        return {"batch_id": batch_id, "batch_results": results, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))