# 📉 Customer Retention Intelligence: End-to-End MLOps Pipeline

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15+-336791.svg?logo=postgresql)](https://www.postgresql.org/)
[![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-2.0+-D71F00.svg?logo=python)](https://www.sqlalchemy.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-FF4B4B.svg?logo=streamlit)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7.2-blue)](https://xgboost.readthedocs.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.7.2-F7931E.svg?logo=scikit-learn)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end, cloud-deployed Machine Learning pipeline designed for the telecommunications industry. This project transcends standard accuracy metrics by optimizing for strict business constraints—specifically, maximizing ROI on retention budgets using a **Recall@20%** ranking strategy.

It features a decoupled architecture with a FastAPI prediction engine, a persistent PostgreSQL data logging layer for live auditing, an interactive Streamlit frontend, a bulletproof Scikit-Learn preprocessing pipeline, and a live Concept Drift simulator.

### 🔗 Live Deployments
* **Frontend UI (Streamlit):** https://priyanshu-retention-intelligence.streamlit.app/
* **Backend API (FastAPI/Render):** https://api-service-for-xgb-pipeline.onrender.com
* **Interactive API Documentation (Swagger UI):** https://api-service-for-xgb-pipeline.onrender.com/docs

---

## 📊 Model Performance

The final model was evaluated using both traditional machine learning metrics and a business-oriented ranking metric.

| Metric | Score | Why it matters |
|---------|:-----:|----------------|
| **ROC-AUC** | **0.8487** | Measures the model's ability to distinguish between customers who churn and those who stay. |
| **Recall** | **0.8500** | Captures 85% of all actual churners in the test set. |
| **Recall@20%** | **0.5036** | Measures how many actual churners are captured within the top 20% highest-risk customers ranked by the model. This aligns with real-world retention campaigns where businesses have limited outreach capacity. |

## 🏗 System Architecture & Tech Stack

This project is built on a decoupled, cloud-ready architecture to ensure distinct separation of concerns between the user interface, the inference engine, and the telemetry storage layer.

```mermaid
graph TD
    %% Client Node
    Client((Client / Browser))

    %% Streamlit Environment
    subgraph Streamlit ["Frontend: Streamlit Community Cloud"]
        direction TB
        App[app.py - Main Dashboard]
    end

    %% Render Environment
    subgraph Render ["Production Cloud: Render Framework"]
        direction TB
        FastAPI[api.py - FastAPI Server]
        XGBoost[(XGBoost Pipeline .pkl)]
        DB[(PostgreSQL Database)]
    end

    %% Core Prediction Flow
    Client -- "1. Uploads CSV / JSON Action" --> App
    App -- "2. POST /predict Payload" --> FastAPI
    FastAPI -- "3. Executes Preprocessing & Model" --> XGBoost
    XGBoost -- "4. Returns Risk Probabilities" --> FastAPI
    FastAPI -- "5. Triggers Non-Blocking Logging Task" --> DB
    FastAPI -- "6. Displays Recall@20 Ranks Immediately" --> App

    %% Styling Elements (Forced Black Text for Contrast)
    classDef browser fill:#ececff,stroke:#9370db,stroke-width:2px,color:#000000;
    classDef frontend fill:#ffe6e6,stroke:#ff4b4b,stroke-width:2px,color:#000000;
    classDef backend fill:#e6ffe6,stroke:#009688,stroke-width:2px,color:#000000;

    class Client browser;
    class App frontend;
    class FastAPI,XGBoost,DB backend;
```

* **The Brain (Backend):** A FastAPI server hosting a serialized, structural XGBoost pipeline. It automatically generates interactive Swagger documentation at `/docs` for seamless third-party testing.
* **The Storage (Telemetry):** A managed PostgreSQL instance connected asynchronously to log features, metadata, and probabilities in real time.
* **The Face (Frontend):** A modular Streamlit dashboard that orchestrates JSON payloads to the API for real-time and batch predictions.
* **Core Stack:** `scikit-learn`, `xgboost`, `sqlalchemy`, `asyncpg`, `pandas`, `fastapi`, `streamlit`, `plotly`.

### 🔄 End-to-End MLOps Lifecycle

Beyond a static Jupyter Notebook, this project treats machine learning as a continuous software engineering lifecycle. The pipeline is broken into four distinct enterprise phases, tracking how raw data is transformed into explicit, deployable artifacts, and ultimately monitored by an automated feedback loop.

```mermaid
graph TD
    %% Phase 1
    subgraph Phase 1: Data Engineering
        direction TB
        D1[Raw Telecom Data] --> D2[Shared Custom Preprocessing Modules]
        D2 -.-> R1>Artifact: Cleaned Feature Matrix & Encoders]
    end

    %% Phase 2
    subgraph Phase 2: Model Development
        direction TB
        D2 --> M1[XGBoost Classifier]
        M1 --> M2[Threshold Tuning for Recall@20]
        M2 -.-> R2>Artifact: production_pipeline.pkl]
    end

    %% Phase 3
    subgraph Phase 3: Cloud Deployment
        direction TB
        M2 --> P1[FastAPI Web Service]
        P1 --> P2[Streamlit Business UI]
        P1 --> P3[Asynchronous PostgreSQL Registry]
        P3 -.-> R3>Artifact: Persistent Telemetry Audit Trails]
    end

    %% Phase 4
    subgraph Phase 4: Continuous MLOps
        direction TB
        P2 --> O1[Live Concept Drift Simulator]
        O1 --> O2[SLA Drop Trigger & Shadow Retrain]
        O2 -.-> R4>Artifact: retrained_challenger.pkl]
    end

    %% Styling Elements (Forced Black Text & Distinct Shapes)
    classDef default color:#000000;
    classDef process fill:#e3f2fd,stroke:#1e88e5,stroke-width:2px,color:#000000;
    classDef artifact fill:#fffde7,stroke:#fbc02d,stroke-width:2px,stroke-dasharray: 5 5,color:#000000;

    class D1,D2,M1,M2,P1,P2,P3,O1,O2 process;
    class R1,R2,R3,R4 artifact;
```

---

## 📊 The Business Logic: Why Recall@20?

A high-classification metric (like F1 or overall Recall) is meaningless if it doesn't align with business reality. Standard 0.5-threshold optimization assumes an unlimited budget to contact every single customer flagged as a risk, leading to expensive false positives.

**The "Gift Basket" Constraint:**
> Imagine a database of 100 customers. Secretly, 25 are planning to cancel next week. Marketing has a strict budget to send VIP retention gift baskets to exactly 20 people (a 20% resource constraint). If we hand them out randomly, we waste money. 
> 
> Instead, our model ranks customers by raw churn probability. We target the top 20. Out of those 20 baskets, 13 go to the secret group of 25 actual churners. Our **Recall@20 is 52% (13/25)**. 
> 
> **Conclusion:** By leveraging probabilistic ranking, we can intercept more than half of the total revenue about to walk out the door while utilizing only 1/5th of the budget.

---

## 🧠 Machine Learning Philosophy & Model Selection

### 1. The Structured Pipeline Architecture
Messy DataFrame manipulation invalidates model comparison and prevents production scaling. To solve "Array Stripping" and "Prefix Leaks", all preprocessing is locked inside a strict structural pipeline utilizing custom Scikit-Learn classes imported explicitly from independent modules to ensure thread-safe unpickling operations in high-throughput servers.
* **Phase 1 (Sequential Cleaning):** Custom preprocessors execute sequentially to engineer behavioral features (e.g., combining Partner and Dependents into a "Stability" metric) and fill nulls before mathematical scaling.
* **Phase 2 (Parallel Master Transformer):** Scaling (`StandardScaler`) and encoding (`OneHotEncoder`) run in parallel via a `ColumnTransformer` with `verbose_feature_names_out=False` to preserve downstream namespace integrity.

### 2. The SMOTE Trap
While synthetic oversampling (SMOTE) improved standard overall Recall, experimentation proved it injected calibration noise into the highest probability bounds, effectively *dropping* our critical Recall@20% metric. We opted for native algorithm weights (`class_weight='balanced'`) for cleaner probability ranking.

### 3. Model Showdown: XGBoost vs. Random Forest
* **Random Forest:** ROC-AUC: 0.8473 | Overall Recall: 83.5% | Recall@20: 51.4%
* **XGBoost:** ROC-AUC: 0.8487 | Overall Recall: 85.0% | Recall@20: 50.3%
* **The Decision:** XGBoost was crowned the champion. The 1.1% difference in Recall@20 was a statistical tie (less than 4 customers). XGBoost provided a wider safety net (85% overall recall) and better global calibration (ROC-AUC), meaning it scales safer if the business budget suddenly increases to 30% or 40%.

---

## 📊 Database Telemetry Schema

Every live customer lookup and batch inference pipeline evaluation is recorded persistently into PostgreSQL for audit trails, drift analysis, and model performance tracking. The environment dynamically configures itself at boot using an asynchronous connection string matrix.

### Database Layout: `churn_predictions`

| Column Name | Data Type | Configuration | Engineering Objective |
| :--- | :--- | :--- | :--- |
| `id` | `VARCHAR` | Primary Key | Unique UUID tracking string per prediction event. |
| `timestamp` | `TIMESTAMP` | Indexed | Tracks inference timing; indexed to optimize high-volume time-series drift queries. |
| `prediction_mode` | `VARCHAR` | Default: `'single'` | Categorizes tracking via UI origins (`'single'` or `'batch'`). |
| `batch_id` | `VARCHAR` | Indexed, Nullable | Groups row logs belonging to a single bulk CSV pipeline upload together. |
| `input_features` | `JSONB` | Not Null | Stores the raw dictionary input payload; uses `JSONB` for schematic audit elasticity. |
| `churn_probability`| `FLOAT` | Not Null | Captured raw prediction probability float for ranking calibration checks. |
| `churn_prediction` | `INTEGER` | Not Null | The final binary prediction assignment (0 = Retained, 1 = Risk). |
| `model_version` | `VARCHAR` | Default: `'v1.0'` | Tracks production version provenance for legacy auditing. |

---

## 🖥 User Interface (Streamlit)

The UI is divided into 5 distinct operational modules:

1. **High-Level Dashboard:** Project introduction and system status.
2. **Business Strategy Studio:** Real-time Single and Batch CSV inference strictly focused on isolating the Top 20% highest flight-risk accounts. 
3. **Model Mechanics:** Standard threshold classification insights, detailing our pipeline engineering and the decision to forgo SMOTE.
4. **Data Insights (EDA):** Interactive Plotly visualizations highlighting anomalies: The Electronic Check Anomaly, Month-to-Month Senior Citizens, and the Zero-Tenure Data Trap.
5. **Concept Drift Matrix (Live Simulator):** An interactive module that injects mathematical drift into a live data stream (`stream_data.csv`), triggering an SLA failure, and successfully executing a shadow deployment to retrain a Challenger model using base data.

---

## 🛠 Engineering Hurdles Conquered

* **Inference Latency vs. Telemetry Logging:** Writing data inputs and output arrays into a remote database during the live user HTTP lifecycle normally introduces server blocking latency. **Solution:** Leveraged FastAPI's asynchronous `BackgroundTasks` engine pool, handing database transaction calls off to background threads to return results to the UI instantly without blocking.
* **The Async Database Driver Prefix Mismatch:** Cloud platforms like Render supply database strings prefixed with standard `postgres://` or `postgresql://`. Passing this directly to an asynchronous SQLAlchemy engine triggers crashes as it falls back to synchronous `psycopg2` libraries. **Solution:** Engineered a robust runtime parsing regex boundary within `database.py` that dynamically sanitizes connection prefixes to force an `asyncpg` execution driver loop.
* **Environment Drift:** Streamlit Cloud defaulted to `scikit-learn 1.8.0` causing `InconsistentVersionWarning` crashes against our `1.7.2` model. **Solution:** Enforced strict dependency pinning in `requirements.txt`.
* **State Management & Memory:** Unrestricted model loading caused out-of-memory errors on EDA pages. **Solution:** Isolated the `production_pipeline.pkl` loading state strictly to the Streamlit pages that require active inference.

---

## 💻 Local Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Priyanshu-Upadhyay-27/end-to-end-churn-ml.git](https://github.com/Priyanshu-Upadhyay-27/end-to-end-churn-ml.git)
   cd end-to-end-churn-ml
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install exact dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Local Environment Credentials:**
   Create a `.env` file in the root of your backend project directory to inject local telemetry strings:
   ```env
   DATABASE_URL="postgresql+asyncpg://your_username:your_password@localhost:5432/your_database"
   ```

5. **Launch the FastAPI Backend:**
   ```bash
   uvicorn api:app --reload --port 8000
   ```

6. **Launch the Streamlit Frontend (In a new terminal):**
   ```bash
   streamlit run app.py
   ```

---

## 🛡️ Creator & Identity

**Created and Maintained by Priyanshu Upadhyay**

This project is the original intellectual property of Priyanshu Upadhyay, developed to showcase end-to-end MLOps deployment, cloud architecture, and business-focused machine learning.

**About the Developer:**
* **Education:** Computer Science and Engineering Undergraduate at KIET Group of Institutions.
* **Specialization:** Machine Learning, MLOps, scalable system architecture, and predictive analytics.

**Connect:**
* **GitHub:** [Priyanshu-Upadhyay-27](https://github.com/Priyanshu-Upadhyay-27)
* **LinkedIn:** [Priyanshu Upadhyay](https://www.linkedin.com/in/priyanshu-upadhyay-cse/) 

---

<div align="center">

*✨ Designed with a focus on scalable MLOps, rigorous software engineering, and quantifiable business value.*

</div>
