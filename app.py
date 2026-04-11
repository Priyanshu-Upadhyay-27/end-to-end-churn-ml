import streamlit as st
import pandas as pd
import requests
import time
import plotly.express as px
import plotly.graph_objects as go

# --- CONFIGURATION & CYBERPUNK CSS ---
st.set_page_config(page_title="Churn OS | Retention Radar", page_icon="🛰️", layout="wide")

# Custom CSS for that dark, cinematic, premium UI
st.markdown("""
    <style>
    /* Main Background & Text */
    .stApp {
        background-color: #0b0c10;
        color: #c5c6c7;
        font-family: 'Inter', sans-serif;
    }

    /* Headings */
    h1, h2, h3 { color: #66fcf1 !important; font-weight: 600; text-transform: uppercase; letter-spacing: 1.5px; }

    /* Neon Accents for Risk/Safe */
    .neon-red { color: #ff003c; text-shadow: 0 0 10px rgba(255, 0, 60, 0.5); font-weight: bold; font-size: 1.2rem; }
    .neon-green { color: #00f0ff; text-shadow: 0 0 10px rgba(0, 240, 255, 0.5); font-weight: bold; font-size: 1.2rem; }

    /* Subheaders */
    .cyber-sub { border-left: 4px solid #45a29e; padding-left: 10px; margin-bottom: 20px; }

    /* Clean up the sidebar */
    [data-testid="stSidebar"] { background-color: #1f2833; border-right: 1px solid #45a29e; }

    /* Metrics */
    [data-testid="stMetricValue"] { color: #66fcf1; }
    </style>
""", unsafe_allow_html=True)

# API Endpoint Configuration
API_URL = "http://127.0.0.1:8000"

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("System // Menu")
page = st.sidebar.radio("Navigation", [
    "01 // The Hook (Overview)",
    "02 // High-Value Targets (Batch)",
    "03 // Global Radar (Single)",
    "04 // Drift Simulation",
    "05 // Data Insights"
])

# ==========================================
# PAGE 1: THE HOOK (OVERVIEW)
# ==========================================
if page == "01 // The Hook (Overview)":
    st.markdown(
        "<h1 style='text-align: center; font-size: 3.5rem; margin-top: 50px;'>ANTICIPATE. INTERVENE. RETAIN.</h1>",
        unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; font-size: 1.2rem; color: #45a29e;'>Advanced Telecommunications Churn Analytics OS</p>",
        unsafe_allow_html=True)

    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Global Recall", value="85.0%", delta="Production Ready")
        st.caption("Captures 85% of all actual churn events.")
    with col2:
        st.metric(label="Top 20% Capture Rate", value="50.4%", delta="High ROI")
        st.caption("Captures half of all churners by targeting only the top 20% highest-risk accounts.")
    with col3:
        st.metric(label="Inference Latency", value="< 50ms", delta="Real-time")
        st.caption("Powered by XGBoost & FastAPI.")

    st.markdown("### ⚡ The Objective")
    st.write(
        "Customer attrition is a silent revenue killer. This OS deploys a highly tuned XGBoost pipeline to scan account metadata, service configurations, and billing histories to identify flight-risk customers *before* they cancel their contracts. Use the sidebar to initiate scans.")

# ==========================================
# PAGE 2: HIGH-VALUE TARGETS (BATCH)
# ==========================================
elif page == "02 // High-Value Targets (Batch)":
    st.markdown("<h2>Batch Risk Assessment</h2>", unsafe_allow_html=True)
    st.markdown(
        "<div class='cyber-sub'>Upload bulk customer data. The system will rank them by risk and isolate the top 20% for immediate retention protocols.</div>",
        unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload Customer CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(f"Loaded {len(df)} customer records.")

        if st.button("Initiate Batch Scan"):
            with st.spinner("Transmitting to FastAPI & running XGBoost pipeline..."):
                try:
                    # Convert to list of dicts for the API
                    payload = df.to_dict(orient="records")
                    response = requests.post(f"{API_URL}/predict_batch", json=payload)

                    if response.status_code == 200:
                        results = response.json()["batch_results"]
                        res_df = pd.DataFrame(results)

                        # Merge back with original data
                        final_df = pd.concat([df, res_df[['probability', 'prediction']]], axis=1)

                        # Sort by highest risk
                        final_df = final_df.sort_values(by="probability", ascending=False).reset_index(drop=True)

                        # Isolate Top 20%
                        top_20_cutoff = int(len(final_df) * 0.20)

                        st.success("Scan Complete.")
                        st.markdown(f"### 🚨 Top 20% Critical Targets ({top_20_cutoff} Accounts)")

                        # Style the dataframe to highlight the danger zone
                        st.dataframe(
                            final_df.head(top_20_cutoff).style.background_gradient(cmap="Reds", subset=["probability"]),
                            use_container_width=True
                        )

                    else:
                        st.error(f"API Error: {response.text}")
                except Exception as e:
                    st.error(f"Server Offline: Ensure Uvicorn is running. Details: {e}")

# ==========================================
# PAGE 3: GLOBAL RADAR (SINGLE PREDICTOR)
# ==========================================
elif page == "03 // Global Radar (Single)":
    st.markdown("<h2>Single Target Radar</h2>", unsafe_allow_html=True)
    st.markdown("<div class='cyber-sub'>Input specific customer telemetry to gauge immediate flight risk.</div>",
                unsafe_allow_html=True)

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### Demographics")
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.slider("Tenure (Months)", 0, 72, 12)

        with col2:
            st.markdown("### Services")
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            multiple = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            device = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])

        with col3:
            st.markdown("### Billing & Tech")
            tech = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)",
                                                      "Credit card (automatic)"])
            monthly = st.number_input("Monthly Charges", value=84.75)
            total = st.text_input("Total Charges", value="1050.50")

        submitted = st.form_submit_button("Engage Radar", use_container_width=True)

        if submitted:
            payload = {
                "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
                "tenure": tenure, "PhoneService": phone, "MultipleLines": multiple, "InternetService": internet,
                "OnlineSecurity": security, "OnlineBackup": backup, "DeviceProtection": device, "TechSupport": tech,
                "StreamingTV": tv, "StreamingMovies": movies, "Contract": contract, "PaperlessBilling": paperless,
                "PaymentMethod": payment, "MonthlyCharges": monthly, "TotalCharges": total
            }

            with st.spinner("Connecting to Vault..."):
                try:
                    response = requests.post(f"{API_URL}/predict", json=payload)
                    if response.status_code == 200:
                        data = response.json()
                        prob = data['probability']

                        st.divider()
                        if data['prediction'] == 1:
                            st.markdown(f"<span class='neon-red'>⚠️ HIGH RISK DETECTED</span>", unsafe_allow_html=True)
                            st.markdown(f"**Churn Probability:** {prob * 100:.2f}%")
                            st.progress(prob)
                            st.write(
                                "This customer sits above the 0.5 threshold. Recommend immediate retention protocol.")
                        else:
                            st.markdown(f"<span class='neon-green'>✅ STABLE ACCOUNT</span>", unsafe_allow_html=True)
                            st.markdown(f"**Churn Probability:** {prob * 100:.2f}%")
                            st.progress(prob)
                            st.write("This customer is likely to renew.")
                    else:
                        st.error("API Error.")
                except Exception as e:
                    st.error("Server Offline. Start FastAPI.")

# ==========================================
# PAGE 4 & 5: SKELETONS FOR NEXT STEPS
# ==========================================
elif page == "04 // Drift Simulation":
    st.markdown("<h2>Data Drift & Retraining Protocol</h2>", unsafe_allow_html=True)
    st.write("Visual simulation of ROC-AUC degradation and automated model recovery.")

    if st.button("Simulate Concept Drift"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Fake animation for now to show the UI vibe
        status_text.text("Injecting 15% manipulated stream data...")
        for i in range(100):
            time.sleep(0.02)
            progress_bar.progress(i + 1)

        st.error("WARNING: Data Distribution Shift Detected. ROC-AUC degraded below 0.70.")
        if st.button("Initiate Retraining Protocol"):
            st.info("Re-aligning weights... Updating MLflow Vault...")
            # We will build the real graph logic here next!

elif page == "05 // Data Insights":
    st.markdown("<h2>Telemetry Insights</h2>", unsafe_allow_html=True)
    st.write("Load data to view distributions.")
    # We will put Plotly graphs here next!