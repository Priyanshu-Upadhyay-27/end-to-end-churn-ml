import streamlit as st
import pandas as pd
import requests
import time

# --- CONFIGURATION & CYBERPUNK CSS ---
st.set_page_config(page_title="Churn OS | Retention Radar", page_icon="🛰️", layout="wide")

st.markdown("""
    <style>
    /* Main Background & Text */
    .stApp { background-color: #0b0c10; color: #c5c6c7; font-family: 'Inter', sans-serif; }

    /* Headings */
    h1, h2, h3 { color: #66fcf1 !important; font-weight: 600; text-transform: uppercase; letter-spacing: 1.5px; }
    h1 { margin-top: 0rem !important; padding-top: 0rem !important; font-size: 3.5rem; }

    /* Neon Accents */
    .neon-red { color: #ff003c; text-shadow: 0 0 10px rgba(255, 0, 60, 0.5); font-weight: bold; font-size: 1.2rem; }
    .neon-green { color: #00f0ff; text-shadow: 0 0 10px rgba(0, 240, 255, 0.5); font-weight: bold; font-size: 1.2rem; }

    /* Sidebar Navigation Font Visibility Fix */
    [data-testid="stSidebar"] { background-color: #1a202c; border-right: 1px solid #45a29e; }
    [data-testid="stSidebar"] .stRadio p { color: #ffffff !important; font-size: 1.1rem !important; margin-bottom: 8px;}

    /* Toggle Button Styling */
    div[role="radiogroup"] { padding-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

API_URL = "http://127.0.0.1:8000"


# --- REUSABLE UI COMPONENTS ---
def render_input_form(key_prefix):
    """Renders the 19-field form and returns the payload if submitted."""
    with st.form(f"{key_prefix}_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### Demographics")
            gender = st.selectbox("Gender", ["Female", "Male"], key=f"{key_prefix}_g")
            senior = st.selectbox("Senior Citizen", [0, 1], key=f"{key_prefix}_sc")
            partner = st.selectbox("Partner", ["Yes", "No"], key=f"{key_prefix}_p")
            dependents = st.selectbox("Dependents", ["Yes", "No"], key=f"{key_prefix}_d")
            tenure = st.slider("Tenure (Months)", 0, 72, 12, key=f"{key_prefix}_t")
        with col2:
            st.markdown("### Services")
            phone = st.selectbox("Phone Service", ["Yes", "No"], key=f"{key_prefix}_ps")
            multiple = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"], key=f"{key_prefix}_ml")
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], key=f"{key_prefix}_is")
            security = st.selectbox("Online Security", ["Yes", "No", "No internet service"], key=f"{key_prefix}_os")
            backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"], key=f"{key_prefix}_ob")
            device = st.selectbox("Device Protection", ["Yes", "No", "No internet service"], key=f"{key_prefix}_dp")
        with col3:
            st.markdown("### Billing & Tech")
            tech = st.selectbox("Tech Support", ["Yes", "No", "No internet service"], key=f"{key_prefix}_ts")
            tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"], key=f"{key_prefix}_stv")
            movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"], key=f"{key_prefix}_sm")
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], key=f"{key_prefix}_c")
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"], key=f"{key_prefix}_pb")
            payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)",
                                                      "Credit card (automatic)"], key=f"{key_prefix}_pm")
            monthly = st.number_input("Monthly Charges", value=84.75, key=f"{key_prefix}_mc")
            total = st.text_input("Total Charges", value="1050.50", key=f"{key_prefix}_tc")

        if st.form_submit_button("Initiate Scan", use_container_width=True):
            return {
                "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
                "tenure": tenure, "PhoneService": phone, "MultipleLines": multiple, "InternetService": internet,
                "OnlineSecurity": security, "OnlineBackup": backup, "DeviceProtection": device, "TechSupport": tech,
                "StreamingTV": tv, "StreamingMovies": movies, "Contract": contract, "PaperlessBilling": paperless,
                "PaymentMethod": payment, "MonthlyCharges": monthly, "TotalCharges": total
            }
    return None


# --- SIDEBAR NAVIGATION ---
st.sidebar.title("System // Menu")
page = st.sidebar.radio("Navigation", [
    "01 // Introduction",
    "02 // Business Strategy (Recall@20)",
    "03 // Model Predictions",
    "04 // Data Insights (EDA)",
    "05 // Concept Drift Matrix"
])

# ==========================================
# PAGE 1: INTRODUCTION
# ==========================================
if page == "01 // Introduction":
    st.markdown("<h1 style='text-align: center;'>ANTICIPATE. INTERVENE. RETAIN.</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; font-size: 1.2rem; color: #45a29e;'>Advanced Telecommunications Churn Analytics OS</p>",
        unsafe_allow_html=True)
    st.divider()

    col1, col2, col3 = st.columns(3)
    col1.metric(label="Global Recall", value="85.0%", delta="Production Ready")
    col2.metric(label="Top 20% Capture Rate", value="50.4%", delta="High ROI")
    col3.metric(label="Inference Latency", value="< 50ms", delta="Real-time")

    st.markdown("### ⚡ The Objective")
    st.write(
        "Customer attrition is a silent revenue killer. This OS deploys a highly tuned XGBoost pipeline to scan account metadata, service configurations, and billing histories to identify flight-risk customers *before* they cancel their contracts. Select a module from the sidebar to begin.")

# ==========================================
# PAGE 2: BUSINESS STRATEGY (RECALL@20)
# ==========================================
elif page == "02 // Business Strategy (Recall@20)":
    st.title("Business Strategy Tracker")
    st.write("Focus: Identifying the Top 20% Highest-Risk Accounts for maximum Retention ROI.")

    input_mode = st.radio("Input Method:", ["Single Target Form", "Batch Processing (CSV Upload)"], horizontal=True)

    if input_mode == "Single Target Form":
        payload = render_input_form("biz")
        if payload:
            with st.spinner("Analyzing..."):
                response = requests.post(f"{API_URL}/predict", json=payload)
                if response.status_code == 200:
                    prob = response.json()['probability']
                    st.divider()
                    if prob >= 0.80:  # Business logic: High risk threshold
                        st.markdown(f"<span class='neon-red'>⚠️ CRITICAL RISK (Top Percentile)</span>",
                                    unsafe_allow_html=True)
                        st.write("Recommend immediate High-Value Retention Protocol (Discount/Upgrade).")
                    else:
                        st.markdown(f"<span class='neon-green'>✅ MODERATE TO LOW RISK</span>", unsafe_allow_html=True)
                    st.progress(prob)
                else:
                    st.error("API Error.")

    elif input_mode == "Batch Processing (CSV Upload)":
        uploaded_file = st.file_uploader("Drop Customer CSV Here", type=["csv"])
        if uploaded_file and st.button("Run Top 20% Batch Scan"):
            df = pd.read_csv(uploaded_file)
            has_ground_truth = "Churn" in df.columns
            api_df = df.drop(columns=["Churn"]) if has_ground_truth else df.copy()

            with st.spinner("Processing massive dataset..."):
                response = requests.post(f"{API_URL}/predict_batch", json=api_df.to_dict(orient="records"))
                if response.status_code == 200:
                    res_df = pd.DataFrame(response.json()["batch_results"])
                    final_df = pd.concat([df, res_df[['probability', 'prediction']]], axis=1)
                    final_df = final_df.sort_values(by="probability", ascending=False).reset_index(drop=True)

                    top_20_cutoff = int(len(final_df) * 0.20)
                    top_20_df = final_df.head(top_20_cutoff)

                    st.success(f"Scan Complete. Isolated {top_20_cutoff} critical targets.")
                    st.dataframe(top_20_df.style.background_gradient(cmap="Reds", subset=["probability"]),
                                 use_container_width=True)

# ==========================================
# PAGE 3: MODEL PREDICTIONS (STANDARD)
# ==========================================
elif page == "03 // Model Predictions":
    st.title("Global Prediction Engine")
    st.write("Focus: Standard classification based on the 0.5 decision threshold.")

    input_mode = st.radio("Input Method:", ["Single Target Form", "Batch Processing (CSV Upload)"], horizontal=True)

    if input_mode == "Single Target Form":
        payload = render_input_form("mod")
        if payload:
            with st.spinner("Inferencing..."):
                response = requests.post(f"{API_URL}/predict", json=payload)
                if response.status_code == 200:
                    data = response.json()
                    st.divider()
                    if data['prediction'] == 1:
                        st.markdown("<span class='neon-red'>⚠️ CHURN DETECTED</span>", unsafe_allow_html=True)
                    else:
                        st.markdown("<span class='neon-green'>✅ ACCOUNT STABLE</span>", unsafe_allow_html=True)
                    st.write(f"Raw Probability: {data['probability']:.4f}")
                    st.progress(data['probability'])
                else:
                    st.error("API Error.")

    elif input_mode == "Batch Processing (CSV Upload)":
        uploaded_file = st.file_uploader("Drop Customer CSV Here", type=["csv"], key="mod_upload")
        if uploaded_file and st.button("Run Global Scan"):
            df = pd.read_csv(uploaded_file)
            api_df = df.drop(columns=["Churn"]) if "Churn" in df.columns else df.copy()
            with st.spinner("Processing dataset..."):
                response = requests.post(f"{API_URL}/predict_batch", json=api_df.to_dict(orient="records"))
                if response.status_code == 200:
                    res_df = pd.DataFrame(response.json()["batch_results"])
                    final_df = pd.concat([df, res_df[['probability', 'prediction']]], axis=1)
                    st.dataframe(final_df, use_container_width=True)

# ==========================================
# PAGE 4 & 5: SKELETONS
# ==========================================
elif page == "04 // Data Insights (EDA)":
    st.title("Telemetry & Exploratory Data Analysis")
    st.write("Interactive visualizations will be built here.")

elif page == "05 // Concept Drift Matrix":
    st.title("Concept Drift Simulation")
    st.write("Automated model degradation and retraining cycle visualization will be built here.")