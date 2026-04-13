import streamlit as st
import pandas as pd
import requests
import time

# --- CONFIGURATION & ENTERPRISE CLEAN CSS ---
st.set_page_config(page_title="Retention Intelligence", page_icon="📊", layout="wide")

st.markdown("""
    <style>
    /* Hide the Streamlit top header */
    [data-testid="stHeader"] { display: none; }

    /* Main Background & Text (Light Mode) */
    .stApp { background-color: #f4f6f9; color: #334155; font-family: 'Inter', sans-serif; }

    /* Headings (Deep Slate) */
    h1, h2, h3 { color: #0f172a !important; font-weight: 700; text-transform: uppercase; letter-spacing: 1.2px; }
    h1 { margin-top: 0rem !important; padding-top: 0rem !important; font-size: 3.5rem; }

    /* Clean Enterprise Accents */
    .neon-red { color: #dc2626; font-weight: bold; font-size: 1.2rem; } /* Brick Red */
    .neon-green { color: #059669; font-weight: bold; font-size: 1.2rem; } /* Emerald Green */

    /* Subheaders */
    .cyber-sub { border-left: 4px solid #2563eb; padding-left: 10px; margin-bottom: 20px; color: #475569; }

    /* Sidebar Navigation */
    [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e2e8f0; }
    [data-testid="stSidebar"] .stRadio p { color: #1e293b !important; font-weight: 600; font-size: 1.1rem !important; margin-bottom: 8px;}

    /* Metrics Numbers (Cobalt Blue) */
    [data-testid="stMetricValue"] { color: #2563eb; font-weight: 800; }

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
    st.markdown(
        "<h1 style='text-align: center; color: #0f172a;'>CUSTOMER RETENTION <span style='color: #2563eb;'>INTELLIGENCE</span></h1>",
        unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #64748b; font-weight: 400;'>ANTICIPATE. INTERVENE. RETAIN.</h3>",
                unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; font-size: 1rem; color: #94a3b8;'>Advanced Telecommunications Churn Analytics</p>",
        unsafe_allow_html=True)

    st.divider()

    col1, col2, col3 = st.columns(3)
    col1.metric(label="Global Recall", value="85.0%", delta="Production Ready")
    col2.metric(label="Top 20% Capture Rate", value="50.4%", delta="High ROI")
    col3.metric(label="Inference Latency", value="< 50ms", delta="Real-time")

    st.markdown("### ⚡ The Objective")
    st.write(
        "Customer attrition is a silent revenue killer. This platform deploys a highly tuned XGBoost pipeline to scan account metadata, service configurations, and billing histories to identify flight-risk customers *before* they cancel their contracts. Select a module from the sidebar to begin.")

# ==========================================
# PAGE 2: BUSINESS STRATEGY (RECALL@20)
# ==========================================
elif page == "02 // Business Strategy (Recall@20)":
    st.title("Business Strategy Tracker")
    st.markdown(
        "<div class='cyber-sub'>Focus: Identifying the Top 20% Highest-Risk Accounts for maximum Retention ROI.</div>",
        unsafe_allow_html=True)

    input_mode = st.radio("Input Method:", ["Single Target Form", "Batch Processing (CSV Upload)"], horizontal=True)

    if input_mode == "Single Target Form":
        payload = render_input_form("biz")
        if payload:
            with st.spinner("Analyzing..."):
                try:
                    response = requests.post(f"{API_URL}/predict", json=payload)
                    if response.status_code == 200:
                        prob = response.json()['probability']
                        st.divider()
                        if prob >= 0.80:
                            st.markdown(f"<span class='neon-red'>⚠️ CRITICAL RISK (Top Percentile)</span>",
                                        unsafe_allow_html=True)
                            st.write("Recommend immediate High-Value Retention Protocol (Discount/Upgrade).")
                        else:
                            st.markdown(f"<span class='neon-green'>✅ MODERATE TO LOW RISK</span>",
                                        unsafe_allow_html=True)
                        st.progress(prob)
                        st.write(f"**Calculated Probability:** {prob * 100:.2f}%")
                    else:
                        st.error(f"API Error: {response.text}")
                except Exception as e:
                    st.error(f"Server Offline: {e}")

    elif input_mode == "Batch Processing (CSV Upload)":
        uploaded_file = st.file_uploader("Drop Customer CSV Here", type=["csv"])
        if uploaded_file and st.button("Run Top 20% Batch Scan"):
            df = pd.read_csv(uploaded_file)
            has_ground_truth = "Churn" in df.columns
            api_df = df.drop(columns=["Churn"]) if has_ground_truth else df.copy()

            with st.spinner("Processing massive dataset..."):
                try:
                    response = requests.post(f"{API_URL}/predict_batch", json=api_df.to_dict(orient="records"))
                    if response.status_code == 200:
                        res_df = pd.DataFrame(response.json()["batch_results"])
                        final_df = pd.concat([df, res_df[['probability', 'prediction']]], axis=1)
                        final_df = final_df.sort_values(by="probability", ascending=False).reset_index(drop=True)

                        top_20_cutoff = int(len(final_df) * 0.20)
                        top_20_df = final_df.head(top_20_cutoff)

                        st.success(f"Scan Complete. Isolated {top_20_cutoff} critical targets.")

                        if has_ground_truth:
                            # Safely convert target to binary for math
                            if final_df['Churn'].dtype == 'object':
                                final_df['Churn_Binary'] = final_df['Churn'].map({'Yes': 1, 'No': 0})
                                top_20_df['Churn_Binary'] = top_20_df['Churn'].map({'Yes': 1, 'No': 0})
                            else:
                                final_df['Churn_Binary'] = final_df['Churn']
                                top_20_df['Churn_Binary'] = top_20_df['Churn']

                            total_actual_churners = final_df['Churn_Binary'].sum()
                            captured_churners = top_20_df['Churn_Binary'].sum()
                            recall_at_20 = (
                                        captured_churners / total_actual_churners * 100) if total_actual_churners > 0 else 0

                            st.markdown("### 📊 Evaluation Metrics")
                            mcol1, mcol2, mcol3 = st.columns(3)
                            mcol1.metric("Total Actual Churners in CSV", int(total_actual_churners))
                            mcol2.metric("Churners Caught in Top 20%", int(captured_churners))
                            mcol3.metric("Recall @ 20", f"{recall_at_20:.2f}%", delta="Matched Notebook")
                            st.divider()

                        # Formatting output dataframe (dropping internal binary column before display)
                        display_df = top_20_df.drop(columns=['Churn_Binary'], errors='ignore')
                        st.dataframe(display_df.style.background_gradient(cmap="Reds", subset=["probability"]),
                                     use_container_width=True)
                    else:
                        st.error(f"API Error: {response.text}")
                except Exception as e:
                    st.error(f"Server Offline: {e}")

# ==========================================
# PAGE 3: MODEL PREDICTIONS (STANDARD)
# ==========================================
elif page == "03 // Model Predictions":
    st.title("Global Prediction Engine")
    st.markdown("<div class='cyber-sub'>Focus: Standard classification based on the 0.5 decision threshold.</div>",
                unsafe_allow_html=True)

    input_mode = st.radio("Input Method:", ["Single Target Form", "Batch Processing (CSV Upload)"], horizontal=True)

    if input_mode == "Single Target Form":
        payload = render_input_form("mod")
        if payload:
            with st.spinner("Inferencing..."):
                try:
                    response = requests.post(f"{API_URL}/predict", json=payload)
                    if response.status_code == 200:
                        data = response.json()
                        st.divider()
                        if data['prediction'] == 1:
                            st.markdown("<span class='neon-red'>⚠️ CHURN DETECTED</span>", unsafe_allow_html=True)
                        else:
                            st.markdown("<span class='neon-green'>✅ ACCOUNT STABLE</span>", unsafe_allow_html=True)
                        st.write(f"**Raw Probability:** {data['probability']:.4f}")
                        st.progress(data['probability'])
                    else:
                        st.error(f"API Error: {response.text}")
                except Exception as e:
                    st.error(f"Server Offline: {e}")

    elif input_mode == "Batch Processing (CSV Upload)":
        uploaded_file = st.file_uploader("Drop Customer CSV Here", type=["csv"], key="mod_upload")
        if uploaded_file and st.button("Run Global Scan"):
            df = pd.read_csv(uploaded_file)
            api_df = df.drop(columns=["Churn"]) if "Churn" in df.columns else df.copy()

            with st.spinner("Processing dataset..."):
                try:
                    response = requests.post(f"{API_URL}/predict_batch", json=api_df.to_dict(orient="records"))
                    if response.status_code == 200:
                        res_df = pd.DataFrame(response.json()["batch_results"])
                        final_df = pd.concat([df, res_df[['probability', 'prediction']]], axis=1)
                        st.success("Global Classification Complete.")
                        st.dataframe(final_df, use_container_width=True)
                    else:
                        st.error(f"API Error: {response.text}")
                except Exception as e:
                    st.error(f"Server Offline: {e}")

# ==========================================
# PAGE 4 & 5: SKELETONS
# ==========================================
elif page == "04 // Data Insights (EDA)":
    st.title("Telemetry & Exploratory Data Analysis")
    st.write("Interactive visualizations will be built here.")

elif page == "05 // Concept Drift Matrix":
    st.title("Concept Drift Simulation")
    st.write("Automated model degradation and retraining cycle visualization will be built here.")