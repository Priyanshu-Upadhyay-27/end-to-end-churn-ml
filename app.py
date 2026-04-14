import streamlit as st
import pandas as pd
import requests
import time
import os
import numpy as np
import joblib
import plotly.graph_objects as go
import xgboost as xgb
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.metrics import recall_score, roc_auc_score



########################Custom Functions for app.py#####################
from sklearn.preprocessing import FunctionTransformer


def preprocessing_raw_data(X):
    df = X.copy()

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    internet_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    phone_cols = ["MultipleLines"]
    binary_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
        "MultipleLines", "Partner", "Dependents",
        "PhoneService", "PaperlessBilling"
    ]

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


new_feature_clean_transformer = FunctionTransformer(preprocessing_raw_data)


def binaryEncoder(X):
    df = X.copy()
    binary_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
        "MultipleLines", "Partner", "Dependents",
        "PhoneService", "PaperlessBilling"
    ]

    mapping = {"no": 0, "yes": 1}

    for col in binary_cols:
        if col in df.columns:
            # map it, and used fillna just in case a weird value sneaks in
            df[col] = df[col].map(mapping).fillna(df[col])

    return df


binary_encoder_transformer = FunctionTransformer(binaryEncoder)

def inject_systematic_drift(df_batch, batch_index, total_batches):
    """
    Mathematically alters telemetry to simulate a shifting business environment.
    Scenario: Competitor slashes prices. Customers are paying less and staying longer,
    but still churning. Champion model falsely assumes they are safe.
    """
    df_mod = df_batch.copy()
    drift_intensity = (batch_index + 1) / total_batches  # Scales from >0.0 to 1.0

    if 'MonthlyCharges' in df_mod.columns:
        # Progressively lower monthly charges by up to 40%
        df_mod['MonthlyCharges'] = df_mod['MonthlyCharges'] * (1 - (0.40 * drift_intensity))

    if 'tenure' in df_mod.columns:
        # Progressively add up to 24 months of artificial tenure
        df_mod['tenure'] = df_mod['tenure'] + (24 * drift_intensity)

    return df_mod


#--- CONFIGURATION & ENTERPRISE CLEAN CSS ---
st.set_page_config(page_title="Retention Intelligence", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    /* NOTE: We have completely removed the code that hides the header. 
       The native Streamlit navigation button will now appear normally. */

    /* Main Background & Text */
    .stApp { background-color: #f4f6f9; color: #334155; font-family: 'Inter', sans-serif; }

    /* Headings */
    h1, h2, h3 { color: #0f172a !important; font-weight: 700; text-transform: uppercase; letter-spacing: 1.2px; }
    h1 { margin-top: 0rem !important; padding-top: 0rem !important; font-size: 3.5rem; }

    /* Clean Enterprise Accents */
    .neon-red { color: #dc2626; font-weight: bold; font-size: 1.2rem; } 
    .neon-green { color: #059669; font-weight: bold; font-size: 1.2rem; } 

    /* Subheaders */
    .cyber-sub { border-left: 4px solid #2563eb; padding-left: 10px; margin-bottom: 20px; color: #475569; }

    /* Sidebar Navigation */
    [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e2e8f0; }
    [data-testid="stSidebar"] .stRadio p { color: #1e293b !important; font-weight: 600; font-size: 1.1rem !important; margin-bottom: 8px;}

    /* Metrics Numbers */
    [data-testid="stMetricValue"] { color: #2563eb; font-weight: 800; }
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
    "Introduction",
    "Business Strategy (Recall@20)",
    "Model Predictions",
    "Data Insights (EDA)",
    "Concept Drift Matrix"
], key="main_nav")

# Global Reset Button placed at the bottom of the sidebar
st.sidebar.divider()
if st.sidebar.button("🔄 Reset System Memory", use_container_width=True):
    # This deletes all saved variables (like the drifted charts on Page 5)
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    # Rerun the app from the top
    st.rerun()

# ==========================================
# PAGE 1: INTRODUCTION
# ==========================================
if page == "Introduction":
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
elif page == "Business Strategy (Recall@20)":
    st.title("Business Strategy Tracker")
    st.markdown(
        "<div class='cyber-sub'>Focus: Identifying the Top 20% Highest-Risk Accounts for maximum Retention ROI.</div>",
        unsafe_allow_html=True)

    input_mode = st.radio("Input Method:", ["Single Target Form", "Batch Processing (CSV Upload)"], horizontal=True, key="input_mode_biz")

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
elif page == "Model Predictions":
    st.title("Global Prediction Engine")
    st.markdown("<div class='cyber-sub'>Focus: Standard classification based on the 0.5 decision threshold.</div>",
                unsafe_allow_html=True)

    input_mode = st.radio("Input Method:", ["Single Target Form", "Batch Processing (CSV Upload)"], horizontal=True, key="input_mode_global")

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
# PAGE 4: SKELETONS
# ==========================================
elif page == "Data Insights (EDA)":
    st.title("Telemetry & Exploratory Data Analysis")
    st.write("Interactive visualizations will be built here.")


# ==========================================
# PAGE 05: CONCEPT DRIFT MATRIX (THE COMMAND CENTER)
# ==========================================
elif "Concept Drift Matrix" == page:
    # --- 1. STATE MANAGEMENT ---
    if "sim_phase" not in st.session_state:
        st.session_state.sim_phase = "init"
    if "terminal_text" not in st.session_state:
        st.session_state.terminal_text = "> [SYSTEM] Awaiting command...\n"

    # --- 2. DYNAMIC CSS ---
    color_1 = "#2563eb" if st.session_state.sim_phase == "init" else "#e2e8f0"
    color_2 = "#eab308" if st.session_state.sim_phase == "streaming" else "#e2e8f0"
    color_3 = "#059669" if st.session_state.sim_phase in ["streaming", "drifted"] else "#e2e8f0"
    color_4 = "#dc2626" if st.session_state.sim_phase == "drifted" else "#e2e8f0"
    color_5 = "#8b5cf6" if st.session_state.sim_phase in ["retraining", "testing", "resolved"] else "#e2e8f0"
    color_6 = "#10b981" if st.session_state.sim_phase == "deployed" else "#e2e8f0"

    st.markdown(f"""
        <style>
        .title-sim {{ margin-top: -2.5rem; font-size: 3.2rem; color: #0f172a; font-weight: 800; text-transform: uppercase;}}
        .theory-quote {{ border-left: 4px solid #2563eb; padding-left: 15px; font-size: 1.1rem; color: #475569; margin-bottom: 2rem; margin-top: 1rem;}}
        .flow-box {{
            padding: 10px; border-radius: 8px; text-align: center; font-weight: 600; font-size: 0.85rem;
            transition: all 0.3s ease; border: 2px solid transparent; background-color: #f8fafc; color: #334155;
            min-height: 110px; display: flex; flex-direction: column; justify-content: center;
        }}
        .box-1 {{ border-color: {color_1}; box-shadow: 0 0 10px {color_1}40; }}
        .box-2 {{ border-color: {color_2}; box-shadow: 0 0 10px {color_2}40; }}
        .box-3 {{ border-color: {color_3}; box-shadow: 0 0 10px {color_3}40; }}
        .box-4 {{ border-color: {color_4}; box-shadow: 0 0 10px {color_4}40; }}
        .box-5 {{ border-color: {color_5}; box-shadow: 0 0 10px {color_5}40; }}
        .box-6 {{ border-color: {color_6}; box-shadow: 0 0 10px {color_6}40; }}
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='title-sim'>Concept Drift & Retraining Simulator</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='theory-quote'>Observe dynamic SLA failure as systematic mathematical drift is injected into the telemetry stream. Evaluate the Challenger model via a sequential blind holdout comparison.</div>",
        unsafe_allow_html=True)

    # --- 3. THE ARCHITECTURE FLOWCHART ---
    flow_col1, flow_col2, flow_col3, flow_col4, flow_col5, flow_col6 = st.columns(6)

    with flow_col1:
        st.markdown(
            f"<div class='flow-box box-1'>📊 1. Data Source<br><br><span style='font-size:0.8em; font-weight:normal;'>1047 Raw Stream</span></div>",
            unsafe_allow_html=True)

    with flow_col2:
        st.markdown(
            f"<div class='flow-box box-2'>🔄 2. Stream<br><br><span style='font-size:0.8em; font-weight:normal;'>Math Drift Inject</span></div>",
            unsafe_allow_html=True)
        if st.session_state.sim_phase == "init":
            start_stream = st.button("▶️ Start", use_container_width=True, type="primary")
        else:
            st.button("▶️ Done", use_container_width=True, disabled=True)
            start_stream = False

    with flow_col3:
        st.markdown(
            f"<div class='flow-box box-3'>🤖 3. Champion<br><br><span style='font-size:0.8em; font-weight:normal;'>Standard Pipeline</span></div>",
            unsafe_allow_html=True)

    with flow_col4:
        st.markdown(
            f"<div class='flow-box box-4'>📉 4. Monitor<br><br><span style='font-size:0.8em; font-weight:normal;'>Recall SLA</span></div>",
            unsafe_allow_html=True)

    with flow_col5:
        st.markdown(
            f"<div class='flow-box box-5'>⚔️ 5. Challenger<br><br><span style='font-size:0.8em; font-weight:normal;'>Retrain Protocol</span></div>",
            unsafe_allow_html=True)
        if st.session_state.sim_phase == "drifted":
            start_retrain = st.button("⚔️ Retrain", use_container_width=True, type="primary")
        else:
            st.button("⚔️ Locked", use_container_width=True, disabled=True)
            start_retrain = False

    with flow_col6:
        st.markdown(
            f"<div class='flow-box box-6'>🚀 6. Production<br><br><span style='font-size:0.8em; font-weight:normal;'>Deploy Winner</span></div>",
            unsafe_allow_html=True)
        if st.session_state.sim_phase == "resolved":
            deploy_model = st.button("🚀 Deploy", use_container_width=True, type="primary")
        elif st.session_state.sim_phase == "deployed":
            st.button("✅ Active", use_container_width=True, disabled=True)
            deploy_model = False
        else:
            st.button("🔒 Locked", use_container_width=True, disabled=True)
            deploy_model = False

    st.divider()

    # --- 4. THE COMMAND CENTER UI ---
    col_logs, col_graphs = st.columns([1, 2])
    with col_logs:
        st.markdown("### 🖥️ System Terminal")
        log_box = st.empty()
        log_box.code(st.session_state.terminal_text, language="bash")
    with col_graphs:
        st.markdown("### 📈 Live Telemetry (Recall vs. Time)")
        graph_box_1 = st.empty()
        graph_box_2 = st.empty()
        success_box = st.empty()

    # --- LOGIC TRIGGER: START STREAM (PHASE 1) ---
    if start_stream:
        st.session_state.sim_phase = "streaming"
        st.rerun()

    if st.session_state.sim_phase == "streaming":
        st.session_state.terminal_text = "> [SYSTEM] Initializing Data Pipeline...\n"
        try:
            # 1. LOAD DATA
            DEFAULT_DATA_PATH = r"C:\Users\priya\Desktop\PyCh_Pro\Churn_Analysis_and_Modelling\data\raw\stream_data.csv"
            st.session_state.terminal_text += f"> [SYSTEM] Loading 1047-row native stream...\n"
            log_box.code(st.session_state.terminal_text, language="bash")

            if not os.path.exists(DEFAULT_DATA_PATH):
                st.error(f"CRITICAL ERROR: Cannot find '{DEFAULT_DATA_PATH}'.")
                st.stop()
            df = pd.read_csv(DEFAULT_DATA_PATH)

            if 'Churn' in df.columns:
                df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0, 1: 1, 0: 0})
                X_stream = df.drop('Churn', axis=1)
                y_stream = df['Churn']

                # Split: 750 rows for Degradation, 297 for Final Testing
                X_phase1, y_phase1 = X_stream.iloc[:750], y_stream.iloc[:750]
                X_phase2, y_phase2 = X_stream.iloc[750:], y_stream.iloc[750:]
            else:
                st.error("Dataset must contain a 'Churn' column.")
                st.stop()

            # 2. LOAD PIPELINE
            PIPELINE_PATH = "production_pipeline.pkl"
            st.session_state.terminal_text += f"> [SYSTEM] Loading pre-trained Master Pipeline (70% Base)...\n"
            log_box.code(st.session_state.terminal_text, language="bash")
            champion = joblib.load(PIPELINE_PATH)

            # 3. STREAM & DRIFT PHASE 1 (15 Batches)
            drift_batches_X = np.array_split(X_phase1, 15)
            drift_batches_y = np.array_split(y_phase1, 15)

            live_metric_history = []
            time_steps = []
            drifted_X_history = []  # Save the modified data for Challenger retraining!
            drift_detected = False

            st.session_state.terminal_text += "> [WARNING] Injecting systematic mathematical drift...\n"
            log_box.code(st.session_state.terminal_text, language="bash")

            for i in range(15):
                # Apply the mathematical drift
                current_batch_X = inject_systematic_drift(drift_batches_X[i], i, 15)
                drifted_X_history.append(current_batch_X)

                preds = champion.predict(current_batch_X)
                try:
                    current_recall = recall_score(drift_batches_y[i], preds, zero_division=0)
                except:
                    current_recall = live_metric_history[-1] if live_metric_history else 0.85

                live_metric_history.append(current_recall)
                time_steps.append(f"Batch {i + 1}")
                st.session_state.terminal_text += f"> [METRIC] {time_steps[-1]} | Champion Recall: {current_recall:.3f}\n"

                if current_recall < 0.65 and not drift_detected:
                    st.session_state.terminal_text += "\n> [CRITICAL] 🚨 SLA BREACH. Recall dropped below 0.65.\n> [CRITICAL] RETRAINING REQUIRED.\n"
                    drift_detected = True

                log_box.code(st.session_state.terminal_text, language="bash")

                fig1 = go.Figure(data=go.Scatter(x=time_steps, y=live_metric_history, mode='lines+markers',
                                                 line=dict(color='#dc2626' if drift_detected else '#2563eb', width=3)))
                fig1.add_hline(y=0.65, line_dash="dash", line_color="red", annotation_text="Recall Failure Threshold")
                fig1.update_layout(title="Phase 1: Time-Series Model Degradation",
                                   xaxis_title="Chronological Data Stream", yaxis_title="Recall Score",
                                   yaxis=dict(range=[0.0, 1.0]), height=300, margin=dict(l=0, r=0, t=40, b=0))
                graph_box_1.plotly_chart(fig1, use_container_width=True)
                time.sleep(0.3)

                # Save state for Phase 2
            st.session_state.sim_phase = "drifted"
            st.session_state.fig1 = fig1
            st.session_state.X_phase1_drifted = pd.concat(drifted_X_history)  # Challenger needs this!
            st.session_state.y_phase1 = y_phase1
            st.session_state.X_phase2 = X_phase2
            st.session_state.y_phase2 = y_phase2
            st.session_state.champion = champion
            st.rerun()

        except Exception as e:
            st.error(f"Pipeline Processing Error: {str(e)}")
            st.stop()

    # Restore Graph 1
    if st.session_state.sim_phase in ["drifted", "testing", "resolved", "deployed"] and 'fig1' in st.session_state:
        graph_box_1.plotly_chart(st.session_state.fig1, use_container_width=True)

    # --- LOGIC TRIGGER: START RETRAIN & TESTING (PHASE 2) ---
    if start_retrain:
        st.session_state.sim_phase = "testing"
        st.rerun()

    if st.session_state.sim_phase == "testing":
        st.session_state.terminal_text += "\n> [ACTION] Retraining protocol authorized...\n"
        st.session_state.terminal_text += "> [SYSTEM] Cloning pipeline architecture for Challenger...\n"
        log_box.code(st.session_state.terminal_text, language="bash")

        champion = st.session_state.champion
        challenger = clone(champion)

        st.session_state.terminal_text += "> [SYSTEM] Fitting Challenger on recent drifted telemetry...\n"
        log_box.code(st.session_state.terminal_text, language="bash")

        # Train Challenger on the 750 rows that were modified in Phase 1
        challenger.fit(st.session_state.X_phase1_drifted, st.session_state.y_phase1)

        st.session_state.terminal_text += "> [SYSTEM] Executing Dual-Stream Holdout Evaluation (297 rows)...\n"
        log_box.code(st.session_state.terminal_text, language="bash")

        # Split the remaining 297 rows into 10 testing batches
        test_batches_X = np.array_split(st.session_state.X_phase2, 10)
        test_batches_y = np.array_split(st.session_state.y_phase2, 10)

        champ_history = []
        challenger_history = []
        test_time_steps = []

        # Dual-Stream Animation Loop
        for i in range(10):
            # Apply Maximum Drift to the test set to represent the "new normal"
            current_test_X = inject_systematic_drift(test_batches_X[i], 15, 15)

            # Both models predict
            champ_preds = champion.predict(current_test_X)
            chall_preds = challenger.predict(current_test_X)

            try:
                champ_recall = recall_score(test_batches_y[i], champ_preds, zero_division=0)
                chall_recall = recall_score(test_batches_y[i], chall_preds, zero_division=0)
            except:
                champ_recall, chall_recall = 0, 0

            champ_history.append(champ_recall)
            challenger_history.append(chall_recall)
            test_time_steps.append(f"Test {i + 1}")

            st.session_state.terminal_text += f"> [TEST] Batch {i + 1} | Champion: {champ_recall:.2f} | Challenger: {chall_recall:.2f}\n"
            log_box.code(st.session_state.terminal_text, language="bash")

            # The Dual-Line Graph
            fig2 = go.Figure()
            fig2.add_trace(
                go.Scatter(x=test_time_steps, y=champ_history, mode='lines+markers', name='Champion (Degraded)',
                           line=dict(color='#dc2626', width=2, dash='dot')))
            fig2.add_trace(
                go.Scatter(x=test_time_steps, y=challenger_history, mode='lines+markers', name='Challenger (Restored)',
                           line=dict(color='#059669', width=3)))
            fig2.add_hline(y=0.65, line_dash="dash", line_color="red", annotation_text="SLA")
            fig2.update_layout(title="Phase 2: Live Holdout Testing Showdown", xaxis_title="Future Data Stream",
                               yaxis_title="Recall Score", yaxis=dict(range=[0.0, 1.0]), height=300,
                               margin=dict(l=0, r=0, t=40, b=0),
                               legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99))

            graph_box_2.plotly_chart(fig2, use_container_width=True)
            time.sleep(0.4)

        avg_champ = np.mean(champ_history)
        avg_chall = np.mean(challenger_history)

        st.session_state.terminal_text += f"\n> [RESULT] Final Avg Champion Recall: {avg_champ:.3f}\n"
        st.session_state.terminal_text += f"> [RESULT] Final Avg Challenger Recall: {avg_chall:.3f}\n"

        if avg_chall > avg_champ:
            st.session_state.terminal_text += "> [DECISION] SUCCESS. Challenger dominates. New weights approved.\n"
        else:
            st.session_state.terminal_text += "> [DECISION] WARNING. Challenger underperformed.\n"

        log_box.code(st.session_state.terminal_text, language="bash")

        st.session_state.fig2 = fig2
        st.session_state.sim_phase = "resolved"
        st.rerun()

    # Restore Graph 2
    if st.session_state.sim_phase in ["resolved", "deployed"] and 'fig2' in st.session_state:
        graph_box_2.plotly_chart(st.session_state.fig2, use_container_width=True)

    # --- LOGIC TRIGGER: DEPLOYMENT ---
    if deploy_model:
        st.session_state.sim_phase = "deployed"
        st.rerun()

    if st.session_state.sim_phase == "deployed":
        st.session_state.terminal_text += "\n> [DEPLOYMENT] Initializing API override...\n"
        st.session_state.terminal_text += "> [DEPLOYMENT] System state synchronized. New weights active in production.\n"
        log_box.code(st.session_state.terminal_text, language="bash")
        success_box.success(
            "✅ **Deployment Successful:** The Challenger model is now serving live traffic. (Simulation Complete)")

        if st.button("Reset System Simulator", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()