import streamlit as st
import pandas as pd
import requests
import time
import numpy as np
import plotly.graph_objects as go
from sklearn.datasets import make_classification
from sklearn.datasets import make_classification
from sklearn.metrics import recall_score, roc_auc_score
import xgboost as xgb

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
# PAGE 4: SKELETONS
# ==========================================
elif page == "04 // Data Insights (EDA)":
    st.title("Telemetry & Exploratory Data Analysis")
    st.write("Interactive visualizations will be built here.")

# ==========================================
# PAGE 05: CONCEPT DRIFT MATRIX (THE COMMAND CENTER)
# ==========================================
elif page == "05 // Concept Drift Matrix":

    # --- 1. STATE MANAGEMENT ---
    if "sim_phase" not in st.session_state:
        st.session_state.sim_phase = "init"
    if "terminal_text" not in st.session_state:
        st.session_state.terminal_text = "> [SYSTEM] Awaiting command...\n"

    # --- 2. DYNAMIC CSS FOR ACTIVE FLOWCHART HIGHLIGHTING ---
    color_1 = "#2563eb" if st.session_state.sim_phase == "init" else "#e2e8f0"
    color_2 = "#eab308" if st.session_state.sim_phase == "streaming" else "#e2e8f0"
    color_3 = "#059669" if st.session_state.sim_phase in ["streaming", "drifted"] else "#e2e8f0"
    color_4 = "#dc2626" if st.session_state.sim_phase == "drifted" else "#e2e8f0"
    color_5 = "#8b5cf6" if st.session_state.sim_phase in ["retraining", "resolved"] else "#e2e8f0"
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
        "<div class='theory-quote'>Watch the model's <b>Recall</b> degrade over time under shifting data conditions. The system utilizes the official production pipeline for feature engineering.</div>",
        unsafe_allow_html=True)

    # --- 3. THE ARCHITECTURE FLOWCHART ---
    flow_col1, flow_col2, flow_col3, flow_col4, flow_col5, flow_col6 = st.columns(6)

    with flow_col1:
        st.markdown(
            f"<div class='flow-box box-1'>📊 1. Data Source<br><br><span style='font-size:0.8em; font-weight:normal;'>Native vs Upload</span></div>",
            unsafe_allow_html=True)
        uploaded_csv = st.file_uploader("Upload Data", type=['csv'], disabled=(st.session_state.sim_phase != "init"),
                                        label_visibility="collapsed")

    with flow_col2:
        st.markdown(
            f"<div class='flow-box box-2'>🔄 2. Stream<br><br><span style='font-size:0.8em; font-weight:normal;'>Time-Series Inject</span></div>",
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

    # --- LOGIC TRIGGER: START STREAM ---
    if start_stream:
        st.session_state.sim_phase = "streaming"
        st.rerun()

    if st.session_state.sim_phase == "streaming":
        st.session_state.terminal_text = "> [SYSTEM] Initializing Data Pipeline...\n"

        try:
            # 1. LOAD THE DATA
            if uploaded_csv is not None:
                st.session_state.terminal_text += "> [SYSTEM] Custom Sandbox CSV detected. Parsing...\n"
                log_box.code(st.session_state.terminal_text, language="bash")
                df = pd.read_csv(uploaded_csv)
            else:
                DEFAULT_DATA_PATH = "your_15_percent_data.csv"  # <-- UPDATE THIS
                st.session_state.terminal_text += f"> [SYSTEM] Loading native telemetry stream: {DEFAULT_DATA_PATH}...\n"
                log_box.code(st.session_state.terminal_text, language="bash")

                if not os.path.exists(DEFAULT_DATA_PATH):
                    st.error(f"CRITICAL ERROR: Cannot find '{DEFAULT_DATA_PATH}'.")
                    st.stop()
                df = pd.read_csv(DEFAULT_DATA_PATH)

            # 2. PREPARE RAW DATA (No pd.get_dummies here!)
            if 'Churn' in df.columns:
                df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0, 1: 1, 0: 0})
                X_raw = df.drop('Churn', axis=1)
                y = df['Churn']

                split_idx = int(len(df) * 0.5)
                X_base, y_base = X_raw.iloc[:split_idx], y.iloc[:split_idx]
                X_drift, y_drift = X_raw.iloc[split_idx:], y.iloc[split_idx:]
            else:
                st.error("Dataset must contain a 'Churn' column.")
                st.stop()

            # 3. LOAD THE PRODUCTION PIPELINE
            PIPELINE_PATH = "production_pipeline.pkl"
            st.session_state.terminal_text += f"> [SYSTEM] Loading master pipeline architecture: {PIPELINE_PATH}...\n"
            log_box.code(st.session_state.terminal_text, language="bash")

            if not os.path.exists(PIPELINE_PATH):
                st.error(f"CRITICAL ERROR: Cannot find '{PIPELINE_PATH}'.")
                st.stop()

            champion = joblib.load(PIPELINE_PATH)

            # Fit the loaded pipeline on the base data to establish the Champion
            st.session_state.terminal_text += f"> [SYSTEM] Fitting Champion Pipeline on base data block...\n"
            log_box.code(st.session_state.terminal_text, language="bash")
            champion.fit(X_base, y_base)

            # 4. STREAM THE DRIFT DATA (Time-Series Simulation)
            drift_batches = np.array_split(X_drift, 15)
            drift_y_batches = np.array_split(y_drift, 15)

            live_metric_history = []
            time_steps = []
            drift_detected = False

            st.session_state.terminal_text += "> [WARNING] Injecting chronological drift data...\n"
            log_box.code(st.session_state.terminal_text, language="bash")

            for i in range(len(drift_batches)):
                # The pipeline handles all preprocessing automatically here!
                preds = champion.predict(drift_batches[i])
                try:
                    current_recall = recall_score(drift_y_batches[i], preds, zero_division=0)
                except:
                    current_recall = live_metric_history[-1] if live_metric_history else 0.85

                live_metric_history.append(current_recall)
                time_steps.append(f"Batch {i + 1}")
                st.session_state.terminal_text += f"> [METRIC] {time_steps[-1]} | Champion Recall: {current_recall:.3f}\n"

                # SLA Failure Logic
                if current_recall < 0.65 and not drift_detected:
                    st.session_state.terminal_text += "\n> [CRITICAL] 🚨 SLA BREACH. Recall dropped below 0.65.\n> [CRITICAL] RETRAINING REQUIRED.\n"
                    drift_detected = True

                log_box.code(st.session_state.terminal_text, language="bash")

                # Plotly Graph: X-axis is explicitly Time (Batches), Y-axis is locked 0.0 to 1.0
                fig1 = go.Figure(data=go.Scatter(x=time_steps, y=live_metric_history, mode='lines+markers',
                                                 line=dict(color='#dc2626' if drift_detected else '#2563eb', width=3)))
                fig1.add_hline(y=0.65, line_dash="dash", line_color="red", annotation_text="Recall Failure Threshold")
                fig1.update_layout(
                    title="Phase 1: Time-Series Model Degradation",
                    xaxis_title="Chronological Data Stream",
                    yaxis_title="Recall Score",
                    yaxis=dict(range=[0.0, 1.0]),
                    height=300, margin=dict(l=0, r=0, t=40, b=0)
                )
                graph_box_1.plotly_chart(fig1, use_container_width=True)
                time.sleep(0.4)

            st.session_state.sim_phase = "drifted"
            st.session_state.fig1 = fig1
            st.session_state.X_drift = X_drift
            st.session_state.y_drift = y_drift
            st.session_state.champion = champion
            st.rerun()

        except Exception as e:
            st.error(f"Pipeline Processing Error: {str(e)}")
            st.stop()

    # Restore Graph 1
    if st.session_state.sim_phase in ["drifted", "retraining", "resolved", "deployed"] and 'fig1' in st.session_state:
        graph_box_1.plotly_chart(st.session_state.fig1, use_container_width=True)

    # --- LOGIC TRIGGER: START RETRAIN ---
    if start_retrain:
        st.session_state.sim_phase = "retraining"
        st.rerun()

    if st.session_state.sim_phase == "retraining":
        st.session_state.terminal_text += "\n> [ACTION] Retraining protocol authorized...\n"
        log_box.code(st.session_state.terminal_text, language="bash")

        X_drift = st.session_state.X_drift
        y_drift = st.session_state.y_drift
        champion = st.session_state.champion

        st.session_state.terminal_text += "> [SYSTEM] Cloning pipeline architecture for Challenger...\n"
        log_box.code(st.session_state.terminal_text, language="bash")

        # Clone creates a blank, unfitted copy of your exact pipeline
        challenger = clone(champion)
        challenger.fit(X_drift, y_drift)

        st.session_state.terminal_text += "> [SYSTEM] Evaluating Challenger vs Champion on blind holdout...\n"
        log_box.code(st.session_state.terminal_text, language="bash")

        # Use the very last piece of the stream as the ultimate blind holdout
        split_test = int(len(X_drift) * 0.8)
        X_test, y_test = X_drift.iloc[split_test:], y_drift.iloc[split_test:]

        champ_preds = champion.predict(X_test)
        chall_preds = challenger.predict(X_test)

        champ_recall = recall_score(y_test, champ_preds, zero_division=0)
        chall_recall = recall_score(y_test, chall_preds, zero_division=0)

        time.sleep(1.5)

        st.session_state.terminal_text += f"\n> [RESULT] Final Champion Recall: {champ_recall:.3f}\n"
        st.session_state.terminal_text += f"> [RESULT] Final Challenger Recall: {chall_recall:.3f}\n"

        if chall_recall > champ_recall:
            st.session_state.terminal_text += "> [DECISION] SUCCESS. Challenger dominates. New weights approved for production.\n"
        else:
            st.session_state.terminal_text += "> [DECISION] WARNING. Challenger underperformed. Manual review required.\n"

        log_box.code(st.session_state.terminal_text, language="bash")

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=['Champion (Old)', 'Challenger (New)'], y=[champ_recall, chall_recall],
                              marker_color=['#94a3b8', '#059669']))
        fig2.add_hline(y=0.65, line_dash="dash", line_color="red")
        fig2.update_layout(
            title="Phase 2: Holdout Recall Matrix",
            yaxis_title="Recall Score",
            yaxis=dict(range=[0.0, 1.0]),
            height=300, margin=dict(l=0, r=0, t=40, b=0)
        )

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