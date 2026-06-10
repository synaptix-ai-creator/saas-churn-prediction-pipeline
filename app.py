import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
import os
import numpy as np
from datetime import datetime
from src.data_processor import DataProcessor

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="ChurnIQ | Enterprise Retention Command Center",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Enterprise SaaS Theme CSS Injection ---
st.markdown("""
    <style>

/* Import Typography */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Main App Background */
.stApp {
    background-color: #0F172A;
    color: #E2E8F0;
}

/* Main Container */
.block-container {
    max-width: 1500px;
    margin: auto;
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

/* Hide Default Streamlit Elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

[data-testid="stHeader"] {
    background: transparent;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid rgba(255,255,255,0.05);
}

/* Premium Card Styling */
.metric-card, .insight-card, .action-card, .chart-card {
    background-color: rgba(17, 24, 39, 0.85);
    backdrop-filter: blur(10px);
    border-radius: 8px;
    padding: 20px;
    border: 1px solid #1F2937;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    margin-bottom: 16px;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    min-height: 135px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

/* Hover Effect */
.metric-card:hover,
.insight-card:hover,
.action-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.35);
}

/* Metric Typography */
.metric-title {
    color: #94A3B8;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 12px;
}

.metric-value {
    color: #FFFFFF;
    font-size: 34px;
    font-weight: 700;
    line-height: 1.2;
}

/* Dashboard Header */
.dashboard-header {
    font-size: 38px;
    font-weight: 700;
    color: #FFFFFF;
    margin-bottom: 6px;
    letter-spacing: -0.5px;
}

.dashboard-subheader {
    color: #94A3B8;
    font-size: 15px;
    margin-bottom: 28px;
}

/* Section Headers */
.section-title {
    color: #F8FAFC;
    font-size: 20px;
    font-weight: 600;
    margin-bottom: 18px;
    margin-top: 10px;
}

/* Insight Cards */
.insight-card {
    border-left: 4px solid #3B82F6;
}

.insight-title {
    font-size: 15px;
    font-weight: 600;
    color: #FFFFFF;
    margin-bottom: 6px;
}

.insight-text {
    font-size: 14px;
    color: #CBD5E1;
    line-height: 1.5;
}

/* Action Cards */
.action-card {
    border-left: 4px solid #10B981;
}

.action-list {
    color: #E2E8F0;
    line-height: 1.8;
    font-size: 14px;
    margin-bottom: 0;
}

/* Table Styling */
[data-testid="stDataFrame"] {
    background-color: #111827;
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.05);
}

/* Risk Colors */
.status-high {
    color: #EF4444;
}

.status-medium {
    color: #F59E0B;
}

.status-low {
    color: #10B981;
}

/* Smooth Scroll */
html {
    scroll-behavior: smooth;
}

/* Premium Widget Styles */
div.stButton > button {
    background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.25) !important;
}

div.stButton > button:hover {
    transform: translateY(-1.5px) !important;
    box-shadow: 0 8px 20px rgba(59, 130, 246, 0.4) !important;
    background: linear-gradient(135deg, #60A5FA 0%, #2563EB 100%) !important;
}

div.stButton > button:active {
    transform: translateY(0.5px) !important;
}

[data-testid="stFileUploader"] {
    background-color: rgba(17, 24, 39, 0.5) !important;
    border: 1px dashed rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    padding: 15px !important;
}

/* Custom Scrollbars */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: #0F172A;
}
::-webkit-scrollbar-thumb {
    background: #1F2937;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #374151;
}

    </style>
""", unsafe_allow_html=True)

# --- 3. Model Loading ---
@st.cache_resource
def load_assets():
    model_path = 'models/lightgbm_churn_model.pkl'
    features_path = 'models/model_features.pkl'
    
    if not os.path.exists(model_path) or not os.path.exists(features_path):
        st.error("System Error: Production model files not found in /models directory.")
        st.stop()
        
    model = joblib.load(model_path)
    features = joblib.load(features_path)
    return model, features

model, required_features = load_assets()
processor = DataProcessor('models/model_features.pkl')

# --- 4. Sidebar Architecture (Stateful Sync) ---
with st.sidebar:
    st.markdown("<h2 style='color: #E2E8F0; margin-top:0;'>Account Assessment</h2>", unsafe_allow_html=True)
    
    # Multi-Format File Uploader
    st.markdown("### Portfolio Ingest")
    uploaded_file = st.file_uploader(
        "Upload CRM Export", 
        type=['csv', 'xlsx', 'xls'],
        help="Accepts .csv or Excel files exported from your billing system or CRM."
    )
    
    # State Management for selected customer sync
    upload_successful = False
    null_rate = 0.0

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                raw_df = pd.read_csv(uploaded_file)
            else:
                raw_df = pd.read_excel(uploaded_file)
            
            # Standardize columns using DataProcessor
            batch_df = processor.standardize_columns(raw_df)
            
            # Verify required columns exist
            missing_cols = [col for col in required_features + ['Customer ID'] if col not in batch_df.columns]
            if missing_cols:
                # Beautiful error card for schema alignment failure
                display_names = {
                    'Customer ID': 'Customer ID (or Account ID)',
                    'total_transactions': 'Billing Cycles (or Transactions)',
                    'total_revenue': 'Lifetime Value (LTV / Revenue)',
                    'currently_auto_renews': 'Auto-Renew Status (or Auto Renew)',
                    'has_cancelled_before': 'Prior Cancellations (or Cancelled Before)',
                    'total_active_days': 'Active Sessions (30-day) (or Active Sessions)',
                    'total_songs_skipped': 'Failed Sessions / Errors (or Feature Drops)',
                    'total_songs_completed': 'Core Feature Adoptions (or Features Completed)',
                    'total_listen_time_secs': 'Product Usage Duration (Mins) (or Usage Duration)'
                }
                missing_details = "".join(f"<li style='margin-bottom: 4px;'>{display_names.get(c, c)}</li>" for c in missing_cols)
                
                st.markdown(f"""
                <div style="background-color: rgba(239, 68, 68, 0.12); border-left: 4px solid #EF4444; padding: 16px; border-radius: 6px; margin-top: 10px; margin-bottom: 15px;">
                    <h5 style="color: #F8FAFC; margin-top: 0; margin-bottom: 6px; font-weight: 600;">⚠️ Schema Alignment Failure</h5>
                    <p style="color: #94A3B8; margin-bottom: 12px; font-size: 13px; line-height: 1.4;">
                        The uploaded file structure does not match the model's requirements. The following parameters are missing:
                    </p>
                    <ul style="color: #F8FAFC; font-size: 12.5px; margin-top: 0; margin-bottom: 0; padding-left: 20px;">
                        {missing_details}
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                # Build template df with SaaS terms
                template_data = {
                    'Customer ID': ['CUS-9910', 'CUS-9911', 'CUS-9912'],
                    'Billing Cycles': [48, 2, 15],
                    'LTV ($)': [2400.00, 40.00, 750.00],
                    'Auto-Renew Status': [1, 0, 1],
                    'Prior Cancellations': [0, 0, 1],
                    'Active Sessions (30-day)': [29, 1, 14],
                    'Failed Sessions / Errors': [12, 30, 55],
                    'Core Feature Adoptions': [450, 4, 120],
                    'Product Usage Duration (Mins)': [1800, 16, 480]
                }
                template_df = pd.DataFrame(template_data)
                template_csv = template_df.to_csv(index=False).encode('utf-8')
                
                st.download_button(
                    label="Download Standard SaaS CSV Template", 
                    data=template_csv, 
                    file_name="churn_saas_data_template.csv", 
                    mime="text/csv",
                    type="primary",
                    use_container_width=True
                )
                st.warning("Please align headers or use simulator data.")
            else:
                # Fill missing/NaN fields in batch_df
                fill_values = {
                    'total_transactions': 0, 'total_revenue': 0, 
                    'currently_auto_renews': 0, 'has_cancelled_before': 0,
                    'total_active_days': 0, 'total_songs_skipped': 0,
                    'total_songs_completed': 0, 'total_listen_time_secs': 0
                }
                batch_df = batch_df.fillna(value=fill_values)
                
                st.markdown("### Focus Account")
                selected_customer = st.selectbox("Search / Select ID:", batch_df['Customer ID'].tolist())
                
                cust_data = batch_df[batch_df['Customer ID'] == selected_customer].iloc[0]
                
                def_rev = float(cust_data['total_revenue'])
                def_txns = int(cust_data['total_transactions'])
                def_auto = int(cust_data['currently_auto_renews'])
                def_cancel = int(cust_data['has_cancelled_before'])
                def_days = int(cust_data['total_active_days'])
                def_listen = float(cust_data['total_listen_time_secs'])
                def_comp = int(cust_data['total_songs_completed'])
                def_skip = int(cust_data['total_songs_skipped'])
                
                data_source = f"CRM Export ({uploaded_file.name})"
                
                # Batch Predictions for Macro View
                predict_df = batch_df[required_features]
                raw_probs = model.predict_proba(predict_df)[:, 1] * 100
                batch_df['Risk Probability'] = [
                    processor.calibrate_probability(row.to_dict(), p)
                    for p, (_, row) in zip(raw_probs, batch_df.iterrows())
                ]
                
                total_analyzed = len(batch_df)
                high_risk_count = len(batch_df[batch_df['Risk Probability'] >= 70])
                revenue_at_risk = batch_df[batch_df['Risk Probability'] >= 70]['total_revenue'].sum()
                retention_opp = batch_df[(batch_df['Risk Probability'] >= 40) & (batch_df['Risk Probability'] < 70)]['total_revenue'].sum()
                
                # Segmentation Math
                healthy_pct = len(batch_df[batch_df['Risk Probability'] < 40]) / total_analyzed * 100
                at_risk_pct = len(batch_df[(batch_df['Risk Probability'] >= 40) & (batch_df['Risk Probability'] < 70)]) / total_analyzed * 100
                critical_pct = len(batch_df[batch_df['Risk Probability'] >= 70]) / total_analyzed * 100
                
                # Calculate Null Rate for Scorecard
                total_nulls = raw_df.isnull().sum().sum()
                total_cells = raw_df.size
                null_rate = (total_nulls / total_cells * 100) if total_cells > 0 else 0.0

                upload_successful = True

        except Exception as e:
            st.error(f"⚠️ Upload Failed: {str(e)}")
            
            # Build template with mock data matching required structure
            template_data = {
                'Customer ID': ['CUS-9910', 'CUS-9911', 'CUS-9912'],
                'Billing Cycles': [48, 2, 15],
                'LTV ($)': [2400.00, 40.00, 750.00],
                'Auto-Renew Status': [1, 0, 1],
                'Prior Cancellations': [0, 0, 1],
                'Active Sessions (30-day)': [29, 1, 14],
                'Failed Sessions / Errors': [12, 30, 55],
                'Core Feature Adoptions': [450, 4, 120],
                'Product Usage Duration (Mins)': [1800, 16, 480]
            }
            template_df = pd.DataFrame(template_data)
            template_csv = template_df.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="Download Standard SaaS CSV Template", 
                data=template_csv, 
                file_name="churn_saas_data_template.csv", 
                mime="text/csv",
                type="primary",
                use_container_width=True
            )
            st.warning("Loading default simulation data...")

    if not upload_successful:
        # Default State for Simulation
        selected_customer = "CUS-8924"
        def_rev = 1250.0
        def_txns = 24
        def_auto = 1
        def_cancel = 0
        def_days = 8
        def_listen = 15000.0  # 250 minutes
        def_comp = 45
        def_skip = 60
        data_source = "AI Platform Simulation"
        
        total_analyzed = 12482
        high_risk_count = 1204
        revenue_at_risk = 248500.00
        retention_opp = 91200.00
        healthy_pct = 62.0
        at_risk_pct = 24.0
        critical_pct = 14.0
        null_rate = 0.0

    st.markdown("---")
    st.markdown("### Account Simulator")
    
    customer_id = st.text_input("Account ID", value=selected_customer)
    
    st.markdown("#### Subscription History")
    rev = st.number_input("Customer LTV ($)", min_value=0.0, value=def_rev)
    txns = st.number_input("Billing Cycles", min_value=0, value=def_txns)
    auto_renew = st.selectbox("Auto-Renew Status", options=[1, 0], index=0 if def_auto == 1 else 1, format_func=lambda x: "Active" if x==1 else "Disabled")
    cancelled_before = st.selectbox("Prior Cancellations", options=[0, 1], index=0 if def_cancel == 0 else 1, format_func=lambda x: "None" if x==0 else "Yes")

    st.markdown("#### Product Telemetry (30-Day)")
    active_days = st.slider("Active Sessions (30-day)", 0, 30, def_days)
    listen_time_mins = st.number_input("Product Usage Duration (Mins)", min_value=0.0, value=def_listen / 60.0)
    listen_time = listen_time_mins * 60.0
    songs_completed = st.number_input("Core Feature Adoptions", min_value=0, value=def_comp)
    songs_skipped = st.number_input("Failed Sessions / Errors", min_value=0, value=def_skip)
    
    run_assessment = st.button("Calculate Risk Profile", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.markdown("""
    <div style="background-color: rgba(17, 24, 39, 0.85); border: 1px solid #1F2937; border-radius: 8px; padding: 15px; margin-top: 10px;">
        <h4 style="color: #FFFFFF; font-size: 14px; font-weight: 600; margin-top: 0; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.5px; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 8px;">🤖 Model Authority</h4>
        <div style="display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 12.5px;">
            <span style="color: #94A3B8;">Algorithm</span>
            <span style="color: #FFFFFF; font-weight: 600;">LightGBM Production</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 12.5px;">
            <span style="color: #94A3B8;">ROC-AUC Score</span>
            <span style="color: #3B82F6; font-weight: 600; font-family: monospace;">0.9224</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 12.5px;">
            <span style="color: #94A3B8;">Model Accuracy</span>
            <span style="color: #10B981; font-weight: 600; font-family: monospace;">88.21%</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 12.5px;">
            <span style="color: #94A3B8;">Catch Rate (Recall)</span>
            <span style="color: #FFFFFF; font-weight: 600; font-family: monospace;">63.73%</span>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 12.5px;">
            <span style="color: #94A3B8;">Test Set Size</span>
            <span style="color: #FFFFFF; font-weight: 600; font-family: monospace;">194,192 accounts</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# --- 5. Main Content Tabs ---
tab1, tab2 = st.tabs(["📊 Command Center", "🔌 Developer API Docs"])

with tab1:
    current_date = datetime.now().strftime("%B %d, %Y")
    
    st.markdown("<div class='dashboard-header'>Retention Command Center</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='dashboard-subheader'>Last Updated: {current_date} | Source: {data_source}</div>", unsafe_allow_html=True)
    
    # Calculate single account indicators
    engagement_ratio = (songs_completed / max(1, (songs_completed + songs_skipped)) * 100)
    
    input_dict = {
        'total_transactions': txns,
        'total_revenue': rev,
        'currently_auto_renews': auto_renew,
        'has_cancelled_before': cancelled_before,
        'total_active_days': active_days,
        'total_songs_skipped': songs_skipped,
        'total_songs_completed': songs_completed,
        'total_listen_time_secs': listen_time
    }
    input_df = pd.DataFrame([input_dict])[required_features]
    
    # Single prediction
    churn_probability = model.predict_proba(input_df)[0][1] * 100
    
    # Calibrate probability using business rules
    churn_probability = processor.calibrate_probability(input_dict, churn_probability)
    is_churner = churn_probability >= 50.0
    
    if churn_probability >= 70:
        risk_category = "High Flight Risk"
        theme_hex = "#EF4444" 
        status_class = "status-high"
    elif churn_probability >= 40:
        risk_category = "At Risk"
        theme_hex = "#F59E0B"
        status_class = "status-medium"
    else:
        risk_category = "Healthy"
        theme_hex = "#10B981"
        status_class = "status-low"
        
    # --- Data Diagnostic Report ---
    if upload_successful:
        avg_ltv = batch_df['total_revenue'].mean()
        max_ltv = batch_df['total_revenue'].max()
        st.markdown(f"""
        <div style="background-color: rgba(16, 185, 129, 0.06); border-left: 4px solid #10B981; padding: 18px; border-radius: 8px; margin-bottom: 25px; border: 1px solid rgba(16, 185, 129, 0.12);">
            <h5 style="color: #10B981; margin-top: 0; margin-bottom: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; font-size: 13px;">⚙️ CRM Ingest Diagnostic Scorecard</h5>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; text-align: center;">
                <div style="border-right: 1px solid rgba(255,255,255,0.08);">
                    <div style="font-size: 10.5px; color: #94A3B8; text-transform: uppercase; font-weight: 500; margin-bottom: 4px;">Schema Alignment</div>
                    <div style="font-size: 18px; font-weight: 700; color: #10B981;">100% Matched</div>
                </div>
                <div style="border-right: 1px solid rgba(255,255,255,0.08);">
                    <div style="font-size: 10.5px; color: #94A3B8; text-transform: uppercase; font-weight: 500; margin-bottom: 4px;">Null/Missing Rate</div>
                    <div style="font-size: 18px; font-weight: 700; color: #10B981;">{null_rate:.1f}%</div>
                </div>
                <div style="border-right: 1px solid rgba(255,255,255,0.08);">
                    <div style="font-size: 10.5px; color: #94A3B8; text-transform: uppercase; font-weight: 500; margin-bottom: 4px;">Active Rows Loaded</div>
                    <div style="font-size: 18px; font-weight: 700; color: #FFFFFF;">{len(batch_df):,}</div>
                </div>
                <div>
                    <div style="font-size: 10.5px; color: #94A3B8; text-transform: uppercase; font-weight: 500; margin-bottom: 4px;">Average / Max LTV</div>
                    <div style="font-size: 18px; font-weight: 700; color: #3B82F6;">${avg_ltv:,.0f} / ${max_ltv:,.0f}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # --- Portfolio Calculations ---
    if upload_successful:
        batch_probabilities = batch_df['Risk Probability'].values
        batch_predictions = np.where(batch_probabilities >= 50.0, 1, 0)
        table_df = pd.DataFrame({
            'Account ID': batch_df['Customer ID'],
            'Health Score': batch_probabilities,
            'Status': np.where(batch_probabilities >= 70, 'High Risk', 
                      np.where(batch_probabilities >= 40, 'At Risk', 'Healthy')),
            'LTV': batch_df['total_revenue'].apply(lambda x: f"${x:,.2f}"),
            'Velocity Drop-off': batch_df['total_active_days'].apply(lambda x: f"{30-x} days inactive"),
            'CS Action': np.where(batch_predictions == 1, 'Intervene', 'Monitor')
        })
    else:
        mock_data = {
            'Account ID': [customer_id],
            'Health Score': [churn_probability],
            'Status': [risk_category],
            'LTV': [f"${rev:,.2f}"],
            'Velocity Drop-off': [f"{30 - active_days} days inactive"],
            'CS Action': ['Intervene' if is_churner else 'Monitor']
        }
        table_df = pd.DataFrame(mock_data)

    # --- Portfolio KPI Row ---
    st.markdown("<div class='section-title'>Portfolio Health Metrics</div>", unsafe_allow_html=True)
    portfolio1, portfolio2, portfolio3, portfolio4 = st.columns(4)
    
    total_customers = len(table_df) if uploaded_file is not None else 1
    high_risk_accounts = len(table_df[table_df['Status'].isin(['High Risk', 'High Flight Risk'])]) if uploaded_file is not None else int(churn_probability >= 70)
    rev_at_risk_calculated = rev * (churn_probability / 100) if not upload_successful else batch_df[batch_df['Risk Probability'] >= 70]['total_revenue'].sum()
    opp_calculated = rev_at_risk_calculated * 0.35
    
    with portfolio1:
        st.markdown(f"<div class='metric-card'><div class='metric-title'>Accounts Analyzed</div><div class='metric-value'>{total_customers:,}</div></div>", unsafe_allow_html=True)
    with portfolio2:
        st.markdown(f"<div class='metric-card'><div class='metric-title'>High Risk Accounts</div><div class='metric-value status-high'>{high_risk_accounts:,}</div></div>", unsafe_allow_html=True)
    with portfolio3:
        st.markdown(f"<div class='metric-card'><div class='metric-title'>Revenue At Risk</div><div class='metric-value'>${rev_at_risk_calculated:,.2f}</div></div>", unsafe_allow_html=True)
    with portfolio4:
        st.markdown(f"<div class='metric-card'><div class='metric-title'>Retention Opportunity</div><div class='metric-value status-low'>${opp_calculated:,.2f}</div></div>", unsafe_allow_html=True)

    # --- Individual KPI Row ---
    st.markdown("<div class='section-title'>Focus Account Deep-Dive Metrics</div>", unsafe_allow_html=True)
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f"<div class='metric-card'><div class='metric-title'>Lifetime Value</div><div class='metric-value'>${rev:,.2f}</div></div>", unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"<div class='metric-card'><div class='metric-title'>Active Sessions</div><div class='metric-value'>{active_days} Days</div></div>", unsafe_allow_html=True)
    with kpi3:
        st.markdown(f"<div class='metric-card'><div class='metric-title'>Engagement Rate</div><div class='metric-value'>{engagement_ratio:.1f}%</div></div>", unsafe_allow_html=True)
    with kpi4:
        st.markdown(f"<div class='metric-card' style='border-left: 4px solid {theme_hex};'><div class='metric-title'>Account Health</div><div class='metric-value {status_class}'>{risk_category}</div></div>", unsafe_allow_html=True)

    # --- Visualizations & Explainable AI (XAI) ---
    col_chart, col_trend, col_insights = st.columns([1, 1.2, 1])
    
    with col_chart:
        st.markdown("<div class='section-title'>Explainable AI Risk Factors</div>", unsafe_allow_html=True)
        
        # Zero-dependency, lightweight calculation of local feature contributions
        baselines = {
            'total_transactions': 15,
            'total_revenue': 500.0,
            'currently_auto_renews': 1,
            'has_cancelled_before': 0,
            'total_active_days': 15,
            'total_songs_skipped': 20,
            'total_songs_completed': 100,
            'total_listen_time_secs': 10000.0
        }
        importances = {
            'total_transactions': 857,
            'total_revenue': 1211,
            'currently_auto_renews': 97,
            'has_cancelled_before': 154,
            'total_active_days': 264,
            'total_songs_skipped': 170,
            'total_songs_completed': 148,
            'total_listen_time_secs': 99
        }
        directions = {
            'total_transactions': 1,
            'total_revenue': 1,
            'currently_auto_renews': 1,
            'has_cancelled_before': -1,
            'total_active_days': 1,
            'total_songs_skipped': -1,
            'total_songs_completed': 1,
            'total_listen_time_secs': 1
        }
        scales = {
            'total_transactions': 10,
            'total_revenue': 500.0,
            'currently_auto_renews': 1,
            'has_cancelled_before': 1,
            'total_active_days': 10,
            'total_songs_skipped': 30,
            'total_songs_completed': 150,
            'total_listen_time_secs': 12000.0
        }
        
        contrib_labels = {
            'total_transactions': 'Billing Cycles',
            'total_revenue': 'Lifetime Value (LTV)',
            'currently_auto_renews': 'Auto-Renew Status',
            'has_cancelled_before': 'Prior Cancellations',
            'total_active_days': 'Active Sessions (30-day)',
            'total_songs_skipped': 'Failed Sessions / Errors',
            'total_songs_completed': 'Core Feature Adoptions',
            'total_listen_time_secs': 'Product Usage Duration'
        }
        
        xai_data = []
        for feat in required_features:
            val = input_dict[feat]
            base = baselines[feat]
            dir_val = directions[feat]
            scale = scales[feat]
            imp = importances[feat]
            
            diff = (val - base) / scale * dir_val
            contrib = -diff * imp
            # Adjust label logic for auto-renew
            lbl = contrib_labels[feat]
            if feat == 'currently_auto_renews':
                lbl = 'Auto-Renew Active' if val == 1 else 'Auto-Renew Disabled'
                
            xai_data.append({
                'Feature': lbl,
                'Contribution': contrib,
                'Impact': 'Increases Risk' if contrib > 0 else 'Reduces Risk'
            })
            
        xai_df = pd.DataFrame(xai_data).sort_values(by='Contribution', ascending=True)
        
        fig_xai = px.bar(
            xai_df,
            x='Contribution',
            y='Feature',
            orientation='h',
            color='Impact',
            color_discrete_map={'Increases Risk': '#EF4444', 'Reduces Risk': '#10B981'},
            labels={'Contribution': 'Risk Factor Impact Score', 'Feature': ''}
        )
        fig_xai.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#E2E8F0'),
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(tickfont=dict(size=11)),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_xai, use_container_width=True)

    with col_trend:
        st.markdown("<div class='section-title'>Engagement Velocity</div>", unsafe_allow_html=True)
        
        np.random.seed(42)
        dates = pd.date_range(end=pd.Timestamp.now(), periods=30)
        base_trend = np.linspace(10, 2 if is_churner else 15, 30)
        noise = np.random.normal(0, 1.5, 30)
        usage = np.maximum(0, base_trend + noise)
        
        trend_df = pd.DataFrame({'Date': dates, 'Sessions': usage})
        fig_trend = px.area(
            trend_df,
            x='Date',
            y='Sessions'
        )
        fig_trend.update_traces(
            line_color='#3B82F6',
            line_width=4,
            fill='tozeroy'
        )
        fig_trend.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=10, b=0), 
            plot_bgcolor='rgba(0,0,0,0)', 
            paper_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified',
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(255,255,255,0.05)',
                title=None,
                tickfont=dict(color='#94A3B8')
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(255,255,255,0.05)',
                title=None,
                tickfont=dict(color='#94A3B8')
            )
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with col_insights:
        st.markdown("<div class='section-title'>Key Risk Factors</div>", unsafe_allow_html=True)
        
        if auto_renew == 0:
            st.markdown("<div class='insight-card' style='border-color: #EF4444;'><div class='insight-title'>Auto-Renew Disabled</div><div class='insight-text'>Immediate revenue contraction risk.</div></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='insight-card' style='border-color: #10B981;'><div class='insight-title'>Auto-Renew Active</div><div class='insight-text'>Subscription intent remains stable.</div></div>", unsafe_allow_html=True)
            
        st.markdown("<div class='section-title'>Success Team Playbook</div>", unsafe_allow_html=True)
        
        if is_churner:
            st.markdown("""
            <div class='action-card' style='border-color: #F59E0B;'>
                <ul class='action-list'>
                    <li>Authorize 20% Rescue Discount</li>
                    <li>Enroll in re-onboarding email sequence</li>
                    <li>Flag for executive CS outreach</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='action-card'>
                <ul class='action-list'>
                    <li>Maintain standard communication lifecycle</li>
                    <li>Identify upsell potential based on LTV</li>
                    <li>Request NPS / Product Feedback</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    # --- Business ROI Calculator ---
    st.markdown("---")
    st.markdown("<div class='section-title'>💰 Business ROI & Financial Rescue Impact</div>", unsafe_allow_html=True)
    roi_col1, roi_col2, roi_col3, roi_col4 = st.columns(4)
    
    with roi_col1:
        arpu = st.number_input("Average Revenue Per User (ARPU) / Month ($)", min_value=1.0, value=100.0, step=5.0)
    with roi_col2:
        rescue_rate = st.slider("Target Rescue Success Rate (%)", min_value=0, max_value=100, value=15)
    with roi_col3:
        acq_cost = st.number_input("Customer Acquisition Cost (CAC) ($)", min_value=1.0, value=250.0, step=10.0)
    with roi_col4:
        campaign_cost = st.number_input("Outreach Cost / Customer ($)", min_value=0.0, value=15.0, step=1.0)

    # Calculations
    monthly_at_risk = high_risk_count * arpu
    rescued_customers = int(high_risk_count * (rescue_rate / 100))
    monthly_saved_revenue = rescued_customers * arpu
    annual_saved_revenue = monthly_saved_revenue * 12
    cac_savings = rescued_customers * acq_cost
    
    # Outreach cost
    total_campaign_cost = high_risk_count * campaign_cost
    net_savings = annual_saved_revenue + cac_savings - total_campaign_cost
    campaign_roi = (net_savings / total_campaign_cost * 100) if total_campaign_cost > 0 else 100.0

    val_col1, val_col2, val_col3, val_col4 = st.columns(4)
    with val_col1:
        st.markdown(f"""
        <div class='metric-card' style='border-left: 4px solid #EF4444;'>
            <div class='metric-title'>Monthly MRR At Risk</div>
            <div class='metric-value'>${monthly_at_risk:,.2f}</div>
            <div style='color: #94A3B8; font-size: 11px; margin-top: 4px;'>Based on {high_risk_count} high-risk accounts</div>
        </div>
        """, unsafe_allow_html=True)
    with val_col2:
        st.markdown(f"""
        <div class='metric-card' style='border-left: 4px solid #10B981;'>
            <div class='metric-title'>Annual ARR Recovered</div>
            <div class='metric-value status-low'>${annual_saved_revenue:,.2f}</div>
            <div style='color: #94A3B8; font-size: 11px; margin-top: 4px;'>Rescuing {rescued_customers} customers</div>
        </div>
        """, unsafe_allow_html=True)
    with val_col3:
        st.markdown(f"""
        <div class='metric-card' style='border-left: 4px solid #3B82F6;'>
            <div class='metric-title'>CAC Reacquisition Savings</div>
            <div class='metric-value'>${cac_savings:,.2f}</div>
            <div style='color: #94A3B8; font-size: 11px; margin-top: 4px;'>Marketing pipeline costs saved</div>
        </div>
        """, unsafe_allow_html=True)
    with val_col4:
        roi_color = "#10B981" if campaign_roi >= 0 else "#EF4444"
        st.markdown(f"""
        <div class='metric-card' style='border-left: 4px solid {roi_color};'>
            <div class='metric-title'>Campaign Net Benefit</div>
            <div class='metric-value' style='color: {roi_color};'>${net_savings:,.2f}</div>
            <div style='color: #94A3B8; font-size: 11px; margin-top: 4px;'>ROI: {campaign_roi:,.1f}% (Cost: ${total_campaign_cost:,.0f})</div>
        </div>
        """, unsafe_allow_html=True)

    # --- Action Center & Webhook Simulator ---
    st.markdown("---")
    st.markdown("<div class='section-title'>⚡ CRM Integration & Webhook Simulator</div>", unsafe_allow_html=True)
    
    act_col1, act_col2 = st.columns([1, 1.2])
    with act_col1:
        st.markdown("""
        Select a target destination platform and simulate triggering an automated CRM/outreach alert. In a production pipeline, this fires a JSON webhook payload to trigger immediate customer rescue playbooks.
        """)
        webhook_target = st.selectbox("Select Target CRM / Platform:", options=["Slack Alerts", "HubSpot Contacts", "Salesforce Tasks"])
        trigger_webhook = st.button("🔌 Trigger Customer Rescue Webhook Event", type="secondary", use_container_width=True)
        
    with act_col2:
        if trigger_webhook:
            import json
            if webhook_target == "Slack Alerts":
                webhook_payload = {
                    "text": f"🚨 *High Churn Risk Alert* for customer *{customer_id}*",
                    "attachments": [
                        {
                            "color": "#EF4444" if churn_probability >= 70 else "#F59E0B",
                            "blocks": [
                                {
                                    "type": "header",
                                    "text": {
                                        "type": "plain_text",
                                        "text": "🚨 ChurnIQ Retention Alert",
                                        "emoji": True
                                    }
                                },
                                {
                                    "type": "section",
                                    "text": {
                                        "type": "mrkdwn",
                                        "text": f"*Customer ID:* `{customer_id}`\n*Lifetime Value (LTV):* `${rev:,.2f}`\n*Risk Probability:* *{churn_probability:.2f}%* ({risk_category})"
                                    }
                                },
                                {
                                    "type": "section",
                                    "text": {
                                        "type": "mrkdwn",
                                        "text": f"*Playbook Actions:*\n• " + ("Authorize 20% Rescue Discount" if is_churner else "Enroll in CSM engagement check-in") + "\n• Deploy re-onboarding outreach campaign"
                                    }
                                }
                            ]
                        }
                    ]
                }
            elif webhook_target == "HubSpot Contacts":
                webhook_payload = {
                    "properties": {
                        "email": f"contact_{customer_id.lower()}@saasclient.com",
                        "churniq_risk_score": f"{churn_probability:.2f}",
                        "churniq_risk_category": risk_category,
                        "churniq_last_updated": datetime.now().isoformat(),
                        "lifecycle_stage": "customer_at_risk" if is_churner else "customer",
                        "customer_success_owner": "HubSpot CS Automated Agent",
                        "notes_last_contacted": f"Automated Alert: Risk ({churn_probability:.1f}%) detected due to activity deceleration."
                    }
                }
            else: # Salesforce Tasks
                webhook_payload = {
                    "attributes": {
                        "type": "Account",
                        "referenceId": f"ref_{customer_id}"
                    },
                    "Id": f"sf_acc_{customer_id.lower()}",
                    "ChurnIQ_Risk_Probability__c": round(churn_probability, 2),
                    "ChurnIQ_Risk_Category__c": risk_category,
                    "LTV__c": rev,
                    "Customer_Success_Action__c": "Schedule High Risk CS Task",
                    "Description": f"Auto-alert from ChurnIQ: Risk score {churn_probability:.1f}% is critical. Please trigger outreach."
                }
                
            st.success(f"✅ Webhook triggered successfully to {webhook_target}!")
            st.code(json.dumps(webhook_payload, indent=2), language="json")

    # --- Customer Health Distribution ---
    st.markdown("---")
    dist_col1, dist_col2 = st.columns([1.2, 1])
    with dist_col1:
        st.markdown("<div class='section-title'>Customer Health Distribution</div>", unsafe_allow_html=True)
        if uploaded_file is not None:
            health_counts = table_df['Status'].value_counts()
            donut_fig = px.pie(
                values=health_counts.values,
                names=health_counts.index,
                hole=0.7,
                color=health_counts.index,
                color_discrete_map={'Healthy': '#10B981', 'At Risk': '#F59E0B', 'High Risk': '#EF4444'}
            )
            donut_fig.update_layout(
                height=320,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                legend=dict(font=dict(color='white'))
            )
            st.plotly_chart(donut_fig, use_container_width=True)
        else:
            st.info("Upload a CRM Export file to view the portfolio distribution charts.")

    with dist_col2:
        st.markdown("<div class='section-title'>Portfolio Health Summary</div>", unsafe_allow_html=True)
        if uploaded_file is not None:
            avg_risk = batch_df['Risk Probability'].mean()
            vulnerable_idx = batch_df['Risk Probability'].idxmax()
            vulnerable_cust = batch_df.loc[vulnerable_idx]
            healthy_idx = batch_df['Risk Probability'].idxmin()
            healthy_cust = batch_df.loc[healthy_idx]
            
            st.markdown(f"""
            <div class='metric-card' style='min-height: 90px; padding: 15px; margin-bottom: 10px;'>
                <div class='metric-title' style='margin-bottom: 4px;'>Average Portfolio Risk</div>
                <div style='font-size: 24px; font-weight: bold; color: {"#EF4444" if avg_risk >= 50 else "#10B981"};'>{avg_risk:.1f}%</div>
            </div>
            <div class='metric-card' style='min-height: 90px; padding: 15px; margin-bottom: 10px; border-left: 4px solid #EF4444;'>
                <div class='metric-title' style='margin-bottom: 4px;'>Most Vulnerable Account</div>
                <div style='font-size: 15px; font-weight: bold; color: #FFFFFF;'>{vulnerable_cust['Customer ID']}</div>
                <div style='font-size: 11px; color: #94A3B8;'>Risk Score: {vulnerable_cust['Risk Probability']:.1f}% | LTV: ${float(vulnerable_cust['total_revenue']):,.2f}</div>
            </div>
            <div class='metric-card' style='min-height: 90px; padding: 15px; border-left: 4px solid #10B981;'>
                <div class='metric-title' style='margin-bottom: 4px;'>Model-Identified Champion Account</div>
                <div style='font-size: 15px; font-weight: bold; color: #FFFFFF;'>{healthy_cust['Customer ID']}</div>
                <div style='font-size: 11px; color: #94A3B8;'>Risk Score: {healthy_cust['Risk Probability']:.1f}% | LTV: ${float(healthy_cust['total_revenue']):,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Upload a CRM Export file to view the portfolio diagnostics summary.")

    # --- 8. Bottom Row: Batch Portfolio Analysis ---
    st.markdown("<div class='section-title'>Portfolio Risk Roster</div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="
    background: #111827;
    padding: 14px 18px;
    border-radius: 12px;
    margin-bottom: 14px;
    border: 1px solid rgba(255,255,255,0.05);
    display:flex;
    justify-content:space-between;
    align-items:center;
    ">
    <div style="color:#E2E8F0;font-weight:600;">
    Showing {len(table_df)} customer accounts
    </div>
    <div style="color:#94A3B8;">
    AI Portfolio Monitoring Active
    </div>
    </div>
    """, unsafe_allow_html=True)

    def format_risk(val):
        return f"{val:.1f}% Risk"

    def highlight_status(val):
        if val == 'High Risk' or val == 'High Flight Risk':
            return 'color: #EF4444; font-weight: bold;'
        elif val == 'At Risk':
            return 'color: #F59E0B; font-weight: bold;'
        elif val == 'Healthy':
            return 'color: #10B981; font-weight: bold;'
        return ''

    styled_table = table_df.style.format({'Health Score': format_risk}).map(highlight_status, subset=['Status'])

    st.dataframe(
        styled_table,
        use_container_width=True,
        hide_index=True,
        height=min((len(table_df) + 1) * 35 + 3, 500)
    )

    csv = table_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Export Full Roster to CSV",
        data=csv,
        file_name='portfolio_risk_report.csv',
        mime='text/csv',
    )

with tab2:
    st.markdown("## 🔌 Developer Integration API Documentation")
    st.markdown("""
    To integrate **ChurnIQ** predictions directly into your production CRM, billing portal, or internal database pipelines, you can connect directly to our REST API service.
    """)
    
    st.markdown("### 1. Service Health Check")
    st.markdown("Verify that the prediction service is online and running the correct LightGBM model binary.")
    st.code("""
curl -X GET http://localhost:8000/health
    """, language="bash")
    st.markdown("**Expected Response:**")
    st.code("""
{
  "status": "healthy",
  "model": "LightGBM Production"
}
    """, language="json")
    
    st.markdown("---")
    st.markdown("### 2. Request Single Account Risk Profile")
    st.markdown("Send active telemetry for a single customer account to get real-time prediction and risk classification. You can use standard SaaS vocabulary fields.")
    st.code("""
curl -X POST http://localhost:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "Customer ID": "CUS-8924",
    "Billing Cycles": 24,
    "LTV ($)": 1250.0,
    "Auto-Renew Status": 1,
    "Prior Cancellations": 0,
    "Active Sessions (30-day)": 8,
    "Failed Sessions / Errors": 60,
    "Core Feature Adoptions": 45,
    "Product Usage Duration (Mins)": 250.0
  }'
    """, language="bash")
    st.markdown("**Expected Response:**")
    st.code("""
{
  "customer_id": "CUS-8924",
  "churn_probability": 99.37,
  "is_churner": true,
  "risk_category": "High Flight Risk"
}
    """, language="json")
    
    st.markdown("---")
    st.markdown("### 3. Production Python Integration Snippet")
    st.markdown("Integrate this script directly into your HubSpot automated tasks or Stripe billing listener.")
    st.code("""
import requests

url = "http://localhost:8000/predict"
payload = {
    "Customer ID": "CUS-8924",
    "Billing Cycles": 24,
    "LTV ($)": 1250.0,
    "Auto-Renew Status": 1,
    "Prior Cancellations": 0,
    "Active Sessions (30-day)": 8,
    "Failed Sessions / Errors": 60,
    "Core Feature Adoptions": 45,
    "Product Usage Duration (Mins)": 250.0
}

response = requests.post(url, json=payload)
data = response.json()

if data["churn_probability"] >= 70:
    print(f"Alert: Account {data['customer_id']} is at critical risk ({data['churn_probability']}%). Launching retention play.")
else:
    print(f"Account {data['customer_id']} is healthy. Risk: {data['churn_probability']}%")
    """, language="python")