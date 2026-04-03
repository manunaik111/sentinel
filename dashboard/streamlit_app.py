# dashboard/streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(
    page_title="Sentinel — Churn Analytics",
    page_icon="⬛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Premium B&W CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #000000 !important;
    color: #ffffff !important;
    font-family: 'DM Mono', monospace !important;
}

[data-testid="stAppViewContainer"] {
    background: #000000 !important;
}

[data-testid="stMain"] {
    background: #000000 !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0a0a0a !important;
    border-right: 1px solid #1a1a1a !important;
}

[data-testid="stSidebar"] * {
    color: #ffffff !important;
    font-family: 'DM Mono', monospace !important;
}

/* ── Typography ── */
h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
    color: #ffffff !important;
    letter-spacing: -0.02em;
}

/* ── Animated Title ── */
.sentinel-title {
    font-family: 'DM Serif Display', serif !important;
    font-size: 3.2rem;
    color: #ffffff;
    letter-spacing: -0.03em;
    line-height: 1;
    animation: fadeSlideDown 0.8s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    opacity: 0;
}

.sentinel-subtitle {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #555555;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-top: 0.4rem;
    animation: fadeSlideDown 0.8s 0.15s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    opacity: 0;
}

/* ── Divider Line ── */
.divider-line {
    width: 100%;
    height: 1px;
    background: linear-gradient(90deg, #ffffff 0%, #333333 60%, transparent 100%);
    margin: 1.5rem 0;
    animation: expandWidth 1s 0.3s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    transform-origin: left;
    transform: scaleX(0);
}

/* ── KPI Cards ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    background: #1a1a1a;
    border: 1px solid #1a1a1a;
    margin: 1.5rem 0;
    animation: fadeIn 0.8s 0.4s forwards;
    opacity: 0;
}

.kpi-card {
    background: #000000;
    padding: 1.8rem 1.5rem;
    position: relative;
    overflow: hidden;
    transition: background 0.3s ease;
}

.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 2px;
    background: #ffffff;
    transform: scaleX(0);
    transform-origin: left;
    transition: transform 0.4s cubic-bezier(0.16, 1, 0.3, 1);
}

.kpi-card:hover::before { transform: scaleX(1); }
.kpi-card:hover { background: #080808; }

.kpi-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #555555;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}

.kpi-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    color: #ffffff;
    line-height: 1;
    letter-spacing: -0.03em;
}

.kpi-delta {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    margin-top: 0.4rem;
}

.kpi-delta.negative { color: #888888; }
.kpi-delta.positive { color: #ffffff; }

/* ── Section Header ── */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 0.65rem;
    color: #444444;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.8rem;
}

.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #1a1a1a;
}

/* ── Predict Page ── */
.predict-header {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem;
    color: #ffffff;
    letter-spacing: -0.02em;
    animation: fadeSlideDown 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}

.input-section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #444444;
    border-bottom: 1px solid #1a1a1a;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}

/* ── Result Card ── */
.result-card {
    border: 1px solid #1f1f1f;
    padding: 2.5rem;
    position: relative;
    overflow: hidden;
    animation: fadeIn 0.5s forwards;
}

.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: #ffffff;
}

.result-probability {
    font-family: 'Syne', sans-serif;
    font-size: 4rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    line-height: 1;
}

.result-probability.high { color: #ffffff; }
.result-probability.medium { color: #aaaaaa; }
.result-probability.low { color: #666666; }

.result-risk-badge {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    padding: 0.3rem 0.8rem;
    margin-top: 0.8rem;
    border: 1px solid;
}

.result-risk-badge.high {
    border-color: #ffffff;
    color: #ffffff;
    background: rgba(255,255,255,0.05);
}

.result-risk-badge.medium {
    border-color: #888888;
    color: #888888;
    background: rgba(255,255,255,0.02);
}

.result-risk-badge.low {
    border-color: #444444;
    color: #444444;
}

/* ── Sidebar styling ── */
.sidebar-logo {
    font-family: 'DM Serif Display', serif;
    font-size: 1.6rem;
    color: #ffffff;
    letter-spacing: -0.02em;
    margin-bottom: 0.2rem;
}

.sidebar-tagline {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    color: #444444;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}

/* ── Streamlit overrides ── */
[data-testid="stMetric"] {
    background: transparent !important;
}

div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stNumberInput"] label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.68rem !important;
    color: #666666 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}

div[data-testid="stSelectbox"] > div > div,
div[data-testid="stNumberInput"] input {
    background: #080808 !important;
    border: 1px solid #1f1f1f !important;
    color: #ffffff !important;
    border-radius: 0 !important;
    font-family: 'DM Mono', monospace !important;
}

div[data-testid="stSelectbox"] > div > div:focus-within,
div[data-testid="stNumberInput"] input:focus {
    border-color: #ffffff !important;
    box-shadow: none !important;
}

/* Slider */
div[data-testid="stSlider"] > div > div > div {
    background: #333333 !important;
}

div[data-testid="stSlider"] > div > div > div > div {
    background: #ffffff !important;
}

/* Button */
div[data-testid="stButton"] > button {
    background: #ffffff !important;
    color: #000000 !important;
    border: none !important;
    border-radius: 0 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    padding: 0.8rem 2rem !important;
    transition: all 0.2s ease !important;
}

div[data-testid="stButton"] > button:hover {
    background: #dddddd !important;
    transform: translateY(-1px) !important;
}

/* Radio */
div[data-testid="stRadio"] label {
    font-family: 'DM Mono', monospace !important;
    color: #888888 !important;
    font-size: 0.75rem !important;
}

div[data-testid="stRadio"] label:has(input:checked) {
    color: #ffffff !important;
}

/* Divider */
hr {
    border-color: #1a1a1a !important;
}

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* ── Keyframes ── */
@keyframes fadeSlideDown {
    from { opacity: 0; transform: translateY(-16px); }
    to   { opacity: 1; transform: translateY(0); }
}

@keyframes expandWidth {
    from { transform: scaleX(0); }
    to   { transform: scaleX(1); }
}

@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}

@keyframes pulseGlow {
    0%, 100% { box-shadow: 0 0 0 0 rgba(255,255,255,0); }
    50%       { box-shadow: 0 0 20px 2px rgba(255,255,255,0.06); }
}

.result-card { animation: fadeIn 0.5s forwards, pulseGlow 3s 0.5s infinite; }
</style>
""", unsafe_allow_html=True)

# ── Plotly B&W Theme ──────────────────────────────────────────────────────────
BW_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='DM Mono, monospace', color='#666666', size=11),
    title_font=dict(family='Syne, sans-serif', color='#ffffff', size=14),
    xaxis=dict(
        gridcolor='#111111', linecolor='#1f1f1f', tickcolor='#333333',
        title_font=dict(color='#444444', size=10),
        tickfont=dict(color='#444444', size=10),
        showgrid=True, zeroline=False
    ),
    yaxis=dict(
        gridcolor='#111111', linecolor='#1f1f1f', tickcolor='#333333',
        title_font=dict(color='#444444', size=10),
        tickfont=dict(color='#444444', size=10),
        showgrid=True, zeroline=False
    ),
    legend=dict(
        bgcolor='rgba(0,0,0,0)', bordercolor='#1f1f1f', borderwidth=1,
        font=dict(color='#666666', size=10)
    ),
    margin=dict(l=20, r=20, t=40, b=20),
    hoverlabel=dict(
        bgcolor='#111111', bordercolor='#333333',
        font=dict(family='DM Mono, monospace', color='#ffffff', size=11)
    )
)

# Graph color scales — B&W with accent
MONO_COLORS = ['#ffffff', '#cccccc', '#999999', '#666666', '#333333']
CHURN_COLORS = {'Retained': '#444444', 'Churned': '#ffffff'}

# ── DB & Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def get_engine():
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD").replace("@", "%40")
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = os.getenv("DB_PORT")
    DB_NAME = os.getenv("DB_NAME")
    return create_engine(
        f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

@st.cache_data
def load_data():
    return pd.read_sql("SELECT * FROM cleaned_customers", get_engine())

@st.cache_resource
def load_model():
    return joblib.load("src/best_model.pkl"), joblib.load("src/scaler.pkl")

df = load_data()
model, scaler = load_model()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 1.5rem 0 1rem 0; border-bottom: 1px solid #1a1a1a; margin-bottom: 1.5rem;'>
        <div class='sidebar-logo'>Sentinel</div>
        <div class='sidebar-tagline'>Customer Retention Engine</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("", ["Dashboard", "Predict Churn"], label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family: DM Mono, monospace; font-size: 0.6rem; color: #2a2a2a; 
                letter-spacing: 0.12em; text-transform: uppercase; 
                border-top: 1px solid #111111; padding-top: 1rem;'>
        Model — Logistic Regression<br>
        AUC — 0.846<br>
        Dataset — 7,043 records
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "Dashboard":

    st.markdown("""
    <div class='sentinel-title'>Churn Analytics</div>
    <div class='sentinel-subtitle'>Real-time intelligence · Sentinel v1.0</div>
    <div class='divider-line'></div>
    """, unsafe_allow_html=True)

    # ── KPI Cards ──
    total = len(df)
    churned = int(df['Churn'].sum())
    retained = total - churned
    churn_rate = churned / total * 100
    avg_monthly = df[df['Churn'] == 1]['MonthlyCharges'].mean()

    st.markdown(f"""
    <div class='kpi-grid'>
        <div class='kpi-card'>
            <div class='kpi-label'>Total Customers</div>
            <div class='kpi-value'>{total:,}</div>
            <div class='kpi-delta negative'>↗ Full dataset</div>
        </div>
        <div class='kpi-card'>
            <div class='kpi-label'>Churned</div>
            <div class='kpi-value'>{churned:,}</div>
            <div class='kpi-delta negative'>↓ {churn_rate:.1f}% of base</div>
        </div>
        <div class='kpi-card'>
            <div class='kpi-label'>Retained</div>
            <div class='kpi-value'>{retained:,}</div>
            <div class='kpi-delta positive'>↑ {100-churn_rate:.1f}% of base</div>
        </div>
        <div class='kpi-card'>
            <div class='kpi-label'>Avg Churn Bill</div>
            <div class='kpi-value'>${avg_monthly:.0f}</div>
            <div class='kpi-delta negative'>↑ Monthly charges</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Row 1 ──
    st.markdown("<div class='section-header'>Distribution & Contract</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        labels = ['Retained', 'Churned']
        values = [retained, churned]
        fig = go.Figure(go.Pie(
            labels=labels, values=values,
            hole=0.6,
            marker=dict(colors=['#1a1a1a', '#ffffff'],
                        line=dict(color='#000000', width=3)),
            textfont=dict(family='DM Mono, monospace', size=11, color='#000000'),
            hovertemplate='<b>%{label}</b><br>%{value:,} customers<br>%{percent}<extra></extra>'
        ))
        fig.add_annotation(
            text=f"<b>{churn_rate:.1f}%</b><br><span style='font-size:10px'>churn</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(family='Syne, sans-serif', size=18, color='#ffffff')
        )
        fig.update_layout(**BW_LAYOUT, title='Churn Split', showlegend=True)
        fig.update_layout(legend=dict(orientation='h', y=-0.1, x=0.5, xanchor='center',
                                      font=dict(color='#666666', size=10)))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with col2:
        contract_churn = df.groupby('Contract')['Churn'].mean().reset_index()
        contract_churn.columns = ['Contract', 'Churn Rate']
        contract_churn = contract_churn.sort_values('Churn Rate', ascending=True)

        fig = go.Figure(go.Bar(
            x=contract_churn['Churn Rate'],
            y=contract_churn['Contract'],
            orientation='h',
            marker=dict(
                color=contract_churn['Churn Rate'],
                colorscale=[[0, '#1a1a1a'], [0.5, '#888888'], [1, '#ffffff']],
                line=dict(color='#000000', width=0)
            ),
            text=[f"{v:.1%}" for v in contract_churn['Churn Rate']],
            textposition='outside',
            textfont=dict(family='DM Mono, monospace', size=10, color='#666666'),
            hovertemplate='<b>%{y}</b><br>Churn Rate: %{x:.1%}<extra></extra>'
        ))
        fig.update_layout(**BW_LAYOUT, title='Churn Rate by Contract')
        fig.update_layout(xaxis=dict(**BW_LAYOUT['xaxis'], tickformat='.0%'))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # ── Row 2 ──
    st.markdown("<div class='section-header'>Tenure & Charges Analysis</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        for churn_val, label, color in [(0, 'Retained', '#333333'), (1, 'Churned', '#ffffff')]:
            fig.add_trace(go.Histogram(
                x=df[df['Churn'] == churn_val]['tenure'],
                name=label, nbinsx=30,
                marker_color=color,
                opacity=0.85,
                hovertemplate=f'<b>{label}</b><br>Tenure: %{{x}} months<br>Count: %{{y}}<extra></extra>'
            ))
        fig.update_layout(**BW_LAYOUT, title='Tenure Distribution by Churn', barmode='overlay')
        fig.update_layout(xaxis=dict(**BW_LAYOUT['xaxis'], title='Tenure (months)'),
                          yaxis=dict(**BW_LAYOUT['yaxis'], title='Count'))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with col2:
        fig = go.Figure()
        for churn_val, label, color in [(0, 'Retained', '#333333'), (1, 'Churned', '#ffffff')]:
            d = df[df['Churn'] == churn_val]['MonthlyCharges']
            fig.add_trace(go.Box(
                y=d, name=label,
                marker_color=color,
                line=dict(color=color, width=1.5),
                fillcolor=f'rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.08)',
                hovertemplate=f'<b>{label}</b><br>%{{y:.2f}}<extra></extra>'
            ))
        fig.update_layout(**BW_LAYOUT, title='Monthly Charges by Churn')
        fig.update_layout(yaxis=dict(**BW_LAYOUT['yaxis'], title='Monthly Charges ($)'))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # ── Row 3 — Full width ──
    st.markdown("<div class='section-header'>Internet Service & Payment Risk</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        internet_churn = df.groupby('InternetService')['Churn'].mean().reset_index()
        internet_churn.columns = ['Service', 'Churn Rate']
        fig = go.Figure(go.Bar(
            x=internet_churn['Service'],
            y=internet_churn['Churn Rate'],
            marker=dict(
                color=internet_churn['Churn Rate'],
                colorscale=[[0, '#111111'], [1, '#ffffff']],
                line=dict(color='#000000', width=0)
            ),
            text=[f"{v:.1%}" for v in internet_churn['Churn Rate']],
            textposition='outside',
            textfont=dict(family='DM Mono, monospace', size=10, color='#666666'),
            hovertemplate='<b>%{x}</b><br>Churn Rate: %{y:.1%}<extra></extra>'
        ))
        fig.update_layout(**BW_LAYOUT, title='Churn Rate by Internet Service')
        fig.update_layout(yaxis=dict(**BW_LAYOUT['yaxis'], tickformat='.0%'))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with col2:
        pay_churn = df.groupby('PaymentMethod')['Churn'].mean().reset_index()
        pay_churn.columns = ['Method', 'Churn Rate']
        pay_churn = pay_churn.sort_values('Churn Rate', ascending=True)
        pay_churn['Method'] = pay_churn['Method'].str.replace(' (automatic)', '', regex=False)

        fig = go.Figure(go.Bar(
            x=pay_churn['Churn Rate'],
            y=pay_churn['Method'],
            orientation='h',
            marker=dict(
                color=pay_churn['Churn Rate'],
                colorscale=[[0, '#111111'], [1, '#ffffff']],
            ),
            text=[f"{v:.1%}" for v in pay_churn['Churn Rate']],
            textposition='outside',
            textfont=dict(family='DM Mono, monospace', size=10, color='#666666'),
            hovertemplate='<b>%{y}</b><br>Churn Rate: %{x:.1%}<extra></extra>'
        ))
        fig.update_layout(**BW_LAYOUT, title='Churn Rate by Payment Method')
        fig.update_layout(xaxis=dict(**BW_LAYOUT['xaxis'], tickformat='.0%'))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # ── Tenure vs Monthly Charges Scatter ──
    st.markdown("<div class='section-header'>Scatter — Tenure vs Monthly Charges</div>",
                unsafe_allow_html=True)
    sample = df.sample(min(1500, len(df)), random_state=42)
    fig = go.Figure()
    for churn_val, label, color, opacity in [(0, 'Retained', '#333333', 0.5),
                                              (1, 'Churned', '#ffffff', 0.8)]:
        d = sample[sample['Churn'] == churn_val]
        fig.add_trace(go.Scatter(
            x=d['tenure'], y=d['MonthlyCharges'],
            mode='markers', name=label,
            marker=dict(color=color, size=4, opacity=opacity,
                        line=dict(width=0)),
            hovertemplate=f'<b>{label}</b><br>Tenure: %{{x}}m<br>Monthly: $%{{y:.0f}}<extra></extra>'
        ))
    fig.update_layout(**BW_LAYOUT, title='Tenure vs Monthly Charges', height=320)
    fig.update_layout(xaxis=dict(**BW_LAYOUT['xaxis'], title='Tenure (months)'),
                      yaxis=dict(**BW_LAYOUT['yaxis'], title='Monthly Charges ($)'))
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Predict Churn":

    st.markdown("""
    <div class='predict-header'>Churn Risk Predictor</div>
    <div class='sentinel-subtitle'>Enter customer profile · Real-time inference</div>
    <div class='divider-line'></div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<div class='input-section-label'>Demographics</div>", unsafe_allow_html=True)
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)

    with col2:
        st.markdown("<div class='input-section-label'>Services</div>", unsafe_allow_html=True)
        internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        online_sec = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_bk = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multi_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

    with col3:
        st.markdown("<div class='input-section-label'>Billing</div>", unsafe_allow_html=True)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        monthly = st.number_input("Monthly Charges ($)", 0.0, 150.0, 65.0, step=0.5)
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0,
                                        float(monthly * max(tenure, 1)), step=1.0)
        device = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_mv = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Run Prediction →", use_container_width=True):
        encode_map = {
            'gender': {'Male': 1, 'Female': 0},
            'partner': {'Yes': 1, 'No': 0},
            'dependents': {'Yes': 1, 'No': 0},
            'phone': {'Yes': 1, 'No': 0},
            'multi_lines': {'Yes': 2, 'No': 1, 'No phone service': 0},
            'internet': {'Fiber optic': 2, 'DSL': 1, 'No': 0},
            'online_sec': {'Yes': 2, 'No': 1, 'No internet service': 0},
            'online_bk': {'Yes': 2, 'No': 1, 'No internet service': 0},
            'device': {'Yes': 2, 'No': 1, 'No internet service': 0},
            'tech_support': {'Yes': 2, 'No': 1, 'No internet service': 0},
            'streaming_tv': {'Yes': 2, 'No': 1, 'No internet service': 0},
            'streaming_mv': {'Yes': 2, 'No': 1, 'No internet service': 0},
            'contract': {'Two year': 2, 'One year': 1, 'Month-to-month': 0},
            'paperless': {'Yes': 1, 'No': 0},
            'payment': {
                'Electronic check': 3, 'Mailed check': 2,
                'Bank transfer (automatic)': 1, 'Credit card (automatic)': 0
            }
        }

        tenure_group = pd.cut([tenure], bins=[0, 12, 24, 48, 72],
                               labels=[0, 1, 2, 3])[0]
        charges_per_tenure = total_charges / (tenure + 1)
        has_support = (int(tech_support == 'Yes') +
                       int(online_sec == 'Yes') +
                       int(online_bk == 'Yes'))
        is_echeque = int(payment == 'Electronic check')

        features = np.array([[
            encode_map['gender'][gender], senior,
            encode_map['partner'][partner], encode_map['dependents'][dependents],
            tenure, encode_map['phone'][phone],
            encode_map['multi_lines'][multi_lines], encode_map['internet'][internet],
            encode_map['online_sec'][online_sec], encode_map['online_bk'][online_bk],
            encode_map['device'][device], encode_map['tech_support'][tech_support],
            encode_map['streaming_tv'][streaming_tv], encode_map['streaming_mv'][streaming_mv],
            encode_map['contract'][contract], encode_map['paperless'][paperless],
            encode_map['payment'][payment], monthly, total_charges,
            tenure_group, charges_per_tenure, has_support, is_echeque
        ]])

        features_scaled = scaler.transform(features)
        prediction = int(model.predict(features_scaled)[0])
        probability = float(model.predict_proba(features_scaled)[0][1])
        risk = "high" if probability >= 0.7 else "medium" if probability >= 0.4 else "low"
        risk_label = risk.capitalize()

        st.markdown("<div class='divider-line'></div>", unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown(f"""
            <div class='result-card'>
                <div style='font-family: DM Mono, monospace; font-size: 0.6rem;
                            color: #444444; letter-spacing: 0.2em; text-transform: uppercase;
                            margin-bottom: 1rem;'>Prediction Output</div>
                <div class='result-probability {risk}'>{probability*100:.1f}%</div>
                <div style='font-family: DM Mono, monospace; font-size: 0.7rem;
                            color: #444444; margin-top: 0.4rem;'>churn probability</div>
                <div class='result-risk-badge {risk}'>{risk_label} Risk</div>
                <div style='margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid #111111;
                            font-family: DM Mono, monospace; font-size: 0.65rem; color: #444444;'>
                    {'⬛ Intervention recommended' if prediction == 1 else '⬜ Customer likely stable'}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            # Gauge in B&W
            gauge_color = '#ffffff' if risk == 'high' else '#888888' if risk == 'medium' else '#333333'
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                number=dict(suffix="%", font=dict(
                    family='Syne, sans-serif', size=32, color='#ffffff')),
                gauge=dict(
                    axis=dict(range=[0, 100], tickwidth=1, tickcolor='#333333',
                              tickfont=dict(family='DM Mono, monospace',
                                            color='#444444', size=9)),
                    bar=dict(color=gauge_color, thickness=0.25),
                    bgcolor='#000000',
                    borderwidth=1,
                    bordercolor='#1f1f1f',
                    steps=[
                        dict(range=[0, 40], color='#080808'),
                        dict(range=[40, 70], color='#0f0f0f'),
                        dict(range=[70, 100], color='#141414'),
                    ],
                    threshold=dict(
                        line=dict(color='#ffffff', width=2),
                        thickness=0.8, value=probability * 100
                    )
                )
            ))
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='DM Mono, monospace', color='#666666'),
                height=240,
                margin=dict(l=20, r=20, t=20, b=10)
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

            # Risk factor breakdown
            factors = []
            if payment == 'Electronic check': factors.append("E-cheque payment")
            if contract == 'Month-to-month': factors.append("No long-term contract")
            if internet == 'Fiber optic': factors.append("Fiber optic service")
            if tenure < 12: factors.append("New customer (<12mo)")
            if monthly > 75: factors.append("High monthly bill")
            if tech_support == 'No': factors.append("No tech support")

            if factors:
                factors_html = "".join([
                    f"<div style='padding: 0.3rem 0; border-bottom: 1px solid #0f0f0f; "
                    f"font-family: DM Mono, monospace; font-size: 0.65rem; color: #555555;'>"
                    f"↗ {f}</div>"
                    for f in factors[:4]
                ])
                st.markdown(f"""
                <div style='border: 1px solid #1a1a1a; padding: 1rem; margin-top: 0.5rem;'>
                    <div style='font-family: DM Mono, monospace; font-size: 0.58rem;
                                color: #333333; letter-spacing: 0.18em; text-transform: uppercase;
                                margin-bottom: 0.6rem;'>Risk Factors</div>
                    {factors_html}
                </div>
                """, unsafe_allow_html=True)