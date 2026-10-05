"""
TrendPulse: Multi-Modal Social Media Trend Dynamics & Virality Intelligence Framework
Machine Learning Major Project - Case Study No. 127
Author: Tellur Om | AIML Major Project

Strictly styled according to DESIGN.md (Claude.com / Anthropic Editorial Design System)
- Canvas: Tinted warm cream (#faf9f5)
- Primary CTA & Accent: Signature warm coral (#cc785c)
- Surface Cards: Light cream (#efe9de)
- Product Chrome: Dark navy (#181715)
- Typography: Cormorant Garamond (Serif 400, negative tracking) + Inter (Sans 400/500) + JetBrains Mono
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Add src to system path
sys.path.insert(0, os.path.abspath("src"))

from text_processor import clean_text, extract_nlp_features, extract_hashtags
from preprocessor import TrendDataPreprocessor
from utils import (
    load_metrics_and_metadata,
    create_model_comparison_fig,
    create_confusion_matrix_fig,
    create_feature_importance_fig,
    create_temporal_heatmap,
    create_sentiment_virality_fig,
    create_platform_media_fig,
    generate_virality_recommendations
)

# Page Configuration
st.set_page_config(
    page_title="TrendPulse | Social Media Trend ML Framework",
    page_icon="✳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Claude.com / Anthropic Editorial Design System Injection
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,500;0,600;1,400&family=EB+Garamond:ital,wght@0,400;0,500;1,400&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
    
    :root {
        --canvas: #faf9f5;
        --surface-soft: #f5f0e8;
        --surface-card: #efe9de;
        --surface-cream-strong: #e8e0d2;
        --surface-dark: #181715;
        --surface-dark-elevated: #252320;
        --surface-dark-soft: #1f1e1b;
        --hairline: #e6dfd8;
        --hairline-soft: #ebe6df;
        --primary: #cc785c;
        --primary-active: #a9583e;
        --primary-disabled: #e6dfd8;
        --accent-teal: #5db8a6;
        --accent-amber: #e8a55a;
        --ink: #141413;
        --body-strong: #252523;
        --body: #3d3d3a;
        --muted: #6c6a64;
        --muted-soft: #8e8b82;
        --on-primary: #ffffff;
        --on-dark: #faf9f5;
        --on-dark-soft: #a09d96;
    }
    
    /* Base App Floor - Tinted Cream Canvas */
    .stApp {
        background-color: var(--canvas) !important;
        color: var(--ink) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    /* Serif Display Headers (Cormorant Garamond 400, negative tracking, never bold) */
    h1, h2, h3, .serif-display {
        font-family: 'Cormorant Garamond', 'EB Garamond', Garamond, serif !important;
        font-weight: 400 !important;
        letter-spacing: -0.025em !important;
        color: var(--ink) !important;
    }
    
    h1 { font-size: 2.75rem !important; line-height: 1.1 !important; margin-bottom: 0.5rem !important; }
    h2 { font-size: 2.1rem !important; line-height: 1.15 !important; margin-top: 1.5rem !important; margin-bottom: 0.5rem !important; }
    h3 { font-size: 1.55rem !important; line-height: 1.25 !important; margin-top: 1rem !important; }
    h4, h5, h6 { font-family: 'Inter', sans-serif !important; font-weight: 500 !important; color: var(--body-strong) !important; }
    
    p, span, label, div {
        color: var(--body);
        font-family: 'Inter', sans-serif;
    }
    
    /* Code font */
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.88rem !important;
    }
    
    /* Top Navigation / Hero Band */
    .hero-container {
        background-color: var(--canvas);
        border: 1px solid var(--hairline);
        border-radius: 16px;
        padding: 40px;
        margin-bottom: 32px;
    }
    
    .brand-mark {
        font-size: 1.25rem;
        color: var(--ink);
        display: inline-block;
        margin-right: 6px;
        vertical-align: middle;
    }
    
    .badge-pill {
        display: inline-flex;
        align-items: center;
        background-color: var(--surface-card);
        color: var(--ink);
        padding: 5px 14px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 500;
        letter-spacing: 0.5px;
        margin-bottom: 16px;
        border: 1px solid var(--hairline);
    }
    
    .badge-coral {
        background-color: var(--primary);
        color: var(--on-primary);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 500;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    
    .hero-title {
        font-family: 'Cormorant Garamond', 'EB Garamond', serif !important;
        font-size: 3.2rem !important;
        font-weight: 400 !important;
        letter-spacing: -0.03em !important;
        color: var(--ink) !important;
        line-height: 1.05 !important;
        margin-bottom: 14px !important;
    }
    
    .hero-subtitle {
        color: var(--body);
        font-size: 1.12rem;
        max-width: 880px;
        line-height: 1.6;
        font-weight: 400;
    }

    /* Light Cream Feature Cards (Surface Card) */
    .feature-card {
        background-color: var(--surface-card);
        border: 1px solid var(--hairline);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 16px;
        transition: border-color 0.2s ease;
    }
    .feature-card:hover {
        border-color: #d8cfc4;
    }
    .kpi-value {
        font-family: 'Cormorant Garamond', serif;
        font-size: 2.6rem;
        font-weight: 400;
        color: var(--ink);
        line-height: 1.0;
        margin-top: 6px;
    }
    .kpi-label {
        font-size: 0.8rem;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 500;
    }

    /* Dark Navy Product Surfaces (Surface Dark) */
    .dark-mockup-card {
        background-color: var(--surface-dark);
        color: var(--on-dark);
        border-radius: 12px;
        padding: 28px;
        margin-bottom: 24px;
    }
    .dark-mockup-card * {
        color: var(--on-dark) !important;
    }
    .dark-mockup-card .code-panel {
        background-color: var(--surface-dark-soft);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 8px;
        padding: 16px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        line-height: 1.6;
        overflow-x: auto;
    }
    .dark-mockup-card .code-status {
        color: var(--on-dark-soft) !important;
        font-size: 0.8rem;
        margin-top: 8px;
    }

    /* Full-Bleed Coral Callout Card */
    .coral-callout-card {
        background-color: var(--primary);
        color: var(--on-primary);
        border-radius: 12px;
        padding: 32px;
        margin-bottom: 24px;
    }
    .coral-callout-card * {
        color: var(--on-primary) !important;
    }

    /* Primary Coral Buttons */
    button[kind="primary"], .stButton > button {
        background-color: var(--primary) !important;
        color: var(--on-primary) !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 22px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.92rem !important;
        font-weight: 500 !important;
        transition: background-color 0.15s ease !important;
    }
    button[kind="primary"]:hover, .stButton > button:hover {
        background-color: var(--primary-active) !important;
        color: var(--on-primary) !important;
    }

    /* Form Inputs - Clean Cream & Hairline */
    input, textarea, [data-baseweb="select"] {
        background-color: var(--canvas) !important;
        border-color: var(--hairline) !important;
        color: var(--ink) !important;
        border-radius: 8px !important;
    }
    input:focus, textarea:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 3px rgba(204, 120, 92, 0.18) !important;
    }
    
    /* Result Badges */
    .tier-badge-viral {
        background-color: var(--primary);
        color: white;
        padding: 8px 20px;
        border-radius: 9999px;
        font-size: 1.15rem;
        font-weight: 500;
        display: inline-block;
        letter-spacing: 0.5px;
    }
    .tier-badge-moderate {
        background-color: var(--accent-teal);
        color: white;
        padding: 8px 20px;
        border-radius: 9999px;
        font-size: 1.15rem;
        font-weight: 500;
        display: inline-block;
        letter-spacing: 0.5px;
    }
    .tier-badge-low {
        background-color: var(--surface-card);
        color: var(--ink);
        border: 1px solid var(--hairline);
        padding: 8px 20px;
        border-radius: 9999px;
        font-size: 1.15rem;
        font-weight: 500;
        display: inline-block;
        letter-spacing: 0.5px;
    }

    /* Advice Cards */
    .advice-card {
        background-color: var(--surface-card);
        border: 1px solid var(--hairline);
        border-left: 4px solid var(--primary);
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .advice-title {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: var(--ink);
        font-size: 0.96rem;
        margin-bottom: 4px;
    }
    .advice-desc {
        color: var(--body);
        font-size: 0.88rem;
        line-height: 1.5;
    }

    /* Sidebar - Surface Soft (#f5f0e8) with Hairline Border */
    section[data-testid="stSidebar"] {
        background-color: var(--surface-soft) !important;
        border-right: 1px solid var(--hairline) !important;
    }
    section[data-testid="stSidebar"] * {
        color: var(--ink) !important;
    }
    
    /* Metrics override */
    [data-testid="stMetricValue"] {
        font-family: 'Cormorant Garamond', serif !important;
        font-size: 2.2rem !important;
        font-weight: 400 !important;
        color: var(--ink) !important;
    }
    [data-testid="stMetricLabel"] {
        color: var(--muted) !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.8px !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: var(--surface-card) !important;
        border-radius: 8px !important;
        border: 1px solid var(--hairline) !important;
        color: var(--ink) !important;
        font-weight: 500 !important;
    }
    
    /* Table Styling */
    table {
        border-collapse: collapse !important;
        width: 100% !important;
        background-color: var(--canvas) !important;
        border: 1px solid var(--hairline) !important;
        border-radius: 8px !important;
    }
    th {
        background-color: var(--surface-card) !important;
        color: var(--ink) !important;
        font-weight: 500 !important;
        padding: 10px 14px !important;
        border-bottom: 1px solid var(--hairline) !important;
        font-size: 0.88rem !important;
    }
    td {
        padding: 10px 14px !important;
        border-bottom: 1px solid var(--hairline-soft) !important;
        color: var(--body) !important;
        font-size: 0.88rem !important;
    }

    /* Dark Footer */
    .dark-footer {
        background-color: var(--surface-dark);
        color: var(--on-dark-soft);
        border-radius: 12px;
        padding: 40px;
        margin-top: 48px;
    }
    .dark-footer * {
        color: var(--on-dark-soft) !important;
    }
    .dark-footer h4 {
        color: var(--on-dark) !important;
        font-family: 'Cormorant Garamond', serif !important;
        font-size: 1.4rem !important;
    }
</style>
""", unsafe_allow_html=True)


# Cache data & model loader
@st.cache_data
def load_cached_dataset():
    df = pd.read_csv("data/social_media_trends.csv")
    return df


@st.cache_resource
def load_all_artifacts():
    preprocessor = TrendDataPreprocessor.load("models/preprocessor_pipeline.joblib")
    models = {
        'Logistic Regression': joblib.load("models/logistic_regression.joblib"),
        'Support Vector Machine (SVM)': joblib.load("models/support_vector_machine_svm.joblib"),
        'Random Forest': joblib.load("models/random_forest.joblib"),
        'Gradient Boosting': joblib.load("models/gradient_boosting.joblib"),
        'Multi-Layer Perceptron (MLP)': joblib.load("models/multi-layer_perceptron_mlp.joblib")
    }
    lda_model = joblib.load("models/lda_topic_model.joblib")
    lda_vectorizer = joblib.load("models/lda_vectorizer.joblib")
    metrics = load_metrics_and_metadata()
    return preprocessor, models, lda_model, lda_vectorizer, metrics


# Load assets
df = load_cached_dataset()
preprocessor, trained_models, lda_model, lda_vectorizer, benchmark_metrics = load_all_artifacts()


# Sidebar Navigation with Claude.com Styling
with st.sidebar:
    st.markdown("""
        <div style="padding: 12px 0 24px 0;">
            <div style="font-family: 'Cormorant Garamond', serif; font-size: 2.1rem; color: #141413; line-height: 1.1;">
                <span style="color: #cc785c; font-size: 1.6rem; vertical-align: middle; margin-right: 4px;">✳</span><b>TrendPulse</b>
            </div>
            <div style="color: #6c6a64; font-size: 0.8rem; font-weight: 500; letter-spacing: 0.8px; text-transform: uppercase; margin-top: 4px;">
                AIML Major Project • Case 127
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<p style='font-size:0.75rem; text-transform:uppercase; letter-spacing:1px; color:#6c6a64; font-weight:600;'>System Navigation</p>", unsafe_allow_html=True)
    navigation_option = st.radio(
        "Navigation",
        [
            "Overview & Problem Definition",
            "Exploratory Data Analysis (EDA)",
            "Model Benchmark Arena",
            "Real-Time Virality Predictor",
            "Unsupervised Topic Discovery"
        ],
        index=0,
        label_visibility="collapsed"
    )
    
    st.markdown("<hr style='border: none; border-top: 1px solid #e6dfd8; margin: 24px 0;'>", unsafe_allow_html=True)
    
    st.markdown("""
        <div style="background-color: #efe9de; padding: 16px; border-radius: 8px; border: 1px solid #e6dfd8;">
            <div style="font-size: 0.72rem; color: #6c6a64; text-transform: uppercase; font-weight: 600; letter-spacing: 0.8px;">Case Study Specification</div>
            <div style="font-size: 0.92rem; font-weight: 500; color: #141413; margin-top: 4px;">Case Study No. 127: Social Media Trend Analysis</div>
            <div style="font-size: 0.78rem; color: #3d3d3a; margin-top: 6px; line-height: 1.4;">
                Supervised Multi-Class Virality Modeling + Unsupervised LDA Topic Discovery
            </div>
        </div>
    """, unsafe_allow_html=True)


# ==========================================
# MODULE 1: PROBLEM STATEMENT & ARCHITECTURE
# ==========================================
if navigation_option == "Overview & Problem Definition":
    st.markdown("""
        <div class="hero-container">
            <span class="badge-pill"><span style="color:#cc785c; margin-right:6px;">✳</span> Case Study No. 127 • Machine Learning Major Project</span>
            <div class="hero-title">TrendPulse: Multi-Modal Social Media Trend Dynamics & Virality Intelligence</div>
            <div class="hero-subtitle">
                An editorial and enterprise-grade machine learning system designed to uncover latent conversational patterns, model engagement velocity, and forecast breakout virality from publicly accessible social media streams.
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # KPI Grid - Light Cream Feature Cards (Surface Card: #efe9de)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
            <div class="feature-card">
                <div class="kpi-label">Analyzed Records</div>
                <div class="kpi-value">5,000</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class="feature-card">
                <div class="kpi-label">Evaluated Models</div>
                <div class="kpi-value">5 + LDA</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
            <div class="feature-card">
                <div class="kpi-label">Top ROC-AUC Score</div>
                <div class="kpi-value">{benchmark_metrics['supervised_benchmark']['Logistic Regression']['roc_auc_macro']*100:.1f}%</div>
            </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
            <div class="feature-card">
                <div class="kpi-label">Mined Topics</div>
                <div class="kpi-value">6 Verticals</div>
            </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 2 Column layout: Formal Problem Formulation & Organizational Justification
    left_col, right_col = st.columns([1.1, 0.9])
    
    with left_col:
        st.markdown("### Formal Machine Learning Problem Definition")
        st.markdown("""
        Social media streams exhibit immense volume, non-stationary topical drift, and extreme power-law engagement distributions. Traditional keyword matching fails to capture contextual nuance, emotional resonance, and temporal dynamics.
        
        **Mathematical Formulation:**  
        Let each social media post $i$ be parameterized by a multi-modal feature vector:
        $$\\mathbf{x}_i = [\\mathbf{x}_{text}, \\mathbf{x}_{sentiment}, \\mathbf{x}_{temporal}, \\mathbf{x}_{author}, \\mathbf{x}_{modality}] \\in \\mathbb{R}^{278}$$
        
        Where:
        - **$\\mathbf{x}_{text} \\in \\mathbb{R}^{250}$**: TF-IDF n-gram vectors with stopword filtering.
        - **$\\mathbf{x}_{temporal}$**: Cyclical trigonometric projections $\\sin(\\frac{2\\pi h}{24}), \\cos(\\frac{2\\pi h}{24}), \\sin(\\frac{2\\pi d}{7}), \\cos(\\frac{2\\pi d}{7})$.
        - **$\\mathbf{x}_{author}$**: Non-linear logarithmic scaling $[\\log(1 + \\text{followers}), \\text{verified}]$.
        - **$\\mathbf{x}_{modality}$**: Categorical indicator vectors for text, image, video, and carousel formats.
        
        **Core Machine Learning Tasks:**
        1. **Multi-Class Virality Classification:** Formulating hypothesis mapping $f: \\mathcal{X} \\rightarrow \\mathcal{Y}$, where $\\mathcal{Y} \\in \\{\\text{Low}, \\text{Moderate}, \\text{Viral}\\}$.
        2. **Unsupervised Topic Discovery:** Latent Dirichlet Allocation (LDA) to extract conversational themes $P(\\text{word}|\\text{topic})$ and thematic mixture distributions $P(\\text{topic}|\\text{post})$.
        """)
        
    with right_col:
        st.markdown("### Enterprise & Organizational Justification")
        st.markdown("""
        Modern organizations face significant information asymmetry in public communications. Publicly available social data is often noisy, unstructured, and volatile.
        
        **Why Organizations Require This Framework:**
        - **PR Crisis Mitigation:** Early detection of polarizing or viral negative themes before public relations escalation.
        - **Publishing Schedule Optimization:** Identifying the exact diurnal windows (12:00-14:00 and 18:00-21:00) where engagement velocity maximizes reach per follower.
        - **Content Resource Allocation:** Evaluating whether high-budget video production yields statistically verified engagement lift over static text.
        - **Algorithmic Trend Forecasting:** Uncovering emerging conversation clusters early to capitalize on consumer sentiment.
        """)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Alternating Rhythm: Dark Product Chrome Surface (Surface Dark: #181715)
    st.markdown("""
        <div class="dark-mockup-card">
            <div style="font-family:'Cormorant Garamond', serif; font-size:1.6rem; margin-bottom:8px;">End-to-End Pipeline Architecture</div>
            <div style="color:#a09d96; font-size:0.9rem; margin-bottom:18px;">
                Complete transformation pipeline from raw public social post to cyclical NLP feature matrices and dual model inference.
            </div>
            <div class="code-panel">
RAW POST STREAM ──> [Text Normalization & Regex Cleaner] ──> TF-IDF 250-D N-Grams
                 ──> [Temporal Coordinate Extractor]    ──> Sin / Cos Cyclical Projections (Hour, Day)
                 ──> [Author & Format Extractor]       ──> Log(1 + Followers) + One-Hot Modalities
                 ─────────────────────────────────────────────────────────────────────────────
                 ──> COMPOSITE 278-D FEATURE MATRIX X
                         ├──> [Supervised Classifier Arena] ──> Virality Tier {Low, Moderate, Viral}
                         └──> [Unsupervised LDA Engine]    ──> 6 Latent Conversational Topics
            </div>
            <div class="code-status">✳ Pipeline Artifacts: models/preprocessor_pipeline.joblib • Feature Dimensions: 278</div>
        </div>
    """, unsafe_allow_html=True)


# ==========================================
# MODULE 2: EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
elif navigation_option == "Exploratory Data Analysis (EDA)":
    st.markdown("## Exploratory Data Analysis & Empirical Observations")
    st.markdown("Statistical exploration across platforms, media types, publication timing, and sentiment dynamics.")
    
    # Filter Controls inside Surface Card
    with st.expander("Filter Dataset for Segmented Exploration", expanded=False):
        fcol1, fcol2, fcol3 = st.columns(3)
        with fcol1:
            sel_platform = st.multiselect("Social Platform:", options=df['platform'].unique(), default=df['platform'].unique())
        with fcol2:
            sel_category = st.multiselect("Thematic Vertical:", options=df['category'].unique(), default=df['category'].unique())
        with fcol3:
            sel_media = st.multiselect("Content Format:", options=df['media_type'].unique(), default=df['media_type'].unique())
            
    filtered_df = df[
        (df['platform'].isin(sel_platform)) &
        (df['category'].isin(sel_category)) &
        (df['media_type'].isin(sel_media))
    ]
    
    st.markdown(f"<p style='color:#6c6a64; font-size:0.88rem;'>Displaying <b>{len(filtered_df):,}</b> of <b>{len(df):,}</b> total benchmark records.</p>", unsafe_allow_html=True)
    
    # Visual Row 1: Class Distribution & Temporal Virality Heatmap
    c1, c2 = st.columns([1, 1.4])
    with c1:
        tier_counts = filtered_df['virality_tier'].value_counts().reset_index()
        tier_counts.columns = ['Tier', 'Count']
        fig_donut = px.pie(
            tier_counts,
            values='Count',
            names='Tier',
            hole=0.55,
            title="Virality Tier Class Distribution",
            color='Tier',
            color_discrete_map={'Low': '#8e8b82', 'Moderate': '#5db8a6', 'Viral': '#cc785c'}
        )
        fig_donut.update_layout(
            font=dict(family="Inter, sans-serif", size=12, color="#141413"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            title=dict(font=dict(family="'Cormorant Garamond', serif", size=20, color="#141413"))
        )
        st.plotly_chart(fig_donut, use_container_width=True)
        
    with c2:
        fig_heat = create_temporal_heatmap(filtered_df)
        st.plotly_chart(fig_heat, use_container_width=True)
        
    # Visual Row 2: Media Type by Platform & Sentiment Dynamics
    c3, c4 = st.columns(2)
    with c3:
        fig_platform = create_platform_media_fig(filtered_df)
        st.plotly_chart(fig_platform, use_container_width=True)
    with c4:
        fig_sentiment = create_sentiment_virality_fig(filtered_df)
        st.plotly_chart(fig_sentiment, use_container_width=True)
        
    # Key Statistical Observations Cards
    st.markdown("### Key Empirical Findings")
    col_obs1, col_obs2, col_obs3 = st.columns(3)
    with col_obs1:
        st.markdown("""
            <div class="feature-card">
                <div style="font-weight:600; color:#141413; margin-bottom:6px;">1. Video & Carousel Format Lift</div>
                <div style="color:#3d3d3a; font-size:0.88rem; line-height:1.5;">
                    Posts featuring Video or Carousel formats achieve <b>2.1x to 2.4x higher median engagement</b> than plain text across all platforms.
                </div>
            </div>
        """, unsafe_allow_html=True)
    with col_obs2:
        st.markdown("""
            <div class="feature-card">
                <div style="font-weight:600; color:#141413; margin-bottom:6px;">2. Bimodal Diurnal Peaks</div>
                <div style="color:#3d3d3a; font-size:0.88rem; line-height:1.5;">
                    Engagement velocity surges predictably during <b>12:00-14:00 (midday)</b> and <b>18:00-21:00 (evening)</b>, validating cyclical temporal encodings.
                </div>
            </div>
        """, unsafe_allow_html=True)
    with col_obs3:
        st.markdown("""
            <div class="feature-card">
                <div style="font-weight:600; color:#141413; margin-bottom:6px;">3. Emotional Subjectivity</div>
                <div style="color:#3d3d3a; font-size:0.88rem; line-height:1.5;">
                    Posts with pronounced subjectivity and polarized sentiment are <b>3.2x more likely</b> to enter the Viral breakout tier than neutral statements.
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    # Data Table Sample
    with st.expander("Inspect Benchmark Sample Data"):
        st.dataframe(
            filtered_df[['post_id', 'platform', 'category', 'author_followers', 'media_type', 'sentiment_polarity', 'raw_engagement', 'virality_tier', 'post_text']].head(20),
            use_container_width=True
        )


# ==========================================
# MODULE 3: MODEL BENCHMARK ARENA
# ==========================================
elif navigation_option == "Model Benchmark Arena":
    st.markdown("## Machine Learning Model Benchmark Arena")
    st.markdown("Empirical comparison of five supervised classification architectures evaluated across 5-fold stratified cross-validation on 278 engineered features.")
    
    # Model Comparison Chart
    fig_comp = create_model_comparison_fig(benchmark_metrics)
    st.plotly_chart(fig_comp, use_container_width=True)
    
    # Editorial Comparison Table
    bench_data = []
    for m_name, m_info in benchmark_metrics['supervised_benchmark'].items():
        bench_data.append({
            'Algorithm': m_name,
            'Test Accuracy': f"{m_info['accuracy']*100:.2f}%",
            'Macro F1-Score': f"{m_info['f1_macro']*100:.2f}%",
            'Weighted F1': f"{m_info['f1_weighted']*100:.2f}%",
            'ROC-AUC (OvR)': f"{m_info['roc_auc_macro']*100:.2f}%",
            '5-Fold CV F1 (μ ± σ)': f"{m_info['cv_f1_mean']*100:.2f}% ± {m_info['cv_f1_std']*100:.2f}%",
            'Training Latency': f"{m_info['training_time_sec']:.2f}s"
        })
    st.table(pd.DataFrame(bench_data))
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Detailed Model Deep Dive
    st.markdown("### Model Diagnostics & Error Breakdown")
    selected_inspect_model = st.selectbox(
        "Select Model for Detailed Inspection:",
        list(benchmark_metrics['supervised_benchmark'].keys())
    )
    
    model_stats = benchmark_metrics['supervised_benchmark'][selected_inspect_model]
    
    dcol1, dcol2 = st.columns([1, 1.2])
    with dcol1:
        cm_fig = create_confusion_matrix_fig(
            model_stats['confusion_matrix'],
            benchmark_metrics['target_classes'],
            selected_inspect_model
        )
        st.plotly_chart(cm_fig, use_container_width=True)
        
    with dcol2:
        if model_stats['top_feature_importance']:
            feat_fig = create_feature_importance_fig(model_stats['top_feature_importance'], selected_inspect_model)
            st.plotly_chart(feat_fig, use_container_width=True)
        else:
            st.info(f"Direct feature importance is not applicable for {selected_inspect_model} due to non-linear kernel representation.")
            
    # Per-Class Precision / Recall Table
    st.markdown(f"#### Class-Level Metrics for {selected_inspect_model}")
    cr = model_stats['classification_report']
    cr_df = pd.DataFrame({
        'Class': ['Low', 'Moderate', 'Viral'],
        'Precision': [f"{cr['Low']['precision']*100:.1f}%", f"{cr['Moderate']['precision']*100:.1f}%", f"{cr['Viral']['precision']*100:.1f}%"],
        'Recall': [f"{cr['Low']['recall']*100:.1f}%", f"{cr['Moderate']['recall']*100:.1f}%", f"{cr['Viral']['recall']*100:.1f}%"],
        'F1-Score': [f"{cr['Low']['f1-score']*100:.1f}%", f"{cr['Moderate']['f1-score']*100:.1f}%", f"{cr['Viral']['f1-score']*100:.1f}%"],
        'Test Support': [int(cr['Low']['support']), int(cr['Moderate']['support']), int(cr['Viral']['support'])]
    })
    st.table(cr_df)
    
    # Error Analysis Card
    st.markdown("""
        <div class="feature-card" style="margin-top: 16px;">
            <div style="font-weight:600; color:#141413; margin-bottom:6px;">Error Analysis & Misclassification Interpretation</div>
            <div style="color:#3d3d3a; font-size:0.88rem; line-height:1.6;">
                <b>• Continuous Boundary Ambiguity:</b> The majority of classification errors occur along the Moderate vs. Viral boundary. Social media reach operates on continuous power-law gradients; posts near the 85th percentile boundary exhibit sensitive threshold behavior.<br>
                <b>• High Follower Flops (False Positives):</b> Accounts with over 500,000 followers occasionally underperform when copy is overly generic, which initial follower weights over-estimate.<br>
                <b>• Niche Account Breakouts (False Negatives):</b> Niche accounts with under 1,000 followers occasionally generate explosive viral hits driven by breaking news or intense emotional controversy that metadata alone cannot foresee.
            </div>
        </div>
    """, unsafe_allow_html=True)


# ==========================================
# MODULE 4: REAL-TIME VIRALITY PREDICTOR
# ==========================================
elif navigation_option == "Real-Time Virality Predictor":
    st.markdown("## Real-Time Virality & Engagement Predictor")
    st.markdown("Test prospective social copy and publishing parameters to forecast reach tier and receive actionable content optimization heuristics.")
    
    col_input, col_pred = st.columns([1.1, 0.9])
    
    with col_input:
        st.markdown("### Post Configuration & Metadata")
        
        user_post_text = st.text_area(
            "Draft Social Copy (including text & hashtags):",
            value="Unbelievable breakthrough! We just open-sourced our autonomous AI agent framework with zero latency and 99.2% benchmark accuracy. How is your team adapting to edge AI in 2026? #ArtificialIntelligence #MachineLearning #TechTrends",
            height=130
        )
        
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            sel_user_platform = st.selectbox("Platform:", ['Twitter/X', 'Instagram', 'LinkedIn', 'Reddit'])
            sel_user_category = st.selectbox("Category:", [
                'Technology & AI', 'Finance & Crypto', 'Health & Wellness',
                'Entertainment & Culture', 'Sports & Gaming', 'Business & Careers'
            ])
            sel_user_media = st.selectbox("Attached Format:", ['Video', 'Image', 'Carousel', 'Text'])
            
        with m_col2:
            user_followers = st.number_input("Author Followers:", min_value=50, max_value=5000000, value=25000, step=500)
            user_hour = st.slider("Publishing Hour (24h):", min_value=0, max_value=23, value=13)
            user_day = st.selectbox("Day of Week:", [
                'Monday (0)', 'Tuesday (1)', 'Wednesday (2)', 'Thursday (3)',
                'Friday (4)', 'Saturday (5)', 'Sunday (6)'
            ], index=2)
            user_day_idx = int(user_day.split('(')[1].replace(')', ''))
            
        sel_active_model = st.selectbox(
            "Classifier Engine:",
            ['Logistic Regression', 'Gradient Boosting', 'Random Forest', 'Support Vector Machine (SVM)', 'Multi-Layer Perceptron (MLP)'],
            index=0
        )
        
        predict_btn = st.button("Forecast Virality Potential", use_container_width=True, type="primary")

    with col_pred:
        st.markdown("### Forecast & Signal Diagnostics")
        
        if predict_btn or user_post_text:
            X_infer = preprocessor.transform_single_post(
                post_text=user_post_text,
                platform=sel_user_platform,
                category=sel_user_category,
                media_type=sel_user_media,
                followers=user_followers,
                hour=user_hour,
                day_of_week=user_day_idx,
                is_verified=1 if user_followers > 50000 else 0
            )
            
            chosen_model = trained_models[sel_active_model]
            pred_idx = chosen_model.predict(X_infer)[0]
            pred_tier = preprocessor.label_encoder.inverse_transform([pred_idx])[0]
            pred_probs = chosen_model.predict_proba(X_infer)[0]
            
            class_prob_dict = {
                cls_name: float(pred_probs[i])
                for i, cls_name in enumerate(preprocessor.label_encoder.classes_)
            }
            
            nlp_stats = extract_nlp_features(user_post_text)
            
            virality_score, recommendations = generate_virality_recommendations(
                predicted_tier=pred_tier,
                probabilities=class_prob_dict,
                nlp_dict=nlp_stats,
                media_type=sel_user_media,
                hour=user_hour,
                follower_count=user_followers
            )
            
            # Display Prediction Badge using Claude Tokens
            st.markdown("<div style='text-align: center; margin: 16px 0;'>", unsafe_allow_html=True)
            if pred_tier == 'Viral':
                st.markdown('<span class="tier-badge-viral">✳ PREDICTED TIER: VIRAL BREAKOUT</span>', unsafe_allow_html=True)
            elif pred_tier == 'Moderate':
                st.markdown('<span class="tier-badge-moderate">✳ PREDICTED TIER: MODERATE TRACTION</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="tier-badge-low">✳ PREDICTED TIER: LOW REACH</span>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Gauges / Metrics
            score_col1, score_col2 = st.columns(2)
            with score_col1:
                st.metric("Virality Index", f"{virality_score} / 100")
            with score_col2:
                st.metric("Viral Probability", f"{class_prob_dict.get('Viral', 0.0)*100:.1f}%")
                
            # Class Probability Distribution Bar Chart
            prob_df = pd.DataFrame({
                'Tier': list(class_prob_dict.keys()),
                'Probability (%)': [v * 100 for v in class_prob_dict.values()]
            })
            prob_fig = px.bar(
                prob_df,
                x='Tier',
                y='Probability (%)',
                color='Tier',
                color_discrete_map={'Low': '#8e8b82', 'Moderate': '#5db8a6', 'Viral': '#cc785c'},
                text_auto='.1f',
                height=200
            )
            prob_fig.update_layout(
                font=dict(family="Inter, sans-serif", size=12, color="#141413"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#efe9de",
                margin=dict(l=20, r=20, t=20, b=20),
                showlegend=False,
                xaxis=dict(gridcolor="#e6dfd8"),
                yaxis=dict(gridcolor="#e6dfd8")
            )
            st.plotly_chart(prob_fig, use_container_width=True)
            
            # Linguistic Signals
            st.markdown("#### Linguistic & Emotional Signals")
            pcol1, pcol2, pcol3 = st.columns(3)
            with pcol1:
                st.markdown(f"**Polarity:** `{nlp_stats['sentiment_polarity']:+.2f}`")
                st.markdown(f"**Subjectivity:** `{nlp_stats['sentiment_subjectivity']:.2f}`")
            with pcol2:
                st.markdown(f"**Word Count:** `{int(nlp_stats['word_count'])}`")
                st.markdown(f"**Char Count:** `{int(nlp_stats['char_count'])}`")
            with pcol3:
                st.markdown(f"**Questions:** `{int(nlp_stats['question_count'])}`")
                st.markdown(f"**Hashtags:** `{int(nlp_stats['hashtag_count'])}`")

    # Recommendations Section
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Editorial Content Optimization Heuristics")
    st.markdown("Actionable suggestions derived from model feature importances to improve distribution reach:")
    
    for tip in recommendations:
        st.markdown(f"""
            <div class="advice-card">
                <div style="font-size: 0.72rem; text-transform: uppercase; color: #cc785c; font-weight: 600; letter-spacing: 0.8px;">{tip['category']}</div>
                <div class="advice-title">{tip['title']}</div>
                <div class="advice-desc">{tip['advice']}</div>
            </div>
        """, unsafe_allow_html=True)


# ==========================================
# MODULE 5: UNSUPERVISED TOPIC DISCOVERY
# ==========================================
elif navigation_option == "Unsupervised Topic Discovery":
    st.markdown("## Unsupervised Topic Discovery & Conversational Clustering")
    st.markdown("Discovering latent conversational patterns across social media data using **Latent Dirichlet Allocation (LDA)** without human supervision.")
    
    topics_dict = benchmark_metrics['unsupervised_topics']
    
    st.markdown("""
    LDA models each social media post as a random mixture over latent topics, and each topic as a discrete probability distribution over vocabulary tokens:
    $$P(w_i | d) = \\sum_{k=1}^{K} P(w_i | z_i = k) \\cdot P(z_i = k | d)$$
    """)
    
    # 6 Topic Cards in Grid (Surface Card: #efe9de)
    tcols1 = st.columns(3)
    tcols2 = st.columns(3)
    grid_cols = tcols1 + tcols2
    
    topic_titles = [
        "AI Agents & Computing Benchmarks",
        "Pop Culture & Cinematic Reviews",
        "Macro Crypto, Bitcoin & Financial Markets",
        "Executive Leadership & Future of Work",
        "Global Esports & Athletic Championships",
        "Viral Breakthroughs & Tech Portfolios"
    ]
    
    for idx, (t_name, t_content) in enumerate(topics_dict.items()):
        with grid_cols[idx]:
            st.markdown(f"""
                <div class="feature-card">
                    <div style="font-size: 0.75rem; color: #cc785c; font-weight: 600; text-transform: uppercase; letter-spacing: 0.8px;">{t_name}</div>
                    <div style="font-family: 'Cormorant Garamond', serif; font-size: 1.35rem; font-weight: 500; color: #141413; margin: 4px 0 10px 0;">{topic_titles[idx]}</div>
                    <div style="display: flex; flex-wrap: wrap; gap: 5px;">
                        {' '.join([f'<span style=\"background-color: #faf9f5; border: 1px solid #e6dfd8; color: #141413; padding: 3px 8px; border-radius: 9999px; font-size: 0.75rem;\">#{w}</span>' for w in t_content['keywords'][:6]])}
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
    # Interactive Keyword Weight Visualizer
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Latent Topic Vocabulary Weights")
    sel_topic = st.selectbox("Inspect Topic Vocabulary:", list(topics_dict.keys()))
    
    top_words = topics_dict[sel_topic]['keywords']
    top_weights = topics_dict[sel_topic]['top_weights']
    
    topic_bar_df = pd.DataFrame({'Keyword': top_words[::-1], 'Weight': top_weights[::-1]})
    topic_fig = px.bar(
        topic_bar_df,
        x='Weight',
        y='Keyword',
        orientation='h',
        title=f"Vocabulary Distribution for {sel_topic}",
        color_discrete_sequence=['#cc785c']
    )
    topic_fig.update_layout(
        font=dict(family="Inter, sans-serif", size=12, color="#141413"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#efe9de",
        xaxis=dict(gridcolor="#e6dfd8"),
        yaxis=dict(gridcolor="#e6dfd8"),
        title=dict(font=dict(family="'Cormorant Garamond', serif", size=20, color="#141413"))
    )
    st.plotly_chart(topic_fig, use_container_width=True)


# Dark Footer (Surface Dark: #181715)
st.markdown("""
    <div class="dark-footer">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:20px;">
            <div>
                <div style="font-family:'Cormorant Garamond', serif; font-size:1.8rem; color:#faf9f5;">
                    <span style="color:#cc785c; margin-right:4px;">✳</span> TrendPulse
                </div>
                <div style="font-size:0.85rem; color:#a09d96; margin-top:4px;">
                    Multi-Modal Social Media Trend Dynamics & Virality Intelligence
                </div>
            </div>
            <div style="font-size:0.82rem; color:#a09d96; text-align:right;">
                Case Study No. 127 • AIML Major Project<br>
                Department of Artificial Intelligence & Machine Learning
            </div>
        </div>
        <hr style="border:none; border-top:1px solid rgba(255,255,255,0.1); margin:24px 0 16px 0;">
        <div style="font-size:0.78rem; color:#6c6a64; text-align:center;">
            Designed in accordance with Claude.com Editorial Design System (Cream Canvas, Warm Coral, Dark Navy Surface)
        </div>
    </div>
""", unsafe_allow_html=True)
