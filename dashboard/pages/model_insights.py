"""
Model Insights Page
===================
Model performance metrics and feature importance.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Model Insights", page_icon="📈", layout="wide")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.api_client import get_api_client

st.markdown("## 📈 Model Insights")
st.markdown("Model performance, feature importance, and training details")

client = get_api_client()

with st.spinner("Loading model info..."):
    model_info = client.get_model_info()

if not model_info:
    st.error("⚠️ Unable to fetch model information")
    st.stop()

# Model Overview
st.markdown("### 🤖 Model Overview")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Model", model_info.get('model_name', 'Unknown'))
with col2:
    st.metric("Version", model_info.get('version', '1.0.0'))
with col3:
    st.metric("Features", model_info.get('n_features', 0))
with col4:
    date = model_info.get('training_date', 'Unknown')
    st.metric("Trained", date[:10] if len(str(date)) > 10 else date)

st.markdown("---")

# Performance Metrics
st.markdown("### 📊 Performance Metrics")
metrics = model_info.get('metrics', {})

col1, col2, col3, col4 = st.columns(4)

targets = {'roc_auc': 0.80, 'recall': 0.75, 'precision': 0.60, 'f1': 0.65}
labels = {'roc_auc': 'ROC-AUC', 'recall': 'Recall', 'precision': 'Precision', 'f1': 'F1 Score'}

for col, (key, target) in zip([col1, col2, col3, col4], targets.items()):
    with col:
        value = metrics.get(key, 0)
        status = "✅" if value >= target else "❌"
        st.metric(
            label=f"{labels[key]} {status}",
            value=f"{value:.4f}",
            delta=f"vs {target:.2f}",
            delta_color="normal" if value >= target else "inverse"
        )

st.markdown("---")

# Charts
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📈 Metrics vs Targets")
    
    metrics_df = pd.DataFrame([
        {"Metric": "ROC-AUC", "Actual": metrics.get('roc_auc', 0), "Target": 0.80},
        {"Metric": "Recall", "Actual": metrics.get('recall', 0), "Target": 0.75},
        {"Metric": "Precision", "Actual": metrics.get('precision', 0), "Target": 0.60},
        {"Metric": "F1 Score", "Actual": metrics.get('f1', 0), "Target": 0.65}
    ])
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Actual', x=metrics_df['Metric'], y=metrics_df['Actual'],
                         marker_color='#1f77b4', text=metrics_df['Actual'].apply(lambda x: f'{x:.3f}'),
                         textposition='outside'))
    fig.add_trace(go.Scatter(name='Target', x=metrics_df['Metric'], y=metrics_df['Target'],
                              mode='markers', marker=dict(size=15, color='red', symbol='line-ew-open')))
    fig.update_layout(yaxis_range=[0, 1], height=350, barmode='group',
                      legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### 🎯 Performance Radar")
    
    categories = ['ROC-AUC', 'Recall', 'Precision', 'F1', 'ROC-AUC']
    actuals = [metrics.get('roc_auc', 0), metrics.get('recall', 0), 
               metrics.get('precision', 0), metrics.get('f1', 0), metrics.get('roc_auc', 0)]
    targets_radar = [0.80, 0.75, 0.60, 0.65, 0.80]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=actuals, theta=categories, fill='toself', name='Actual',
                                   fillcolor='rgba(31, 119, 180, 0.3)', line=dict(color='#1f77b4')))
    fig.add_trace(go.Scatterpolar(r=targets_radar, theta=categories, fill='toself', name='Target',
                                   fillcolor='rgba(255, 0, 0, 0.1)', line=dict(color='red', dash='dash')))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1])), height=350)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Feature Categories
st.markdown("### 🔑 Feature Importance")
feature_names = model_info.get('feature_names', [])
st.markdown(f"The model uses **{len(feature_names)} features** to predict churn.")

categories = {
    "Demographics": ["gender", "senior", "partner", "dependent", "family"],
    "Services": ["phone", "internet", "security", "backup", "protection", "support", "streaming"],
    "Contract": ["contract", "mtm", "long"],
    "Payment": ["payment", "paperless", "auto"],
    "Tenure": ["tenure", "new", "established", "loyal"],
    "Financial": ["charge", "spend", "clv", "value"],
    "Risk": ["risk"],
}

counts = {}
for cat, keywords in categories.items():
    count = sum(1 for f in feature_names if any(k in f.lower() for k in keywords))
    if count > 0:
        counts[cat] = count

fig = px.bar(x=list(counts.keys()), y=list(counts.values()), 
             color=list(counts.values()), color_continuous_scale='Blues')
fig.update_layout(height=250, showlegend=False, coloraxis_showscale=False,
                  xaxis_title="Category", yaxis_title="Features")
fig.update_traces(texttemplate='%{y}', textposition='outside')
st.plotly_chart(fig, use_container_width=True)

with st.expander("📋 View All Features"):
    st.write(feature_names)

st.markdown("---")

# Interpretation Guide
st.markdown("### 📚 Quick Guide")
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    | Metric | Meaning |
    |--------|---------|
    | **ROC-AUC** | Overall quality (higher = better) |
    | **Recall** | % churners caught |
    | **Precision** | % predictions correct |
    | **F1** | Balance of precision/recall |
    """)

with col2:
    st.markdown("""
    | Risk Level | Range | Action |
    |------------|-------|--------|
    | 🔴 Critical | ≥75% | Immediate |
    | 🟠 High | 50-75% | Proactive |
    | 🟡 Medium | 25-50% | Monitor |
    | 🟢 Low | <25% | Standard |
    """)

# Improvement suggestions
if metrics.get('precision', 0) < 0.60 or metrics.get('f1', 0) < 0.65:
    st.markdown("---")
    st.warning("⚠️ **Improvement Opportunities:** Consider threshold tuning or hyperparameter optimization")
