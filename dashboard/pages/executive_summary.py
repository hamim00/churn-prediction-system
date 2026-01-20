"""
Executive Summary Page
======================
KPIs, charts, and high-level churn insights.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Executive Summary", page_icon="📊", layout="wide")

# Add parent directory to path for utils import
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.api_client import get_api_client

st.markdown("## 📊 Executive Summary")
st.markdown("Real-time churn analytics and customer insights")

client = get_api_client()

# Fetch at-risk customers for analysis
with st.spinner("Loading customer data..."):
    at_risk = client.get_at_risk_customers(min_probability=0.0, limit=100)
    model_info = client.get_model_info()

if not at_risk:
    st.warning("⚠️ Unable to fetch customer data. Please ensure the API is running.")
    st.code("uvicorn api.main:app --reload --port 8000", language="bash")
    st.stop()

# Convert to DataFrame
df = pd.DataFrame(at_risk)

# Calculate KPIs
total_customers = len(df)
high_risk_count = len(df[df['churn_probability'] >= 0.5])
critical_risk_count = len(df[df['churn_probability'] >= 0.75])
avg_probability = df['churn_probability'].mean()

# Revenue at risk
if 'monthly_charges' in df.columns:
    revenue_at_risk = df[df['churn_probability'] >= 0.5]['monthly_charges'].sum()
else:
    revenue_at_risk = high_risk_count * 70

# KPI Row
st.markdown("### 🎯 Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Total Customers", value=f"{total_customers:,}")

with col2:
    st.metric(
        label="High Risk (≥50%)",
        value=f"{high_risk_count:,}",
        delta=f"{high_risk_count/total_customers*100:.1f}%" if total_customers > 0 else "0%",
        delta_color="inverse"
    )

with col3:
    st.metric(
        label="Critical Risk (≥75%)",
        value=f"{critical_risk_count:,}",
        delta=f"{critical_risk_count/total_customers*100:.1f}%" if total_customers > 0 else "0%",
        delta_color="inverse"
    )

with col4:
    st.metric(label="Monthly Revenue at Risk", value=f"${revenue_at_risk:,.0f}")

st.markdown("---")

# Charts Row 1
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📊 Risk Distribution")
    
    df['risk_category'] = pd.cut(
        df['churn_probability'],
        bins=[0, 0.25, 0.5, 0.75, 1.0],
        labels=['Low (0-25%)', 'Medium (25-50%)', 'High (50-75%)', 'Critical (75-100%)']
    )
    
    risk_counts = df['risk_category'].value_counts().reindex(
        ['Low (0-25%)', 'Medium (25-50%)', 'High (50-75%)', 'Critical (75-100%)']
    ).fillna(0)
    
    fig_pie = px.pie(
        values=risk_counts.values,
        names=risk_counts.index,
        color_discrete_sequence=['#00cc44', '#ffcc00', '#ff8800', '#ff4444'],
        hole=0.4
    )
    fig_pie.update_layout(
        margin=dict(t=20, b=20, l=20, r=20),
        height=300,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2)
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    st.markdown("### 📈 Probability Distribution")
    
    fig_hist = px.histogram(df, x='churn_probability', nbins=20, color_discrete_sequence=['#1f77b4'])
    fig_hist.update_layout(
        xaxis_title="Churn Probability",
        yaxis_title="Number of Customers",
        margin=dict(t=20, b=20, l=20, r=20),
        height=300
    )
    fig_hist.add_vline(x=0.5, line_dash="dash", line_color="red", annotation_text="High Risk")
    st.plotly_chart(fig_hist, use_container_width=True)

st.markdown("---")

# Charts Row 2
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📋 Risk by Contract Type")
    if 'contract_type' in df.columns:
        contract_risk = df.groupby('contract_type').agg({
            'churn_probability': 'mean',
            'customer_id': 'count'
        }).reset_index()
        contract_risk.columns = ['Contract Type', 'Avg Churn Risk', 'Count']
        
        fig_contract = px.bar(
            contract_risk, x='Contract Type', y='Avg Churn Risk',
            color='Avg Churn Risk', color_continuous_scale='RdYlGn_r',
            text=contract_risk['Avg Churn Risk'].apply(lambda x: f'{x:.1%}')
        )
        fig_contract.update_layout(height=300, margin=dict(t=20, b=20), coloraxis_showscale=False)
        fig_contract.update_traces(textposition='outside')
        st.plotly_chart(fig_contract, use_container_width=True)

with col2:
    st.markdown("### 💳 Risk by Payment Method")
    if 'payment_method' in df.columns:
        payment_risk = df.groupby('payment_method').agg({
            'churn_probability': 'mean',
            'customer_id': 'count'
        }).reset_index()
        payment_risk.columns = ['Payment Method', 'Avg Churn Risk', 'Count']
        
        fig_payment = px.bar(
            payment_risk, x='Payment Method', y='Avg Churn Risk',
            color='Avg Churn Risk', color_continuous_scale='RdYlGn_r',
            text=payment_risk['Avg Churn Risk'].apply(lambda x: f'{x:.1%}')
        )
        fig_payment.update_layout(height=300, margin=dict(t=20, b=20), coloraxis_showscale=False)
        fig_payment.update_traces(textposition='outside')
        st.plotly_chart(fig_payment, use_container_width=True)

st.markdown("---")

# Model Performance
if model_info:
    st.markdown("### 🤖 Model Performance")
    metrics = model_info.get('metrics', {})
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        roc_auc = metrics.get('roc_auc', 0)
        st.metric("ROC-AUC", f"{roc_auc:.3f}", "✅" if roc_auc >= 0.8 else "⚠️")
    with col2:
        recall = metrics.get('recall', 0)
        st.metric("Recall", f"{recall:.3f}", "✅" if recall >= 0.75 else "⚠️")
    with col3:
        precision = metrics.get('precision', 0)
        st.metric("Precision", f"{precision:.3f}", "✅" if precision >= 0.6 else "⚠️")
    with col4:
        f1 = metrics.get('f1', 0)
        st.metric("F1 Score", f"{f1:.3f}", "✅" if f1 >= 0.65 else "⚠️")

st.markdown("---")

# Top At-Risk Customers
st.markdown("### 🚨 Top 10 At-Risk Customers")

top_risk = df.nlargest(10, 'churn_probability')[['customer_id', 'churn_probability', 'risk_level']].copy()

display_cols = ['customer_id', 'churn_probability', 'risk_level']
if 'contract_type' in df.columns:
    top_risk['contract_type'] = df.nlargest(10, 'churn_probability')['contract_type'].values
    display_cols.append('contract_type')
if 'monthly_charges' in df.columns:
    top_risk['monthly_charges'] = df.nlargest(10, 'churn_probability')['monthly_charges'].values
    display_cols.append('monthly_charges')

top_risk['churn_probability'] = top_risk['churn_probability'].apply(lambda x: f"{x:.1%}")

st.dataframe(
    top_risk[display_cols],
    use_container_width=True,
    hide_index=True,
    column_config={
        "customer_id": "Customer ID",
        "churn_probability": "Churn Risk",
        "risk_level": "Risk Level",
        "contract_type": "Contract",
        "monthly_charges": st.column_config.NumberColumn("Monthly $", format="$%.2f")
    }
)
