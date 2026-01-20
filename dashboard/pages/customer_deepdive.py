"""
Customer Deep-dive Page
=======================
Individual customer analysis with SHAP explanations.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Customer Deep-dive", page_icon="🔍", layout="wide")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.api_client import get_api_client

st.markdown("## 🔍 Customer Deep-dive")
st.markdown("Individual customer analysis and churn explanation")

client = get_api_client()

# Customer ID input
col1, col2 = st.columns([2, 1])
with col1:
    customer_id = st.text_input("Enter Customer ID", placeholder="e.g., 7590-VHVEG")
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)

# Quick select high-risk customers
st.markdown("---")
st.markdown("#### 🚨 Or select from top at-risk customers:")

with st.spinner("Loading..."):
    at_risk = client.get_at_risk_customers(min_probability=0.7, limit=10)

selected = ""
if at_risk and len(at_risk) > 0:
    risk_df = pd.DataFrame(at_risk)
    if 'customer_id' in risk_df.columns:
        selected = st.selectbox(
            "Select high-risk customer:",
            options=[""] + risk_df['customer_id'].tolist(),
            format_func=lambda x: f"{x} ({risk_df[risk_df['customer_id']==x]['churn_probability'].values[0]:.1%})" if x else "Choose..."
        )
        if selected:
            customer_id = selected

st.markdown("---")

# Analyze customer
if customer_id and (analyze_btn or selected):
    with st.spinner(f"Analyzing {customer_id}..."):
        customer = client.get_customer(customer_id)
        prediction = client.get_customer_prediction(customer_id)
    
    if not customer:
        st.error(f"❌ Customer '{customer_id}' not found")
        st.stop()
    
    if not prediction:
        st.error(f"❌ Unable to get prediction")
        st.stop()
    
    prob = prediction.get('churn_probability', 0)
    risk = prediction.get('risk_level', 'UNKNOWN').upper()
    
    # Gauge chart
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={'text': "Churn Risk Score"},
            number={'suffix': '%', 'font': {'size': 48}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#1f77b4"},
                'steps': [
                    {'range': [0, 25], 'color': '#00cc44'},
                    {'range': [25, 50], 'color': '#ffcc00'},
                    {'range': [50, 75], 'color': '#ff8800'},
                    {'range': [75, 100], 'color': '#ff4444'}
                ]
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(t=50, b=0))
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    # Risk badge
    if risk == 'CRITICAL':
        st.error("🚨 **CRITICAL RISK** - Immediate attention required")
    elif risk == 'HIGH':
        st.warning("⚠️ **HIGH RISK** - Customer likely to churn")
    elif risk == 'MEDIUM':
        st.info("📊 **MEDIUM RISK** - Monitor closely")
    else:
        st.success("✅ **LOW RISK** - Customer appears stable")
    
    st.markdown("---")
    
    # Customer details
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 👤 Customer Profile")
        st.markdown("#### Demographics")
        st.markdown(f"**Customer ID:** {customer.get('customer_id')}")
        st.markdown(f"**Gender:** {customer.get('gender')}")
        st.markdown(f"**Senior Citizen:** {'Yes' if customer.get('senior_citizen') else 'No'}")
        st.markdown(f"**Partner:** {'Yes' if customer.get('partner') else 'No'}")
        st.markdown(f"**Dependents:** {'Yes' if customer.get('dependents') else 'No'}")
        
        st.markdown("#### Contract & Billing")
        st.markdown(f"**Contract:** {customer.get('contract_type')}")
        st.markdown(f"**Tenure:** {customer.get('tenure_months')} months")
        st.markdown(f"**Monthly Charges:** ${customer.get('monthly_charges', 0):.2f}")
        st.markdown(f"**Total Charges:** ${customer.get('total_charges', 0):.2f}")
        st.markdown(f"**Payment Method:** {customer.get('payment_method')}")
    
    with col2:
        st.markdown("### 📡 Services")
        services = [
            ("Phone Service", customer.get('phone_service', False)),
            ("Multiple Lines", customer.get('multiple_lines', False)),
            ("Online Security", customer.get('online_security', False)),
            ("Online Backup", customer.get('online_backup', False)),
            ("Device Protection", customer.get('device_protection', False)),
            ("Tech Support", customer.get('tech_support', False)),
            ("Streaming TV", customer.get('streaming_tv', False)),
            ("Streaming Movies", customer.get('streaming_movies', False))
        ]
        for name, value in services:
            icon = "✅" if value else "❌"
            st.markdown(f"{icon} {name}")
        
        internet = customer.get('internet_service', 'No')
        st.markdown(f"📶 **Internet:** {internet}")
    
    st.markdown("---")
    
    # SHAP Explanation
    st.markdown("### 🧠 Why This Prediction?")
    reasons = prediction.get('top_reasons', [])
    
    if reasons:
        reason_df = pd.DataFrame(reasons)
        reason_df['color'] = reason_df['direction'].apply(lambda x: '#ff4444' if x == 'increases' else '#00cc44')
        reason_df = reason_df.sort_values('impact', ascending=True)
        
        fig = go.Figure(go.Bar(
            x=reason_df['impact'],
            y=reason_df['description'],
            orientation='h',
            marker_color=reason_df['color']
        ))
        fig.update_layout(title="Feature Impact", height=250, margin=dict(l=20, r=20, t=40, b=20))
        fig.add_vline(x=0, line_color="gray")
        st.plotly_chart(fig, use_container_width=True)
        
        for r in reasons:
            icon = "🔴" if r['direction'] == 'increases' else "🟢"
            st.markdown(f"- {icon} {r['description']}")
    
    st.markdown("---")
    
    # Recommendations
    st.markdown("### 💡 Retention Recommendations")
    recs = []
    if customer.get('contract_type') == 'Month-to-month':
        recs.append("📝 Offer contract upgrade with discount")
    if customer.get('payment_method') == 'Electronic check':
        recs.append("💳 Suggest auto-payment for small discount")
    if not customer.get('tech_support'):
        recs.append("🛠️ Offer free tech support trial")
    if not customer.get('online_security'):
        recs.append("🔒 Bundle discounted security services")
    if customer.get('tenure_months', 0) < 12:
        recs.append("🎁 New customer engagement program")
    
    for rec in recs or ["✅ Customer profile looks healthy"]:
        st.markdown(rec)

else:
    st.info("👆 Enter a customer ID to analyze their churn risk")
