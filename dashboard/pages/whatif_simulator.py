"""
What-if Simulator Page
======================
Interactive simulator to explore churn risk changes.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="What-if Simulator", page_icon="🔮", layout="wide")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.api_client import get_api_client

st.markdown("## 🔮 What-if Simulator")
st.markdown("Explore how changes affect churn risk")

client = get_api_client()

# Load customer option
col1, col2 = st.columns(2)
with col1:
    load_id = st.text_input("Load existing customer", placeholder="e.g., 7590-VHVEG")
    load_btn = st.button("📥 Load")
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    reset_btn = st.button("🔄 Reset to Defaults")

# Session state
if 'sim_data' not in st.session_state or reset_btn:
    st.session_state.sim_data = {
        'gender': 'Male', 'senior_citizen': False, 'partner': False, 'dependents': False,
        'tenure_months': 12, 'phone_service': True, 'multiple_lines': False,
        'internet_service': 'Fiber optic', 'online_security': False, 'online_backup': False,
        'device_protection': False, 'tech_support': False, 'streaming_tv': False,
        'streaming_movies': False, 'contract_type': 'Month-to-month', 'paperless_billing': True,
        'payment_method': 'Electronic check', 'monthly_charges': 70.0, 'total_charges': 840.0
    }

# Load customer
if load_btn and load_id:
    cust = client.get_customer(load_id)
    if cust:
        st.session_state.sim_data = {
            'gender': cust.get('gender', 'Male'),
            'senior_citizen': cust.get('senior_citizen', False),
            'partner': cust.get('partner', False),
            'dependents': cust.get('dependents', False),
            'tenure_months': cust.get('tenure_months', 12),
            'phone_service': cust.get('phone_service', True),
            'multiple_lines': cust.get('multiple_lines', False),
            'internet_service': cust.get('internet_service', 'No'),
            'online_security': cust.get('online_security', False),
            'online_backup': cust.get('online_backup', False),
            'device_protection': cust.get('device_protection', False),
            'tech_support': cust.get('tech_support', False),
            'streaming_tv': cust.get('streaming_tv', False),
            'streaming_movies': cust.get('streaming_movies', False),
            'contract_type': cust.get('contract_type', 'Month-to-month'),
            'paperless_billing': cust.get('paperless_billing', True),
            'payment_method': cust.get('payment_method', 'Electronic check'),
            'monthly_charges': cust.get('monthly_charges', 70.0),
            'total_charges': cust.get('total_charges', 840.0)
        }
        st.success(f"✅ Loaded {load_id}")
        st.rerun()
    else:
        st.error(f"❌ Customer not found")

st.markdown("---")

# Input form
st.markdown("### 👤 Customer Attributes")

# Demographics
col1, col2, col3, col4 = st.columns(4)
with col1:
    gender = st.selectbox("Gender", ['Male', 'Female'], 
                          index=0 if st.session_state.sim_data['gender'] == 'Male' else 1)
with col2:
    senior = st.checkbox("Senior Citizen", value=st.session_state.sim_data['senior_citizen'])
with col3:
    partner = st.checkbox("Has Partner", value=st.session_state.sim_data['partner'])
with col4:
    dependents = st.checkbox("Has Dependents", value=st.session_state.sim_data['dependents'])

st.markdown("---")

# Contract
st.markdown("### 📝 Contract & Payment")
col1, col2, col3 = st.columns(3)

with col1:
    contract = st.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'],
                            index=['Month-to-month', 'One year', 'Two year'].index(st.session_state.sim_data['contract_type']))
with col2:
    payment = st.selectbox("Payment Method", 
                           ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
                           index=['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'].index(st.session_state.sim_data['payment_method']))
with col3:
    paperless = st.checkbox("Paperless Billing", value=st.session_state.sim_data['paperless_billing'])

col1, col2, col3 = st.columns(3)
with col1:
    tenure = st.slider("Tenure (months)", 0, 72, st.session_state.sim_data['tenure_months'])
with col2:
    monthly = st.slider("Monthly Charges ($)", 18.0, 120.0, float(st.session_state.sim_data['monthly_charges']), 5.0)
with col3:
    total = st.number_input("Total Charges ($)", 0.0, 10000.0, float(st.session_state.sim_data['total_charges']), 100.0)

st.markdown("---")

# Services
st.markdown("### 📡 Services")
col1, col2 = st.columns(2)

with col1:
    phone = st.checkbox("Phone Service", value=st.session_state.sim_data['phone_service'])
    multiple = st.checkbox("Multiple Lines", value=st.session_state.sim_data['multiple_lines'], disabled=not phone)
    internet = st.selectbox("Internet", ['No', 'DSL', 'Fiber optic'],
                            index=['No', 'DSL', 'Fiber optic'].index(st.session_state.sim_data['internet_service']))

has_net = internet != 'No'
with col2:
    security = st.checkbox("Online Security", value=st.session_state.sim_data['online_security'], disabled=not has_net)
    backup = st.checkbox("Online Backup", value=st.session_state.sim_data['online_backup'], disabled=not has_net)
    protection = st.checkbox("Device Protection", value=st.session_state.sim_data['device_protection'], disabled=not has_net)
    support = st.checkbox("Tech Support", value=st.session_state.sim_data['tech_support'], disabled=not has_net)
    tv = st.checkbox("Streaming TV", value=st.session_state.sim_data['streaming_tv'], disabled=not has_net)
    movies = st.checkbox("Streaming Movies", value=st.session_state.sim_data['streaming_movies'], disabled=not has_net)

st.markdown("---")

# Predict
if st.button("🔮 Predict Churn Risk", type="primary", use_container_width=True):
    data = {
        "gender": gender, "senior_citizen": senior, "partner": partner, "dependents": dependents,
        "tenure_months": tenure, "phone_service": phone, "multiple_lines": multiple if phone else False,
        "internet_service": internet,
        "online_security": security if has_net else False, "online_backup": backup if has_net else False,
        "device_protection": protection if has_net else False, "tech_support": support if has_net else False,
        "streaming_tv": tv if has_net else False, "streaming_movies": movies if has_net else False,
        "contract_type": contract, "paperless_billing": paperless, "payment_method": payment,
        "monthly_charges": monthly, "total_charges": total
    }
    
    with st.spinner("Calculating..."):
        result = client.predict(data)
    
    if result:
        st.markdown("---")
        st.markdown("### 🎯 Prediction Results")
        
        prob = result.get('churn_probability', 0)
        risk = result.get('risk_level', 'UNKNOWN').upper()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=prob * 100,
                title={'text': "Churn Probability"},
                number={'suffix': '%', 'font': {'size': 40}},
                gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#1f77b4"},
                       'steps': [{'range': [0, 25], 'color': '#00cc44'}, {'range': [25, 50], 'color': '#ffcc00'},
                                 {'range': [50, 75], 'color': '#ff8800'}, {'range': [75, 100], 'color': '#ff4444'}]}
            ))
            fig.update_layout(height=250, margin=dict(t=50, b=0))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if risk == 'CRITICAL':
                st.error("🚨 **CRITICAL**")
            elif risk == 'HIGH':
                st.warning("⚠️ **HIGH**")
            elif risk == 'MEDIUM':
                st.info("📊 **MEDIUM**")
            else:
                st.success("✅ **LOW**")
            
            st.markdown(f"**Prediction:** {'Will Churn' if prob >= 0.5 else 'Will Stay'}")
        
        # Reasons
        reasons = result.get('top_reasons', [])
        if reasons:
            st.markdown("### 📊 Key Factors")
            for r in reasons:
                icon = "🔴" if r['direction'] == 'increases' else "🟢"
                st.markdown(f"- {icon} {r['description']}")
        
        # Suggestions
        st.markdown("### 💡 Try These Changes")
        suggestions = []
        if contract == 'Month-to-month':
            suggestions.append("📝 Change to **One year** or **Two year** contract")
        if payment == 'Electronic check':
            suggestions.append("💳 Switch to **automatic payment**")
        if not support and has_net:
            suggestions.append("🛠️ Add **Tech Support**")
        if not security and has_net:
            suggestions.append("🔒 Add **Online Security**")
        
        for s in suggestions or ["✅ Profile already optimized!"]:
            st.markdown(f"- {s}")
    else:
        st.error("❌ Prediction failed")

# Scenario comparison
st.markdown("---")
if st.button("📊 Compare Scenarios"):
    scenarios = [
        {"name": "MTM + E-check", "contract_type": "Month-to-month", "payment_method": "Electronic check"},
        {"name": "MTM + Auto-pay", "contract_type": "Month-to-month", "payment_method": "Bank transfer (automatic)"},
        {"name": "1-year + Auto-pay", "contract_type": "One year", "payment_method": "Bank transfer (automatic)"},
        {"name": "2-year + Auto-pay", "contract_type": "Two year", "payment_method": "Bank transfer (automatic)"}
    ]
    
    results = []
    with st.spinner("Comparing..."):
        for s in scenarios:
            data = {
                "gender": gender, "senior_citizen": senior, "partner": partner, "dependents": dependents,
                "tenure_months": tenure, "phone_service": phone, "multiple_lines": multiple if phone else False,
                "internet_service": internet,
                "online_security": security if has_net else False, "online_backup": backup if has_net else False,
                "device_protection": protection if has_net else False, "tech_support": support if has_net else False,
                "streaming_tv": tv if has_net else False, "streaming_movies": movies if has_net else False,
                "contract_type": s["contract_type"], "paperless_billing": paperless, "payment_method": s["payment_method"],
                "monthly_charges": monthly, "total_charges": total
            }
            r = client.predict(data)
            if r:
                results.append({"Scenario": s["name"], "Risk": r['churn_probability']})
    
    if results:
        df = pd.DataFrame(results).sort_values('Risk')
        fig = go.Figure(go.Bar(x=df['Risk'], y=df['Scenario'], orientation='h',
                                marker_color=df['Risk'].apply(lambda x: '#ff4444' if x >= 0.5 else '#00cc44'),
                                text=df['Risk'].apply(lambda x: f'{x:.1%}'), textposition='outside'))
        fig.update_layout(xaxis_range=[0, 1], height=250, xaxis_title="Churn Probability")
        st.plotly_chart(fig, use_container_width=True)
        st.success(f"✅ Best: **{df.iloc[0]['Scenario']}** ({df.iloc[0]['Risk']:.1%})")
