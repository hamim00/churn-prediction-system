"""
Customer Risk List Page
=======================
Searchable and filterable table of all customers with risk scores.
"""

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Customer Risk List", page_icon="👥", layout="wide")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.api_client import get_api_client

st.markdown("## 👥 Customer Risk List")
st.markdown("Search and filter customers by risk level")

client = get_api_client()

# Filters
st.markdown("### 🔍 Filters")
col1, col2, col3, col4 = st.columns(4)

with col1:
    min_prob = st.slider(
        "Min Churn Probability",
        min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        format="%.0f%%"
    )

with col2:
    risk_filter = st.multiselect(
        "Risk Level",
        options=["Critical", "High", "Medium", "Low"],
        default=["Critical", "High", "Medium", "Low"]
    )

with col3:
    max_results = st.selectbox("Max Results", options=[10, 25, 50, 75, 100], index=4)

with col4:
    search_id = st.text_input("Search Customer ID", placeholder="e.g., 7590-VHVEG")

st.markdown("---")

# Fetch data
with st.spinner("Loading customers..."):
    customers = client.get_at_risk_customers(min_probability=min_prob, limit=max_results)

if not customers:
    st.warning("⚠️ No customers found or API unavailable.")
    st.stop()

df = pd.DataFrame(customers)

# Apply filters
if risk_filter:
    risk_filter_upper = [r.upper() for r in risk_filter]
    df = df[df['risk_level'].str.upper().isin(risk_filter_upper)]

if search_id:
    df = df[df['customer_id'].str.contains(search_id, case=False, na=False)]

# Stats
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Showing", f"{len(df):,} customers")
with col2:
    critical = len(df[df['risk_level'].str.upper() == 'CRITICAL'])
    st.metric("Critical", f"{critical:,}")
with col3:
    high = len(df[df['risk_level'].str.upper() == 'HIGH'])
    st.metric("High", f"{high:,}")
with col4:
    avg_prob = df['churn_probability'].mean() if len(df) > 0 else 0
    st.metric("Avg Probability", f"{avg_prob:.1%}")

st.markdown("---")

# Display table
display_df = df.copy()
display_df['churn_probability'] = display_df['churn_probability'].apply(lambda x: f"{x:.1%}")

display_cols = ['customer_id', 'churn_probability', 'risk_level']
for col in ['contract_type', 'tenure_months', 'monthly_charges', 'payment_method']:
    if col in display_df.columns:
        display_cols.append(col)

st.dataframe(
    display_df[display_cols],
    use_container_width=True,
    hide_index=True,
    height=400,
    column_config={
        "customer_id": "Customer ID",
        "churn_probability": "Churn Risk",
        "risk_level": "Risk Level",
        "contract_type": "Contract",
        "tenure_months": "Tenure (mo)",
        "monthly_charges": st.column_config.NumberColumn("Monthly $", format="$%.2f"),
        "payment_method": "Payment"
    }
)

# Export
st.markdown("---")
col1, col2 = st.columns([1, 3])
with col1:
    csv = df.to_csv(index=False)
    st.download_button("📥 Download CSV", data=csv, file_name="at_risk_customers.csv", mime="text/csv")
with col2:
    st.caption(f"Export {len(df):,} customers with current filters")

# Quick lookup
st.markdown("---")
st.markdown("### 🔎 Quick Customer Lookup")
lookup_id = st.text_input("Enter Customer ID:", placeholder="e.g., 7590-VHVEG", key="lookup")

if lookup_id:
    with st.spinner(f"Looking up {lookup_id}..."):
        customer = client.get_customer(lookup_id)
        prediction = client.get_customer_prediction(lookup_id)
    
    if customer and prediction:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Customer Details")
            st.json(customer)
        with col2:
            st.markdown("#### Prediction")
            prob = prediction.get('churn_probability', 0)
            risk = prediction.get('risk_level', 'UNKNOWN').upper()
            
            if risk == 'CRITICAL':
                st.error(f"🚨 **CRITICAL** - {prob:.1%}")
            elif risk == 'HIGH':
                st.warning(f"⚠️ **HIGH** - {prob:.1%}")
            elif risk == 'MEDIUM':
                st.info(f"📊 **MEDIUM** - {prob:.1%}")
            else:
                st.success(f"✅ **LOW** - {prob:.1%}")
    else:
        st.error(f"Customer '{lookup_id}' not found")
