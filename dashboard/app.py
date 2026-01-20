"""
Customer Churn Prediction Dashboard
====================================
Main entry point - redirects to Executive Summary.
"""

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    div[data-testid="metric-container"] {
        background-color: #f0f2f6;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 10px;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
    }
    .risk-critical { color: #ff4444; font-weight: bold; }
    .risk-high { color: #ff8800; font-weight: bold; }
    .risk-medium { color: #ffcc00; font-weight: bold; }
    .risk-low { color: #00cc44; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Main landing page
st.markdown("# 🎯 Customer Churn Prediction Dashboard")
st.markdown("---")

st.markdown("""
Welcome to the **Customer Churn Prediction System**. Use the sidebar to navigate:

| Page | Description |
|------|-------------|
| 📊 **Executive Summary** | KPIs, charts, and high-level insights |
| 👥 **Customer Risk List** | Searchable table of at-risk customers |
| 🔍 **Customer Deep-dive** | Individual customer analysis with SHAP |
| 📈 **Model Insights** | Model performance and feature importance |
| 🔮 **What-if Simulator** | Test how changes affect churn risk |

👈 **Select a page from the sidebar to get started!**
""")

# API Status in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ API Status")

from utils.api_client import get_api_client
client = get_api_client()
health = client.health_check()

if health and health.get("status") == "healthy":
    st.sidebar.success("✅ API Connected")
    st.sidebar.caption(f"Version: {health.get('version', 'N/A')}")
else:
    st.sidebar.error("❌ API Disconnected")
    st.sidebar.code("uvicorn api.main:app --port 8000")

st.sidebar.markdown("---")
st.sidebar.caption("v1.0.0 | Churn Prediction System")
