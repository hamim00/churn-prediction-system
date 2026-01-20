"""
API Client for Dashboard
========================
Handles all communication with the FastAPI backend.
"""

import requests
from typing import Dict, List, Optional, Any
import streamlit as st


class APIClient:
    """Client for interacting with the Churn Prediction API."""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url
        self.timeout = 30
    
    def _get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Make GET request to API."""
        try:
            response = requests.get(
                f"{self.base_url}{endpoint}",
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            return None
        except requests.exceptions.RequestException as e:
            st.error(f"API Error: {e}")
            return None
    
    def _post(self, endpoint: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Make POST request to API."""
        try:
            response = requests.post(
                f"{self.base_url}{endpoint}",
                json=data,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            return None
        except requests.exceptions.RequestException as e:
            st.error(f"API Error: {e}")
            return None
    
    # Health & Info
    def health_check(self) -> Optional[Dict[str, Any]]:
        """Check API health status."""
        return self._get("/health")
    
    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """Get model metadata and metrics."""
        return self._get("/api/v1/model/info")
    
    # Customer endpoints
    def get_customer(self, customer_id: str) -> Optional[Dict[str, Any]]:
        """Get customer by ID."""
        return self._get(f"/api/v1/customers/{customer_id}")
    
    def get_customer_prediction(self, customer_id: str) -> Optional[Dict[str, Any]]:
        """Get prediction for existing customer (hybrid endpoint)."""
        return self._get(f"/api/v1/customers/{customer_id}/predict")
    
    def get_at_risk_customers(
        self, 
        min_probability: float = 0.5, 
        limit: int = 100
    ) -> Optional[List[Dict[str, Any]]]:
        """Get list of at-risk customers."""
        # API uses 'threshold' parameter, max limit is 100
        response = self._get(
            "/api/v1/customers/at-risk/list",
            params={"threshold": min_probability, "limit": min(limit, 100)}
        )
        # Extract customers array from nested response
        if response and isinstance(response, dict):
            customers = response.get("customers", [])
            return customers if isinstance(customers, list) else []
        return None
    
    # Prediction endpoints
    def predict(self, customer_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Make prediction from customer features."""
        return self._post("/api/v1/predict", customer_data)
    
    def predict_batch(self, customers: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Make batch predictions."""
        return self._post("/api/v1/predict/batch", {"customers": customers})


@st.cache_resource
def get_api_client() -> APIClient:
    """Get cached API client instance."""
    return APIClient()
