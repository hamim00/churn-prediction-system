"""
Pydantic Schemas for Request/Response Validation
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class RiskLevel(str, Enum):
    """Risk level categories."""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


# ============================================================================
# Request Schemas
# ============================================================================

class CustomerFeatures(BaseModel):
    """Input features for churn prediction."""
    
    # Demographics
    gender: str = Field(..., description="Customer gender: Male/Female")
    senior_citizen: bool = Field(..., description="Is senior citizen (65+)")
    partner: bool = Field(..., description="Has partner")
    dependents: bool = Field(..., description="Has dependents")
    
    # Account
    tenure_months: int = Field(..., ge=0, le=72, description="Months as customer")
    monthly_charges: float = Field(..., ge=0, description="Monthly charges in $")
    total_charges: float = Field(..., ge=0, description="Total charges to date in $")
    
    # Contract
    contract_type: str = Field(..., description="Month-to-month, One year, Two year")
    payment_method: str = Field(..., description="Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic)")
    paperless_billing: bool = Field(..., description="Uses paperless billing")
    
    # Services
    phone_service: bool = Field(True, description="Has phone service")
    multiple_lines: bool = Field(False, description="Has multiple lines")
    internet_service: str = Field(..., description="DSL, Fiber optic, No")
    online_security: bool = Field(False, description="Has online security")
    online_backup: bool = Field(False, description="Has online backup")
    device_protection: bool = Field(False, description="Has device protection")
    tech_support: bool = Field(False, description="Has tech support")
    streaming_tv: bool = Field(False, description="Has streaming TV")
    streaming_movies: bool = Field(False, description="Has streaming movies")
    
    class Config:
        json_schema_extra = {
            "example": {
                "gender": "Female",
                "senior_citizen": False,
                "partner": True,
                "dependents": False,
                "tenure_months": 12,
                "monthly_charges": 70.35,
                "total_charges": 844.20,
                "contract_type": "Month-to-month",
                "payment_method": "Electronic check",
                "paperless_billing": True,
                "phone_service": True,
                "multiple_lines": False,
                "internet_service": "Fiber optic",
                "online_security": False,
                "online_backup": False,
                "device_protection": False,
                "tech_support": False,
                "streaming_tv": True,
                "streaming_movies": True
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request for batch predictions."""
    customers: List[CustomerFeatures] = Field(..., min_length=1, max_length=100)


# ============================================================================
# Response Schemas
# ============================================================================

class ChurnReason(BaseModel):
    """Explanation for churn prediction."""
    feature: str = Field(..., description="Feature name")
    impact: float = Field(..., description="Impact magnitude")
    direction: str = Field(..., description="increases/decreases churn risk")
    description: str = Field(..., description="Human-readable explanation")


class PredictionResponse(BaseModel):
    """Response for single prediction."""
    customer_id: Optional[str] = Field(None, description="Customer ID if available")
    churn_prediction: int = Field(..., description="0=No Churn, 1=Churn")
    churn_probability: float = Field(..., ge=0, le=1, description="Probability of churn")
    risk_level: RiskLevel = Field(..., description="Risk category")
    top_reasons: List[ChurnReason] = Field(..., description="Top 3 reasons for prediction")
    
    class Config:
        json_schema_extra = {
            "example": {
                "customer_id": None,
                "churn_prediction": 1,
                "churn_probability": 0.73,
                "risk_level": "High",
                "top_reasons": [
                    {
                        "feature": "contract_type",
                        "impact": 0.45,
                        "direction": "increases",
                        "description": "Month-to-month contract increases churn risk"
                    },
                    {
                        "feature": "tenure_months",
                        "impact": 0.32,
                        "direction": "increases",
                        "description": "Short tenure (12 months) increases churn risk"
                    },
                    {
                        "feature": "payment_method",
                        "impact": 0.28,
                        "direction": "increases",
                        "description": "Electronic check payment increases churn risk"
                    }
                ]
            }
        }


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions."""
    total: int = Field(..., description="Total predictions made")
    predictions: List[PredictionResponse]
    summary: Dict[str, int] = Field(..., description="Summary by risk level")


class CustomerResponse(BaseModel):
    """Customer details response."""
    customer_id: str
    gender: str
    senior_citizen: bool
    partner: bool
    dependents: bool
    tenure_months: int
    monthly_charges: float
    total_charges: float
    contract_type: str
    payment_method: str
    churn: Optional[bool] = None


class CustomerWithPrediction(CustomerResponse):
    """Customer details with churn prediction."""
    churn_probability: float
    risk_level: RiskLevel
    top_reasons: List[ChurnReason]


class AtRiskCustomersResponse(BaseModel):
    """Response for at-risk customers list."""
    total: int
    risk_threshold: float
    customers: List[CustomerWithPrediction]


class ModelInfoResponse(BaseModel):
    """Model information response."""
    model_name: str
    version: str
    training_date: str
    n_features: int
    metrics: Dict[str, float]
    feature_names: List[str]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    database_connected: bool
    version: str


class FeatureImportanceResponse(BaseModel):
    """Feature importance response."""
    features: List[Dict[str, Any]]
