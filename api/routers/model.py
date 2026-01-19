"""
Model Router
============
Endpoints for model information and feature importance.
"""

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
import pandas as pd
from pathlib import Path

from api.schemas.schemas import ModelInfoResponse, FeatureImportanceResponse
from api.services.prediction_service import prediction_service
from api.core.config import settings

router = APIRouter()


@router.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """
    Get information about the loaded model.
    
    Returns model name, metrics, and feature list.
    """
    if not prediction_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    info = prediction_service.get_model_info()
    
    return ModelInfoResponse(
        model_name=info['model_name'],
        version=info['version'],
        training_date=info['training_date'],
        n_features=info['n_features'],
        metrics=info['metrics'],
        feature_names=info['feature_names']
    )


@router.get("/model/features")
async def get_required_features():
    """
    Get list of required features for prediction.
    
    Useful for clients to know what data to send.
    """
    return {
        "required_features": [
            {
                "name": "gender",
                "type": "string",
                "description": "Customer gender",
                "values": ["Male", "Female"]
            },
            {
                "name": "senior_citizen",
                "type": "boolean",
                "description": "Is senior citizen (65+)"
            },
            {
                "name": "partner",
                "type": "boolean",
                "description": "Has partner"
            },
            {
                "name": "dependents",
                "type": "boolean",
                "description": "Has dependents"
            },
            {
                "name": "tenure_months",
                "type": "integer",
                "description": "Months as customer",
                "range": [0, 72]
            },
            {
                "name": "monthly_charges",
                "type": "float",
                "description": "Monthly charges in $"
            },
            {
                "name": "total_charges",
                "type": "float",
                "description": "Total charges to date in $"
            },
            {
                "name": "contract_type",
                "type": "string",
                "description": "Contract type",
                "values": ["Month-to-month", "One year", "Two year"]
            },
            {
                "name": "payment_method",
                "type": "string",
                "description": "Payment method",
                "values": ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
            },
            {
                "name": "paperless_billing",
                "type": "boolean",
                "description": "Uses paperless billing"
            },
            {
                "name": "phone_service",
                "type": "boolean",
                "description": "Has phone service"
            },
            {
                "name": "multiple_lines",
                "type": "boolean",
                "description": "Has multiple phone lines"
            },
            {
                "name": "internet_service",
                "type": "string",
                "description": "Internet service type",
                "values": ["DSL", "Fiber optic", "No"]
            },
            {
                "name": "online_security",
                "type": "boolean",
                "description": "Has online security add-on"
            },
            {
                "name": "online_backup",
                "type": "boolean",
                "description": "Has online backup add-on"
            },
            {
                "name": "device_protection",
                "type": "boolean",
                "description": "Has device protection"
            },
            {
                "name": "tech_support",
                "type": "boolean",
                "description": "Has tech support add-on"
            },
            {
                "name": "streaming_tv",
                "type": "boolean",
                "description": "Has streaming TV"
            },
            {
                "name": "streaming_movies",
                "type": "boolean",
                "description": "Has streaming movies"
            }
        ]
    }


@router.get("/model/importance", response_model=FeatureImportanceResponse)
async def get_feature_importance():
    """
    Get feature importance from saved CSV.
    """
    importance_file = Path(settings.MODEL_PATH) / "feature_importance.csv"
    
    if not importance_file.exists():
        raise HTTPException(status_code=404, detail="Feature importance file not found")
    
    try:
        df = pd.read_csv(importance_file)
        features = df.head(20).to_dict('records')  # Top 20
        return FeatureImportanceResponse(features=features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading feature importance: {str(e)}")


@router.get("/model/metrics")
async def get_model_metrics():
    """
    Get detailed model performance metrics.
    """
    if not prediction_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    info = prediction_service.get_model_info()
    metrics = info.get('metrics', {})
    
    return {
        "model_name": info['model_name'],
        "metrics": {
            "roc_auc": metrics.get('roc_auc', 'N/A'),
            "recall": metrics.get('recall', 'N/A'),
            "precision": metrics.get('precision', 'N/A'),
            "f1": metrics.get('f1', 'N/A'),
            "accuracy": metrics.get('accuracy', 'N/A')
        },
        "targets": {
            "roc_auc": {"value": 0.80, "met": metrics.get('roc_auc', 0) >= 0.80},
            "recall": {"value": 0.75, "met": metrics.get('recall', 0) >= 0.75},
            "precision": {"value": 0.60, "met": metrics.get('precision', 0) >= 0.60},
            "f1": {"value": 0.65, "met": metrics.get('f1', 0) >= 0.65}
        }
    }
