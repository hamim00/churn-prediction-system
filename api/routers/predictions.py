"""
Predictions Router
==================
Endpoints for churn predictions.
"""

from fastapi import APIRouter, HTTPException
from typing import List

from api.schemas.schemas import (
    CustomerFeatures, 
    PredictionResponse, 
    BatchPredictionRequest,
    BatchPredictionResponse,
    RiskLevel
)
from api.services.prediction_service import prediction_service

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
async def predict_churn(customer: CustomerFeatures):
    """
    Predict churn for a single customer.
    
    Accepts customer features and returns:
    - Churn prediction (0/1)
    - Churn probability (0-1)
    - Risk level (Low/Medium/High/Critical)
    - Top 3 reasons explaining the prediction
    """
    if not prediction_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        prediction, probability, risk_level, reasons = prediction_service.predict(customer)
        
        return PredictionResponse(
            customer_id=None,
            churn_prediction=prediction,
            churn_probability=round(probability, 4),
            risk_level=risk_level,
            top_reasons=reasons
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict churn for multiple customers (up to 100).
    
    Returns predictions for all customers with a summary.
    """
    if not prediction_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = prediction_service.predict_batch(request.customers)
        
        predictions = []
        risk_summary = {level.value: 0 for level in RiskLevel}
        
        for i, (pred, prob, risk, reasons) in enumerate(results):
            predictions.append(PredictionResponse(
                customer_id=f"batch_{i}",
                churn_prediction=pred,
                churn_probability=round(prob, 4),
                risk_level=risk,
                top_reasons=reasons
            ))
            risk_summary[risk.value] += 1
        
        return BatchPredictionResponse(
            total=len(predictions),
            predictions=predictions,
            summary=risk_summary
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")
