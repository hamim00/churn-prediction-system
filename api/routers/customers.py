"""
Customers Router
================
Endpoints for customer data and predictions from database.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional

from api.schemas.schemas import (
    CustomerFeatures,
    CustomerResponse,
    CustomerWithPrediction,
    PredictionResponse,
    AtRiskCustomersResponse
)
from api.services.customer_service import customer_service
from api.services.prediction_service import prediction_service

router = APIRouter()


def _db_to_customer_features(db_record: dict) -> CustomerFeatures:
    """Convert database record to CustomerFeatures."""
    # Determine internet service type
    if db_record.get('has_fiber'):
        internet_service = 'Fiber optic'
    elif db_record.get('has_dsl'):
        internet_service = 'DSL'
    else:
        internet_service = 'No'
    
    return CustomerFeatures(
        gender=db_record['gender'],
        senior_citizen=bool(db_record['senior_citizen']),
        partner=bool(db_record['partner']),
        dependents=bool(db_record['dependents']),
        tenure_months=db_record['tenure_months'],
        monthly_charges=float(db_record['monthly_charges']),
        total_charges=float(db_record['total_charges']),
        contract_type=db_record['contract_type'],
        payment_method=db_record['payment_method'],
        paperless_billing=bool(db_record.get('paperless_billing', False)),
        phone_service=bool(db_record.get('has_phone_service', True)),
        multiple_lines=bool(db_record.get('has_multiple_lines', False)),
        internet_service=internet_service,
        online_security=bool(db_record.get('has_online_security', False)),
        online_backup=bool(db_record.get('has_online_backup', False)),
        device_protection=bool(db_record.get('has_device_protection', False)),
        tech_support=bool(db_record.get('has_tech_support', False)),
        streaming_tv=bool(db_record.get('has_streaming_tv', False)),
        streaming_movies=bool(db_record.get('has_streaming_movies', False))
    )


@router.get("/customers/{customer_id}", response_model=CustomerResponse)
async def get_customer(customer_id: str):
    """
    Get customer details by ID.
    """
    if not customer_service.is_connected:
        raise HTTPException(status_code=503, detail="Database not connected")
    
    customer = customer_service.get_customer_by_id(customer_id)
    if not customer:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")
    
    return CustomerResponse(
        customer_id=customer['customer_id'],
        gender=customer['gender'],
        senior_citizen=bool(customer['senior_citizen']),
        partner=bool(customer['partner']),
        dependents=bool(customer['dependents']),
        tenure_months=customer['tenure_months'],
        monthly_charges=float(customer['monthly_charges']),
        total_charges=float(customer['total_charges']),
        contract_type=customer['contract_type'],
        payment_method=customer['payment_method'],
        churn=bool(customer['churn']) if customer.get('churn') is not None else None
    )


@router.get("/customers/{customer_id}/predict", response_model=PredictionResponse)
async def predict_customer(customer_id: str):
    """
    Fetch customer from database and predict churn.
    
    This is the hybrid endpoint - fetches customer by ID and returns prediction.
    """
    if not customer_service.is_connected:
        raise HTTPException(status_code=503, detail="Database not connected")
    
    if not prediction_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Fetch customer
    db_customer = customer_service.get_customer_by_id(customer_id)
    if not db_customer:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")
    
    # Convert to features
    customer_features = _db_to_customer_features(db_customer)
    
    # Predict
    try:
        prediction, probability, risk_level, reasons = prediction_service.predict(customer_features)
        
        return PredictionResponse(
            customer_id=customer_id,
            churn_prediction=prediction,
            churn_probability=round(probability, 4),
            risk_level=risk_level,
            top_reasons=reasons
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("/customers", response_model=List[CustomerResponse])
async def list_customers(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0)
):
    """
    List customers with pagination.
    """
    if not customer_service.is_connected:
        raise HTTPException(status_code=503, detail="Database not connected")
    
    customers = customer_service.get_customers(limit=limit, offset=offset)
    
    return [
        CustomerResponse(
            customer_id=c['customer_id'],
            gender=c['gender'],
            senior_citizen=bool(c['senior_citizen']),
            partner=bool(c['partner']),
            dependents=bool(c['dependents']),
            tenure_months=c['tenure_months'],
            monthly_charges=float(c['monthly_charges']),
            total_charges=float(c['total_charges']),
            contract_type=c['contract_type'],
            payment_method=c['payment_method'],
            churn=bool(c['churn']) if c.get('churn') is not None else None
        )
        for c in customers
    ]


@router.get("/customers/at-risk/list", response_model=AtRiskCustomersResponse)
async def get_at_risk_customers(
    limit: int = Query(default=20, ge=1, le=100),
    threshold: float = Query(default=0.5, ge=0.0, le=1.0)
):
    """
    Get list of high-risk customers with predictions.
    
    Fetches customers with high-risk profiles (month-to-month, new) 
    and returns their churn predictions.
    """
    if not customer_service.is_connected:
        raise HTTPException(status_code=503, detail="Database not connected")
    
    if not prediction_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Fetch at-risk customers
    db_customers = customer_service.get_at_risk_customers(limit=limit)
    
    results = []
    for db_customer in db_customers:
        try:
            customer_features = _db_to_customer_features(db_customer)
            prediction, probability, risk_level, reasons = prediction_service.predict(customer_features)
            
            if probability >= threshold:
                results.append(CustomerWithPrediction(
                    customer_id=db_customer['customer_id'],
                    gender=db_customer['gender'],
                    senior_citizen=bool(db_customer['senior_citizen']),
                    partner=bool(db_customer['partner']),
                    dependents=bool(db_customer['dependents']),
                    tenure_months=db_customer['tenure_months'],
                    monthly_charges=float(db_customer['monthly_charges']),
                    total_charges=float(db_customer['total_charges']),
                    contract_type=db_customer['contract_type'],
                    payment_method=db_customer['payment_method'],
                    churn=bool(db_customer['churn']) if db_customer.get('churn') is not None else None,
                    churn_probability=round(probability, 4),
                    risk_level=risk_level,
                    top_reasons=reasons
                ))
        except Exception:
            continue
    
    return AtRiskCustomersResponse(
        total=len(results),
        risk_threshold=threshold,
        customers=results
    )


@router.get("/customers/search/{search_term}")
async def search_customers(search_term: str, limit: int = Query(default=10, ge=1, le=50)):
    """
    Search customers by ID.
    """
    if not customer_service.is_connected:
        raise HTTPException(status_code=503, detail="Database not connected")
    
    customers = customer_service.search_customers(search_term, limit)
    return customers
