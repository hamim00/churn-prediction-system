"""
Health Router
=============
Health check and readiness endpoints.
"""

from fastapi import APIRouter
from datetime import datetime

from api.schemas.schemas import HealthResponse
from api.services.prediction_service import prediction_service
from api.services.customer_service import customer_service
from api.core.config import settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Basic health check endpoint.
    
    Returns status of model and database connections.
    """
    return HealthResponse(
        status="healthy" if prediction_service.is_loaded else "degraded",
        model_loaded=prediction_service.is_loaded,
        database_connected=customer_service.is_connected,
        version=settings.VERSION
    )


@router.get("/health/ready")
async def readiness_check():
    """
    Readiness probe for Kubernetes/Docker.
    
    Returns 200 if service is ready to accept traffic.
    """
    is_ready = prediction_service.is_loaded
    
    return {
        "ready": is_ready,
        "timestamp": datetime.utcnow().isoformat(),
        "checks": {
            "model": prediction_service.is_loaded,
            "database": customer_service.is_connected
        }
    }


@router.get("/health/live")
async def liveness_check():
    """
    Liveness probe for Kubernetes/Docker.
    
    Returns 200 if service is alive.
    """
    return {
        "alive": True,
        "timestamp": datetime.utcnow().isoformat()
    }
