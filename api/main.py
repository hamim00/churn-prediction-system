"""
Customer Churn Prediction API
==============================
FastAPI service for churn predictions with SHAP explanations.

Author: Mahmudul Hasan
Project: Customer Churn Prediction System
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from api.routers import predictions, customers, model, health
from api.core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup: Load model
    print("🚀 Starting Churn Prediction API...")
    from api.services.prediction_service import prediction_service
    prediction_service.load_model()
    print("✅ Model loaded successfully")
    yield
    # Shutdown
    print("👋 Shutting down API...")


app = FastAPI(
    title=settings.APP_NAME,
    description="""
## Customer Churn Prediction API

This API provides churn predictions for telecom customers using machine learning.

### Features:
- **Single Prediction**: Predict churn for one customer
- **Batch Prediction**: Predict churn for multiple customers
- **Customer Lookup**: Fetch customer from database and predict
- **Explainability**: SHAP-based explanations for predictions
- **At-Risk Customers**: List high-risk customers

### Model Info:
- Algorithm: Logistic Regression
- Features: 61 engineered features
- ROC-AUC: 0.84
    """,
    version=settings.VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(predictions.router, prefix="/api/v1", tags=["Predictions"])
app.include_router(customers.router, prefix="/api/v1", tags=["Customers"])
app.include_router(model.router, prefix="/api/v1", tags=["Model"])


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Customer Churn Prediction API",
        "version": settings.VERSION,
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
