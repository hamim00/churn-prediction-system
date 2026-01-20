# 🎯 Customer Churn Prediction System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15+-blue.svg)](https://postgresql.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Machine Learning system for predicting customer churn in telecommunications, featuring a normalized database, ML pipeline with explainability, REST API, and interactive dashboard.

![Dashboard Preview](docs/images/dashboard-preview.png)

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Module Details](#-module-details)
- [API Documentation](#-api-documentation)
- [Model Performance](#-model-performance)
- [Dashboard](#-dashboard)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [License](#-license)

## 🎯 Overview

Customer churn is a critical business metric for telecom companies. This project demonstrates a complete ML pipeline from data ingestion to deployment, showcasing skills in:

- **Database Design** — Normalized PostgreSQL schema (3NF)
- **SQL Analytics** — Complex queries with CTEs, window functions, RFM segmentation
- **Machine Learning** — Feature engineering (61 features), model training, hyperparameter tuning
- **MLOps** — Experiment tracking with MLflow, model versioning
- **API Development** — FastAPI with hybrid endpoints, SHAP explanations
- **Data Visualization** — Interactive Streamlit dashboard with 5 pages

### Business Impact

- Identify customers with **>75% churn probability** for immediate intervention
- Understand **key churn drivers** through SHAP explainability
- Simulate **retention strategies** with what-if analysis
- Track **monthly revenue at risk** from potential churners

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🗄️ **Normalized Database** | 3NF PostgreSQL schema with views for analytics |
| 📊 **SQL Analytics** | 15+ queries including RFM segmentation, cohort analysis |
| 🤖 **ML Pipeline** | Logistic Regression, Random Forest, XGBoost with 61 engineered features |
| 📈 **MLflow Tracking** | Experiment tracking, model registry, artifact storage |
| 🔍 **SHAP Explainability** | Feature importance and individual prediction explanations |
| 🚀 **FastAPI Backend** | RESTful API with hybrid database-ML endpoints |
| 📱 **Streamlit Dashboard** | 5-page interactive dashboard with real-time predictions |
| 🐳 **Docker Ready** | Containerized deployment with Docker Compose |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PRESENTATION LAYER                          │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                   Streamlit Dashboard                        │   │
│  │  📊 Executive Summary │ 👥 Risk List │ 🔮 What-if Simulator │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                           API LAYER                                 │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                      FastAPI Backend                         │   │
│  │  /predict │ /customers │ /at-risk │ /model/info │ /health   │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                          │                 │
                          ▼                 ▼
┌──────────────────────────────┐  ┌──────────────────────────────────┐
│        ML SERVICE            │  │         DATA LAYER               │
│  ┌────────────────────────┐  │  │  ┌────────────────────────────┐  │
│  │   Prediction Service   │  │  │  │      PostgreSQL DB         │  │
│  │  • Feature Engineering │  │  │  │  • customers               │  │
│  │  • Model Inference     │  │  │  │  • customer_contracts      │  │
│  │  • SHAP Explanations   │  │  │  │  • customer_services       │  │
│  └────────────────────────┘  │  │  │  • Views & Analytics       │  │
│  ┌────────────────────────┐  │  │  └────────────────────────────┘  │
│  │   MLflow Tracking      │  │  └──────────────────────────────────┘
│  │  • Experiments         │  │
│  │  • Model Registry      │  │
│  └────────────────────────┘  │
└──────────────────────────────┘
```

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| **Language** | Python 3.10+ |
| **Database** | PostgreSQL 15, SQLAlchemy |
| **ML/Data** | scikit-learn, XGBoost, pandas, numpy, SHAP |
| **MLOps** | MLflow |
| **API** | FastAPI, Pydantic, Uvicorn |
| **Dashboard** | Streamlit, Plotly |
| **Containerization** | Docker, Docker Compose |
| **Version Control** | Git, GitHub |

## 📁 Project Structure

```
churn-prediction-system/
├── 📂 api/                          # FastAPI application
│   ├── main.py                      # App entry point & lifespan
│   ├── core/
│   │   └── config.py                # Settings & configuration
│   ├── routers/
│   │   ├── predictions.py           # /predict endpoints
│   │   ├── customers.py             # /customers endpoints
│   │   ├── model.py                 # /model endpoints
│   │   └── health.py                # /health endpoint
│   ├── schemas/
│   │   └── schemas.py               # Pydantic models
│   └── services/
│       ├── prediction_service.py    # ML inference & SHAP
│       └── customer_service.py      # Database operations
│
├── 📂 dashboard/                    # Streamlit application
│   ├── app.py                       # Landing page
│   ├── utils/
│   │   └── api_client.py            # API communication
│   └── pages/
│       ├── 1_📊_Executive_Summary.py
│       ├── 2_👥_Customer_Risk_List.py
│       ├── 3_🔍_Customer_Deepdive.py
│       ├── 4_📈_Model_Insights.py
│       └── 5_🔮_Whatif_Simulator.py
│
├── 📂 database/                     # Database setup
│   ├── schema.sql                   # Table definitions
│   ├── seed_data.py                 # Data loading script
│   └── queries/
│       └── churn_analytics.sql      # 15 analytics queries
│
├── 📂 src/                          # ML pipeline source
│   ├── features/
│   │   └── feature_engineering.py   # 61 feature transformations
│   ├── models/
│   │   ├── model_training.py        # Training pipeline
│   │   └── prediction.py            # Inference utilities
│   └── utils/
│       └── database.py              # DB connection manager
│
├── 📂 models/                       # Trained model artifacts
│   ├── best_model.joblib            # Serialized model
│   ├── scaler.joblib                # Feature scaler
│   ├── feature_names.json           # Feature list
│   ├── model_metadata.json          # Training metadata
│   └── *.png                        # Evaluation plots
│
├── 📂 data/
│   ├── raw/                         # Original dataset
│   └── processed/                   # Engineered features
│
├── 📂 mlruns/                       # MLflow experiment tracking
├── 📂 notebooks/                    # Jupyter notebooks (EDA)
├── 📂 tests/                        # Unit tests
│
├── .env                             # Environment variables
├── .gitignore
├── requirements.txt
├── run_pipeline.py                  # ML pipeline runner
├── docker-compose.yml               # Container orchestration
├── Dockerfile.api                   # API container
├── Dockerfile.dashboard             # Dashboard container
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- PostgreSQL 15+
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/hamim00/churn-prediction-system.git
   cd churn-prediction-system
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up PostgreSQL database**
   ```bash
   # Create database and user
   psql -U postgres
   CREATE DATABASE churn_db;
   CREATE USER churn_admin WITH PASSWORD 'churn_password_123';
   GRANT ALL PRIVILEGES ON DATABASE churn_db TO churn_admin;
   \q
   
   # Run schema
   psql -U churn_admin -d churn_db -f database/schema.sql
   
   # Seed data
   python database/seed_data.py
   ```

5. **Configure environment**
   ```bash
   # Create .env file
   cp .env.example .env
   
   # Edit with your settings
   DB_HOST=localhost
   DB_PORT=5433
   DB_NAME=churn_db
   DB_USER=churn_admin
   DB_PASSWORD=churn_password_123
   MODEL_PATH=models
   ```

6. **Train the model (optional - pre-trained model included)**
   ```bash
   python run_pipeline.py
   ```

### Running the Application

**Terminal 1 - Start the API:**
```bash
uvicorn api.main:app --reload --port 8000
```

**Terminal 2 - Start the Dashboard:**
```bash
cd dashboard
streamlit run app.py
```

**Access:**
- 🔗 API Docs: http://localhost:8000/docs
- 🔗 Dashboard: http://localhost:8501

## 📦 Module Details

### Module 1: Database Layer ✅

Normalized PostgreSQL schema following 3NF:

| Table | Description |
|-------|-------------|
| `customers` | Core customer demographics |
| `customer_contracts` | Contract and billing info |
| `customer_services` | Service subscriptions (many-to-many) |
| `contract_types` | Lookup: Month-to-month, One year, Two year |
| `service_types` | Lookup: 9 service types |

**Views:**
- `v_customer_full` — Denormalized view for ML
- `v_customer_services_summary` — Pivoted services
- `v_churn_summary` — Aggregated churn metrics

### Module 2: SQL Analytics ✅

15 analytical queries demonstrating:

```sql
-- Example: RFM Segmentation
WITH rfm AS (
    SELECT customer_id,
           NTILE(4) OVER (ORDER BY tenure_months DESC) as recency_score,
           NTILE(4) OVER (ORDER BY total_charges DESC) as monetary_score
    FROM v_customer_full
)
SELECT * FROM rfm WHERE monetary_score = 4 AND recency_score <= 2;
```

**Query Categories:**
- Basic aggregations and JOINs
- Window functions (NTILE, RANK, LAG)
- CTEs and subqueries
- RFM customer segmentation
- Cohort retention analysis

### Module 3: ML Pipeline ✅

**Feature Engineering (61 features):**

| Category | Features | Examples |
|----------|----------|----------|
| Demographics | 7 | `is_senior`, `has_family`, `senior_alone` |
| Services | 14 | `total_services`, `service_diversity`, `premium_service_count` |
| Contract | 3 | `contract_length`, `is_month_to_month` |
| Payment | 6 | `has_auto_payment`, `payment_electronic_check` |
| Tenure | 7 | `tenure_years`, `is_new_customer`, `tenure_bucket` |
| Financial | 8 | `avg_monthly_spend`, `charge_per_service`, `estimated_clv` |
| Risk Scores | 9 | `risk_score`, `risk_factors_count`, `high_risk_flag` |
| Interactions | 5 | `tenure_contract_interaction`, `mtm_high_value` |

**Models Trained:**

| Model | ROC-AUC | Recall | Precision | F1 |
|-------|---------|--------|-----------|-----|
| Logistic Regression | **0.8421** | 0.7914 | 0.5166 | 0.6251 |
| Random Forest | 0.8156 | 0.7234 | 0.4892 | 0.5837 |
| XGBoost | 0.8289 | 0.7521 | 0.5023 | 0.6021 |

**Best Model:** Logistic Regression (selected for interpretability + performance)

### Module 4: FastAPI Service ✅

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/predict` | Predict from JSON input |
| `POST` | `/api/v1/predict/batch` | Batch predictions |
| `GET` | `/api/v1/customers/{id}` | Get customer details |
| `GET` | `/api/v1/customers/{id}/predict` | Hybrid: DB → Predict |
| `GET` | `/api/v1/customers/at-risk/list` | High-risk customer list |
| `GET` | `/api/v1/model/info` | Model metadata & metrics |
| `GET` | `/health` | Health check |

### Module 5: Streamlit Dashboard ✅

| Page | Features |
|------|----------|
| 📊 **Executive Summary** | KPIs, risk distribution, revenue at risk, top churners |
| 👥 **Customer Risk List** | Filterable table, CSV export, quick lookup |
| 🔍 **Customer Deep-dive** | Individual analysis, SHAP waterfall, recommendations |
| 📈 **Model Insights** | Performance metrics, feature importance, radar chart |
| 🔮 **What-if Simulator** | Interactive predictions, scenario comparison |

## 📖 API Documentation

### Predict Churn

```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "senior_citizen": false,
    "partner": false,
    "dependents": false,
    "tenure_months": 12,
    "monthly_charges": 70.0,
    "total_charges": 840.0,
    "contract_type": "Month-to-month",
    "payment_method": "Electronic check",
    "paperless_billing": true,
    "phone_service": true,
    "multiple_lines": false,
    "internet_service": "Fiber optic",
    "online_security": false,
    "online_backup": false,
    "device_protection": false,
    "tech_support": false,
    "streaming_tv": false,
    "streaming_movies": false
  }'
```

**Response:**
```json
{
  "customer_id": "",
  "churn_prediction": 1,
  "churn_probability": 0.7939,
  "risk_level": "Critical",
  "top_reasons": [
    {
      "feature": "contract_length",
      "impact": 1.376,
      "direction": "increases",
      "description": "Contract length increases churn risk"
    }
  ]
}
```

## 📊 Model Performance

### Current Metrics vs Targets

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| ROC-AUC | 0.8421 | > 0.80 | ✅ |
| Recall | 0.7914 | > 0.75 | ✅ |
| Precision | 0.5166 | > 0.60 | ⚠️ |
| F1 Score | 0.6251 | > 0.65 | ⚠️ |

### Key Insights from SHAP

**Top Churn Drivers:**
1. Month-to-month contract (+)
2. Short tenure (+)
3. Electronic check payment (+)
4. Fiber optic without support services (+)
5. High monthly charges (+)

**Retention Factors:**
1. Long-term contract (−)
2. Auto-payment setup (−)
3. Tech support subscription (−)
4. Online security bundle (−)

## 🖥️ Dashboard

### Executive Summary
![Executive Summary](docs/images/executive-summary.png)

### What-if Simulator
![What-if Simulator](docs/images/whatif-simulator.png)

## 🔮 Future Improvements

### Model Improvements (Planned)
- [ ] Threshold optimization to improve Precision (target: 0.60)
- [ ] Hyperparameter tuning with Optuna
- [ ] Ensemble methods (stacking)
- [ ] Cost-sensitive learning

### Module 6: Deployment (Planned)
- [ ] Dockerize API and Dashboard
- [ ] Docker Compose orchestration
- [ ] Deploy to cloud:
  - PostgreSQL → Neon.tech (free)
  - FastAPI → Render.com (free)
  - Dashboard → Streamlit Cloud (free)
- [ ] CI/CD with GitHub Actions
- [ ] Monitoring with Prometheus/Grafana

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Mahmudul Hasan Hamim**

- GitHub: [@hamim00](https://github.com/hamim00)
- LinkedIn: [Connect with me](https://linkedin.com/in/your-profile)

## 🙏 Acknowledgments

- Dataset: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle
- Inspired by real-world telecom churn prediction systems

---

<p align="center">
  Made with ❤️ for the ML community
</p>
