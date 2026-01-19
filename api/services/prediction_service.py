"""
Prediction Service
==================
Handles model loading, predictions, and SHAP explanations.
"""

import numpy as np
import pandas as pd
import joblib
import json
import shap
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings

from api.core.config import settings
from api.schemas.schemas import CustomerFeatures, ChurnReason, RiskLevel

warnings.filterwarnings('ignore')


class PredictionService:
    """Service for making churn predictions."""
    
    def __init__(self) -> None:
        self.model: Any = None
        self.scaler: Any = None
        self.feature_names: List[str] = []
        self.metadata: Dict[str, Any] = {}
        self.explainer: Any = None
        self._is_loaded: bool = False
    
    def load_model(self, model_path: Optional[str] = None) -> None:
        """Load model and artifacts."""
        if model_path is None:
            model_path = settings.MODEL_PATH
        
        model_dir = Path(model_path)
        print(f"📂 Looking for model in: {model_dir.absolute()}")
        
        # Check if directory exists
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        # List all files in directory
        all_files = list(model_dir.iterdir())
        print(f"📄 Files found: {[f.name for f in all_files]}")
        
        # Load model - find any joblib file with best_model in name
        model_files = list(model_dir.glob("best_model*.joblib"))
        if not model_files:
            model_files = list(model_dir.glob("*model*.joblib"))
        if not model_files:
            model_files = [f for f in all_files if f.suffix == '.joblib' and 'scaler' not in f.name]
        
        if not model_files:
            raise FileNotFoundError(f"No model file found in {model_dir}. Available files: {[f.name for f in all_files]}")
        
        self.model = joblib.load(model_files[0])
        print(f"✅ Model loaded: {model_files[0].name}")
        
        # Load scaler
        scaler_file = model_dir / "scaler.joblib"
        if scaler_file.exists():
            self.scaler = joblib.load(scaler_file)
            print("✅ Scaler loaded")
        
        # Load feature names
        features_file = model_dir / "feature_names.json"
        if features_file.exists():
            with open(features_file, 'r') as f:
                self.feature_names = json.load(f)
            print(f"✅ Feature names loaded ({len(self.feature_names)} features)")
        
        # Load metadata
        metadata_file = model_dir / "model_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
            print("✅ Metadata loaded")
        
        # Initialize SHAP explainer
        if len(self.feature_names) > 0:
            try:
                self.explainer = shap.LinearExplainer(self.model, np.zeros((1, len(self.feature_names))))
                print("✅ SHAP explainer initialized")
            except Exception as e:
                print(f"⚠️ SHAP explainer not initialized: {e}")
                self.explainer = None
        
        self._is_loaded = True
    
    @property
    def is_loaded(self) -> bool:
        return self._is_loaded
    
    def _get_risk_level(self, probability: float) -> RiskLevel:
        """Convert probability to risk level."""
        if probability >= 0.75:
            return RiskLevel.CRITICAL
        elif probability >= 0.50:
            return RiskLevel.HIGH
        elif probability >= 0.25:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _engineer_features(self, customer: CustomerFeatures) -> pd.DataFrame:
        """Convert CustomerFeatures to model features."""
        
        features: Dict[str, Any] = {}
        
        # Demographics
        features['gender_male'] = 1 if customer.gender == 'Male' else 0
        features['is_senior'] = int(customer.senior_citizen)
        features['has_partner'] = int(customer.partner)
        features['has_dependents'] = int(customer.dependents)
        features['has_family'] = int(customer.partner or customer.dependents)
        features['family_size_proxy'] = features['has_partner'] + features['has_dependents']
        features['senior_alone'] = int(features['is_senior'] == 1 and features['has_family'] == 0)
        
        # Services
        features['phone_service_flag'] = int(customer.phone_service)
        features['multiple_lines_flag'] = int(customer.multiple_lines)
        features['has_internet'] = 1 if customer.internet_service != 'No' else 0
        features['internet_fiber'] = 1 if customer.internet_service == 'Fiber optic' else 0
        features['internet_dsl'] = 1 if customer.internet_service == 'DSL' else 0
        features['online_security_flag'] = int(customer.online_security)
        features['online_backup_flag'] = int(customer.online_backup)
        features['device_protection_flag'] = int(customer.device_protection)
        features['tech_support_flag'] = int(customer.tech_support)
        features['streaming_tv_flag'] = int(customer.streaming_tv)
        features['streaming_movies_flag'] = int(customer.streaming_movies)
        
        # Service bundles
        features['has_security_bundle'] = int(customer.online_security or customer.online_backup)
        features['has_support_bundle'] = int(customer.tech_support or customer.device_protection)
        features['has_streaming_bundle'] = int(customer.streaming_tv or customer.streaming_movies)
        
        # Total services
        features['total_services'] = (
            features['phone_service_flag'] + features['multiple_lines_flag'] +
            features['has_internet'] + features['online_security_flag'] +
            features['online_backup_flag'] + features['device_protection_flag'] +
            features['tech_support_flag'] + features['streaming_tv_flag'] +
            features['streaming_movies_flag']
        )
        features['service_diversity'] = features['total_services'] / 9.0
        features['premium_service_count'] = (
            features['online_security_flag'] + features['online_backup_flag'] +
            features['device_protection_flag'] + features['tech_support_flag'] +
            features['streaming_tv_flag'] + features['streaming_movies_flag']
        )
        
        # Contract
        contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
        features['contract_length'] = contract_map.get(customer.contract_type, 0)
        features['is_month_to_month'] = 1 if customer.contract_type == 'Month-to-month' else 0
        features['has_long_contract'] = 1 if features['contract_length'] >= 1 else 0
        
        # Payment
        features['payment_electronic_check'] = 1 if customer.payment_method == 'Electronic check' else 0
        features['payment_mailed_check'] = 1 if customer.payment_method == 'Mailed check' else 0
        features['payment_bank_transfer'] = 1 if customer.payment_method == 'Bank transfer (automatic)' else 0
        features['payment_credit_card'] = 1 if customer.payment_method == 'Credit card (automatic)' else 0
        features['has_auto_payment'] = features['payment_bank_transfer'] or features['payment_credit_card']
        features['paperless_billing_flag'] = int(customer.paperless_billing)
        
        # Tenure
        features['tenure_months'] = customer.tenure_months
        features['tenure_years'] = customer.tenure_months / 12.0
        features['is_new_customer'] = 1 if customer.tenure_months <= 6 else 0
        features['is_established'] = 1 if customer.tenure_months >= 24 else 0
        features['is_loyal'] = 1 if customer.tenure_months >= 48 else 0
        
        # Tenure bucket
        if customer.tenure_months <= 6:
            features['tenure_bucket'] = 1
        elif customer.tenure_months <= 12:
            features['tenure_bucket'] = 2
        elif customer.tenure_months <= 24:
            features['tenure_bucket'] = 3
        elif customer.tenure_months <= 48:
            features['tenure_bucket'] = 4
        elif customer.tenure_months <= 72:
            features['tenure_bucket'] = 5
        else:
            features['tenure_bucket'] = 6
        
        features['tenure_quartile'] = min(4, max(1, (customer.tenure_months // 18) + 1))
        
        # Financial
        features['monthly_charges'] = customer.monthly_charges
        features['total_charges'] = customer.total_charges
        features['avg_monthly_spend'] = (
            customer.total_charges / customer.tenure_months 
            if customer.tenure_months > 0 else customer.monthly_charges
        )
        features['charge_per_service'] = (
            customer.monthly_charges / features['total_services']
            if features['total_services'] > 0 else customer.monthly_charges
        )
        features['is_high_value'] = 1 if customer.monthly_charges > 70 else 0
        features['monthly_charge_quartile'] = min(4, max(1, int(customer.monthly_charges // 25) + 1))
        features['estimated_clv'] = customer.total_charges + (customer.monthly_charges * 12)
        features['spending_trend'] = (
            1 if features['avg_monthly_spend'] > customer.monthly_charges
            else (-1 if features['avg_monthly_spend'] < customer.monthly_charges else 0)
        )
        
        # Risk features
        features['risk_mtm'] = features['is_month_to_month']
        features['risk_new'] = features['is_new_customer']
        features['risk_echeck'] = features['payment_electronic_check']
        features['risk_no_support'] = int(features['has_security_bundle'] == 0 and features['has_support_bundle'] == 0)
        features['risk_high_charge'] = 1 if features['charge_per_service'] > 15 else 0
        features['risk_fiber'] = features['internet_fiber']
        
        features['risk_factors_count'] = (
            features['risk_mtm'] + features['risk_new'] + features['risk_echeck'] +
            features['risk_no_support'] + features['risk_high_charge'] + features['risk_fiber']
        )
        features['risk_score'] = (
            features['risk_mtm'] * 3 + features['risk_new'] * 2 + features['risk_echeck'] * 2 +
            features['risk_no_support'] * 1 + features['risk_high_charge'] * 1 + features['risk_fiber'] * 1
        )
        features['high_risk_flag'] = 1 if features['risk_score'] >= 5 else 0
        
        # Interactions
        features['tenure_contract_interaction'] = customer.tenure_months * features['contract_length']
        features['new_high_value'] = features['is_new_customer'] * features['is_high_value']
        features['mtm_high_value'] = features['is_month_to_month'] * features['is_high_value']
        features['senior_mtm'] = features['is_senior'] * features['is_month_to_month']
        features['fiber_no_support'] = features['internet_fiber'] * (1 - features['has_support_bundle'])
        
        # Create DataFrame with correct column order
        df = pd.DataFrame([features])
        
        # Ensure all required features exist
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0
        
        return df[self.feature_names]
    
    def _get_feature_descriptions(self) -> Dict[str, str]:
        """Human-readable feature descriptions."""
        return {
            'is_month_to_month': 'Month-to-month contract',
            'contract_length': 'Contract length',
            'tenure_months': 'Customer tenure',
            'monthly_charges': 'Monthly charges',
            'total_services': 'Number of services',
            'has_internet': 'Internet service',
            'internet_fiber': 'Fiber optic internet',
            'payment_electronic_check': 'Electronic check payment',
            'tech_support_flag': 'Tech support service',
            'online_security_flag': 'Online security service',
            'has_partner': 'Has partner',
            'is_senior': 'Senior citizen',
            'risk_score': 'Overall risk score',
            'is_new_customer': 'New customer (< 6 months)',
            'has_security_bundle': 'Security bundle',
            'has_support_bundle': 'Support bundle',
            'risk_mtm': 'Month-to-month risk factor',
            'risk_echeck': 'Electronic check risk factor',
            'risk_fiber': 'Fiber optic risk factor',
        }
    
    def _get_top_reasons(self, shap_values: np.ndarray, feature_values: np.ndarray, top_n: int = 3) -> List[ChurnReason]:
        """Extract top reasons for prediction."""
        abs_shap = np.abs(shap_values)
        top_indices = np.argsort(abs_shap)[-top_n:][::-1]
        
        descriptions = self._get_feature_descriptions()
        reasons: List[ChurnReason] = []
        
        for idx in top_indices:
            feature_name = self.feature_names[idx]
            shap_value = float(shap_values[idx])
            
            direction = "increases" if shap_value > 0 else "decreases"
            readable_name = descriptions.get(feature_name, feature_name.replace('_', ' ').title())
            
            if direction == "increases":
                desc = f"{readable_name} increases churn risk"
            else:
                desc = f"{readable_name} decreases churn risk"
            
            reasons.append(ChurnReason(
                feature=feature_name,
                impact=float(abs(shap_value)),
                direction=direction,
                description=desc
            ))
        
        return reasons
    
    def predict(self, customer: CustomerFeatures) -> Tuple[int, float, RiskLevel, List[ChurnReason]]:
        """Make prediction for a single customer."""
        if not self._is_loaded or self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Engineer features
        X = self._engineer_features(customer)
        
        # Scale features
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values
        
        # Predict
        probability = float(self.model.predict_proba(X_scaled)[0, 1])
        prediction = int(probability >= 0.5)
        risk_level = self._get_risk_level(probability)
        
        # Get explanations
        reasons: List[ChurnReason] = []
        if self.explainer is not None:
            try:
                shap_values = self.explainer.shap_values(X_scaled)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                shap_values_flat = np.array(shap_values).flatten()
                reasons = self._get_top_reasons(shap_values_flat, X.values.flatten())
            except Exception:
                pass
        
        return prediction, probability, risk_level, reasons
    
    def predict_batch(self, customers: List[CustomerFeatures]) -> List[Tuple[int, float, RiskLevel, List[ChurnReason]]]:
        """Make predictions for multiple customers."""
        results: List[Tuple[int, float, RiskLevel, List[ChurnReason]]] = []
        for customer in customers:
            result = self.predict(customer)
            results.append(result)
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'model_name': self.metadata.get('model_name', 'Unknown') if self.metadata else 'Unknown',
            'version': settings.VERSION,
            'training_date': self.metadata.get('training_date', 'Unknown') if self.metadata else 'Unknown',
            'n_features': len(self.feature_names),
            'metrics': self.metadata.get('test_metrics', {}) if self.metadata else {},
            'feature_names': self.feature_names
        }


# Singleton instance
prediction_service = PredictionService()