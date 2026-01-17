"""
Churn Prediction Module
========================
This module provides prediction functionality using trained models.

Features:
- Single customer prediction
- Batch predictions
- SHAP-based explanations for predictions
- Risk level classification

Author: Mahmudul Hasan
Project: Customer Churn Prediction System
"""

import pandas as pd
import numpy as np
import joblib
import json
import shap
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings

warnings.filterwarnings('ignore')


class ChurnPredictor:
    """
    Prediction class for customer churn.
    
    Loads trained model and provides prediction with explanations.
    """
    
    def __init__(self, model_dir: str = "models"):
        """
        Initialize predictor with saved model artifacts.
        
        Args:
            model_dir: Directory containing model files
        """
        self.model_dir = Path(model_dir)
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.metadata = None
        self.explainer = None
        
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load model and associated artifacts."""
        print(f"Loading model from {self.model_dir}...")
        
        # Find and load model file
        model_files = list(self.model_dir.glob("best_model_*.joblib"))
        if not model_files:
            raise FileNotFoundError(f"No model file found in {self.model_dir}")
        
        self.model = joblib.load(model_files[0])
        print(f"✓ Model loaded: {model_files[0].name}")
        
        # Load scaler
        scaler_file = self.model_dir / "scaler.joblib"
        if scaler_file.exists():
            self.scaler = joblib.load(scaler_file)
            print("✓ Scaler loaded")
        
        # Load feature names
        features_file = self.model_dir / "feature_names.json"
        if features_file.exists():
            with open(features_file, 'r') as f:
                self.feature_names = json.load(f)
            print(f"✓ Feature names loaded ({len(self.feature_names)} features)")
        
        # Load metadata
        metadata_file = self.model_dir / "model_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
            print(f"✓ Metadata loaded (model: {self.metadata.get('model_name', 'unknown')})")
        
        # Initialize SHAP explainer
        self._init_explainer()
    
    def _init_explainer(self):
        """Initialize SHAP explainer based on model type."""
        model_name = self.metadata.get('model_name', '') if self.metadata else ''
        
        try:
            if 'XGBoost' in model_name or 'RandomForest' in model_name:
                self.explainer = shap.TreeExplainer(self.model)
            else:
                # For linear models, we'll initialize explainer on first prediction
                self.explainer = None
            print("✓ SHAP explainer initialized")
        except Exception as e:
            print(f"⚠ Could not initialize SHAP explainer: {e}")
            self.explainer = None
    
    def _get_risk_level(self, probability: float) -> str:
        """
        Convert probability to risk level.
        
        Args:
            probability: Churn probability (0-1)
            
        Returns:
            Risk level string
        """
        if probability >= 0.75:
            return "Critical"
        elif probability >= 0.50:
            return "High"
        elif probability >= 0.25:
            return "Medium"
        else:
            return "Low"
    
    def _get_top_reasons(self, 
                         shap_values: np.ndarray, 
                         feature_values: np.ndarray,
                         top_n: int = 3) -> List[Dict]:
        """
        Extract top reasons for churn prediction.
        
        Args:
            shap_values: SHAP values for prediction
            feature_values: Feature values
            top_n: Number of top reasons to return
            
        Returns:
            List of dictionaries with feature, impact, and direction
        """
        # Get indices of features with highest absolute SHAP values
        abs_shap = np.abs(shap_values)
        top_indices = np.argsort(abs_shap)[-top_n:][::-1]
        
        reasons = []
        for idx in top_indices:
            feature_name = self.feature_names[idx]
            shap_value = shap_values[idx]
            feature_value = feature_values[idx]
            
            # Determine direction
            direction = "increases" if shap_value > 0 else "decreases"
            
            reasons.append({
                'feature': feature_name,
                'value': float(feature_value),
                'impact': float(abs(shap_value)),
                'direction': direction,
                'shap_value': float(shap_value)
            })
        
        return reasons
    
    def _format_reason_text(self, reasons: List[Dict]) -> List[str]:
        """
        Format reasons into human-readable text.
        
        Args:
            reasons: List of reason dictionaries
            
        Returns:
            List of formatted reason strings
        """
        reason_texts = []
        
        # Feature name to readable name mapping
        readable_names = {
            'is_month_to_month': 'Month-to-month contract',
            'contract_length': 'Contract length',
            'tenure_months': 'Customer tenure',
            'monthly_charges': 'Monthly charges',
            'total_services': 'Number of services',
            'has_internet': 'Has internet service',
            'internet_fiber': 'Has fiber optic internet',
            'payment_electronic_check': 'Pays by electronic check',
            'has_tech_support': 'Has tech support',
            'online_security_flag': 'Has online security',
            'has_partner': 'Has partner',
            'is_senior': 'Is senior citizen',
            'risk_score': 'Overall risk score',
            'is_new_customer': 'Is new customer',
        }
        
        for reason in reasons:
            feature = reason['feature']
            readable = readable_names.get(feature, feature.replace('_', ' ').title())
            direction = reason['direction']
            
            if direction == 'increases':
                reason_texts.append(f"{readable} increases churn risk")
            else:
                reason_texts.append(f"{readable} decreases churn risk")
        
        return reason_texts
    
    def predict_single(self, 
                       features: Union[Dict, pd.Series, np.ndarray],
                       explain: bool = True) -> Dict:
        """
        Predict churn for a single customer.
        
        Args:
            features: Customer features as dict, Series, or array
            explain: Whether to include SHAP explanations
            
        Returns:
            Dictionary with prediction results
        """
        # Convert to array if needed
        if isinstance(features, dict):
            feature_array = np.array([features.get(f, 0) for f in self.feature_names])
        elif isinstance(features, pd.Series):
            feature_array = features[self.feature_names].values
        else:
            feature_array = features
        
        feature_array = feature_array.reshape(1, -1)
        
        # Scale features
        if self.scaler:
            feature_scaled = self.scaler.transform(feature_array)
        else:
            feature_scaled = feature_array
        
        # Make prediction
        probability = self.model.predict_proba(feature_scaled)[0, 1]
        prediction = int(probability >= 0.5)
        risk_level = self._get_risk_level(probability)
        
        result = {
            'churn_prediction': prediction,
            'churn_probability': float(probability),
            'risk_level': risk_level,
            'model_name': self.metadata.get('model_name', 'unknown') if self.metadata else 'unknown'
        }
        
        # Add explanations if requested
        if explain and self.explainer:
            try:
                shap_values = self.explainer.shap_values(feature_scaled)
                
                # Handle different SHAP output formats
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # For classification, take positive class
                
                shap_values = shap_values.flatten()
                reasons = self._get_top_reasons(shap_values, feature_array.flatten())
                reason_texts = self._format_reason_text(reasons)
                
                result['top_reasons'] = reasons
                result['explanation'] = reason_texts
            except Exception as e:
                result['explanation_error'] = str(e)
        
        return result
    
    def predict_batch(self, 
                      features: Union[pd.DataFrame, np.ndarray],
                      explain: bool = False) -> pd.DataFrame:
        """
        Predict churn for multiple customers.
        
        Args:
            features: Feature DataFrame or array
            explain: Whether to include explanations (slower)
            
        Returns:
            DataFrame with predictions
        """
        if isinstance(features, pd.DataFrame):
            # Ensure correct column order
            feature_array = features[self.feature_names].values
            index = features.index
        else:
            feature_array = features
            index = range(len(features))
        
        # Scale features
        if self.scaler:
            feature_scaled = self.scaler.transform(feature_array)
        else:
            feature_scaled = feature_array
        
        # Make predictions
        probabilities = self.model.predict_proba(feature_scaled)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        risk_levels = [self._get_risk_level(p) for p in probabilities]
        
        results = pd.DataFrame({
            'churn_prediction': predictions,
            'churn_probability': probabilities,
            'risk_level': risk_levels
        }, index=index)
        
        # Add explanations if requested
        if explain and self.explainer:
            print("Computing SHAP explanations for batch (this may take a while)...")
            try:
                shap_values = self.explainer.shap_values(feature_scaled)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                
                # Get top feature for each prediction
                top_features = []
                for i in range(len(shap_values)):
                    top_idx = np.argmax(np.abs(shap_values[i]))
                    top_features.append(self.feature_names[top_idx])
                
                results['top_reason'] = top_features
            except Exception as e:
                print(f"⚠ Could not compute explanations: {e}")
        
        return results
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        info = {
            'model_name': self.metadata.get('model_name', 'unknown') if self.metadata else 'unknown',
            'n_features': len(self.feature_names) if self.feature_names else 0,
            'feature_names': self.feature_names,
            'has_scaler': self.scaler is not None,
            'has_explainer': self.explainer is not None
        }
        
        if self.metadata:
            info.update({
                'training_date': self.metadata.get('training_date'),
                'test_metrics': self.metadata.get('test_metrics'),
                'n_training_samples': self.metadata.get('n_training_samples'),
                'n_test_samples': self.metadata.get('n_test_samples')
            })
        
        return info


def load_predictor(model_dir: str = "models") -> ChurnPredictor:
    """
    Convenience function to load predictor.
    
    Args:
        model_dir: Directory containing model files
        
    Returns:
        ChurnPredictor instance
    """
    return ChurnPredictor(model_dir)


if __name__ == "__main__":
    # Test prediction
    predictor = ChurnPredictor(model_dir="models")
    
    print("\nModel Info:")
    info = predictor.get_model_info()
    print(f"  Model: {info['model_name']}")
    print(f"  Features: {info['n_features']}")
    print(f"  Test ROC-AUC: {info.get('test_metrics', {}).get('roc_auc', 'N/A')}")
    
    # Example prediction (you would use real feature values)
    print("\nExample prediction with dummy data:")
    dummy_features = {f: 0 for f in predictor.feature_names}
    dummy_features['tenure_months'] = 3
    dummy_features['monthly_charges'] = 85
    dummy_features['is_month_to_month'] = 1
    dummy_features['payment_electronic_check'] = 1
    
    result = predictor.predict_single(dummy_features)
    print(f"  Churn Probability: {result['churn_probability']:.2%}")
    print(f"  Risk Level: {result['risk_level']}")
    if 'explanation' in result:
        print("  Top Reasons:")
        for reason in result['explanation']:
            print(f"    - {reason}")
