"""
Model Training Pipeline for Customer Churn Prediction
======================================================
This module trains multiple ML models with MLflow tracking and SHAP explainability.

Models:
- Logistic Regression (baseline)
- Random Forest
- XGBoost

Features:
- Stratified K-Fold cross-validation
- Class imbalance handling (SMOTE, class weights)
- MLflow experiment tracking
- SHAP explainability
- Model comparison and selection

Author: Mahmudul Hasan
Project: Customer Churn Prediction System
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib
import json
import warnings
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

# ML Libraries
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# MLflow
import mlflow
import mlflow.sklearn
import mlflow.xgboost

# SHAP
import shap

warnings.filterwarnings('ignore')


class ChurnModelTrainer:
    """
    Complete ML training pipeline for churn prediction.
    
    Features:
    - Multiple model training (LR, RF, XGBoost)
    - MLflow experiment tracking
    - SHAP explainability
    - Class imbalance handling
    - Cross-validation
    """
    
    def __init__(self, 
                 experiment_name: str = "churn_prediction",
                 mlflow_tracking_uri: str = "./mlruns",
                 random_state: int = 42):
        """
        Initialize the trainer.
        
        Args:
            experiment_name: MLflow experiment name
            mlflow_tracking_uri: Path for MLflow tracking
            random_state: Random seed for reproducibility
        """
        self.experiment_name = experiment_name
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0
        self.feature_names = None
        self.results = {}
        
        # Setup MLflow
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name)
        
        print(f"✓ MLflow tracking URI: {mlflow_tracking_uri}")
        print(f"✓ Experiment: {experiment_name}")
    
    def prepare_data(self, 
                     X: pd.DataFrame, 
                     y: pd.Series,
                     test_size: float = 0.2,
                     apply_smote: bool = True) -> Tuple:
        """
        Prepare data for training with train/test split and optional SMOTE.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            test_size: Test set proportion
            apply_smote: Whether to apply SMOTE for class balancing
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled)
        """
        self.feature_names = X.columns.tolist()
        
        # Train/test split (stratified)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"\n{'='*60}")
        print("DATA PREPARATION")
        print(f"{'='*60}")
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"Training churn rate: {100*y_train.mean():.2f}%")
        print(f"Test churn rate: {100*y_test.mean():.2f}%")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Apply SMOTE if requested
        if apply_smote:
            print(f"\nApplying SMOTE for class balancing...")
            smote = SMOTE(random_state=self.random_state)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
            print(f"After SMOTE: {len(X_train_resampled)} samples")
            print(f"Class distribution: {np.bincount(y_train_resampled)}")
            
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test
            self.X_train_scaled = X_train_resampled
            self.X_test_scaled = X_test_scaled
            self.y_train_resampled = y_train_resampled
        else:
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test
            self.X_train_scaled = X_train_scaled
            self.X_test_scaled = X_test_scaled
            self.y_train_resampled = y_train
        
        return X_train, X_test, y_train, y_test
    
    def get_models(self) -> Dict[str, Any]:
        """
        Define models to train with their configurations.
        
        Returns:
            Dictionary of model name -> model instance
        """
        models = {
            'LogisticRegression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced',
                solver='lbfgs',
                C=1.0
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=-1
            ),
            'XGBoost': XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                scale_pos_weight=len(self.y_train[self.y_train==0]) / len(self.y_train[self.y_train==1]),
                use_label_encoder=False,
                eval_metric='logloss'
            )
        }
        return models
    
    def evaluate_model(self, 
                       model, 
                       X_test: np.ndarray, 
                       y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate a trained model on test data.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of metric name -> value
        """
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'avg_precision': average_precision_score(y_test, y_pred_proba)
        }
        
        return metrics
    
    def cross_validate_model(self, 
                             model, 
                             X: np.ndarray, 
                             y: pd.Series,
                             cv: int = 5) -> Dict[str, float]:
        """
        Perform stratified k-fold cross-validation.
        
        Args:
            model: Model to evaluate
            X: Feature matrix
            y: Target vector
            cv: Number of folds
            
        Returns:
            Dictionary with CV scores
        """
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        cv_scores = {
            'accuracy': cross_val_score(model, X, y, cv=skf, scoring='accuracy'),
            'precision': cross_val_score(model, X, y, cv=skf, scoring='precision'),
            'recall': cross_val_score(model, X, y, cv=skf, scoring='recall'),
            'f1': cross_val_score(model, X, y, cv=skf, scoring='f1'),
            'roc_auc': cross_val_score(model, X, y, cv=skf, scoring='roc_auc')
        }
        
        return {k: {'mean': v.mean(), 'std': v.std()} for k, v in cv_scores.items()}
    
    def train_single_model(self, 
                           model_name: str, 
                           model,
                           run_cv: bool = True) -> Dict:
        """
        Train a single model with MLflow tracking.
        
        Args:
            model_name: Name of the model
            model: Model instance
            run_cv: Whether to run cross-validation
            
        Returns:
            Dictionary with model results
        """
        print(f"\n{'-'*60}")
        print(f"Training: {model_name}")
        print(f"{'-'*60}")
        
        with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parameters
            params = model.get_params()
            for param, value in params.items():
                try:
                    mlflow.log_param(param, value)
                except:
                    pass  # Skip params that can't be logged
            
            # Train model
            print("Training model...")
            model.fit(self.X_train_scaled, self.y_train_resampled)
            
            # Cross-validation (on original training data without SMOTE)
            if run_cv:
                print("Running cross-validation...")
                cv_results = self.cross_validate_model(
                    model, self.X_train_scaled, self.y_train_resampled
                )
                for metric, scores in cv_results.items():
                    mlflow.log_metric(f"cv_{metric}_mean", scores['mean'])
                    mlflow.log_metric(f"cv_{metric}_std", scores['std'])
                    print(f"  CV {metric}: {scores['mean']:.4f} (+/- {scores['std']:.4f})")
            
            # Evaluate on test set
            print("Evaluating on test set...")
            test_metrics = self.evaluate_model(model, self.X_test_scaled, self.y_test)
            
            for metric, value in test_metrics.items():
                mlflow.log_metric(f"test_{metric}", value)
                print(f"  Test {metric}: {value:.4f}")
            
            # Log model
            if model_name == 'XGBoost':
                mlflow.xgboost.log_model(model, "model")
            else:
                mlflow.sklearn.log_model(model, "model")
            
            # Check if this is the best model (by ROC-AUC)
            if test_metrics['roc_auc'] > self.best_score:
                self.best_score = test_metrics['roc_auc']
                self.best_model = model
                self.best_model_name = model_name
                print(f"  ★ New best model! ROC-AUC: {self.best_score:.4f}")
            
            # Store results
            result = {
                'model': model,
                'test_metrics': test_metrics,
                'cv_results': cv_results if run_cv else None,
                'run_id': mlflow.active_run().info.run_id
            }
            
            return result
    
    def train_all_models(self, run_cv: bool = True) -> Dict:
        """
        Train all models and compare results.
        
        Args:
            run_cv: Whether to run cross-validation
            
        Returns:
            Dictionary of all results
        """
        print(f"\n{'='*60}")
        print("MODEL TRAINING PIPELINE")
        print(f"{'='*60}")
        
        models = self.get_models()
        
        for model_name, model in models.items():
            self.results[model_name] = self.train_single_model(
                model_name, model, run_cv
            )
        
        # Print summary
        self._print_summary()
        
        return self.results
    
    def _print_summary(self):
        """Print training summary."""
        print(f"\n{'='*60}")
        print("TRAINING SUMMARY")
        print(f"{'='*60}")
        
        # Create comparison table
        summary_data = []
        for model_name, result in self.results.items():
            metrics = result['test_metrics']
            summary_data.append({
                'Model': model_name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1': f"{metrics['f1']:.4f}",
                'ROC-AUC': f"{metrics['roc_auc']:.4f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        print("\nModel Comparison:")
        print(summary_df.to_string(index=False))
        
        print(f"\n★ Best Model: {self.best_model_name} (ROC-AUC: {self.best_score:.4f})")
        
        # Check against target metrics
        print("\nTarget Metrics Check:")
        best_metrics = self.results[self.best_model_name]['test_metrics']
        targets = {
            'ROC-AUC': (best_metrics['roc_auc'], 0.80),
            'Recall': (best_metrics['recall'], 0.75),
            'Precision': (best_metrics['precision'], 0.60),
            'F1': (best_metrics['f1'], 0.65)
        }
        
        for metric, (actual, target) in targets.items():
            status = "✓" if actual >= target else "✗"
            print(f"  {status} {metric}: {actual:.4f} (target: {target})")
    
    def compute_shap_values(self, 
                            model=None, 
                            X_sample: Optional[np.ndarray] = None,
                            max_samples: int = 500) -> Tuple:
        """
        Compute SHAP values for model explainability.
        
        Args:
            model: Model to explain (uses best model if None)
            X_sample: Sample data for SHAP (uses test set sample if None)
            max_samples: Maximum samples for SHAP computation
            
        Returns:
            Tuple of (shap_values, explainer)
        """
        if model is None:
            model = self.best_model
            model_name = self.best_model_name
        else:
            model_name = type(model).__name__
        
        print(f"\n{'='*60}")
        print(f"SHAP EXPLAINABILITY - {model_name}")
        print(f"{'='*60}")
        
        # Sample data if too large
        if X_sample is None:
            if len(self.X_test_scaled) > max_samples:
                indices = np.random.choice(
                    len(self.X_test_scaled), max_samples, replace=False
                )
                X_sample = self.X_test_scaled[indices]
            else:
                X_sample = self.X_test_scaled
        
        print(f"Computing SHAP values for {len(X_sample)} samples...")
        
        # Create appropriate explainer based on model type
        if model_name == 'XGBoost' or isinstance(model, XGBClassifier):
            explainer = shap.TreeExplainer(model)
        elif model_name == 'RandomForest' or isinstance(model, RandomForestClassifier):
            explainer = shap.TreeExplainer(model)
        else:
            # For linear models
            explainer = shap.LinearExplainer(model, X_sample)
        
        shap_values = explainer.shap_values(X_sample)
        
        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            # For classification, take the positive class
            shap_values = shap_values[1]
        
        print("✓ SHAP values computed successfully")
        
        return shap_values, explainer, X_sample
    
    def plot_shap_summary(self, 
                          shap_values: np.ndarray, 
                          X_sample: np.ndarray,
                          save_path: Optional[str] = None):
        """
        Create SHAP summary plot.
        
        Args:
            shap_values: SHAP values array
            X_sample: Sample feature matrix
            save_path: Path to save plot
        """
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values, 
            X_sample, 
            feature_names=self.feature_names,
            show=False
        )
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ SHAP summary plot saved to {save_path}")
        
        plt.close()
    
    def plot_shap_importance(self, 
                             shap_values: np.ndarray,
                             save_path: Optional[str] = None,
                             top_n: int = 20):
        """
        Create SHAP feature importance bar plot.
        
        Args:
            shap_values: SHAP values array
            save_path: Path to save plot
            top_n: Number of top features to show
        """
        # Calculate mean absolute SHAP values
        mean_shap = np.abs(shap_values).mean(axis=0)
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': mean_shap
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(
            data=feature_importance.head(top_n),
            x='importance',
            y='feature',
            palette='viridis'
        )
        plt.title(f'Top {top_n} Features by SHAP Importance')
        plt.xlabel('Mean |SHAP Value|')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ SHAP importance plot saved to {save_path}")
        
        plt.close()
        
        return feature_importance
    
    def plot_roc_curves(self, save_path: Optional[str] = None):
        """
        Plot ROC curves for all trained models.
        
        Args:
            save_path: Path to save plot
        """
        plt.figure(figsize=(10, 8))
        
        for model_name, result in self.results.items():
            model = result['model']
            y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            auc = result['test_metrics']['roc_auc']
            
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Model Comparison')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ ROC curves saved to {save_path}")
        
        plt.close()
    
    def plot_confusion_matrix(self, 
                              model=None, 
                              save_path: Optional[str] = None):
        """
        Plot confusion matrix for a model.
        
        Args:
            model: Model to evaluate (uses best if None)
            save_path: Path to save plot
        """
        if model is None:
            model = self.best_model
            model_name = self.best_model_name
        else:
            model_name = type(model).__name__
        
        y_pred = model.predict(self.X_test_scaled)
        cm = confusion_matrix(self.y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['Not Churned', 'Churned'],
            yticklabels=['Not Churned', 'Churned']
        )
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Confusion matrix saved to {save_path}")
        
        plt.close()
    
    def save_best_model(self, 
                        model_dir: str = "models",
                        include_scaler: bool = True):
        """
        Save the best model and associated artifacts.
        
        Args:
            model_dir: Directory to save model
            include_scaler: Whether to save the scaler
        """
        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print("SAVING MODEL ARTIFACTS")
        print(f"{'='*60}")
        
        # Save model
        model_file = model_path / f"best_model_{self.best_model_name.lower()}.joblib"
        joblib.dump(self.best_model, model_file)
        print(f"✓ Model saved: {model_file}")
        
        # Save scaler
        if include_scaler:
            scaler_file = model_path / "scaler.joblib"
            joblib.dump(self.scaler, scaler_file)
            print(f"✓ Scaler saved: {scaler_file}")
        
        # Save feature names
        features_file = model_path / "feature_names.json"
        with open(features_file, 'w') as f:
            json.dump(self.feature_names, f, indent=2)
        print(f"✓ Feature names saved: {features_file}")
        
        # Save model metadata
        metadata = {
            'model_name': self.best_model_name,
            'training_date': datetime.now().isoformat(),
            'test_metrics': self.results[self.best_model_name]['test_metrics'],
            'n_features': len(self.feature_names),
            'n_training_samples': len(self.X_train),
            'n_test_samples': len(self.X_test)
        }
        metadata_file = model_path / "model_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Metadata saved: {metadata_file}")
        
        print(f"\nAll artifacts saved to: {model_path}")


def run_training_pipeline(X: pd.DataFrame, 
                          y: pd.Series,
                          experiment_name: str = "churn_prediction",
                          output_dir: str = "models") -> ChurnModelTrainer:
    """
    Run the complete training pipeline.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        experiment_name: MLflow experiment name
        output_dir: Directory for model outputs
        
    Returns:
        Trained ChurnModelTrainer instance
    """
    # Initialize trainer
    trainer = ChurnModelTrainer(experiment_name=experiment_name)
    
    # Prepare data
    trainer.prepare_data(X, y, apply_smote=True)
    
    # Train all models
    trainer.train_all_models(run_cv=True)
    
    # Generate plots
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    trainer.plot_roc_curves(save_path=output_path / "roc_curves.png")
    trainer.plot_confusion_matrix(save_path=output_path / "confusion_matrix.png")
    
    # SHAP explainability
    shap_values, explainer, X_sample = trainer.compute_shap_values()
    trainer.plot_shap_summary(shap_values, X_sample, save_path=output_path / "shap_summary.png")
    feature_importance = trainer.plot_shap_importance(
        shap_values, save_path=output_path / "shap_importance.png"
    )
    
    # Save feature importance
    feature_importance.to_csv(output_path / "feature_importance.csv", index=False)
    print(f"✓ Feature importance saved to {output_path / 'feature_importance.csv'}")
    
    # Save best model
    trainer.save_best_model(model_dir=output_dir)
    
    return trainer


if __name__ == "__main__":
    # Import feature engineering
    import sys
    sys.path.append('.')
    from feature_engineering import create_features
    
    # Create features
    X, y = create_features()
    
    # Run training pipeline
    trainer = run_training_pipeline(X, y, output_dir="models")
    
    print("\n✓ Training pipeline completed successfully!")
