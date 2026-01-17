"""
Customer Churn Prediction System - Main Pipeline Runner
========================================================
This script runs the complete ML pipeline:
1. Feature Engineering
2. Model Training with MLflow
3. Model Evaluation and Comparison
4. SHAP Explainability
5. Save Best Model

Usage:
    python run_pipeline.py

Author: Mahmudul Hasan
Project: Customer Churn Prediction System
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Now import our modules
from src.features.feature_engineering import FeatureEngineer, create_features
from src.models.model_training import ChurnModelTrainer, run_training_pipeline
from src.models.prediction import ChurnPredictor


def print_banner():
    """Print project banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║     CUSTOMER CHURN PREDICTION SYSTEM                          ║
    ║     ML Pipeline Runner                                        ║
    ║                                                               ║
    ║     Author: Mahmudul Hasan                                    ║
    ║     Project: Portfolio Demonstration                          ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def run_full_pipeline(output_dir: str = "models", 
                      data_dir: str = "data/processed",
                      experiment_name: str = "churn_prediction"):
    """
    Run the complete ML pipeline.
    
    Args:
        output_dir: Directory for model outputs
        data_dir: Directory for processed data
        experiment_name: MLflow experiment name
    """
    print_banner()
    start_time = datetime.now()
    print(f"Pipeline started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create directories
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # STEP 1: Feature Engineering
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 1: FEATURE ENGINEERING")
    print("="*70)
    
    feature_file = Path(data_dir) / "features.csv"
    X, y = create_features(save_path=str(feature_file))
    
    print(f"\nFeature engineering complete!")
    print(f"  - Samples: {len(X)}")
    print(f"  - Features: {X.shape[1]}")
    print(f"  - Saved to: {feature_file}")
    
    # =========================================================================
    # STEP 2: Model Training
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 2: MODEL TRAINING")
    print("="*70)
    
    trainer = run_training_pipeline(
        X=X,
        y=y,
        experiment_name=experiment_name,
        output_dir=output_dir
    )
    
    # =========================================================================
    # STEP 3: Verify Saved Model
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 3: MODEL VERIFICATION")
    print("="*70)
    
    print("\nVerifying saved model can be loaded and used...")
    predictor = ChurnPredictor(model_dir=output_dir)
    
    # Test with a sample from test set
    sample_idx = trainer.X_test.index[0]
    sample_features = trainer.X_test.loc[sample_idx]
    
    prediction = predictor.predict_single(sample_features)
    actual = trainer.y_test.loc[sample_idx]
    
    print(f"\nSample Prediction Test:")
    print(f"  - Customer index: {sample_idx}")
    print(f"  - Predicted probability: {prediction['churn_probability']:.2%}")
    print(f"  - Risk level: {prediction['risk_level']}")
    print(f"  - Actual churn: {'Yes' if actual else 'No'}")
    
    if 'explanation' in prediction:
        print(f"  - Top reasons:")
        for reason in prediction['explanation'][:3]:
            print(f"      • {reason}")
    
    # =========================================================================
    # PIPELINE COMPLETE
    # =========================================================================
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"\nDuration: {duration}")
    print(f"\nArtifacts saved in '{output_dir}':")
    
    for file in Path(output_dir).glob("*"):
        print(f"  - {file.name}")
    
    print(f"\nMLflow experiments saved in './mlruns'")
    print(f"View MLflow UI with: mlflow ui --port 5001")
    
    # Print final summary
    best_metrics = trainer.results[trainer.best_model_name]['test_metrics']
    print(f"\n{'='*70}")
    print("FINAL MODEL PERFORMANCE")
    print(f"{'='*70}")
    print(f"Best Model: {trainer.best_model_name}")
    print(f"\nMetrics:")
    print(f"  ROC-AUC:   {best_metrics['roc_auc']:.4f}")
    print(f"  Recall:    {best_metrics['recall']:.4f}")
    print(f"  Precision: {best_metrics['precision']:.4f}")
    print(f"  F1 Score:  {best_metrics['f1']:.4f}")
    print(f"  Accuracy:  {best_metrics['accuracy']:.4f}")
    
    # Check targets
    print(f"\nTarget Achievement:")
    targets = [
        ('ROC-AUC > 0.80', best_metrics['roc_auc'] >= 0.80),
        ('Recall > 0.75', best_metrics['recall'] >= 0.75),
        ('Precision > 0.60', best_metrics['precision'] >= 0.60),
        ('F1 Score > 0.65', best_metrics['f1'] >= 0.65)
    ]
    
    for target, achieved in targets:
        status = "✓" if achieved else "✗"
        print(f"  {status} {target}")
    
    achieved_count = sum(1 for _, achieved in targets if achieved)
    print(f"\nTargets achieved: {achieved_count}/4")
    
    return trainer


if __name__ == "__main__":
    # Run the pipeline
    trainer = run_full_pipeline(
        output_dir="models",
        data_dir="data/processed",
        experiment_name="churn_prediction"
    )
