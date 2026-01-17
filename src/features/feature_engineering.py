"""
Feature Engineering Pipeline for Customer Churn Prediction
===========================================================
Author: Mahmudul Hasan
Project: Customer Churn Prediction System
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.database import DatabaseManager


class FeatureEngineer:
    """Feature engineering pipeline for churn prediction."""
    
    def __init__(self):
        self.db = DatabaseManager()
        self.feature_names = []
        self.categorical_columns = []
        self.numerical_columns = []
    
    def extract_base_data(self) -> pd.DataFrame:
        df = self.db.get_feature_data()
        print(f"✓ Extracted {len(df)} records from database")
        return df
    
    def extract_base_data_simple(self) -> pd.DataFrame:
        try:
            df = self.db.get_feature_data()
            print(f"✓ Extracted {len(df)} records using get_feature_data()")
            return df
        except Exception as e:
            print(f"Error extracting data: {e}")
            raise
    
    def create_demographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        df['gender_male'] = (df['gender'] == 'Male').astype(int)
        df['is_senior'] = df['senior_citizen'].astype(int)
        df['has_partner'] = df['partner'].astype(int)
        df['has_dependents'] = df['dependents'].astype(int)
        df['has_family'] = (df['has_partner'] | df['has_dependents']).astype(int)
        df['family_size_proxy'] = df['has_partner'] + df['has_dependents']
        df['senior_alone'] = ((df['is_senior'] == 1) & (df['has_family'] == 0)).astype(int)
        
        print("✓ Created 7 demographic features")
        return df
    
    def create_service_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        service_cols = ['has_phone_service', 'has_multiple_lines', 'has_online_security', 
                       'has_online_backup', 'has_device_protection', 'has_tech_support',
                       'has_streaming_tv', 'has_streaming_movies']
        
        for col in service_cols:
            if col in df.columns:
                flag_name = col.replace('has_', '') + '_flag'
                df[flag_name] = df[col].astype(int)
        
        df['has_internet'] = 0
        if 'has_dsl' in df.columns:
            df['has_internet'] = df['has_internet'] | df['has_dsl'].astype(int)
        if 'has_fiber' in df.columns:
            df['has_internet'] = df['has_internet'] | df['has_fiber'].astype(int)
        
        df['internet_fiber'] = df['has_fiber'].astype(int) if 'has_fiber' in df.columns else 0
        df['internet_dsl'] = df['has_dsl'].astype(int) if 'has_dsl' in df.columns else 0
        
        service_flags = [col.replace('has_', '') + '_flag' for col in service_cols 
                        if col.replace('has_', '') + '_flag' in df.columns]
        df['total_services'] = df[service_flags].sum(axis=1) + df['has_internet']
        
        df['has_security_bundle'] = 0
        if 'has_online_security' in df.columns:
            df['has_security_bundle'] = df['has_security_bundle'] | df['has_online_security'].astype(int)
        if 'has_online_backup' in df.columns:
            df['has_security_bundle'] = df['has_security_bundle'] | df['has_online_backup'].astype(int)
        
        df['has_support_bundle'] = 0
        if 'has_tech_support' in df.columns:
            df['has_support_bundle'] = df['has_support_bundle'] | df['has_tech_support'].astype(int)
        if 'has_device_protection' in df.columns:
            df['has_support_bundle'] = df['has_support_bundle'] | df['has_device_protection'].astype(int)
        
        df['has_streaming_bundle'] = 0
        if 'has_streaming_tv' in df.columns:
            df['has_streaming_bundle'] = df['has_streaming_bundle'] | df['has_streaming_tv'].astype(int)
        if 'has_streaming_movies' in df.columns:
            df['has_streaming_bundle'] = df['has_streaming_bundle'] | df['has_streaming_movies'].astype(int)
        
        df['service_diversity'] = df['total_services'] / 9.0
        
        premium_cols = ['has_online_security', 'has_online_backup', 'has_device_protection', 
                       'has_tech_support', 'has_streaming_tv', 'has_streaming_movies']
        df['premium_service_count'] = sum(
            df[col].astype(int) for col in premium_cols if col in df.columns
        )
        
        print("✓ Created 15+ service features")
        return df
    
    def create_contract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        contract_col = 'contract_type' if 'contract_type' in df.columns else 'contract_name'
        contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
        df['contract_length'] = df[contract_col].map(contract_map)
        df['is_month_to_month'] = (df[contract_col] == 'Month-to-month').astype(int)
        df['has_long_contract'] = (df['contract_length'] >= 1).astype(int)
        
        df['payment_electronic_check'] = (df['payment_method'] == 'Electronic check').astype(int)
        df['payment_mailed_check'] = (df['payment_method'] == 'Mailed check').astype(int)
        df['payment_bank_transfer'] = (df['payment_method'] == 'Bank transfer (automatic)').astype(int)
        df['payment_credit_card'] = (df['payment_method'] == 'Credit card (automatic)').astype(int)
        
        df['has_auto_payment'] = (df['payment_bank_transfer'] | df['payment_credit_card']).astype(int)
        df['paperless_billing_flag'] = df['paperless_billing'].astype(int)
        
        print("✓ Created 9 contract/billing features")
        return df
    
    def create_financial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        df['avg_monthly_spend'] = np.where(
            df['tenure_months'] > 0,
            df['total_charges'] / df['tenure_months'],
            df['monthly_charges']
        )
        
        df['charge_per_service'] = np.where(
            df['total_services'] > 0,
            df['monthly_charges'] / df['total_services'],
            df['monthly_charges']
        )
        
        median_charges = df['monthly_charges'].median()
        df['is_high_value'] = (df['monthly_charges'] > median_charges).astype(int)
        
        df['monthly_charge_quartile'] = pd.qcut(
            df['monthly_charges'], q=4, labels=[1, 2, 3, 4], duplicates='drop'
        ).astype(int)
        
        df['estimated_clv'] = df['total_charges'] + (df['monthly_charges'] * 12)
        
        df['spending_trend'] = np.where(
            df['avg_monthly_spend'] > df['monthly_charges'], 1,
            np.where(df['avg_monthly_spend'] < df['monthly_charges'], -1, 0)
        )
        
        print("✓ Created 7 financial features")
        return df
    
    def create_tenure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        df['tenure_years'] = df['tenure_months'] / 12.0
        df['is_new_customer'] = (df['tenure_months'] <= 6).astype(int)
        df['is_established'] = (df['tenure_months'] >= 24).astype(int)
        df['is_loyal'] = (df['tenure_months'] >= 48).astype(int)
        
        bins = [0, 6, 12, 24, 48, 72, float('inf')]
        labels = [1, 2, 3, 4, 5, 6]
        df['tenure_bucket'] = pd.cut(
            df['tenure_months'], bins=bins, labels=labels, include_lowest=True
        ).astype(int)
        
        df['tenure_quartile'] = pd.qcut(
            df['tenure_months'].rank(method='first'), q=4, labels=[1, 2, 3, 4]
        ).astype(int)
        
        print("✓ Created 7 tenure features")
        return df
    
    def create_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        risk_factors = []
        
        df['risk_mtm'] = df['is_month_to_month']
        risk_factors.append('risk_mtm')
        
        df['risk_new'] = df['is_new_customer']
        risk_factors.append('risk_new')
        
        df['risk_echeck'] = df['payment_electronic_check']
        risk_factors.append('risk_echeck')
        
        df['risk_no_support'] = ((df['has_security_bundle'] == 0) & 
                                 (df['has_support_bundle'] == 0)).astype(int)
        risk_factors.append('risk_no_support')
        
        charge_per_service_median = df['charge_per_service'].median()
        df['risk_high_charge'] = (df['charge_per_service'] > charge_per_service_median * 1.5).astype(int)
        risk_factors.append('risk_high_charge')
        
        df['risk_fiber'] = df['internet_fiber']
        risk_factors.append('risk_fiber')
        
        df['risk_factors_count'] = df[risk_factors].sum(axis=1)
        
        df['risk_score'] = (
            df['risk_mtm'] * 3 +
            df['risk_new'] * 2 +
            df['risk_echeck'] * 2 +
            df['risk_no_support'] * 1 +
            df['risk_high_charge'] * 1 +
            df['risk_fiber'] * 1
        )
        
        df['high_risk_flag'] = (df['risk_score'] >= 5).astype(int)
        
        print("✓ Created 9 risk features")
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        df['tenure_contract_interaction'] = df['tenure_months'] * df['contract_length']
        df['new_high_value'] = (df['is_new_customer'] * df['is_high_value']).astype(int)
        df['mtm_high_value'] = (df['is_month_to_month'] * df['is_high_value']).astype(int)
        df['senior_mtm'] = (df['is_senior'] * df['is_month_to_month']).astype(int)
        df['fiber_no_support'] = (df['internet_fiber'] * (1 - df['has_support_bundle'])).astype(int)
        
        print("✓ Created 5 interaction features")
        return df
    
    def get_feature_columns(self) -> dict:
        numerical_features = [
            'tenure_months', 'tenure_years', 'tenure_bucket', 'tenure_quartile',
            'monthly_charges', 'total_charges', 'avg_monthly_spend',
            'charge_per_service', 'estimated_clv', 'monthly_charge_quartile',
            'total_services', 'service_diversity', 'premium_service_count',
            'risk_score', 'risk_factors_count',
            'tenure_contract_interaction',
        ]
        
        binary_features = [
            'gender_male', 'is_senior', 'has_partner', 'has_dependents',
            'has_family', 'senior_alone',
            'phone_service_flag', 'multiple_lines_flag', 'has_internet',
            'internet_fiber', 'internet_dsl',
            'online_security_flag', 'online_backup_flag',
            'device_protection_flag', 'tech_support_flag',
            'streaming_tv_flag', 'streaming_movies_flag',
            'has_security_bundle', 'has_support_bundle', 'has_streaming_bundle',
            'contract_length', 'is_month_to_month', 'has_long_contract',
            'payment_electronic_check', 'payment_mailed_check',
            'payment_bank_transfer', 'payment_credit_card',
            'has_auto_payment', 'paperless_billing_flag',
            'is_new_customer', 'is_established', 'is_loyal',
            'is_high_value', 'spending_trend',
            'risk_mtm', 'risk_new', 'risk_echeck', 'risk_no_support',
            'risk_high_charge', 'risk_fiber', 'high_risk_flag',
            'new_high_value', 'mtm_high_value', 'senior_mtm', 'fiber_no_support',
        ]
        
        all_features = numerical_features + binary_features
        
        return {
            'numerical': numerical_features,
            'binary': binary_features,
            'all_features': all_features,
            'target': 'churn'
        }
    
    def build_features(self, save_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
        print("\n" + "="*60)
        print("FEATURE ENGINEERING PIPELINE")
        print("="*60 + "\n")
        
        print("Step 1: Extracting data from PostgreSQL...")
        df = self.extract_base_data_simple()
        
        print("\nStep 2: Creating features...")
        df = self.create_demographic_features(df)
        df = self.create_service_features(df)
        df = self.create_contract_features(df)
        df = self.create_financial_features(df)
        df = self.create_tenure_features(df)
        df = self.create_risk_features(df)
        df = self.create_interaction_features(df)
        
        print("\nStep 3: Selecting features for model...")
        feature_cols = self.get_feature_columns()
        
        available_features = [col for col in feature_cols['all_features'] if col in df.columns]
        missing_features = [col for col in feature_cols['all_features'] if col not in df.columns]
        
        if missing_features:
            print(f"⚠ Warning: {len(missing_features)} features not available: {missing_features[:5]}...")
        
        print(f"✓ Selected {len(available_features)} features for model")
        
        X = df[available_features].copy()
        y = df['churn'].astype(int)
        
        if X.isnull().sum().sum() > 0:
            print("\nStep 4: Handling missing values...")
            null_counts = X.isnull().sum()
            null_cols = null_counts[null_counts > 0]
            print(f"⚠ Found NaN in columns: {null_cols.to_dict()}")
            X = X.fillna(0)
            print("✓ Filled NaN values with 0")
        
        if save_path:
            print(f"\nSaving processed data to {save_path}...")
            output_df = X.copy()
            output_df['churn'] = y
            output_df['customer_id'] = df['customer_id']
            output_df.to_csv(save_path, index=False)
            print(f"✓ Saved {len(output_df)} records")
        
        print("\n" + "="*60)
        print("FEATURE ENGINEERING COMPLETE")
        print("="*60)
        print(f"Total samples: {len(X)}")
        print(f"Total features: {len(available_features)}")
        print(f"Target distribution:")
        print(f"  - Not Churned (0): {(y == 0).sum()} ({100*(y == 0).mean():.1f}%)")
        print(f"  - Churned (1): {(y == 1).sum()} ({100*(y == 1).mean():.1f}%)")
        print("="*60 + "\n")
        
        return X, y
    
    def get_feature_importance_names(self) -> list:
        return self.get_feature_columns()['all_features']


def create_features(save_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    engineer = FeatureEngineer()
    return engineer.build_features(save_path)


if __name__ == "__main__":
    X, y = create_features(save_path="data/processed/features.csv")
    print("\nFeature DataFrame Info:")
    print(X.info())
    print("\nSample of features:")
    print(X.head())