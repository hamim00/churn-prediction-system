#!/usr/bin/env python3
"""
Data Seeding Script for Customer Churn Prediction System
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import random

import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()


class DatabaseSeeder:
    """Handles database schema creation and data loading."""
    
    def __init__(self):
        self.db_url = self._build_db_url()
        self.engine = create_engine(self.db_url, echo=False)
        
        # File paths
        self.project_root = Path(__file__).parent.parent
        self.schema_path = self.project_root / 'database' / 'schema.sql'
        self.raw_data_path = self.project_root / 'data' / 'raw' / 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
        
    def _build_db_url(self) -> str:
        host = os.getenv('DB_HOST', 'localhost')
        port = os.getenv('DB_PORT', '5433')
        name = os.getenv('DB_NAME', 'churn_db')
        user = os.getenv('DB_USER', 'churn_admin')
        password = os.getenv('DB_PASSWORD', 'churn_password_123')
        return f"postgresql://{user}:{password}@{host}:{port}/{name}"
    
    def test_connection(self) -> bool:
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                print("✓ Database connection successful")
                return True
        except Exception as e:
            print(f"✗ Database connection failed: {e}")
            return False
    
    def create_schema(self):
        print("\n" + "="*50)
        print("Creating Database Schema")
        print("="*50)
        
        if not self.schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {self.schema_path}")
        
        with open(self.schema_path, 'r') as f:
            schema_sql = f.read()
        
        with self.engine.connect() as conn:
            statements = [s.strip() for s in schema_sql.split(';') if s.strip()]
            
            for statement in tqdm(statements, desc="Executing SQL"):
                if statement and not statement.startswith('--'):
                    try:
                        conn.execute(text(statement))
                    except Exception as e:
                        if "does not exist" not in str(e):
                            print(f"\nWarning: {str(e)[:80]}")
            
            conn.commit()
        
        print("✓ Schema created successfully")
    
    def load_raw_data(self) -> pd.DataFrame:
        print("\n" + "="*50)
        print("Loading Raw Data")
        print("="*50)
        
        if not self.raw_data_path.exists():
            raise FileNotFoundError(
                f"Raw data file not found: {self.raw_data_path}\n"
                "Please download the Telco Customer Churn dataset from Kaggle "
                "and place it in data/raw/"
            )
        
        df = pd.read_csv(self.raw_data_path)
        print(f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
        
        # Clean data
        df = self._clean_data(df)
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Cleaning data...")
        df = df.copy()
        
        # Standardize column names
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        
        # Handle TotalCharges
        df['totalcharges'] = pd.to_numeric(df['totalcharges'], errors='coerce')
        mask = df['totalcharges'].isna()
        df.loc[mask, 'totalcharges'] = df.loc[mask, 'monthlycharges'] * df.loc[mask, 'tenure']
        df['totalcharges'] = df['totalcharges'].fillna(0)
        
        # Convert Yes/No to boolean
        yes_no_columns = ['partner', 'dependents', 'phoneservice', 'paperlessbilling', 'churn']
        for col in yes_no_columns:
            if col in df.columns:
                df[col] = df[col].map({'Yes': True, 'No': False})
        
        df['seniorcitizen'] = df['seniorcitizen'].astype(bool)
        
        print(f"✓ Data cleaned")
        return df
    
    def insert_customers(self, df: pd.DataFrame):
        print("\n" + "="*50)
        print("Inserting Customer Data")
        print("="*50)
        
        customers_data = df[[
            'customerid', 'gender', 'seniorcitizen', 'partner', 
            'dependents', 'tenure', 'monthlycharges', 'totalcharges', 'churn'
        ]].copy()
        
        customers_data.columns = [
            'customer_id', 'gender', 'senior_citizen', 'partner',
            'dependents', 'tenure_months', 'monthly_charges', 'total_charges', 'churn'
        ]
        
        def generate_churn_date(row):
            if row['churn']:
                days_ago = random.randint(1, 90)
                return datetime.now().date() - timedelta(days=days_ago)
            return None
        
        customers_data['churn_date'] = customers_data.apply(generate_churn_date, axis=1)
        
        with self.engine.connect() as conn:
            conn.execute(text("TRUNCATE TABLE predictions CASCADE"))
            conn.execute(text("TRUNCATE TABLE customer_services CASCADE"))
            conn.execute(text("TRUNCATE TABLE customer_contracts CASCADE"))
            conn.execute(text("TRUNCATE TABLE customers CASCADE"))
            conn.commit()
            
            batch_size = 500
            for i in tqdm(range(0, len(customers_data), batch_size), desc="Inserting customers"):
                batch = customers_data.iloc[i:i+batch_size]
                batch.to_sql('customers', conn, if_exists='append', index=False, method='multi')
            
            conn.commit()
        
        print(f"✓ Inserted {len(customers_data):,} customers")
    
    def insert_contracts(self, df: pd.DataFrame):
        print("\n" + "="*50)
        print("Inserting Contract Data")
        print("="*50)
        
        with self.engine.connect() as conn:
            contract_types = pd.read_sql(
                "SELECT contract_type_id, contract_name FROM contract_types",
                conn
            )
        
        contract_type_map = dict(zip(
            contract_types['contract_name'], 
            contract_types['contract_type_id']
        ))
        
        contracts_data = df[['customerid', 'contract', 'paymentmethod', 'paperlessbilling']].copy()
        contracts_data.columns = ['customer_id', 'contract_name', 'payment_method', 'paperless_billing']
        
        contracts_data['contract_type_id'] = contracts_data['contract_name'].map(contract_type_map)
        
        contracts_data['contract_start_date'] = df['tenure'].apply(
            lambda t: datetime.now().date() - timedelta(days=t * 30)
        )
        contracts_data['contract_end_date'] = None
        
        contracts_data = contracts_data[[
            'customer_id', 'contract_type_id', 'payment_method', 
            'paperless_billing', 'contract_start_date', 'contract_end_date'
        ]]
        
        with self.engine.connect() as conn:
            batch_size = 500
            for i in tqdm(range(0, len(contracts_data), batch_size), desc="Inserting contracts"):
                batch = contracts_data.iloc[i:i+batch_size]
                batch.to_sql('customer_contracts', conn, if_exists='append', index=False, method='multi')
            
            conn.commit()
        
        print(f"✓ Inserted {len(contracts_data):,} contracts")
    
    def insert_services(self, df: pd.DataFrame):
        print("\n" + "="*50)
        print("Inserting Service Data")
        print("="*50)
        
        with self.engine.connect() as conn:
            service_types = pd.read_sql(
                "SELECT service_type_id, service_name FROM service_types",
                conn
            )
        
        service_type_map = dict(zip(
            service_types['service_name'],
            service_types['service_type_id']
        ))
        
        services_records = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing services"):
            customer_id = row['customerid']
            
            # Phone services
            if row['phoneservice'] == True:
                services_records.append({
                    'customer_id': customer_id,
                    'service_type_id': service_type_map['PhoneService'],
                    'is_subscribed': True
                })
                
                if row['multiplelines'] == 'Yes':
                    services_records.append({
                        'customer_id': customer_id,
                        'service_type_id': service_type_map['MultipleLines'],
                        'is_subscribed': True
                    })
            
            # Internet service
            internet_service = row['internetservice']
            if internet_service == 'DSL':
                services_records.append({
                    'customer_id': customer_id,
                    'service_type_id': service_type_map['InternetService_DSL'],
                    'is_subscribed': True
                })
            elif internet_service == 'Fiber optic':
                services_records.append({
                    'customer_id': customer_id,
                    'service_type_id': service_type_map['InternetService_Fiber'],
                    'is_subscribed': True
                })
            else:
                services_records.append({
                    'customer_id': customer_id,
                    'service_type_id': service_type_map['InternetService_No'],
                    'is_subscribed': True
                })
            
            # Internet-dependent services
            if internet_service != 'No':
                internet_services = [
                    ('onlinesecurity', 'OnlineSecurity'),
                    ('onlinebackup', 'OnlineBackup'),
                    ('deviceprotection', 'DeviceProtection'),
                    ('techsupport', 'TechSupport'),
                    ('streamingtv', 'StreamingTV'),
                    ('streamingmovies', 'StreamingMovies'),
                ]
                
                for col_name, service_name in internet_services:
                    if row[col_name] == 'Yes':
                        services_records.append({
                            'customer_id': customer_id,
                            'service_type_id': service_type_map[service_name],
                            'is_subscribed': True
                        })
        
        services_df = pd.DataFrame(services_records)
        
        with self.engine.connect() as conn:
            batch_size = 1000
            for i in tqdm(range(0, len(services_df), batch_size), desc="Inserting services"):
                batch = services_df.iloc[i:i+batch_size]
                batch.to_sql('customer_services', conn, if_exists='append', index=False, method='multi')
            
            conn.commit()
        
        print(f"✓ Inserted {len(services_df):,} service subscriptions")
    
    def validate_data(self):
        print("\n" + "="*50)
        print("Data Validation")
        print("="*50)
        
        with self.engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM customers"))
            print(f"  Customers: {result.scalar():,}")
            
            result = conn.execute(text("SELECT COUNT(*) FROM customer_contracts"))
            print(f"  Contracts: {result.scalar():,}")
            
            result = conn.execute(text("SELECT COUNT(*) FROM customer_services"))
            print(f"  Services: {result.scalar():,}")
            
            result = conn.execute(text("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN churn THEN 1 ELSE 0 END) as churned,
                    ROUND(100.0 * SUM(CASE WHEN churn THEN 1 ELSE 0 END) / COUNT(*), 2) as churn_rate
                FROM customers
            """))
            row = result.fetchone()
            print(f"  Churn rate: {row[2]}% ({row[1]:,} churned out of {row[0]:,})")
        
        print("\n✓ Data validation complete")
    
    def run(self):
        print("\n" + "="*60)
        print(" Customer Churn Prediction System - Database Seeder")
        print("="*60)
        
        if not self.test_connection():
            print("\nPlease check your database configuration and try again.")
            sys.exit(1)
        
        try:
            self.create_schema()
            df = self.load_raw_data()
            self.insert_customers(df)
            self.insert_contracts(df)
            self.insert_services(df)
            self.validate_data()
            
            print("\n" + "="*60)
            print(" ✓ Database seeding completed successfully!")
            print("="*60)
            
        except Exception as e:
            print(f"\n✗ Error during seeding: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == '__main__':
    seeder = DatabaseSeeder()
    seeder.run()
