"""
Customer Service
================
Handles customer data operations from database.
"""

import pandas as pd
from typing import List, Optional, Dict, Any
from sqlalchemy import create_engine, text

from api.core.config import settings


class CustomerService:
    """Service for customer database operations."""
    
    def __init__(self):
        self.engine = None
        self._connect()
    
    def _connect(self):
        """Create database connection."""
        try:
            self.engine = create_engine(
                settings.DATABASE_URL,
                pool_size=5,
                pool_pre_ping=True
            )
        except Exception as e:
            print(f"Database connection failed: {e}")
            self.engine = None
    
    @property
    def is_connected(self) -> bool:
        """Check if database is connected."""
        if self.engine is None:
            return False
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False
    
    def get_customer_by_id(self, customer_id: str) -> Optional[Dict[str, Any]]:
        """Fetch customer by ID."""
        if not self.engine:
            return None
        
        query = """
        SELECT 
            c.customer_id, c.gender, c.senior_citizen, c.partner, c.dependents,
            c.tenure_months, c.monthly_charges, c.total_charges, c.churn,
            ct.contract_name AS contract_type, cc.payment_method, cc.paperless_billing,
            COALESCE(css.has_phone_service, FALSE) AS has_phone_service,
            COALESCE(css.has_multiple_lines, FALSE) AS has_multiple_lines,
            COALESCE(css.has_dsl, FALSE) AS has_dsl,
            COALESCE(css.has_fiber, FALSE) AS has_fiber,
            COALESCE(css.has_online_security, FALSE) AS has_online_security,
            COALESCE(css.has_online_backup, FALSE) AS has_online_backup,
            COALESCE(css.has_device_protection, FALSE) AS has_device_protection,
            COALESCE(css.has_tech_support, FALSE) AS has_tech_support,
            COALESCE(css.has_streaming_tv, FALSE) AS has_streaming_tv,
            COALESCE(css.has_streaming_movies, FALSE) AS has_streaming_movies
        FROM customers c
        LEFT JOIN customer_contracts cc ON c.customer_id = cc.customer_id
        LEFT JOIN contract_types ct ON cc.contract_type_id = ct.contract_type_id
        LEFT JOIN v_customer_services_summary css ON c.customer_id = css.customer_id
        WHERE c.customer_id = :customer_id
        """
        
        try:
            with self.engine.connect() as conn:
                result = pd.read_sql(text(query), conn, params={'customer_id': customer_id})
                if result.empty:
                    return None
                return result.iloc[0].to_dict()
        except Exception as e:
            print(f"Error fetching customer: {e}")
            return None
    
    def get_customers(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Fetch multiple customers."""
        if not self.engine:
            return []
        
        query = """
        SELECT 
            c.customer_id, c.gender, c.senior_citizen, c.partner, c.dependents,
            c.tenure_months, c.monthly_charges, c.total_charges, c.churn,
            ct.contract_name AS contract_type, cc.payment_method
        FROM customers c
        LEFT JOIN customer_contracts cc ON c.customer_id = cc.customer_id
        LEFT JOIN contract_types ct ON cc.contract_type_id = ct.contract_type_id
        ORDER BY c.customer_id
        LIMIT :limit OFFSET :offset
        """
        
        try:
            with self.engine.connect() as conn:
                result = pd.read_sql(text(query), conn, params={'limit': limit, 'offset': offset})
                return result.to_dict('records')
        except Exception as e:
            print(f"Error fetching customers: {e}")
            return []
    
    def get_at_risk_customers(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Fetch customers with high-risk profile."""
        if not self.engine:
            return []
        
        query = """
        SELECT 
            c.customer_id, c.gender, c.senior_citizen, c.partner, c.dependents,
            c.tenure_months, c.monthly_charges, c.total_charges, c.churn,
            ct.contract_name AS contract_type, cc.payment_method, cc.paperless_billing,
            COALESCE(css.has_phone_service, FALSE) AS has_phone_service,
            COALESCE(css.has_multiple_lines, FALSE) AS has_multiple_lines,
            COALESCE(css.has_dsl, FALSE) AS has_dsl,
            COALESCE(css.has_fiber, FALSE) AS has_fiber,
            COALESCE(css.has_online_security, FALSE) AS has_online_security,
            COALESCE(css.has_online_backup, FALSE) AS has_online_backup,
            COALESCE(css.has_device_protection, FALSE) AS has_device_protection,
            COALESCE(css.has_tech_support, FALSE) AS has_tech_support,
            COALESCE(css.has_streaming_tv, FALSE) AS has_streaming_tv,
            COALESCE(css.has_streaming_movies, FALSE) AS has_streaming_movies
        FROM customers c
        LEFT JOIN customer_contracts cc ON c.customer_id = cc.customer_id
        LEFT JOIN contract_types ct ON cc.contract_type_id = ct.contract_type_id
        LEFT JOIN v_customer_services_summary css ON c.customer_id = css.customer_id
        WHERE c.churn = FALSE
          AND ct.contract_name = 'Month-to-month'
          AND c.tenure_months <= 12
        ORDER BY c.monthly_charges DESC
        LIMIT :limit
        """
        
        try:
            with self.engine.connect() as conn:
                result = pd.read_sql(text(query), conn, params={'limit': limit})
                return result.to_dict('records')
        except Exception as e:
            print(f"Error fetching at-risk customers: {e}")
            return []
    
    def search_customers(self, search_term: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search customers by ID."""
        if not self.engine:
            return []
        
        query = """
        SELECT 
            c.customer_id, c.gender, c.senior_citizen,
            c.tenure_months, c.monthly_charges, c.churn,
            ct.contract_name AS contract_type
        FROM customers c
        LEFT JOIN customer_contracts cc ON c.customer_id = cc.customer_id
        LEFT JOIN contract_types ct ON cc.contract_type_id = ct.contract_type_id
        WHERE c.customer_id ILIKE :search_term
        LIMIT :limit
        """
        
        try:
            with self.engine.connect() as conn:
                result = pd.read_sql(
                    text(query), 
                    conn, 
                    params={'search_term': f'%{search_term}%', 'limit': limit}
                )
                return result.to_dict('records')
        except Exception as e:
            print(f"Error searching customers: {e}")
            return []


# Singleton instance
customer_service = CustomerService()
