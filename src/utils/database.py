"""
Database Utilities Module
"""

import os
from contextlib import contextmanager
from typing import Optional, Dict, Any, List

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()


class DatabaseManager:
    """Manages database connections and provides utility methods."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.db_url = self._build_db_url()
        self.engine = create_engine(self.db_url, pool_size=5, pool_pre_ping=True)
        self.Session = sessionmaker(bind=self.engine)
        self._initialized = True
    
    @staticmethod
    def _build_db_url() -> str:
        host = os.getenv('DB_HOST', 'localhost')
        port = os.getenv('DB_PORT', '5433')
        name = os.getenv('DB_NAME', 'churn_db')
        user = os.getenv('DB_USER', 'churn_admin')
        password = os.getenv('DB_PASSWORD', 'churn_password_123')
        return f"postgresql://{user}:{password}@{host}:{port}/{name}"
    
    @contextmanager
    def get_connection(self):
        conn = self.engine.connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def query_to_df(self, query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        with self.engine.connect() as conn:
            return pd.read_sql(text(query), conn, params=params)
    
    def get_churn_summary(self) -> Dict[str, Any]:
        query = "SELECT * FROM v_churn_summary"
        df = self.query_to_df(query)
        return df.iloc[0].to_dict()
    
    def get_customers(self, limit: int = 100, churn_only: bool = False) -> pd.DataFrame:
        where = "WHERE churn = true" if churn_only else ""
        query = f"SELECT * FROM v_customer_full {where} LIMIT {limit}"
        return self.query_to_df(query)
    
    def get_feature_data(self) -> pd.DataFrame:
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
        """
        return self.query_to_df(query)
    
    def health_check(self) -> Dict[str, Any]:
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                result = conn.execute(text("""
                    SELECT 
                        (SELECT COUNT(*) FROM customers) as customers,
                        (SELECT COUNT(*) FROM customer_contracts) as contracts,
                        (SELECT COUNT(*) FROM customer_services) as services
                """))
                counts = result.fetchone()
                
            return {
                "status": "healthy",
                "tables": {"customers": counts[0], "contracts": counts[1], "services": counts[2]}
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}


def get_db() -> DatabaseManager:
    return DatabaseManager()


if __name__ == '__main__':
    db = get_db()
    print("Database Health:", db.health_check())
