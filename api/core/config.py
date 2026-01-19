"""
API Configuration Settings
"""

import os
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings."""
    
    APP_NAME: str = "Customer Churn Prediction API"
    VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # Model settings
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models")
    
    # Database settings
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_PORT: str = os.getenv("DB_PORT", "5433")
    DB_NAME: str = os.getenv("DB_NAME", "churn_db")
    DB_USER: str = os.getenv("DB_USER", "churn_admin")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "churn_password_123")
    
    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
    
    class Config:
        env_file = ".env"
        extra = "allow"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
