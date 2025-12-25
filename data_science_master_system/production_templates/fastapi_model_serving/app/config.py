"""
Configuration settings for the ML API.
"""

from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    """Application settings."""
    
    # API Settings
    API_TITLE: str = "ML Model Serving API"
    API_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # CORS
    ALLOWED_ORIGINS: List[str] = ["*"]
    
    # Model Settings
    MODELS_DIR: str = "models"
    DEFAULT_MODEL: str = "demo_classifier"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 100
    
    # Authentication (optional)
    API_KEY: str = ""
    ENABLE_AUTH: bool = False
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
