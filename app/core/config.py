"""
Core configuration management for the AI Chatbot Backend.
Loads environment variables and provides application-wide settings.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field, validator


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # ================================
    # DATABASE CONFIGURATION
    # ================================
    DB_HOST: str = Field(default="localhost", description="Database host")
    DB_PORT: int = Field(default=3306, description="Database port")
    DB_USER: str = Field(default="root", description="Database user")
    DB_PASSWORD: str = Field(default="", description="Database password")
    DB_NAME: str = Field(default="krushi_node", description="Database name")
    DB_POOL_SIZE: int = Field(default=10, description="Connection pool size")
    DB_MAX_OVERFLOW: int = Field(default=20, description="Max overflow connections")
    DB_POOL_TIMEOUT: int = Field(default=30, description="Connection timeout in seconds")
    
    # ================================
    # LLM API KEYS
    # ================================
    GROQ_API_KEY: Optional[str] = Field(default=None, description="Groq API key (primary)")
    OPENAI_API_KEY: Optional[str] = Field(default=None, description="OpenAI API key (fallback)")
    
    # ================================
    # SARVAM AI (Primary: STT + TTS + Translation)
    # ================================
    SARVAM_API_KEY: Optional[str] = Field(default=None, description="Sarvam AI API key — primary for Indian language STT/TTS/Translation")
    SARVAM_STT_MODEL: str = Field(default="saarika:v2", description="Sarvam STT model")
    SARVAM_TTS_MODEL: str = Field(default="bulbul:v1", description="Sarvam TTS model")
    SARVAM_TRANSLATE_MODEL: str = Field(default="mayura:v1", description="Sarvam translation model")
    SARVAM_DEFAULT_SPEAKER: str = Field(default="meera", description="Default Sarvam TTS speaker voice")
    SARVAM_DEFAULT_LANGUAGE: str = Field(default="hi-IN", description="Default language for Sarvam services")
    
    # ================================
    # SPEECH-TO-TEXT (Fallbacks)
    # ================================
    DEEPGRAM_API_KEY: Optional[str] = Field(default=None, description="Deepgram API key (fallback)")
    
    # ================================
    # APPLICATION SETTINGS
    # ================================
    APP_HOST: str = Field(default="0.0.0.0", description="Application host")
    APP_PORT: int = Field(default=8000, description="Application port")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    ENVIRONMENT: str = Field(default="development", description="Environment (development/production)")
    
    # ================================
    # FEATURE FLAGS
    # ================================
    # SQL flow: database queries for prices, products, orders, etc.
    # "false" = bot only answers NAVIGATION + GENERAL + GREETING
    # "true"  = bot also queries the database for live data
    # To enable later: set ENABLE_SQL_FLOW=true in .env and restart
    ENABLE_SQL_FLOW: str = Field(default="false", description="Enable SQL database query flow (true/false)")
    
    # ================================
    # RATE LIMITING CONFIGURATION
    # ================================
    GROQ_RPM_LIMIT: int = Field(default=30, description="Groq requests per minute")
    GROQ_TOKENS_PER_MINUTE: int = Field(default=14400, description="Groq tokens per minute")
    OPENAI_RPM_LIMIT: int = Field(default=60, description="OpenAI requests per minute")
    OPENAI_TOKENS_PER_MINUTE: int = Field(default=90000, description="OpenAI tokens per minute")
    
    # ================================
    # LLM MODEL CONFIGURATION
    # ================================
    GROQ_MODEL: str = Field(default="llama-3.3-70b-versatile", description="Groq model name")
    GOOGLE_TRANSLATE_API_KEY: Optional[str] = Field(default=None, description="Google Translate API key for language detection")
    OPENAI_MODEL: str = Field(default="gpt-4o", description="OpenAI model for SQL generation — gpt-4o gives best SQL accuracy")
    # SQL_PROVIDER controls which LLM is used for Step 3 (SQL generation)
    # Options: "openai" | "groq" | "auto" (auto = openai if key exists, else groq)
    SQL_PROVIDER: str = Field(default="auto", description="LLM provider for SQL query generation")
    
    # ================================
    # WEBSOCKET CONFIGURATION
    # ================================
    WS_MESSAGE_QUEUE_SIZE: int = Field(default=100, description="WebSocket message queue size")
    WS_PING_INTERVAL: int = Field(default=30, description="WebSocket ping interval")
    WS_PING_TIMEOUT: int = Field(default=10, description="WebSocket ping timeout")
    
    # ================================
    # TTS CONFIGURATION
    # ================================
    TTS_PROVIDER: str = Field(default="gtts", description="TTS provider (gtts/pyttsx3)")
    TTS_LANGUAGE: str = Field(default="en", description="TTS language code")
    TTS_SPEED: float = Field(default=1.0, description="TTS speech speed")
    
    # ================================
    # SCHEMA PATHS
    # ================================
    SCHEMA_DIR: str = Field(default="app/schemas", description="Schema directory path")
    TOOLS_DIR: str = Field(default="app/schemas/tools", description="Tools directory path")
    
    @validator("LOG_LEVEL")
    def validate_log_level(cls, v):
        """Validate log level is one of the standard levels."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of {valid_levels}")
        return v.upper()
    
    @validator("ENVIRONMENT")
    def validate_environment(cls, v):
        """Validate environment is development or production."""
        valid_envs = ["development", "production"]
        if v.lower() not in valid_envs:
            raise ValueError(f"ENVIRONMENT must be one of {valid_envs}")
        return v.lower()
    @property
    def has_google_translate_key(self) -> bool:
       return self.GOOGLE_TRANSLATE_API_KEY is not None and len(self.GOOGLE_TRANSLATE_API_KEY) > 0
    
    @property
    def database_url(self) -> str:
        """Generate database URL for SQLAlchemy."""
        password = f":{self.DB_PASSWORD}" if self.DB_PASSWORD else ""
        return f"mysql+aiomysql://{self.DB_USER}{password}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.ENVIRONMENT == "development"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.ENVIRONMENT == "production"
    
    @property
    def has_groq_key(self) -> bool:
        """Check if Groq API key is available."""
        return self.GROQ_API_KEY is not None and len(self.GROQ_API_KEY) > 0
    
    @property
    def has_openai_key(self) -> bool:
        """Check if OpenAI API key is available."""
        return self.OPENAI_API_KEY is not None and len(self.OPENAI_API_KEY) > 0
    
    @property
    def has_sarvam_key(self) -> bool:
        """Check if Sarvam API key is available."""
        return self.SARVAM_API_KEY is not None and len(self.SARVAM_API_KEY) > 0

    @property
    def has_deepgram_key(self) -> bool:
        """Check if Deepgram API key is available."""
        return self.DEEPGRAM_API_KEY is not None and len(self.DEEPGRAM_API_KEY) > 0
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings instance."""
    return settings