import os
from dotenv import load_dotenv
from typing import Optional

# Load environment variables from .env file in the parent directory
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

class Settings:
    """Application settings loaded from environment variables."""
    
    # API Keys
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    QDRANT_URL: str = os.getenv("QDRANT_URL", "localhost")  # Use QDRANT_URL as per the user's request
    
    # Database
    NEON_DB_URL: Optional[str] = os.getenv("NEON_DB_URL")
    
    # Application
    APP_NAME: str = "RAG Chatbot API"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # Qdrant settings
    QDRANT_COLLECTION_NAME: str = "physical_ai_book"
    
    # Session settings
    SESSION_EXPIRATION_HOURS: int = 24

    def __init__(self):
        # Validate required environment variables
        if not self.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        if not self.QDRANT_API_KEY:
            raise ValueError("QDRANT_API_KEY environment variable is required")
        if not self.QDRANT_URL:
            raise ValueError("QDRANT_URL environment variable is required")

# Create a single instance of settings
settings = Settings()