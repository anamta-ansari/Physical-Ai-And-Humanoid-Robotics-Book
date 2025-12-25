from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class ChatSession(BaseModel):
    """Model for chat session management."""
    session_id: str = Field(..., description="Unique identifier for the session")
    created_at: datetime = Field(default_factory=datetime.now, description="Session creation timestamp")
    last_interaction: datetime = Field(default_factory=datetime.now, description="Timestamp of last interaction")
    active: bool = Field(default=True, description="Whether the session is still active")
    
    class Config:
        # Allow datetime serialization
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }