from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class ChatRequest(BaseModel):
    """Model for chat requests."""
    message: str = Field(..., description="The user's question or message")
    selected_text: Optional[str] = Field(None, description="Optional selected text from the book")
    session_id: Optional[str] = Field(None, description="Optional session identifier for conversation history")


class ChatResponse(BaseModel):
    """Model for chat responses."""
    response: str = Field(..., description="The AI-generated response")
    session_id: str = Field(..., description="Session identifier")
    sources: List[str] = Field(default_factory=list, description="List of source documents used")


class IngestResponse(BaseModel):
    """Model for ingestion responses."""
    status: str = Field(..., description="Status of the ingestion process")
    message: str = Field(..., description="Descriptive message about the result")
    count: int = Field(..., description="Number of documents processed")


class HealthResponse(BaseModel):
    """Model for health check responses."""
    status: str = Field(..., description="Health status of the service")
    timestamp: str = Field(..., description="Timestamp of the health check")


class ChatSession(BaseModel):
    """Model for chat session data."""
    session_id: str
    created_at: datetime = Field(default_factory=datetime.now)
    last_interaction: datetime = Field(default_factory=datetime.now)
    active: bool = True