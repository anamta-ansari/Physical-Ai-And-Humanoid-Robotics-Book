from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime


class DocumentChunk(BaseModel):
    """Model for document chunks stored in the vector database."""
    chunk_id: str = Field(..., description="Unique identifier for the chunk")
    content: str = Field(..., description="The actual text content of the chunk")
    source_file: str = Field(..., description="Path to the original source file")
    part: str = Field(..., description="Part of the book this chunk belongs to")
    chapter: str = Field(..., description="Chapter of the book this chunk belongs to")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    embedding: Optional[list] = Field(None, description="Embedding vector (if stored separately)")


class DocumentIngestionRequest(BaseModel):
    """Model for document ingestion requests."""
    force_reindex: bool = Field(default=False, description="Whether to force re-indexing of documents")
    metadata_filter: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata filter for selective ingestion")


class DocumentIngestionResponse(BaseModel):
    """Model for document ingestion responses."""
    status: str = Field(..., description="Status of the ingestion process")
    message: str = Field(..., description="Descriptive message about the result")
    documents_processed: int = Field(..., description="Number of documents processed")
    chunks_created: int = Field(..., description="Number of chunks created")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of ingestion")