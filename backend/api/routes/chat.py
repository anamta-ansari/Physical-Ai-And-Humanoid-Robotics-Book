from fastapi import APIRouter, HTTPException, Depends
from typing import Optional
import uuid
from datetime import datetime
from models.chat import ChatRequest, ChatResponse
from models.chat_session import ChatSession
from services.rag_service import RAGService
from services.history_service import HistoryService
from utils.logging import log_prompt_response
from config.settings import settings

router = APIRouter()

# Initialize services
rag_service = RAGService()
history_service = HistoryService()


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatRequest):
    """Endpoint for chat functionality."""
    session_id = None
    try:
        # Validate input
        if not chat_request.message or not chat_request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        if len(chat_request.message) > 10000:  # Set a reasonable limit
            raise HTTPException(status_code=400, detail="Message too long")

        if chat_request.selected_text and len(chat_request.selected_text) > 5000:
            raise HTTPException(status_code=400, detail="Selected text too long")

        # Generate a session ID if not provided
        session_id = chat_request.session_id or str(uuid.uuid4())

        # Create or update session in history
        if settings.NEON_DB_URL:
            try:
                history_service.create_session(session_id)
            except Exception as e:
                print(f"Warning: Could not create session in history: {str(e)}")
                # Continue without session history if DB is unavailable

        # Query the RAG service
        result = rag_service.query(chat_request.message, chat_request.selected_text)

        # Log the prompt and response
        log_prompt_response(chat_request.message, result["response"], session_id)

        # Return the response
        return ChatResponse(
            response=result["response"],
            session_id=session_id,
            sources=[source["source"] for source in result["sources"]]
        )
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Log unexpected errors
        error_msg = f"Error processing chat request: {str(e)}"
        print(error_msg)  # Also print to console for debugging

        # Log the error in prompts.md
        log_prompt_response(
            f"ERROR processing request: {chat_request.message if chat_request else 'No message'}",
            f"Exception: {str(e)}",
            session_id or getattr(chat_request, 'session_id', 'unknown')
        )

        raise HTTPException(
            status_code=500,
            detail=error_msg
        )


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Test if Qdrant collection exists
        collection_exists = rag_service.check_collection_exists()

        if not collection_exists:
            return {
                "status": "warning",
                "message": "Qdrant collection does not exist. Ingest documents first.",
                "timestamp": datetime.now().isoformat()
            }

        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }


@router.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get session information."""
    if not settings.NEON_DB_URL:
        raise HTTPException(status_code=400, detail="Session history not enabled")
    
    session = history_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return session


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    if not settings.NEON_DB_URL:
        raise HTTPException(status_code=400, detail="Session history not enabled")
    
    history_service.delete_session(session_id)
    return {"message": "Session deleted successfully"}