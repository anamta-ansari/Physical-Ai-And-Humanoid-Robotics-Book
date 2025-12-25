from fastapi import APIRouter, HTTPException
from models.documents import DocumentIngestionResponse
from services.document_loader import DocumentLoaderService
from services.rag_service import RAGService
from utils.logging import log_ingestion_event
from config.settings import settings
import os

router = APIRouter()

# Initialize services
loader_service = DocumentLoaderService()  # Use default docs path
rag_service = RAGService()


@router.post("/ingest", response_model=DocumentIngestionResponse)
async def ingest_documents():
    """Endpoint to ingest all documents from the docs directory into the vector store."""
    try:
        print("Starting document ingestion process...")

        # Check if RAG service is initialized
        if not rag_service.initialized:
            error_msg = "RAG service is not properly initialized. Please check Qdrant connection and API keys."
            print(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)

        # Load documents
        documents = loader_service.load_documents()
        if not documents:
            print("No documents found in docs directory")
            raise HTTPException(status_code=404, detail="No documents found in docs directory")

        print(f"Loaded {len(documents)} documents from docs directory")

        # Chunk documents
        chunks = loader_service.chunk_documents(documents)
        print(f"Created {len(chunks)} chunks from documents")

        # Ingest into RAG service
        count = rag_service.ingest_documents(chunks)
        print(f"Successfully ingested {count} document chunks into vector store")

        # Log the ingestion event
        log_ingestion_event(
            status="success",
            message=f"Successfully ingested {len(chunks)} document chunks",
            count=count
        )

        return DocumentIngestionResponse(
            status="success",
            message=f"Successfully ingested {len(chunks)} document chunks",
            documents_processed=len(documents),
            chunks_created=len(chunks)
        )
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        error_msg = f"Error ingesting documents: {str(e)}"
        print(error_msg)  # Print to console for debugging

        log_ingestion_event(
            status="error",
            message=error_msg,
            count=0
        )
        raise HTTPException(status_code=500, detail=error_msg)


@router.get("/status")
async def ingestion_status():
    """Check the status of the document collection."""
    try:
        collection_exists = rag_service.check_collection_exists()

        if not collection_exists:
            return {
                "status": "warning",
                "message": "Qdrant collection does not exist. Run ingestion first.",
                "collection_exists": collection_exists,
                "document_count": 0
            }

        # Get the count of documents in the collection
        try:
            collection_info = rag_service.qdrant_client.get_collection(settings.QDRANT_COLLECTION_NAME)
            document_count = collection_info.points_count
        except:
            document_count = 0  # If we can't get the count, default to 0

        return {
            "status": "ready",
            "message": "Qdrant collection is ready for queries.",
            "collection_exists": collection_exists,
            "document_count": document_count
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "collection_exists": False,
            "document_count": 0
        }