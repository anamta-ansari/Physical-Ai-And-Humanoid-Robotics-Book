from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config.settings import settings
from api.routes import chat, ingest
import asyncio
import atexit
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    print("Starting up RAG Chatbot API...")

    # Startup: Run initial ingestion
    import threading
    import requests
    import time

    def run_initial_ingestion():
        """Run initial document ingestion when the application starts."""
        try:
            import time
            import requests

            # Wait a moment for the server to start
            time.sleep(2)

            # First check if collection exists and has documents
            status_response = requests.get("http://localhost:8000/api/v1/ingest/status")
            if status_response.status_code == 200:
                status_data = status_response.json()
                if status_data.get("status") == "ready" and status_data.get("document_count", 0) > 0:
                    print(f"Qdrant collection already exists with {status_data.get('document_count', 0)} documents. Skipping initial ingestion.")
                    return
                elif status_data.get("status") == "ready":
                    print("Qdrant collection exists but is empty. Proceeding with initial ingestion...")
            else:
                print("Could not check collection status, proceeding with ingestion...")

            # Run the ingestion endpoint to index documents
            response = requests.post("http://localhost:8000/api/v1/ingest")
            if response.status_code == 200:
                print("Initial document ingestion completed successfully")
            else:
                print(f"Initial document ingestion failed with status: {response.status_code}")
                print(f"Response: {response.text}")
        except requests.exceptions.ConnectionError:
            print("Could not connect to the server for initial ingestion. Documents will need to be ingested manually.")
        except Exception as e:
            print(f"Error during initial ingestion: {str(e)}")
            print("Server will continue running but documents may not be ingested.")

    # Run ingestion in a separate thread to not block startup
    thread = threading.Thread(target=run_initial_ingestion)
    thread.start()

    yield  # This is where the application runs

    # Shutdown: Perform cleanup if needed
    print("Shutting down RAG Chatbot API...")

app = FastAPI(
    title=settings.APP_NAME,
    lifespan=lifespan
)

# Add CORS middleware to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:8000"],  # Allow frontend and backend origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(chat.router, prefix=settings.API_V1_STR)
app.include_router(ingest.router, prefix=settings.API_V1_STR)

# Mount chat router at root /chat path as well for direct access
app.include_router(chat.router, prefix="/chat")

@app.get("/")
def read_root():
    return {"message": "RAG Chatbot API is running!"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": "2025-12-24T00:00:00Z"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)