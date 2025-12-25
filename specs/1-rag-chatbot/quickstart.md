# Quickstart Guide: RAG Chatbot

## Prerequisites
- Python 3.11+
- Node.js 18+ (for Docusaurus frontend)
- Google Gemini API key
- Qdrant Cloud account and API key
- (Optional) Neon Postgres account and connection string

## Setup Instructions

### 1. Clone the repository
```bash
git clone <repository-url>
cd <repository-name>
git checkout 1-rag-chatbot
```

### 2. Backend Setup
```bash
# Navigate to the backend directory
mkdir backend
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install fastapi uvicorn langchain langchain-google-genai langchain-community qdrant-client psycopg2-binary python-dotenv pydantic

# Create .env file with your API keys
touch .env
```

Add the following to your `.env` file:
```
GEMINI_API_KEY=your_gemini_api_key_here
QDRANT_API_KEY=your_qdrant_api_key_here
QDRANT_URL=your_qdrant_cloud_url
NEON_DB_URL=your_neon_postgres_connection_string (optional)
```

### 3. Frontend Setup
```bash
# From repository root
npm install
```

### 4. Index the book content
```bash
# From backend directory
python -c "
import sys
sys.path.append('.')
from api.routes import ingest
# Run the ingestion process
ingest.index_documents()
"
```

### 5. Start the backend server
```bash
# From backend directory
uvicorn main:app --reload --port 8000
```

### 6. Start the Docusaurus frontend
```bash
# From repository root
npm start
```

## API Usage

### Ingest Documents
```bash
curl -X POST http://localhost:8000/ingest
```

### Chat Endpoint
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is ROS 2?",
    "selected_text": "Optional text selected by user",
    "session_id": "Optional session ID"
  }'
```

## Frontend Integration

The chatbot is integrated into the Docusaurus site with:
1. A floating button on all pages that opens the chat modal
2. A dedicated /chat route for the full interface
3. Support for sending selected text to the backend

## Troubleshooting

### Common Issues
- API keys not loaded: Ensure .env file is properly configured and in the right location
- Document indexing fails: Check that docs/ directory exists and contains Markdown files
- Chat responses are slow: Verify your API keys and network connection

### Environment Variables
Make sure these environment variables are set:
- `GEMINI_API_KEY`: Your Google Gemini API key
- `QDRANT_API_KEY`: Your Qdrant Cloud API key
- `QDRANT_URL`: Your Qdrant Cloud URL
- `NEON_DB_URL`: (Optional) Neon Postgres connection string