# RAG Chatbot for Physical AI & Humanoid Robotics Book

This project implements a Retrieval-Augmented Generation (RAG) chatbot for the Physical AI & Humanoid Robotics book. The solution includes a FastAPI backend with document ingestion and chat endpoints, using Google's Gemini API for embeddings and generation, Qdrant Cloud for vector storage, and React frontend components integrated into the Docusaurus site.

## Prerequisites

- Python 3.11+
- Node.js 18+ (for Docusaurus frontend)
- Google Gemini API key
- Qdrant Cloud account and API key
- (Optional) Neon Postgres account and connection string

## Setup Instructions

### 1. Backend Setup

```bash
# Navigate to the backend directory
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file in the backend directory with your API keys:

```env
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

### 4. Start the Services

#### Backend:
```bash
cd backend
source venv/bin/activate  # On Windows: venv\Scripts\activate
uvicorn main:app --reload --port 8000
```

#### Frontend:
```bash
# From repository root
npm start
```

## API Usage

### Ingest Documents
```bash
curl -X POST http://localhost:8000/api/v1/ingest
```

### Chat Endpoint
```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is ROS 2?",
    "selected_text": "Optional text selected by user",
    "session_id": "Optional session ID"
  }'
```

## Features

1. **Document Ingestion**: Automatically indexes all book content from the docs/ directory
2. **Chat Interface**: Available as a floating button on all pages and a dedicated /chat route
3. **Selected Text Support**: Users can select text on any page and ask questions about it
4. **Session Management**: Optional conversation history using Neon Postgres
5. **Responsive Design**: Mobile-friendly interface with neon-themed styling
6. **Logging**: All prompts and responses are logged to prompts.md for audit and improvement

## Architecture

- **Backend**: FastAPI application in the `backend/` directory
- **Frontend**: React components integrated with Docusaurus
- **Vector Store**: Qdrant Cloud for document embeddings
- **LLM**: Google Gemini (gemini-1.5-flash) for generation
- **Embeddings**: Google Gemini for text embeddings
- **Session Storage**: Optional Neon Postgres for conversation history