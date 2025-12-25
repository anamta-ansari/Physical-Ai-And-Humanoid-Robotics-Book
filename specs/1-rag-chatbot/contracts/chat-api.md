# Chat API Contract

## Endpoints

### POST /ingest
**Description**: Index all book content from docs/ directory into Qdrant vector store

**Request**:
- Headers: None required
- Body: None required

**Response**:
- 200: { "status": "success", "message": "Documents indexed successfully", "count": number_of_documents }
- 500: { "status": "error", "message": "Error message" }

### POST /chat
**Description**: Process user query and return AI-generated response with streaming

**Request**:
- Headers: 
  - Content-Type: application/json
- Body:
  ```json
  {
    "message": "string (user's question)",
    "selected_text": "string (optional selected text from book)",
    "session_id": "string (optional session identifier)"
  }
  ```

**Response**:
- 200 (streaming): Stream of response tokens
- 400: { "status": "error", "message": "Invalid request parameters" }
- 500: { "status": "error", "message": "Error processing query" }

### GET /health
**Description**: Check the health status of the service

**Request**:
- Headers: None required
- Body: None required

**Response**:
- 200: { "status": "healthy", "timestamp": "ISO date string" }
- 503: { "status": "unhealthy", "message": "Error message" }