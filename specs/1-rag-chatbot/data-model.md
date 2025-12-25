# Data Model: RAG Chatbot

## Entities

### ChatSession
Represents a conversation session with history and metadata
- **session_id**: string (unique identifier)
- **creation_time**: timestamp
- **last_interaction_time**: timestamp
- **active**: boolean (whether session is still active)

### DocumentChunk
Represents a chunk of book content stored with metadata
- **chunk_id**: string (unique identifier)
- **source_file**: string (path to original document)
- **chapter**: string (chapter name from book)
- **part**: string (part name from book)
- **content**: string (the actual text content)
- **embedding**: vector (the embedding vector for similarity search)
- **metadata**: object (additional metadata)

### UserQuery
Represents a user's input to the system
- **query_id**: string (unique identifier)
- **session_id**: string (reference to ChatSession)
- **message**: string (the user's question/message)
- **selected_text**: string (optional selected text from book)
- **timestamp**: timestamp
- **processed**: boolean (whether the query has been processed)

### ChatResponse
Represents the system's response to a user query
- **response_id**: string (unique identifier)
- **query_id**: string (reference to UserQuery)
- **response_text**: string (the AI-generated response)
- **source_chunks**: array (list of DocumentChunk IDs used to generate response)
- **timestamp**: timestamp
- **confidence_score**: float (confidence in response accuracy)

## Relationships
- ChatSession "has many" UserQuery
- UserQuery "has one" ChatResponse
- ChatResponse "uses many" DocumentChunk

## Validation Rules
- session_id must be unique
- content in DocumentChunk must not exceed token limits for embedding model
- selected_text in UserQuery must be from the book content
- ChatSession must expire after 24 hours of inactivity

## State Transitions
- ChatSession: active (true) → inactive (false) after 24 hours of inactivity