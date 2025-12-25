# Feature Specification: RAG Chatbot

**Feature Branch**: `1-rag-chatbot`
**Created**: 2025-12-24
**Status**: Draft
**Input**: User description: "Write clear specifications for the RAG Chatbot: Backend (backend/ folder): FastAPI app with /ingest (index book content), /chat (POST {message, selected_text?, session_id?} → retrieve from Qdrant → augment → stream Gemini response). Embeddings: Gemini (via langchain-google-genai, model compatible with free tier e.g., embedding-001 or latest free). Generation: gemini-1.5-flash. Vector store: Qdrant Cloud collection "physical_ai_book" with metadata (source chapter/part). Optional history: Neon Postgres table "sessions" for conversation persistence. RAG pipeline: Load/chunk all .md files → embed → upsert Qdrant; on query retrieve top-5-10 → context (prioritize selected_text if provided). Frontend: React component in src/components/RagChatBot/ – floating button opens modal with history/streaming; support selected text (window.getSelection() → send to backend). Add route /chat. Security/Privacy: Keys from .env only. Run: Automatic ingestion on first backend start or via /ingest."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Book Content Q&A (Priority: P1)

As a reader of the Physical AI & Humanoid Robotics book, I want to ask questions about the book content and get accurate answers based on the book's information, so I can better understand complex topics without having to manually search through chapters.

**Why this priority**: This is the core value proposition of the chatbot - providing intelligent answers based on the book's content, which is the primary function users will expect.

**Independent Test**: The system can be tested by asking specific questions about book content and verifying that the responses are accurate, relevant, and sourced from the appropriate chapters. The chatbot should be able to answer questions about fundamental concepts, specific details, and relationships between topics within the book.

**Acceptance Scenarios**:

1. **Given** a user is viewing any page of the book, **When** the user types a question about book content in the chat interface, **Then** the system returns a relevant answer based on the book's content with appropriate context.
2. **Given** a user has selected specific text in the book, **When** the user activates the chat function, **Then** the system prioritizes the selected text in its response and provides relevant information related to that selection.

---

### User Story 2 - Context-Aware Responses (Priority: P2)

As a user, I want the chatbot to provide context-aware responses that take into account the specific part of the book I'm reading, so I can get more targeted and relevant information.

**Why this priority**: This enhances the user experience by making the chatbot more intelligent and responsive to the user's immediate context.

**Independent Test**: When a user is on a specific chapter page, the system should prioritize information from that chapter or related chapters in its responses, rather than providing generic answers from across the entire book.

**Acceptance Scenarios**:

1. **Given** a user is viewing a chapter about ROS 2, **When** the user asks a question about robotic communication, **Then** the system prioritizes information from ROS 2 chapters in its response.
2. **Given** a user has selected text from a specific section, **When** the user asks a follow-up question, **Then** the system maintains context from the selected text in its response.

---

### User Story 3 - Conversation History (Priority: P3)

As a user, I want to maintain conversation history during my session, so I can have natural, multi-turn conversations with the chatbot without losing context.

**Why this priority**: This provides a more natural and efficient user experience, allowing for follow-up questions and complex discussions.

**Independent Test**: The system maintains conversation context across multiple exchanges, allowing users to ask follow-up questions that refer back to previous parts of the conversation.

**Acceptance Scenarios**:

1. **Given** a user has had a conversation with the chatbot, **When** the user asks a follow-up question that references previous exchanges, **Then** the system understands the context and provides an appropriate response.
2. **Given** a user has a conversation history, **When** the user returns to the site in the same session, **Then** the conversation history is preserved and accessible.

---

### Edge Cases

- What happens when the knowledge base service is temporarily unavailable?
- How does the system handle extremely long user queries or selected text?
- What happens when the AI generation service is rate-limited or unavailable?
- How does the system handle queries that have no relevant information in the book?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST provide a chat interface accessible via a floating button on all book pages and a dedicated /chat route
- **FR-002**: System MUST index all book chapters from the docs/ directory with appropriate chunking and chapter/part metadata
- **FR-003**: System MUST retrieve relevant information from the knowledge base when processing user queries (top-5-10 most relevant chunks)
- **FR-004**: System MUST generate responses using an AI service
- **FR-005**: System MUST accept user queries via a /chat endpoint that takes message, selected_text (optional), and session_id (optional)
- **FR-006**: System MUST support selected text functionality where users can select text in the book and send it to the backend for context
- **FR-007**: System MUST stream responses from the AI service to the frontend for real-time display
- **FR-008**: System MUST have an /ingest endpoint to index book content (run automatically on first backend start)
- **FR-009**: System MUST support optional conversation session persistence
- **FR-010**: System MUST securely manage API keys without exposing configuration file contents
- **FR-011**: System MUST prioritize selected text context when provided in user queries
- **FR-012**: System MUST log all prompts and responses for audit and improvement purposes

*Example of marking unclear requirements:*

- **FR-013**: System MUST determine session persistence duration (default: 24 hours of inactivity before session expires)
- **FR-014**: System MUST handle embedding model selection (default: use latest free tier compatible model available)

### Key Entities *(include if feature involves data)*

- **ChatSession**: Represents a conversation session with history and metadata (session_id, creation_time, last_interaction_time)
- **DocumentChunk**: Represents a chunk of book content stored with metadata (source_file, chapter, part, content)
- **UserQuery**: Represents a user's input to the system (message, selected_text, session_id, timestamp)
- **ChatResponse**: Represents the system's response to a user query (response_text, source_chunks, timestamp)

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can ask questions about book content and receive relevant answers within 5 seconds
- **SC-002**: The system successfully retrieves relevant information from the book for 90% of user queries
- **SC-003**: Users can maintain coherent multi-turn conversations with the chatbot for at least 5 exchanges
- **SC-004**: The system correctly prioritizes selected text context in 95% of queries where text is selected
- **SC-005**: Users report a satisfaction score of 4 or higher (out of 5) for the helpfulness of chatbot responses
- **SC-006**: The system handles at least 100 concurrent users without degradation in response quality or time