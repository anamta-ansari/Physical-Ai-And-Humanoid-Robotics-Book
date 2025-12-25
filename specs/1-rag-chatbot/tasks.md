---

description: "Task list for RAG Chatbot implementation"
---

# Tasks: RAG Chatbot

**Input**: Design documents from `/specs/1-rag-chatbot/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: The examples below include test tasks. Tests are OPTIONAL - only include them if explicitly requested in the feature specification.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `src/`, `tests/` at repository root
- **Web app**: `backend/src/`, `frontend/src/`
- **Mobile**: `api/src/`, `ios/src/` or `android/src/`
- Paths shown below assume single project - adjust based on plan.md structure

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [x] T001 Create backend/ folder for FastAPI application
- [x] T002 Create backend/requirements.txt with dependencies: fastapi, uvicorn, langchain, langchain-google-genai, langchain-community, qdrant-client, psycopg2-binary, python-dotenv, pydantic
- [x] T003 [P] Install backend dependencies with pip install -r backend/requirements.txt
- [x] T004 Create backend/.env file structure for API keys (not committed)
- [x] T005 Create src/components/RagChatBot/ directory for React components

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T006 Create backend/config/settings.py for environment configuration management
- [x] T007 [P] Implement document loader in backend/services/document_loader.py for Markdown files from docs/ directory
- [x] T008 [P] Setup API routing structure in backend/main.py with FastAPI
- [x] T009 Create backend/models/chat.py with Pydantic models for chat requests/responses
- [x] T010 Configure logging infrastructure to write prompts/responses to prompts.md
- [x] T011 Setup secure credential handling in backend/config/settings.py
- [x] T012 Implement document chunking mechanism in backend/services/document_loader.py
- [x] T013 Create backend/models/documents.py with Pydantic models for document chunks
- [x] T014 Setup Qdrant Cloud connection in backend/services/rag_service.py

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Book Content Q&A (Priority: P1) 🎯 MVP

**Goal**: Enable users to ask questions about book content and receive accurate answers based on the book's information

**Independent Test**: The system can be tested by asking specific questions about book content and verifying that the responses are accurate, relevant, and sourced from the appropriate chapters. The chatbot should be able to answer questions about fundamental concepts, specific details, and relationships between topics within the book.

### Tests for User Story 1 (OPTIONAL - only if tests requested) ⚠️

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T015 [P] [US1] Contract test for /chat endpoint in backend/tests/test_chat.py
- [ ] T016 [P] [US1] Integration test for basic Q&A functionality in backend/tests/test_qa_integration.py

### Implementation for User Story 1

- [x] T017 [P] [US1] Create backend/models/chat_session.py for session management
- [x] T018 [US1] Implement basic RAG service in backend/services/rag_service.py (load, embed, retrieve)
- [x] T019 [US1] Create /chat endpoint in backend/api/routes/chat.py with basic functionality
- [x] T020 [US1] Add validation and error handling for chat requests
- [x] T021 [US1] Add logging for user queries and responses to prompts.md
- [x] T022 [US1] Create src/components/RagChatBot/RagChatBot.jsx with basic chat interface
- [x] T023 [US1] Create src/components/RagChatBot/ChatModal.jsx for modal interface
- [x] T024 [US1] Implement API call functionality to http://localhost:8000/chat in React component

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - Context-Aware Responses (Priority: P2)

**Goal**: Provide context-aware responses that take into account the specific part of the book the user is reading

**Independent Test**: When a user is on a specific chapter page, the system should prioritize information from that chapter or related chapters in its responses, rather than providing generic answers from across the entire book.

### Tests for User Story 2 (OPTIONAL - only if tests requested) ⚠️

- [x] T025 [P] [US2] Contract test for selected text handling in backend/tests/test_selected_text.py
- [x] T026 [P] [US2] Integration test for context-aware responses in backend/tests/test_context_aware.py

### Implementation for User Story 2

- [x] T027 [P] [US2] Create backend/services/history_service.py for conversation persistence
- [x] T028 [US2] Enhance RAG service in backend/services/rag_service.py to prioritize selected text
- [x] T029 [US2] Update /chat endpoint in backend/api/routes/chat.py to handle selected_text parameter
- [x] T030 [US2] Implement selected text capture functionality in src/components/RagChatBot/RagChatBot.jsx using window.getSelection()
- [x] T031 [US2] Add UI indicators for selected text context in React components
- [x] T032 [US2] Create src/components/RagChatBot/FloatingButton.jsx for chat access on all pages

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - Conversation History (Priority: P3)

**Goal**: Maintain conversation history during the user's session for natural, multi-turn conversations

**Independent Test**: The system maintains conversation context across multiple exchanges, allowing users to ask follow-up questions that refer back to previous parts of the conversation.

### Tests for User Story 3 (OPTIONAL - only if tests requested) ⚠️

- [x] T033 [P] [US3] Contract test for session persistence in backend/tests/test_session_persistence.py
- [x] T034 [P] [US3] Integration test for multi-turn conversations in backend/tests/test_multi_turn.py

### Implementation for User Story 3

- [x] T035 [P] [US3] Create backend/models/chat_response.py for response management
- [x] T036 [US3] Enhance history service in backend/services/history_service.py with Neon Postgres integration
- [x] T037 [US3] Update /chat endpoint in backend/api/routes/chat.py to handle session_id parameter
- [x] T038 [US3] Implement conversation history UI in src/components/RagChatBot/ChatHistory.jsx
- [x] T039 [US3] Add session management functionality to React components
- [x] T040 [US3] Integrate with Neon Postgres for optional conversation persistence

**Checkpoint**: All user stories should now be independently functional

---

## Phase 6: Ingestion & Indexing

**Goal**: Implement document ingestion pipeline to index book content into Qdrant

- [x] T041 Create backend/api/routes/ingest.py with /ingest endpoint
- [x] T042 Implement document processing pipeline in backend/services/document_loader.py
- [x] T043 Add embedding functionality using Google Gemini in backend/services/rag_service.py
- [x] T044 Create backend/services/rag_service.py with upsert functionality to Qdrant
- [x] T045 Add automatic ingestion on first backend start in backend/main.py
- [x] T046 Create backend/tests/test_ingest.py for ingestion functionality

---

## Phase 7: Frontend Integration & Styling

**Goal**: Complete frontend integration with Docusaurus and styling to match neon theme

- [x] T047 Create src/components/RagChatBot/styles.css with neon purple futuristic theme
- [x] T048 Add mobile-responsive styling to all React components
- [x] T049 Create dedicated /chat route page in Docusaurus
- [x] T050 Integrate floating button on all Docusaurus pages
- [x] T051 Implement streaming response display in React components
- [x] T052 Add bright white text styling to match neon theme

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [x] T053 [P] Documentation updates in docs/
- [x] T054 Code cleanup and refactoring
- [x] T055 Performance optimization for document retrieval and response generation
- [x] T056 [P] Additional unit tests in backend/tests/ and frontend/tests/
- [x] T057 Security hardening for API key handling
- [x] T058 Run quickstart.md validation

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3)
- **Ingestion & Indexing (Phase 6)**: Can run in parallel with user stories once foundational is complete
- **Frontend Integration (Phase 7)**: Can run in parallel with other phases once foundational is complete
- **Polish (Final Phase)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - Builds upon US1 functionality
- **User Story 3 (P3)**: Can start after Foundational (Phase 2) - Builds upon US1/US2 functionality

### Within Each User Story

- Tests (if included) MUST be written and FAIL before implementation
- Models before services
- Services before endpoints
- Core implementation before integration
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Once Foundational phase completes, all user stories can start in parallel (if team capacity allows)
- All tests for a user story marked [P] can run in parallel
- Models within a story marked [P] can run in parallel
- Different user stories can be worked on in parallel by different team members

---

## Parallel Example: User Story 1

```bash
# Launch all tests for User Story 1 together (if tests requested):
Task: "Contract test for /chat endpoint in backend/tests/test_chat.py"
Task: "Integration test for basic Q&A functionality in backend/tests/test_qa_integration.py"

# Launch all models for User Story 1 together:
Task: "Create backend/models/chat_session.py for session management"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1
4. Complete Phase 6: Ingestion & Indexing (needed for US1 to work)
5. Complete Phase 7: Frontend Integration (basic UI for US1)
6. **STOP and VALIDATE**: Test User Story 1 independently
7. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 + Ingestion + Basic UI → Test independently → Deploy/Demo (MVP!)
3. Add User Story 2 → Test independently → Deploy/Demo
4. Add User Story 3 → Test independently → Deploy/Demo
5. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1
   - Developer B: User Story 2
   - Developer C: User Story 3
   - Developer D: Ingestion & Indexing
   - Developer E: Frontend Integration
3. Stories complete and integrate independently

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Verify tests fail before implementing
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence