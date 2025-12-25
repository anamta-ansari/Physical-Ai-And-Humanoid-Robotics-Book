# Implementation Plan: RAG Chatbot

**Branch**: `1-rag-chatbot` | **Date**: 2025-12-24 | **Spec**: [link to spec](spec.md)
**Input**: Feature specification from `/specs/1-rag-chatbot/spec.md`

**Note**: This template is filled in by the `/sp.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Implementation of a Retrieval-Augmented Generation (RAG) chatbot for the Physical AI & Humanoid Robotics book. The solution will include a FastAPI backend with document ingestion and chat endpoints, using Google's Gemini API for embeddings and generation, Qdrant Cloud for vector storage, and React frontend components integrated into the Docusaurus site. The system will support general queries about book content and prioritized responses based on user-selected text.

## Technical Context

**Language/Version**: Python 3.11, JavaScript/TypeScript for frontend components
**Primary Dependencies**: FastAPI, LangChain, langchain-google-genai, qdrant-client, psycopg2-binary, python-dotenv, React
**Storage**: Qdrant Cloud (vector store), Neon Postgres (optional session history), local Markdown files from docs/
**Testing**: pytest for backend, Jest for frontend components
**Target Platform**: Web application (Docusaurus frontend + FastAPI backend)
**Project Type**: Web application (frontend + backend)
**Performance Goals**: Response time <5 seconds for 95% of queries, support 100+ concurrent users
**Constraints**: Must use free-tier services only (Google Gemini API, Qdrant Cloud free tier, Neon Serverless Postgres)
**Scale/Scope**: Single book content (Physical AI & Humanoid Robotics), multiple chapters and parts

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### RAG Architecture Compliance
- [x] Confirm all components use free-tier services only (Google Gemini API, Qdrant Cloud free tier, Neon Serverless Postgres)
- [x] Verify LangChain pipeline architecture with required integrations (langchain-google-genai, qdrant-client, psycopg2-binary)
- [x] Ensure secure environment management (no .env file access, proper credential handling)
- [x] Confirm document processing & indexing from docs/ directory to Qdrant with metadata
- [x] Validate backend implementation in "backend" folder using FastAPI
- [x] Verify frontend integration with Docusaurus (floating bubble + /chat route)
- [x] Check that all prompts/responses are logged in prompts.md

## Project Structure

### Documentation (this feature)

```text
specs/1-rag-chatbot/
├── plan.md              # This file (/sp.plan command output)
├── research.md          # Phase 0 output (/sp.plan command)
├── data-model.md        # Phase 1 output (/sp.plan command)
├── quickstart.md        # Phase 1 output (/sp.plan command)
├── contracts/           # Phase 1 output (/sp.plan command)
└── tasks.md             # Phase 2 output (/sp.tasks command - NOT created by /sp.plan)
```

### Source Code (repository root)

```text
backend/
├── main.py              # FastAPI application entry point
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (not committed)
├── config/
│   └── settings.py      # Configuration management
├── api/
│   ├── __init__.py
│   ├── routes/
│   │   ├── ingest.py    # Ingestion endpoint
│   │   └── chat.py      # Chat endpoint
├── services/
│   ├── __init__.py
│   ├── rag_service.py   # Core RAG functionality
│   ├── document_loader.py # Document processing
│   └── history_service.py # Conversation history
├── models/
│   ├── __init__.py
│   ├── chat.py          # Chat request/response models
│   └── documents.py     # Document models
└── tests/
    ├── __init__.py
    ├── test_ingest.py
    └── test_chat.py

src/components/RagChatBot/
├── RagChatBot.jsx       # Main chatbot component
├── ChatModal.jsx        # Modal interface
├── FloatingButton.jsx   # Floating button component
├── ChatHistory.jsx      # Conversation history
├── styles.css           # Component styles
└── index.js             # Export component

docs/
└── (existing book content - will be indexed)

prompts.md               # Log file for all prompts/responses
```

**Structure Decision**: Web application with separate backend (FastAPI) and frontend (React components in Docusaurus) following the architecture requirements in the constitution.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| N/A | N/A | N/A |