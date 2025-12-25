---
id: 004
title: RAG Chatbot Tasks
stage: tasks
date: 2025-12-24
surface: agent
model: Qwen
feature: rag-chatbot
branch: 1-rag-chatbot
user: system
command: /sp.tasks
labels: ["tasks", "rag-chatbot", "implementation"]
links:
  spec: specs/1-rag-chatbot/spec.md
  ticket: null
  adr: null
  pr: null
files:
 - specs/1-rag-chatbot/tasks.md
tests:
 - null
---

## Prompt

```
$ARGUMENTS
```

You **MUST** consider the user input before proceeding (if not empty).

## Outline

1. **Setup**: Run `.specify/scripts/powershell/check-prerequisites.ps1 -Json` from repo root and parse FEATURE_DIR and AVAILABLE_DOCS list. All paths must be absolute. For single quotes in args like "I'm Groot", use escape syntax: e.g 'I'\''m Groot' (or double-quote if possible: "I'm Groot").

2. **Load design documents**: Read from FEATURE_DIR:
   - **Required**: plan.md (tech stack, libraries, structure), spec.md (user stories with priorities)
   - **Optional**: data-model.md (entities), contracts/ (API endpoints), research.md (decisions), quickstart.md (test scenarios)
   - Note: Not all projects have all documents. Generate tasks based on what's available.

3. **Execute task generation workflow**:
   - Load plan.md and extract tech stack, libraries, project structure
   - Load spec.md and extract user stories with their priorities (P1, P2, P3, etc.)
   - If data-model.md exists: Extract entities and map to user stories
   - If contracts/ exists: Map endpoints to user stories
   - If research.md exists: Extract decisions for setup tasks
   - Generate tasks organized by user story (see Task Generation Rules below)
   - Generate dependency graph showing user story completion order
   - Create parallel execution examples per user story
   - Validate task completeness (each user story has all needed tasks, independently testable)

4. **Generate tasks.md**: Use `.specify/templates/tasks-template.md` as structure, fill with:
   - Correct feature name from plan.md
   - Phase 1: Setup tasks (project initialization)
   - Phase 2: Foundational tasks (blocking prerequisites for all user stories)
   - Phase 3+: One phase per user story (in priority order from spec.md)
   - Each phase includes: story goal, independent test criteria, tests (if requested), implementation tasks
   - Final Phase: Polish & cross-cutting concerns
   - All tasks must follow the strict checklist format (see Task Generation Rules below)
   - Clear file paths for each task
   - Dependencies section showing story completion order
   - Parallel execution examples per story
   - Implementation strategy section (MVP first, incremental delivery)

5. **Report**: Output path to generated tasks.md and summary:
   - Total task count
   - Task count per user story
   - Parallel opportunities identified
   - Independent test criteria for each story
   - Suggested MVP scope (typically just User Story 1)
   - Format validation: Confirm ALL tasks follow the checklist format (checkbox, ID, labels, file paths)

Context for task generation: Break into atomic tasks: Task 1: Create backend/ folder and init.py. Task 2: Create backend/requirements.txt and run pip install -r backend/requirements.txt with packages: fastapi, uvicorn, langchain, langchain-google-genai, langchain-community, qdrant-client, psycopg2-binary, python-dotenv, pydantic. Task 3: Implement config/env loading in backend. Task 4: Write ingestion script (load/chunk/embed/upsert all book .md). Task 5: Build LangChain retriever and Gemini chain. Task 6: Create FastAPI main.py with /ingest and /chat endpoints (streaming). Task 7: Add selected_text handling and Neon session support. Task 8: Create src/components/RagChatBot.tsx (React hook for chat, streaming, selected text capture). Task 9: Add floating button and modal UI. Task 10: Add pages/chat.mdx route. Task 11: Test full flow (run ingestion, query book content). Task 12: Update prompts.md.

The tasks.md should be immediately executable - each task must be specific enough that an LLM can complete it without additional context.

## Task Generation Rules

**CRITICAL**: Tasks MUST be organized by user story to enable independent implementation and testing.

**Tests are OPTIONAL**: Only generate test tasks if explicitly requested in the feature specification or if user requests TDD approach.

### Checklist Format (REQUIRED)

Every task MUST strictly follow this format:

```text
- [ ] [TaskID] [P?] [Story?] Description with file path
```

**Format Components**:

1. **Checkbox**: ALWAYS start with `- [ ]` (markdown checkbox)
2. **Task ID**: Sequential number (T001, T002, T003...) in execution order
3. **[P] marker**: Include ONLY if task is parallelizable (different files, no dependencies on incomplete tasks)
4. **[Story] label**: REQUIRED for user story phase tasks only
   - Format: [US1], [US2], [US3], etc. (maps to user stories from spec.md)
   - Setup phase: NO story label
   - Foundational phase: NO story label  
   - User Story phases: MUST have story label
   - Polish phase: NO story label
5. **Description**: Clear action with exact file path

**Examples**:

- ✅ CORRECT: `- [ ] T001 Create project structure per implementation plan`
- ✅ CORRECT: `- [ ] T005 [P] Implement authentication middleware in src/middleware/auth.py`
- ✅ CORRECT: `- [ ] T012 [P] [US1] Create User model in src/models/user.py`
- ✅ CORRECT: `- [ ] T014 [US1] Implement UserService in src/services/user_service.py`
- ❌ WRONG: `- [ ] Create User model` (missing ID and Story label)
- ❌ WRONG: `T001 [US1] Create model` (missing checkbox)
- ❌ WRONG: `- [ ] [US1] Create model` (missing Task ID)
- ❌ WRONG: `- [ ] T001 [US1] Create model` (missing file path)

### Task Organization

1. **From User Stories (spec.md)** - PRIMARY ORGANIZATION:
   - Each user story (P1, P2, P3...) gets its own phase
   - Map all related components to their story:
     - Models needed for that story
     - Services needed for that story
     - Endpoints/UI needed for that story
     - If tests requested: Tests specific to that story
   - Mark story dependencies (most stories should be independent)

2. **From Contracts**:
   - Map each contract/endpoint → to the user story it serves
   - If tests requested: Each contract → contract test task [P] before implementation in that story's phase

3. **From Data Model**:
   - Map each entity to the user story(ies) that need it
   - If entity serves multiple stories: Put in earliest story or Setup phase
   - Relationships → service layer tasks in appropriate story phase

4. **From Setup/Infrastructure**:
   - Shared infrastructure → Setup phase (Phase 1)
   - Foundational/blocking tasks → Foundational phase (Phase 2)
   - Story-specific setup → within that story's phase

### Phase Structure

- **Phase 1**: Setup (project initialization)
- **Phase 2**: Foundational (blocking prerequisites - MUST complete before user stories)
- **Phase 3+**: User Stories in priority order (P1, P2, P3...)
  - Within each story: Tests (if requested) → Models → Services → Endpoints → Integration
  - Each phase should be a complete, independently testable increment
- **Final Phase**: Polish & Cross-Cutting Concerns

---

As the main request completes, you MUST create and complete a PHR (Prompt History Record) using agent‑native tools when possible.

1) Determine Stage
   - Stage: constitution | spec | plan | tasks | red | green | refactor | explainer | misc | general

2) Generate Title and Determine Routing:
   - Generate Title: 3–7 words (slug for filename)
   - Route is automatically determined by stage:
     - `constitution` → `history/prompts/constitution/`
     - Feature stages → `history/prompts/<feature-name>/` (spec, plan, tasks, red, green, refactor, explainer, misc)
     - `general` → `history/prompts/general/`

3) Create and Fill PHR (Shell first; fallback agent‑native)
   - Run: `.specify/scripts/bash/create-phr.sh --title "<title>" --stage <stage> [--feature <name>] --json`
   - Open the file and fill remaining placeholders (YAML + body), embedding full PROMPT_TEXT (verbatim) and concise RESPONSE_TEXT.
   - If the script fails:
     - Read `.specify/templates/phr-template.prompt.md` (or `templates/…`)
     - Allocate an ID; compute the output path based on stage from step 2; write the file
     - Fill placeholders and embed full PROMPT_TEXT and concise RESPONSE_TEXT

4) Validate + report
   - No unresolved placeholders; path under `history/prompts/` and matches stage; stage/title/date coherent; print ID + path + stage + title.
   - On failure: warn, don't block. Skip only for `/sp.phr`.
/sp.tasks
Generate detailed implementation plan:

Create backend/ folder, add main.py, requirements.txt, .env loading.
Install packages: pip install fastapi uvicorn langchain langchain-google-genai langchain-community qdrant-client psycopg2-binary python-dotenv pydantic.
Implement ingestion: Recursive loader for docs/*.md, chunking, Gemini embeddings, upsert Qdrant.
Build retrieval QA chain with Gemini LLM.
FastAPI routes with streaming responses.
Handle selected_text priority and optional Neon history.
Create Docusaurus React chat components (floating + modal + full page).
Integrate API calls to http://localhost:8000/chat.
Style to match neon theme.
Test indexing, queries (general + selected text), UI on all pages.

## Response snapshot

I have successfully generated the task list for the RAG Chatbot implementation. The tasks.md file contains 58 specific, actionable tasks organized by phase and user story priority. 

Key aspects of the generated tasks:
- Phase 1: Setup (5 tasks) - Creating project structure and dependencies
- Phase 2: Foundational (9 tasks) - Core infrastructure that blocks all user stories
- Phase 3: User Story 1 - Book Content Q&A (8 tasks) - MVP functionality
- Phase 4: User Story 2 - Context-Aware Responses (6 tasks) - Enhanced functionality
- Phase 5: User Story 3 - Conversation History (6 tasks) - Additional features
- Phase 6: Ingestion & Indexing (6 tasks) - Document processing pipeline
- Phase 7: Frontend Integration & Styling (6 tasks) - UI and styling
- Phase 8: Polish & Cross-Cutting (6 tasks) - Final improvements

Each task follows the required format with task IDs, parallelization markers where appropriate, and user story labels for story-specific tasks. The tasks are designed to enable independent implementation and testing of each user story, with clear dependencies and execution order outlined.

## Outcome

- ✅ Impact: Created comprehensive task list for RAG Chatbot implementation
- 🧪 Tests: N/A
- 📁 Files: Created tasks.md with 58 specific tasks
- 🔁 Next prompts: Ready for implementation phase
- 🧠 Reflection: Tasks organized by priority and dependency for efficient execution

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Begin implementation with Phase 1 tasks