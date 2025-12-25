---
id: 003
title: RAG Chatbot Plan
stage: plan
date: 2025-12-24
surface: agent
model: Qwen
feature: rag-chatbot
branch: 1-rag-chatbot
user: system
command: /sp.plan
labels: ["planning", "rag-chatbot", "architecture", "design"]
links:
  spec: specs/1-rag-chatbot/spec.md
  ticket: null
  adr: null
  pr: null
files:
 - specs/1-rag-chatbot/plan.md
 - specs/1-rag-chatbot/research.md
 - specs/1-rag-chatbot/data-model.md
 - specs/1-rag-chatbot/contracts/chat-api.md
 - specs/1-rag-chatbot/quickstart.md
tests:
 - null
---

## Prompt

```
$ARGUMENTS
```

You **MUST** consider the user input before proceeding (if not empty).

## Outline

1. **Setup**: Run `.specify/scripts/powershell/setup-plan.ps1 -Json` from repo root and parse JSON for FEATURE_SPEC, IMPL_PLAN, SPECS_DIR, BRANCH. For single quotes in args like "I'm Groot", use escape syntax: e.g 'I'\''m Groot' (or double-quote if possible: "I'm Groot").

2. **Load context**: Read FEATURE_SPEC and `.specify/memory/constitution.md`. Load IMPL_PLAN template (already copied).

3. **Execute plan workflow**: Follow the structure in IMPL_PLAN template to:
   - Fill Technical Context (mark unknowns as "NEEDS CLARIFICATION")
   - Fill Constitution Check section from constitution
   - Evaluate gates (ERROR if violations unjustified)
   - Phase 0: Generate research.md (resolve all NEEDS CLARIFICATION)
   - Phase 1: Generate data-model.md, contracts/, quickstart.md
   - Phase 1: Update agent context by running the agent script
   - Re-evaluate Constitution Check post-design

4. **Stop and report**: Command ends after Phase 2 planning. Report branch, IMPL_PLAN path, and generated artifacts.

## Phases

### Phase 0: Outline & Research

1. **Extract unknowns from Technical Context** above:
   - For each NEEDS CLARIFICATION → research task
   - For each dependency → best practices task
   - For each integration → patterns task

2. **Generate and dispatch research agents**:

   ```text
   For each unknown in Technical Context:
     Task: "Research {unknown} for {feature context}"
   For each technology choice:
     Task: "Find best practices for {tech} in {domain}"
   ```

3. **Consolidate findings** in `research.md` using format:
   - Decision: [what was chosen]
   - Rationale: [why chosen]
   - Alternatives considered: [what else evaluated]

**Output**: research.md with all NEEDS CLARIFICATION resolved

### Phase 1: Design & Contracts

**Prerequisites:** `research.md` complete

1. **Extract entities from feature spec** → `data-model.md`:
   - Entity name, fields, relationships
   - Validation rules from requirements
   - State transitions if applicable

2. **Generate API contracts** from functional requirements:
   - For each user action → endpoint
   - Use standard REST/GraphQL patterns
   - Output OpenAPI/GraphQL schema to `/contracts/`

3. **Agent context update**:
   - Run `.specify/scripts/powershell/update-agent-context.ps1 -AgentType qwen`
   - These scripts detect which AI agent is in use
   - Update the appropriate agent-specific context file
   - Add only new technology from current plan
   - Preserve manual additions between markers

**Output**: data-model.md, /contracts/*, quickstart.md, agent-specific file

## Key rules

- Use absolute paths
- ERROR on gate failures or unresolved clarifications

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


/sp.plan
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

I have successfully completed the implementation planning for the RAG Chatbot feature. Here's what was accomplished:

1. Created the implementation plan (plan.md) with:
   - Technical context specifying Python, FastAPI, React, LangChain, etc.
   - Architecture compliance check confirming adherence to constitution
   - Project structure outlining backend and frontend components

2. Created research.md documenting:
   - Technology stack decisions and rationale
   - Alternative technologies considered
   - Implementation approaches selected

3. Created data-model.md defining:
   - Key entities (ChatSession, DocumentChunk, UserQuery, ChatResponse)
   - Relationships between entities
   - Validation rules and state transitions

4. Created API contracts (contracts/chat-api.md) specifying:
   - Endpoints for ingestion and chat functionality
   - Request/response formats
   - Error handling

5. Created quickstart.md guide with:
   - Setup instructions for backend and frontend
   - API usage examples
   - Troubleshooting tips

The implementation plan is comprehensive and ready for the next phase of creating specific tasks. All constitutional requirements have been verified and incorporated into the design.

## Outcome

- ✅ Impact: Created comprehensive implementation plan for RAG Chatbot
- 🧪 Tests: N/A
- 📁 Files: Created plan.md, research.md, data-model.md, contracts/, quickstart.md
- 🔁 Next prompts: Ready for task creation phase with `/sp.tasks`
- 🧠 Reflection: Plan covers all necessary components from backend to frontend

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Proceed to task creation phase