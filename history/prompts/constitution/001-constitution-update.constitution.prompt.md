---
id: 001
title: Constitution Update
stage: constitution
date: 2025-12-24
surface: agent
model: Qwen
feature: none
branch: main
user: system
command: /sp.constitution
labels: ["constitution", "rag-chatbot", "architecture"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - .specify/memory/constitution.md
 - .specify/templates/plan-template.md
 - .specify/templates/spec-template.md
 - .specify/templates/tasks-template.md
tests:
 - null
---

## Prompt

```
$ARGUMENTS
```

You **MUST** consider the user input before proceeding (if not empty).

## Outline

You are updating the project constitution at `.specify/memory/constitution.md`. This file is a TEMPLATE containing placeholder tokens in square brackets (e.g. `[PROJECT_NAME]`, `[PRINCIPLE_1_NAME]`). Your job is to (a) collect/derive concrete values, (b) fill the template precisely, and (c) propagate any amendments across dependent artifacts.

Follow this execution flow:

1. Load the existing constitution template at `.specify/memory/constitution.md`.
   - Identify every placeholder token of the form `[ALL_CAPS_IDENTIFIER]`.
   **IMPORTANT**: The user might require less or more principles than the ones used in the template. If a number is specified, respect that - follow the general template. You will update the doc accordingly.

2. Collect/derive values for placeholders:
   - If user input (conversation) supplies a value, use it.
   - Otherwise infer from existing repo context (README, docs, prior constitution versions if embedded).
   - For governance dates: `RATIFICATION_DATE` is the original adoption date (if unknown ask or mark TODO), `LAST_AMENDED_DATE` is today if changes are made, otherwise keep previous.
   - `CONSTITUTION_VERSION` must increment according to semantic versioning rules:
     - MAJOR: Backward incompatible governance/principle removals or redefinitions.
     - MINOR: New principle/section added or materially expanded guidance.
     - PATCH: Clarifications, wording, typo fixes, non-semantic refinements.
   - If version bump type ambiguous, propose reasoning before finalizing.

3. Draft the updated constitution content:
   - Replace every placeholder with concrete text (no bracketed tokens left except intentionally retained template slots that the project has chosen not to define yet—explicitly justify any left).
   - Preserve heading hierarchy and comments can be removed once replaced unless they still add clarifying guidance.
   - Ensure each Principle section: succinct name line, paragraph (or bullet list) capturing non‑negotiable rules, explicit rationale if not obvious.
   - Ensure Governance section lists amendment procedure, versioning policy, and compliance review expectations.

4. Consistency propagation checklist (convert prior checklist into active validations):
   - Read `.specify/templates/plan-template.md` and ensure any "Constitution Check" or rules align with updated principles.
   - Read `.specify/templates/spec-template.md` for scope/requirements alignment—update if constitution adds/removes mandatory sections or constraints.
   - Read `.specify/templates/tasks-template.md` and ensure task categorization reflects new or removed principle-driven task types (e.g., observability, versioning, testing discipline).
   - Read each command file in `.specify/templates/commands/*.md` (including this one) to verify no outdated references (agent-specific names like CLAUDE only) remain when generic guidance is required.
   - Read any runtime guidance docs (e.g., `README.md`, `docs/quickstart.md`, or agent-specific guidance files if present). Update references to principles changed.

5. Produce a Sync Impact Report (prepend as an HTML comment at top of the constitution file after update):
   - Version change: old → new
   - List of modified principles (old title → new title if renamed)
   - Added sections
   - Removed sections
   - Templates requiring updates (✅ updated / ⚠ pending) with file paths
   - Follow-up TODOs if any placeholders intentionally deferred.

6. Validation before final output:
   - No remaining unexplained bracket tokens.
   - Version line matches report.
   - Dates ISO format YYYY-MM-DD.
   - Principles are declarative, testable, and free of vague language ("should" → replace with MUST/SHOULD rationale where appropriate).

7. Write the completed constitution back to `.specify/memory/constitution.md` (overwrite).

8. Output a final summary to the user with:
   - New version and bump rationale.
   - Any files flagged for manual follow-up.
   - Suggested commit message (e.g., `docs: amend constitution to vX.Y.Z (principle additions + governance update)`).

Formatting & Style Requirements:

- Use Markdown headings exactly as in the template (do not demote/promote levels).
- Wrap long rationale lines to keep readability (<100 chars ideally) but do not hard enforce with awkward breaks.
- Keep a single blank line between sections.
- Avoid trailing whitespace.

If the user supplies partial updates (e.g., only one principle revision), still perform validation and version decision steps.

If critical info missing (e.g., ratification date truly unknown), insert `TODO(<FIELD_NAME>): explanation` and include in the Sync Impact Report under deferred items.

Do not create a new template; always operate on the existing `.specify/memory/constitution.md` file.

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


/sp.constitution
Establish guiding rules and standards for the RAG Chatbot integration phase (building on existing book project):

Develop a Retrieval-Augmented Generation (RAG) chatbot fully embedded in the Docusaurus book site.
Use only free-tier services: Google Gemini API (GEMINI_API_KEY for embeddings/generation with gemini-1.5-flash and compatible embeddings), Qdrant Cloud free tier for vector store, Neon Serverless Postgres free tier for optional session history.
Backend: Create and use "backend" folder with FastAPI.
Chatbot features: Answer questions about full book content; support queries based only on user-selected/highlighted text.
Use LangChain for RAG pipeline, langchain-google-genai for Gemini integration, qdrant-client, psycopg2-binary, python-dotenv.
Index all Markdown chapters from docs/ into Qdrant (chunked, with chapter metadata).
Embed chat UI in Docusaurus: Floating bubble (bottom-right) on all pages + full /chat route.
UI: Match existing neon purple futuristic theme, bright white text, mobile-responsive.
NEVER read, access, display, or output contents of any .env file – load keys only via os.getenv().
Install all packages/libraries via proper pip commands in backend.
Log all prompts/responses in prompts.md.
Run commands: npm for frontend, uvicorn for backend

## Response snapshot

I've successfully updated the project constitution and related templates to reflect the requirements for the RAG Chatbot integration. Here's the summary of changes:

1. **Constitution Updated**:
   - **Version**: Updated from 1.0.0 to 1.1.0 (minor version bump due to new principles and sections)
   - **New Section**: Added "RAG Architecture Principles" with specific guidelines for the chatbot implementation
   - **New Section**: Added "Data Privacy & Security" requirements
   - **Updated Principles**: Modified existing principles to include RAG-specific considerations

2. **Template Updates**:
   - **Plan Template**: Added RAG Architecture Compliance checklist to the Constitution Check section
   - **Spec Template**: Updated functional requirements to include RAG-specific requirements
   - **Tasks Template**: Updated setup and foundational phases with RAG-specific tasks

3. **Version Bump Rationale**: This is a minor version update (1.0.0 → 1.1.0) because we've added new architectural principles and requirements specific to the RAG Chatbot implementation, which materially expand the guidance while maintaining backward compatibility with existing principles.

All changes have been made to align with the requirements for developing a Retrieval-Augmented Generation (RAG) chatbot embedded in the Docusaurus book site, using free-tier services and following the specified architecture.

The constitution and templates are now ready to guide the implementation of the RAG Chatbot feature.

## Outcome

- ✅ Impact: Updated constitution with RAG-specific principles and requirements
- 🧪 Tests: N/A
- 📁 Files: Updated .specify/memory/constitution.md and related templates
- 🔁 Next prompts: Next step would be to create the feature specification
- 🧠 Reflection: The constitution now properly guides RAG chatbot development with security, architecture, and compliance requirements

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Create the feature specification based on the updated constitution