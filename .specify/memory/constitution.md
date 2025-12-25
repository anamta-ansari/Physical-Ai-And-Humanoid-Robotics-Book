<!--
Sync Impact Report:
Version change: 1.0.0 -> 1.1.0 (RAG Chatbot integration principles added)
Modified principles: Spec-First Development, Interface Clarity, Implementation Standards (updated to include RAG-specific considerations)
Added sections: RAG Architecture Principles, Data Privacy & Security
Removed sections: N/A
Templates requiring updates:
- plan-template.md: ✅ Updated to include RAG architecture considerations
- spec-template.md: ✅ Updated to include data privacy requirements
- tasks-template.md: ✅ Updated to reflect RAG-specific tasks
- phr-template.md: ⚠ Pending manual review
- checklist-template.md: ⚠ Pending manual review
Follow-up TODOs: None
-->
# Speckit Constitution

## Core Principles

### Spec-First Development
All development begins with a clear, comprehensive specification that captures user intent, requirements, and acceptance criteria. For RAG implementations, specifications must explicitly define document indexing requirements, embedding strategies, and retrieval quality metrics. Specifications serve as the primary source of truth and must be validated with stakeholders before implementation begins.

### Interface Clarity
Every component must have well-defined, stable interfaces with explicit contracts. For RAG systems, APIs must specify query formats, response structures, and latency expectations. Inputs, outputs, error conditions, and performance characteristics must be clearly documented and agreed upon before implementation. Changes to interfaces require formal approval processes.

### Test-First (NON-NEGOTIABLE)
Test-driven development is mandatory: specifications drive test creation → tests must fail initially → then implementation follows. For RAG systems, tests must validate document retrieval accuracy, response quality, and system performance under load. All code must have corresponding tests that validate behavior against the original specification. Automated test suites must pass before merging.

### Observability & Traceability
All systems must be designed with built-in monitoring, logging, and traceability. Every action must be attributable to a specific specification requirement. For RAG systems, all prompts and responses must be logged in prompts.md for audit and improvement purposes. Metrics must be collected to measure conformance to specifications and identify gaps between intended and actual behavior.

### Minimal Viable Implementation
Implement the simplest solution that satisfies the specification. For RAG implementations, start with basic document indexing and retrieval before adding advanced features like semantic search or conversation history. Avoid over-engineering or anticipating future requirements not explicitly covered in the current specification. Apply YAGNI (You Aren't Gonna Need It) principle rigorously.

### Collaboration-Driven Design
All design decisions must incorporate feedback from relevant stakeholders. Specifications must be reviewed by domain experts, developers, testers, and end-users before implementation. Continuous collaboration ensures solutions meet actual needs rather than assumed requirements.

## RAG Architecture Principles

### Free-Tier Service Integration
All RAG system components must utilize free-tier services only: Google Gemini API for embeddings and generation, Qdrant Cloud free tier for vector storage, and Neon Serverless Postgres free tier for optional session history. This constraint must be maintained throughout the implementation lifecycle.

### Secure Environment Management
NEVER read, access, display, or output contents of any .env file – load keys only via os.getenv(). All API keys and credentials must be managed securely through environment variables. Backend configurations must follow security best practices for credential handling and access control.

### LangChain Pipeline Architecture
Use LangChain for the RAG pipeline implementation with langchain-google-genai for Gemini integration, qdrant-client for vector store operations, and psycopg2-binary for database connectivity. Python-dotenv must be used for environment configuration management. All packages must be installed via proper pip commands in backend setup scripts.

### Document Processing & Indexing
All Markdown chapters from docs/ directory must be indexed into Qdrant with appropriate chunking and chapter metadata. The system must preserve document context and attribution while ensuring efficient retrieval. Content must be processed to maintain original meaning and structure during vectorization.

## Implementation Standards

Technology choices must align with project architecture guidelines. For RAG implementations, use FastAPI for backend services in the "backend" folder. Frontend integration must embed chat UI in Docusaurus with floating bubble on all pages and full /chat route. Code must follow established style guides and pass static analysis tools. All dependencies must be justified with security and licensing reviews. Performance benchmarks must be established and maintained for critical paths, especially for document retrieval and response generation latency.

## Development Workflow

Specifications are captured in dedicated files with clear acceptance criteria. Development follows iterative cycles with frequent stakeholder validation. Code reviews must verify specification compliance. Automated pipelines validate tests, style, and security before deployment. Feature flags enable safe rollouts and rollbacks. For RAG implementations, all prompts and responses must be logged in prompts.md for quality assurance and system improvement purposes.

## Data Privacy & Security

All data processing must comply with privacy regulations. User queries and session data should be handled according to privacy policies. No sensitive information should be stored unnecessarily, and data retention policies must be clearly defined. When using cloud services, ensure compliance with their privacy and security requirements. Log files containing prompts and responses must be protected and handled securely according to organizational policies.

## Governance

This constitution supersedes all other development practices. Amendments require formal documentation, team approval, and migration planning. All pull requests and code reviews must verify constitutional compliance. Specifications must be maintained alongside code changes. Use the Prompt History Records (PHRs) for all development conversations and decisions. RAG-specific implementations must also comply with the additional architectural principles and security requirements outlined in this constitution.

**Version**: 1.1.0 | **Ratified**: 2024-12-20 | **Last Amended**: 2025-12-24
