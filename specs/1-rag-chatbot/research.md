# Research: RAG Chatbot Implementation

## Decision: Tech Stack Selection
**Rationale**: Selected Python with FastAPI for backend due to strong ecosystem for AI/ML applications and excellent async support. React for frontend integration with Docusaurus. LangChain for RAG orchestration due to its comprehensive support for various LLMs, embeddings, and vector stores.

## Decision: Vector Database Implementation
**Rationale**: Qdrant Cloud was chosen as it offers a free tier that meets our requirements for storing document embeddings and performing similarity search. It integrates well with LangChain.

## Decision: LLM and Embeddings Service
**Rationale**: Google's Gemini API (gemini-1.5-flash for generation, latest free-compatible model for embeddings) was selected as it meets the requirement to use free-tier services while providing good performance for both text generation and embeddings.

## Decision: Session Storage
**Rationale**: Neon Serverless Postgres was chosen for optional conversation history as it offers a free tier and integrates well with Python applications using psycopg2-binary.

## Decision: Document Processing
**Rationale**: Using LangChain's document loaders for Markdown files from the docs/ directory. RecursiveCharacterTextSplitter for chunking to maintain context while ensuring chunks fit within model token limits.

## Decision: Frontend Integration
**Rationale**: React components for seamless integration with Docusaurus. Floating button component will appear on all pages with a dedicated /chat route for full interface.

## Decision: Environment Management
**Rationale**: python-dotenv for secure handling of API keys and configuration without exposing sensitive information.

## Alternatives Considered

### Vector Databases
- Pinecone: More expensive, no sufficient free tier
- ChromaDB: Self-hosted option but requires more infrastructure management
- Weaviate: Good alternative but Qdrant had better free tier for our use case

### LLM Services
- OpenAI: More expensive, doesn't meet free-tier requirement
- Anthropic Claude: More expensive, doesn't meet free-tier requirement
- Mistral: Good free tier but less reliable for our use case

### Backend Frameworks
- Flask: Less performant for async operations than FastAPI
- Django: Too heavy for this specific use case
- Express.js: Would require changing to Node.js ecosystem