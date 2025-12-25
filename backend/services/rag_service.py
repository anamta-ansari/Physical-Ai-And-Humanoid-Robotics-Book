from typing import List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import Qdrant
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from config.settings import settings
from models.documents import DocumentChunk
import os


class RAGService:
    """Service for Retrieval Augmented Generation functionality."""

    def __init__(self):
        try:
            # Initialize Qdrant client with HTTPS - fix for Qdrant Cloud connection
            # Extract host from full URL if it contains protocol
            qdrant_url = settings.QDRANT_URL
            if qdrant_url.startswith("https://"):
                # For Qdrant Cloud, use the URL directly and set https=True, prefer_grpc=False
                self.qdrant_client = QdrantClient(
                    url=qdrant_url,
                    api_key=settings.QDRANT_API_KEY,
                    prefer_grpc=False,  # Use REST API only - avoids gRPC port issues
                    https=True,
                    timeout=30
                )
            else:
                # For local Qdrant, use host and port
                self.qdrant_client = QdrantClient(
                    host=qdrant_url,
                    api_key=settings.QDRANT_API_KEY,
                    prefer_grpc=False,
                    https=False,
                    timeout=30
                )

            # Initialize Google Generative AI Embeddings
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=settings.GEMINI_API_KEY
            )

            # Initialize Google Generative AI LLM
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=settings.GEMINI_API_KEY,
                temperature=0.2,
                max_output_tokens=2048
            )

            # Check if collection exists, create if it doesn't
            self._ensure_collection_exists()

            # Initialize Qdrant vector store
            self.vector_store = Qdrant(
                client=self.qdrant_client,
                collection_name=settings.QDRANT_COLLECTION_NAME,
                embeddings=self.embeddings
            )

            # Create a custom RAG prompt
            system_prompt = "You are an expert on Physical AI & Humanoid Robotics. Answer using only the provided context. If selected_text is given, prioritize it heavily.\nContext: {context}"
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}")
            ])

            # Create the RAG chain using the newer LangChain approach
            retriever = self.vector_store.as_retriever(
                search_kwargs={"k": 10}  # Retrieve top 10 most relevant chunks
            )

            # Create the chain
            self.rag_chain = (
                {"context": retriever, "input": RunnablePassthrough()}
                | self.prompt
                | self.llm
                | StrOutputParser()
            )

            # Set initialization status
            self.initialized = True
            print("RAG Service initialized successfully")
        except Exception as e:
            print(f"Error initializing RAG Service: {str(e)}")
            print("RAG Service is not fully initialized. Some features may not work until connection is established.")
            self.initialized = False
            # We don't raise the exception to allow the server to start, but we'll handle it in methods

    def _ensure_collection_exists(self):
        """Ensure the Qdrant collection exists, create if it doesn't."""
        try:
            # Check if collection exists
            collections = self.qdrant_client.get_collections()
            collection_names = [collection.name for collection in collections.collections]

            if settings.QDRANT_COLLECTION_NAME not in collection_names:
                # Create the collection if it doesn't exist
                self.qdrant_client.create_collection(
                    collection_name=settings.QDRANT_COLLECTION_NAME,
                    vectors_config=models.VectorParams(
                        size=768,  # Default size for Google's embedding-001
                        distance=models.Distance.COSINE
                    )
                )
                print(f"Created Qdrant collection: {settings.QDRANT_COLLECTION_NAME}")
            else:
                print(f"Qdrant collection {settings.QDRANT_COLLECTION_NAME} already exists")
        except Exception as e:
            print(f"Error ensuring collection exists: {str(e)}")
            raise e
    
    def ingest_documents(self, documents: List[Document]) -> int:
        """Ingest documents into the Qdrant vector store."""
        try:
            # Check if service is properly initialized
            if not self.initialized:
                print("RAG service not initialized, cannot ingest documents")
                return 0

            # Add documents to the vector store
            self.vector_store.add_documents(documents)

            # Return the number of documents ingested
            return len(documents)
        except Exception as e:
            print(f"Error ingesting documents: {str(e)}")
            return 0
    
    def query(self, query_text: str, selected_text: Optional[str] = None) -> dict:
        """Query the RAG system and return a response."""
        try:
            # Check if service is properly initialized
            if not self.initialized:
                return {
                    "response": "Error: RAG service is not properly initialized. Please check your Qdrant and API key configurations.",
                    "sources": []
                }

            # Prepare the input for the RAG chain
            inputs = query_text

            # If selected text is provided, we'll create a custom chain that includes it in the context
            if selected_text:
                # Create a temporary chain that includes the selected text in the context
                system_prompt_with_selected = f"You are an expert on Physical AI & Humanoid Robotics. Answer using only the provided context. If selected_text is given, prioritize it heavily.\nSelected Text: {selected_text}\nContext: {{context}}"
                temp_prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt_with_selected),
                    ("human", "{input}")
                ])

                # Create the temporary RAG chain
                retriever = self.vector_store.as_retriever(
                    search_kwargs={"k": 10}  # Retrieve top 10 most relevant chunks
                )

                temp_rag_chain = (
                    {"context": retriever, "input": RunnablePassthrough()}
                    | temp_prompt
                    | self.llm
                    | StrOutputParser()
                )

                # Run the temporary RAG chain
                response_text = temp_rag_chain.invoke(inputs)
            else:
                # Run the default RAG chain
                response_text = self.rag_chain.invoke(inputs)

            # For the new approach, we need to separately retrieve source documents
            # Get the relevant documents for the query
            retriever = self.vector_store.as_retriever(
                search_kwargs={"k": 10}  # Retrieve top 10 most relevant chunks
            )
            source_docs = retriever.invoke(query_text)

            # Extract source information
            sources = []
            for doc in source_docs:
                if hasattr(doc, 'page_content'):
                    content = doc.page_content
                elif isinstance(doc, dict) and 'page_content' in doc:
                    content = doc['page_content']
                else:
                    content = str(doc)

                source_info = {
                    "content": content[:200] + "..." if len(content) > 200 else content,
                    "source": getattr(doc, 'metadata', {}).get("source", "unknown"),
                    "part": getattr(doc, 'metadata', {}).get("part", "unknown"),
                    "chapter": getattr(doc, 'metadata', {}).get("chapter", "unknown")
                }
                sources.append(source_info)

            return {
                "response": response_text,
                "sources": sources
            }
        except Exception as e:
            print(f"Error querying RAG system: {str(e)}")
            return {
                "response": f"Error processing query: {str(e)}",
                "sources": []
            }
    
    def check_collection_exists(self) -> bool:
        """Check if the Qdrant collection exists."""
        try:
            # Check if service is properly initialized
            if not self.initialized:
                print("RAG service not initialized, cannot check collection existence")
                return False

            collections = self.qdrant_client.get_collections()
            collection_names = [collection.name for collection in collections.collections]
            return settings.QDRANT_COLLECTION_NAME in collection_names
        except Exception as e:
            print(f"Error checking collection existence: {str(e)}")
            return False