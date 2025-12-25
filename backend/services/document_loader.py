import os
from typing import List, Optional
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document as LCDocument


class DocumentLoaderService:
    """Service for loading and processing Markdown documents from the docs/ directory."""

    def __init__(self, docs_path: str = None):
        # If no docs_path is provided, default to the docs directory in the project root
        if docs_path is None:
            # Get the project root (two levels up from this file: backend/services/document_loader.py)
            project_root = Path(__file__).parent.parent.parent
            self.docs_path = project_root / "docs"
        else:
            self.docs_path = Path(docs_path)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
    
    def load_documents(self) -> List[LCDocument]:
        """Load all Markdown documents from the docs directory recursively."""
        documents = []
        
        # Find all markdown files in the docs directory and subdirectories
        for md_file in self.docs_path.rglob("*.md"):
            try:
                # Create a TextLoader for each markdown file
                loader = TextLoader(str(md_file), encoding='utf-8')
                docs = loader.load()
                
                # Add metadata to each document
                for doc in docs:
                    # Extract part and chapter information from the file path
                    relative_path = md_file.relative_to(self.docs_path)
                    doc.metadata["source"] = str(relative_path)
                    doc.metadata["file_path"] = str(md_file)
                    
                    # Try to extract part and chapter from path structure (e.g., docs/part1/chapter1.md)
                    path_parts = str(relative_path).split(os.sep)
                    if len(path_parts) >= 2:
                        doc.metadata["part"] = path_parts[0] if path_parts[0].startswith("part") else "unknown"
                        doc.metadata["chapter"] = path_parts[1].replace(".md", "") if path_parts[1].endswith(".md") else path_parts[1]
                    else:
                        doc.metadata["part"] = "unknown"
                        doc.metadata["chapter"] = "unknown"
                
                documents.extend(docs)
            except Exception as e:
                print(f"Error loading document {md_file}: {str(e)}")
                continue
        
        return documents
    
    def chunk_documents(self, documents: List[LCDocument]) -> List[LCDocument]:
        """Split documents into chunks."""
        chunks = []
        for doc in documents:
            split_docs = self.text_splitter.split_documents([doc])
            chunks.extend(split_docs)
        return chunks


# Example usage
if __name__ == "__main__":
    loader = DocumentLoaderService()
    documents = loader.load_documents()
    print(f"Loaded {len(documents)} documents")
    
    # Chunk the documents
    chunks = loader.chunk_documents(documents)
    print(f"Created {len(chunks)} chunks")