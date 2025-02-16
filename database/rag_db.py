import os
import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import asyncio

# Constants
SOURCE_PATH = "source_documents"
DB_PATH = "chroma_db"

def get_client():
    """Get ChromaDB client with persistent storage."""
    return chromadb.PersistentClient(path=DB_PATH)

def get_collection(embedding_model, client, collection_name, description):
    """Get or create a ChromaDB collection with specified embedding function."""
    from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
    
    embedding_function = OllamaEmbeddingFunction(model_name=embedding_model)
    
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"description": description},
        embedding_function=embedding_function
    )

def get_bm25_retriever(documents):
    """Create a BM25 retriever from documents."""
    from langchain.retrievers import BM25Retriever
    return BM25Retriever.from_documents(documents)

async def load_documents(chunk_size, overlap, docs, context_llm=None):
    """
    Load and process documents for RAG.
    
    Args:
        chunk_size: Size of text chunks
        overlap: Overlap between chunks
        docs: List of document paths
        context_llm: Optional LLM for generating context
    
    Returns:
        Tuple of (all_splits, ai_splits)
    """
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    
    all_splits = []
    ai_splits = []
    
    # Process each document
    for doc in os.listdir(SOURCE_PATH):
        if doc.endswith('.pdf'):
            loader = PyPDFLoader(os.path.join(SOURCE_PATH, doc))
            pages = loader.load()
            splits = text_splitter.split_documents(pages)
            
            # Extract text content
            for split in splits:
                all_splits.append(split.page_content)
            
            # Generate AI context if LLM is provided
            if context_llm is not None:
                for split in splits:
                    context = context_llm.invoke(
                        f"Generate a concise summary of the following text:\n{split.page_content}"
                    )
                    ai_splits.append(context)
    
    return all_splits, ai_splits

def query_chromadb(query, collection, n_results=3):
    """
    Query the ChromaDB collection.
    
    Args:
        query: Search query
        collection: ChromaDB collection
        n_results: Number of results to return
    
    Returns:
        Tuple of (documents, metadatas)
    """
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    
    return results['documents'][0], results['metadatas'][0]

def query_bm25(retriever, query, n_results=3):
    """
    Query using BM25 retriever.
    
    Args:
        retriever: BM25 retriever
        query: Search query
        n_results: Number of results to return
    
    Returns:
        List of relevant documents
    """
    return retriever.get_relevant_documents(query)[:n_results]

# Create necessary directories
os.makedirs(SOURCE_PATH, exist_ok=True)
os.makedirs(DB_PATH, exist_ok=True) 