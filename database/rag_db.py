import os
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import ollama

# Constants
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_PATH = os.path.join(WORKSPACE_ROOT, "source_documents")
DB_PATH = os.path.join(WORKSPACE_ROOT, "chroma_db")

def get_client():
    """Get ChromaDB client with persistent storage."""
    try:
        return chromadb.PersistentClient(
            path=DB_PATH,
            settings=chromadb.Settings(
                allow_reset=True,
                is_persistent=True
            )
        )
    except Exception as e:
        print(f"Error initializing ChromaDB client: {str(e)}")
        return None

def get_collection(embedding_model, client, collection_name, description):
    """Get or create a ChromaDB collection with specified embedding function."""
    try:
        # First check if client is valid
        if client is None:
            print("ChromaDB client is not initialized")
            return None

        # Ensure model is available
        try:
            # Remove ':latest' if present in the model name
            base_model_name = embedding_model.split(':')[0]
            
            # Check if model is available
            models = ollama.list()
            model_exists = any(m['model'].startswith(base_model_name) for m in models['models'])
            
            if not model_exists:
                print(f"Model {base_model_name} not found, attempting to pull...")
                ollama.pull(base_model_name)
        except Exception as e:
            print(f"Error checking/pulling model: {str(e)}")
            return None

        # Initialize embedding function with error handling
        try:
            embedding_function = embedding_functions.OllamaEmbeddingFunction(
                url="http://localhost:11434/api/embeddings",
                model_name=base_model_name
            )
            
            # Test the embedding function
            test_result = embedding_function(["test"])
            if test_result is None or len(test_result) == 0:
                print("Embedding function test failed")
                return None
        except Exception as e:
            print(f"Error initializing embedding function: {str(e)}")
            return None
        
        # Create or get collection
        try:
            collection = client.get_or_create_collection(
                name=collection_name,
                metadata={"description": description},
                embedding_function=embedding_function
            )
            return collection
        except Exception as e:
            print(f"Error creating collection: {str(e)}")
            return None
            
    except Exception as e:
        print(f"Error in get_collection: {str(e)}")
        return None

def get_bm25_retriever(documents):
    """Create a BM25 retriever from documents."""
    from langchain.retrievers import BM25Retriever
    return BM25Retriever.from_documents(documents)

def load_documents(chunk_size, overlap, docs, context_llm=None):
    """
    Load and process documents for RAG.
    
    Args:
        chunk_size: Size of text chunks
        overlap: Overlap between chunks
        docs: List of document filenames
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
    
    print(f"Processing documents: {docs}")
    print(f"Source path: {SOURCE_PATH}")
    
    # Process each document
    for doc in docs:
        file_path = os.path.join(SOURCE_PATH, doc)
        print(f"Processing file: {file_path}")
        if os.path.exists(file_path):
            loader = PyPDFLoader(file_path)
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
        else:
            print(f"File not found: {file_path}")
    
    print(f"Processed {len(all_splits)} text chunks")
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