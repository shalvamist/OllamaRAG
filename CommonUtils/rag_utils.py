"""
Consolidated RAG utility functions for the OllamaRAG application.
Handles database operations, document processing, embeddings, and model management.
"""

import os
import shutil
import json
import asyncio
from typing import Set, List, Optional, Dict, Tuple
from uuid import uuid4

import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaLLM
from langchain.retrievers import BM25Retriever
import ollama

# Constants
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_PATH = os.path.join(WORKSPACE_ROOT, "source_documents")
DB_PATH = os.path.join(WORKSPACE_ROOT, "chroma_db")

# Create necessary directories
os.makedirs(SOURCE_PATH, exist_ok=True)
os.makedirs(DB_PATH, exist_ok=True)

# ------------------------------------------------------------------------------
# ChromaDB Client & Collection Management
# ------------------------------------------------------------------------------

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

# ------------------------------------------------------------------------------
# Database Management Functions
# ------------------------------------------------------------------------------

def delete_db(db_name: str, collection, chroma_client) -> Tuple[bool, str]:
    """Deletes a database and its associated collection."""
    success = False
    messages = []
    collection_name = f"rag_collection_{db_name}"
    
    # Step 1: Try to delete the collection from ChromaDB
    try:
        if collection:
            try:
                collection.delete()
                messages.append("Collection object deleted")
            except Exception as e:
                messages.append(f"Warning when deleting collection object: {str(e)}")
        
        try:
            chroma_client.delete_collection(collection_name)
            messages.append("Collection deleted from ChromaDB")
        except Exception as e:
            messages.append(f"Note: {str(e)}")
            # Non-fatal error, continue with cleanup
    except Exception as e:
        messages.append(f"Error during collection deletion: {str(e)}")
    
    # Step 2: Reset the ChromaDB client
    try:
        chroma_client.reset()
        messages.append("ChromaDB client reset")
    except Exception as e:
        messages.append(f"Warning when resetting ChromaDB client: {str(e)}")
    
    # Step 3: Delete the database directory with all source documents
    try:
        db_path = os.path.join(SOURCE_PATH, db_name)
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
            messages.append(f"Database directory '{db_path}' deleted")
            success = True  # Mark as success if we at least deleted the files
    except Exception as e:
        messages.append(f"Error deleting database directory: {str(e)}")
        success = False
    
    # Step 4: Remove any ChromaDB persistent files associated with this collection
    try:
        collection_db_path = os.path.join(DB_PATH, collection_name)
        if os.path.exists(collection_db_path):
            shutil.rmtree(collection_db_path)
            messages.append(f"Collection files at '{collection_db_path}' deleted")
            success = True
    except Exception as e:
        messages.append(f"Error cleaning up collection files: {str(e)}")
    
    # Return result
    message = "; ".join(messages)
    if success:
        return True, f"Database '{db_name}' deleted successfully!"
    else:
        return False, f"Error deleting database: {message}"

def get_db_documents(db_name: str) -> List[str]:
    """Gets a list of documents in a database."""
    try:
        db_path = os.path.join(SOURCE_PATH, db_name)
        if os.path.exists(db_path):
            return [f for f in os.listdir(db_path) if f.endswith('.pdf')]
        return []
    except Exception:
        return []

def get_doc_files(db_name):
    """Get list of document files in a database directory."""
    db_path = os.path.join(SOURCE_PATH, db_name)
    if os.path.exists(db_path):
        return [f for f in os.listdir(db_path) if f.endswith('.pdf')]
    return []

def get_db_directory(db_name):
    """Get the directory path for a database."""
    return os.path.join(SOURCE_PATH, db_name)

def add_document_to_db(db_name: str, filename: str, content: bytes) -> Tuple[bool, str]:
    """Adds a document to a database."""
    try:
        # Create database directory if it doesn't exist
        db_path = os.path.join(SOURCE_PATH, db_name)
        if not os.path.exists(db_path):
            os.makedirs(db_path)
        
        # Save the document
        file_path = os.path.join(db_path, filename)
        with open(file_path, 'wb') as f:
            f.write(content)
        
        return True, f"Document '{filename}' added successfully!"
    except Exception as e:
        return False, f"Error adding document: {str(e)}"

def remove_document_from_db(db_name: str, filename: str) -> Tuple[bool, str]:
    """Removes a document from a database."""
    try:
        file_path = os.path.join(SOURCE_PATH, db_name, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            return True, f"Document '{filename}' removed successfully!"
        return False, f"Document '{filename}' not found."
    except Exception as e:
        return False, f"Error removing document: {str(e)}"

# ------------------------------------------------------------------------------
# Document Processing & Embedding Functions
# ------------------------------------------------------------------------------

async def load_documents(chunk_size, overlap, docs, context_llm=None):
    """
    Load and process documents for RAG asynchronously.
    
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
    
    # Process each document
    for doc_path in docs:
        print(f"Processing file: {doc_path}")
        if os.path.exists(doc_path):
            try:
                # Load PDF using PyPDFLoader
                loader = PyPDFLoader(doc_path)
                pages = loader.load()
                
                # Split documents
                splits = text_splitter.split_documents(pages)
                
                # Extract text content
                for split in splits:
                    all_splits.append(split.page_content)
                
                # Generate AI context if LLM is provided
                if context_llm is not None:
                    for i, split in enumerate(splits):
                        try:
                            # Use proper async invocation if available
                            if hasattr(context_llm, 'ainvoke'):
                                context = await context_llm.ainvoke(
                                    f"Generate a concise summary of the following text:\n{split.page_content}"
                                )
                            else:
                                # Fallback to synchronous invoke
                                context = context_llm.invoke(
                                    f"Generate a concise summary of the following text:\n{split.page_content}"
                                )
                            ai_splits.append(context)
                        except Exception as e:
                            print(f"Error generating context for split {i}: {str(e)}")
                            # Add a placeholder to maintain alignment with all_splits
                            ai_splits.append("")
            except Exception as e:
                print(f"Error processing document {doc_path}: {str(e)}")
        else:
            print(f"File not found: {doc_path}")
    
    print(f"Processed {len(all_splits)} text chunks")
    return all_splits, ai_splits

def get_embedding(text, model_name):
    """Get embedding for a text using Ollama."""
    try:
        response = ollama.embeddings(model=model_name, prompt=text)
        return response.get('embedding', [])
    except Exception as e:
        print(f"Error getting embedding: {str(e)}")
        return []

def get_bm25_retriever(documents):
    """Create a BM25 retriever from documents."""
    return BM25Retriever.from_documents(documents)

# ------------------------------------------------------------------------------
# Query Functions
# ------------------------------------------------------------------------------

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

# ------------------------------------------------------------------------------
# Database Update & Model Functions
# ------------------------------------------------------------------------------

async def update_db(
    db_name: str,
    embedding_model: str,
    chroma_client,
    chunk_size: int,
    overlap: int,
    context_model: Optional[str] = None,
    use_contextual_rag: bool = False,
    use_bm25: bool = False,
    progress_callback=None
) -> Tuple[bool, str, Optional[List[str]]]:
    """
    Updates a database with documents and configurations.
    """
    try:
        if not embedding_model:
            return False, "No embedding model selected.", None
            
        if progress_callback:
            progress_callback("Initializing database update...", 0.05)

        # Create database directory if it doesn't exist
        db_path = os.path.join(SOURCE_PATH, db_name)
        if not os.path.exists(db_path):
            os.makedirs(db_path)
            if progress_callback:
                progress_callback(f"Created database directory: {db_path}", 0.1)

        # Get list of PDF files
        source_files = [f for f in os.listdir(db_path) if f.endswith('.pdf')]
        if not source_files:
            return False, "No PDF files found in the database folder.", None
            
        if progress_callback:
            progress_callback(f"Found {len(source_files)} PDF files", 0.15)

        # Initialize context LLM if needed
        context_llm = None
        if context_model and use_contextual_rag:
            try:
                context_llm = OllamaLLM(
                    model=context_model,
                    temperature=0.0,
                    num_predict=1000,
                )
                if progress_callback:
                    progress_callback(f"Initialized context LLM with model: {context_model}", 0.2)
            except Exception as e:
                if progress_callback:
                    progress_callback(f"Error initializing context LLM: {str(e)}", 0.2)
                print(f"Error initializing context LLM: {str(e)}")

        # Process documents
        all_splits = []
        ai_splits = []
        processed_docs = []
        
        for idx, doc in enumerate(source_files):
            doc_path = os.path.join(db_path, doc)
            if progress_callback:
                progress_progress = 0.2 + (0.5 * (idx / len(source_files)))
                progress_callback(f"Processing {idx+1}/{len(source_files)}: {doc}", progress_progress)
            
            try:
                # Process document
                doc_splits, doc_ai_splits = await load_documents(
                    chunk_size,
                    overlap,
                    [doc_path],
                    context_llm
                )
                
                if doc_splits:
                    chunk_count = len(doc_splits)
                    if progress_callback:
                        progress_callback(f"Extracted {chunk_count} chunks from {doc}", progress_progress + 0.02)
                    all_splits.extend(doc_splits)
                    if doc_ai_splits and len(doc_ai_splits) > 0:
                        ai_splits.extend(doc_ai_splits)
                        if progress_callback:
                            progress_callback(f"Generated context for {len(doc_ai_splits)} chunks", progress_progress + 0.03)
                    processed_docs.append(doc)
                else:
                    if progress_callback:
                        progress_callback(f"No content extracted from {doc}", progress_progress)
            except Exception as e:
                if progress_callback:
                    progress_callback(f"Error processing {doc}: {str(e)}", progress_progress)
                print(f"Error processing {doc}: {str(e)}")

        if not all_splits:
            return False, "No text content was extracted from any of the documents.", None

        if progress_callback:
            progress_callback(f"Document processing complete. Extracted {len(all_splits)} text chunks", 0.7)

        # Delete existing collection if it exists
        collection_name = f"rag_collection_{db_name}"
        if progress_callback:
            progress_callback(f"Setting up collection: {collection_name}", 0.75)
        
        try:
            chroma_client.delete_collection(collection_name)
            if progress_callback:
                progress_callback("Deleted existing collection", 0.77)
        except Exception as e:
            # Collection might not exist, continue
            if progress_callback:
                progress_callback(f"Note: {str(e)}", 0.77)
            print(f"Error deleting collection (might not exist): {str(e)}")
            
        # Reset the client to ensure clean state
        try:
            chroma_client.reset()
            if progress_callback:
                progress_callback("Reset ChromaDB client", 0.8)
        except Exception as e:
            if progress_callback:
                progress_callback(f"Error resetting client: {str(e)}", 0.8)
            print(f"Error resetting client: {str(e)}")

        # Create new collection
        if progress_callback:
            progress_callback(f"Creating collection with model: {embedding_model}", 0.82)
            
        collection = get_collection(
            embedding_model,
            chroma_client,
            collection_name,
            f"RAG collection for {db_name}"
        )

        if collection is None:
            return False, "Failed to create ChromaDB collection.", None

        # Generate UUIDs and combine AI context
        if progress_callback:
            progress_callback("Preparing document embeddings", 0.85)
            
        uuids = [str(uuid4()) for _ in range(len(all_splits))]
        
        # Combine AI generated context with original content if available
        if ai_splits and len(ai_splits) == len(all_splits):
            for i in range(len(ai_splits)):
                if ai_splits[i]:  # Only append non-empty context
                    all_splits[i] = f"{all_splits[i]}\n\n{ai_splits[i]}"

        # Add documents to collection
        if progress_callback:
            progress_callback(f"Adding {len(all_splits)} documents to database...", 0.9)
            
        # Add documents in batches to avoid memory issues
        batch_size = 100
        for i in range(0, len(all_splits), batch_size):
            batch_end = min(i + batch_size, len(all_splits))
            batch_progress = 0.9 + (0.1 * (i / len(all_splits)))
            
            if progress_callback:
                progress_callback(f"Adding batch {i//batch_size + 1}/{(len(all_splits) + batch_size - 1)//batch_size}", batch_progress)
                
            try:
                collection.add(
                    documents=all_splits[i:batch_end],
                    ids=uuids[i:batch_end]
                )
            except Exception as e:
                if progress_callback:
                    progress_callback(f"Error adding batch: {str(e)}", batch_progress)
                print(f"Error adding batch {i//batch_size + 1}: {str(e)}")
                # Continue with next batch instead of failing completely
        
        if progress_callback:
            progress_callback("Database update complete!", 1.0)
            
        return True, f"Successfully added {len(processed_docs)} document(s) to the database!", processed_docs
        
    except Exception as e:
        error_msg = f"Error updating database: {str(e)}"
        print(error_msg)
        import traceback
        print(traceback.format_exc())
        return False, error_msg, None

def update_ollama_models() -> Tuple[List[str], List[str]]:
    """
    Updates the list of available Ollama models.
    
    Returns:
        tuple: (embedding_models: List[str], regular_models: List[str])
    """
    try:
        ollama_models = ollama.list()
        embedding_models = []
        regular_models = []
        
        for model in ollama_models['models']:
            if 'embedding' in model['model'] or 'embed' in model['model']:
                embedding_models.append(model['model'])
            else:
                regular_models.append(model['model'])
                
        return embedding_models, regular_models
    except Exception as e:
        return [], [] 