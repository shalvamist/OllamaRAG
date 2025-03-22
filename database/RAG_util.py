"""
Utility functions for RAG (Retrieval-Augmented Generation) operations.
Handles database operations, document processing, and model management.
"""

import os
from typing import Set, List, Optional, Dict
from uuid import uuid4
import asyncio
from langchain_ollama import OllamaLLM
import ollama
import shutil
from database.rag_db import (
    load_documents,
    get_client,
    get_collection,
    get_bm25_retriever,
    SOURCE_PATH
)


def delete_db(db_name: str, collection, chroma_client) -> tuple[bool, str]:
    """
    Deletes a database and its associated collection.
    """
    try:
        # Delete the collection if it exists
        collection_name = f"rag_collection_{db_name}"
        try:
            if collection:
                collection.delete()
            chroma_client.delete_collection(collection_name)
        except Exception:
            pass  # Collection might not exist
        
        # Delete the database directory
        db_path = os.path.join(SOURCE_PATH, db_name)
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
        
        return True, f"Database '{db_name}' deleted successfully!"
    except Exception as e:
        return False, f"Error deleting database: {str(e)}"


def get_db_documents(db_name: str) -> List[str]:
    """
    Gets a list of documents in a database.
    """
    try:
        db_path = os.path.join(SOURCE_PATH, db_name)
        if os.path.exists(db_path):
            return [f for f in os.listdir(db_path) if f.endswith('.pdf')]
        return []
    except Exception:
        return []


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
) -> tuple[bool, str, Optional[List[str]]]:
    """
    Updates a database with documents and configurations.
    """
    try:
        if not embedding_model:
            return False, "No embedding model selected.", None

        # Create database directory if it doesn't exist
        db_path = os.path.join(SOURCE_PATH, db_name)
        if not os.path.exists(db_path):
            os.makedirs(db_path)

        # Get list of PDF files
        source_files = [f for f in os.listdir(db_path) if f.endswith('.pdf')]
        if not source_files:
            return False, "No PDF files found in the database folder.", None

        # Initialize context LLM if needed
        context_llm = None
        if context_model and use_contextual_rag:
            context_llm = OllamaLLM(
                model=context_model,
                temperature=0.0,
                num_predict=1000,
            )

        # Process documents
        all_splits = []
        ai_splits = []
        processed_docs = []
        
        for idx, doc in enumerate(source_files):
            if progress_callback:
                progress_callback(f"Processing: {doc}", (idx + 1) / len(source_files))
            
            try:
                # Process document
                doc_splits, doc_ai_splits = await load_documents(
                    chunk_size,
                    overlap,
                    [os.path.join(db_path, doc)],
                    context_llm
                )
                
                if doc_splits:
                    chunk_count = len(doc_splits)
                    if progress_callback:
                        progress_callback(f"Processing: {doc} ({chunk_count} chunks)", (idx + 1) / len(source_files))
                    all_splits.extend(doc_splits)
                    if doc_ai_splits:
                        ai_splits.extend(doc_ai_splits)
                    processed_docs.append(doc)
                else:
                    if progress_callback:
                        progress_callback(f"No content extracted from {doc}", (idx + 1) / len(source_files))
            except Exception as e:
                if progress_callback:
                    progress_callback(f"Error processing {doc}: {str(e)}", (idx + 1) / len(source_files))

        if not all_splits:
            return False, "No text content was extracted from any of the documents.", None

        # Delete existing collection if it exists
        collection_name = f"rag_collection_{db_name}"
        try:
            chroma_client.delete_collection(collection_name)
        except Exception:
            pass  # Collection might not exist
            
        # Reset the client to ensure clean state
        chroma_client.reset()

        # Create new collection
        collection = get_collection(
            embedding_model,
            chroma_client,
            collection_name,
            f"RAG collection for {db_name}"
        )

        if collection is None:
            return False, "Failed to create ChromaDB collection.", None

        # Generate UUIDs and combine AI context
        uuids = [str(uuid4()) for _ in range(len(all_splits))]
        if ai_splits:
            for i in range(len(ai_splits)):
                all_splits[i] = f"{all_splits[i]}\n\n{ai_splits[i]}"

        # Add documents to collection
        if progress_callback:
            progress_callback("Adding documents to database...", 1.0)
            
        # Add documents in batches to avoid memory issues
        batch_size = 100
        for i in range(0, len(all_splits), batch_size):
            batch_end = min(i + batch_size, len(all_splits))
            collection.add(
                documents=all_splits[i:batch_end],
                ids=uuids[i:batch_end]
            )
        
        return True, f"Successfully added {len(processed_docs)} document(s) to the database!", processed_docs
        
    except Exception as e:
        return False, f"Error updating database: {str(e)}", None


def update_ollama_models() -> tuple[List[str], List[str]]:
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


def add_document_to_db(db_name: str, filename: str, content: bytes) -> tuple[bool, str]:
    """
    Adds a document to a database.
    """
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


def remove_document_from_db(db_name: str, filename: str) -> tuple[bool, str]:
    """
    Removes a document from a database.
    """
    try:
        file_path = os.path.join(SOURCE_PATH, db_name, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            return True, f"Document '{filename}' removed successfully!"
        return False, f"Document '{filename}' not found."
    except Exception as e:
        return False, f"Error removing document: {str(e)}" 