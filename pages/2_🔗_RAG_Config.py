import streamlit as st
import os
from uuid import uuid4
import asyncio
from langchain_ollama import OllamaLLM
from database.rag_db import (
    load_documents,
    get_client,
    get_collection,
    get_bm25_retriever,
    SOURCE_PATH
)

def deleteDB():
    """Deletes the current database and its contents."""
    # Remove sources
    for file in os.listdir(SOURCE_PATH):
        os.remove(os.path.join(SOURCE_PATH, file))

    # Reset docs
    st.session_state.docs = []

    if st.session_state.collection is not None:
        # Delete and reset collection
        st.session_state.collection = st.session_state.chroma_client.get_or_create_collection(
            name="rag_collection_demo",
            metadata={"description": "A collection for RAG with Ollama - Demo1"},
            embedding_function=st.session_state.embedding
        )
        st.session_state.chroma_client.delete_collection("rag_collection_demo")
        st.session_state.chroma_client.reset()
        st.session_state.db_ready = False
    else:
        st.error("No database is currently running")

def get_processed_docs():
    """Get list of documents that have already been processed and added to the database."""
    if not hasattr(st.session_state, 'processed_docs'):
        st.session_state.processed_docs = set()
    return st.session_state.processed_docs

def updateDB():
    """Updates the database with new documents and configurations."""
    # Check if embedding model is selected
    if st.session_state.embeddingModel == "":
        st.error("Please select an embedding model")
        return
    
    # Check if ChromaDB client is initialized
    if st.session_state.chroma_client is None:
        st.error("ChromaDB client is not initialized. Please restart the application.")
        return
    
    # Get list of all PDF files in source directory
    source_files = [f for f in os.listdir(SOURCE_PATH) if f.endswith('.pdf')]
    if not source_files:
        st.error("No PDF files found in the source folder. Please upload PDF files to proceed.")
        return

    # Get list of already processed documents
    processed_docs = get_processed_docs()
    
    # Filter for new documents only
    new_docs = [doc for doc in source_files if doc not in processed_docs]
    
    if not new_docs:
        st.info("No new documents to process.")
        return

    with st.spinner(f"Processing {len(new_docs)} new document(s)..."):    
        try:
            # Initialize context LLM if needed
            contextLLM = None
            if st.session_state.context_model and st.session_state.ContextualRAG:
                contextLLM = OllamaLLM(
                    model=st.session_state.context_model,
                    temperature=0.0,
                    num_predict=1000,
                )

            # Process new documents only
            all_splits, aiSplits = asyncio.run(load_documents(
                int(st.session_state.chunk_size),
                int(st.session_state.overlap),
                new_docs,
                contextLLM
            ))

            if not all_splits:
                st.error("No text content was extracted from the new documents.")
                return

            # Setup BM25 retriever if enabled and not already set
            if st.session_state.ContextualBM25RAG and not hasattr(st.session_state, 'BM25retriver'):
                st.session_state.BM25retriver = get_bm25_retriever(all_splits)

            # Initialize or get existing collection
            if not st.session_state.db_ready:
                st.info(f"Initializing ChromaDB collection with embedding model: {st.session_state.embeddingModel}")
                st.session_state.collection = get_collection(
                    st.session_state.embeddingModel,
                    st.session_state.chroma_client,
                    "rag_collection_demo",
                    "A collection for RAG with Ollama - Demo1"
                )

                if st.session_state.collection is None:
                    st.error(f"""Failed to create ChromaDB collection. Please check:
                    1. Is Ollama running? Run 'ollama serve' in a terminal
                    2. Try pulling the embedding model manually:
                       ```
                       ollama pull nomic-embed-text
                       ```
                    3. If the above doesn't work, try restarting Ollama and the application
                    """)
                    return

            # Generate UUIDs for new documents
            uuids = [str(uuid4()) for _ in range(len(all_splits))]

            # Combine AI-generated context if available
            if len(aiSplits) > 0:
                for i in range(len(aiSplits)):
                    all_splits[i] = all_splits[i] + "\n\n" + aiSplits[i]

            # Add new documents to collection
            st.session_state.collection.add(documents=all_splits, ids=uuids)
            
            # Update processed documents list
            processed_docs.update(new_docs)
            
            st.session_state.db_ready = True
            st.success(f"Successfully added {len(new_docs)} new document(s) to the database!")
            
        except Exception as e:
            st.error(f"Error updating database: {str(e)}")

# Set page config
st.set_page_config(
    page_title="RAG Configuration - OllamaRAG",
    page_icon="🔗",
)

# Initialize session state variables if they don't exist
if 'embeddingModel' not in st.session_state:
    st.session_state.embeddingModel = ""

if 'chunk_size' not in st.session_state:
    st.session_state.chunk_size = 500

if 'overlap' not in st.session_state:
    st.session_state.overlap = 50

if 'dbRetrievalAmount' not in st.session_state:
    st.session_state.dbRetrievalAmount = 3

if 'ContextualRAG' not in st.session_state:
    st.session_state.ContextualRAG = False

if 'ContextualBM25RAG' not in st.session_state:
    st.session_state.ContextualBM25RAG = False

if 'context_model' not in st.session_state:
    st.session_state.context_model = ""

# Sidebar - Source Documents List
with st.sidebar:
    st.header("📚 Source Documents")
    if os.path.exists(SOURCE_PATH):
        docs = [f for f in os.listdir(SOURCE_PATH) if f.endswith('.pdf')]
        if docs:
            st.write(f"Found {len(docs)} documents:")
            for doc in docs:
                col1, col2 = st.columns([4,1])
                with col1:
                    st.text(f"📄 {doc}")
                with col2:
                    if st.button("🗑️", key=f"delete_{doc}"):
                        try:
                            os.remove(os.path.join(SOURCE_PATH, doc))
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error removing {doc}: {str(e)}")
        else:
            st.info("No documents uploaded yet.")
    else:
        st.error("Source documents directory not found.")
    
    # Add Reset System button in sidebar
    st.divider()
    st.header("⚠️ System Reset")
    reset_col1, reset_col2 = st.columns([3,2])
    with reset_col1:
        reset_confirm = st.checkbox("Confirm Reset", help="Check this box to confirm system reset")
    with reset_col2:
        if st.button("Reset All", disabled=not reset_confirm, type="primary"):
            try:
                # Clear all files
                for file in os.listdir(SOURCE_PATH):
                    os.remove(os.path.join(SOURCE_PATH, file))
                
                # Reset ChromaDB
                if st.session_state.collection is not None:
                    st.session_state.chroma_client.delete_collection("rag_collection_demo")
                    st.session_state.chroma_client.reset()
                
                # Reset session state
                st.session_state.collection = None
                st.session_state.db_ready = False
                st.session_state.docs = []
                st.session_state.processed_docs = set()
                
                st.success("System reset successful!")
                st.rerun()
            except Exception as e:
                st.error(f"Error during reset: {str(e)}")

st.title("🔗 RAG Configuration")

st.markdown("""
Configure your RAG (Retrieval-Augmented Generation) settings here. These settings determine how documents are processed and retrieved during conversations.
""")

# Database Status
st.header("Database Status")
if st.session_state.db_ready:
    st.success("Embedding Database is ready")
else:
    st.error("Embedding Database is not ready")

# Embedding Model Selection
st.header("Embedding Configuration")
selected_model = st.selectbox(
    "Select Embedding Model",
    st.session_state.dropDown_embeddingModel_list,
    index=st.session_state.dropDown_embeddingModel_list.index(st.session_state.embeddingModel) if st.session_state.embeddingModel in st.session_state.dropDown_embeddingModel_list else None,
    placeholder="Select model...",
    help="Choose the model for generating document embeddings"
)
# Update session state after selection
if selected_model:
    st.session_state.embeddingModel = selected_model

# Document Processing Settings
st.header("Document Processing")
col1, col2 = st.columns(2)
with col1:
    chunk_size = st.number_input(
        "Chunk Size",
        min_value=100,
        max_value=2000,
        value=st.session_state.chunk_size,
        help="Size of text chunks for processing"
    )
    st.session_state.chunk_size = chunk_size
    
    retrieval_amount = st.number_input(
        "Number of Retrieved Documents",
        min_value=1,
        max_value=10,
        value=st.session_state.dbRetrievalAmount,
        help="Number of documents to retrieve for each query"
    )
    st.session_state.dbRetrievalAmount = retrieval_amount

with col2:
    overlap = st.number_input(
        "Chunk Overlap",
        min_value=0,
        max_value=500,
        value=st.session_state.overlap,
        help="Number of overlapping tokens between chunks"
    )
    st.session_state.overlap = overlap

# Contextual RAG Settings
st.header("Contextual RAG Settings")
col1, col2 = st.columns(2)
with col1:
    contextual_rag = st.checkbox(
        "Enable Contextual RAG",
        value=st.session_state.ContextualRAG,
        help="Use AI to generate additional context for chunks"
    )
    st.session_state.ContextualRAG = contextual_rag
    
    if contextual_rag:
        selected_context_model = st.selectbox(
            "Context Generation Model",
            st.session_state.dropDown_model_list,
            index=st.session_state.dropDown_model_list.index(st.session_state.context_model) if st.session_state.context_model in st.session_state.dropDown_model_list else None,
            placeholder="Select model...",
            help="Model used for generating context"
        )
        if selected_context_model:
            st.session_state.context_model = selected_context_model

with col2:
    bm25_rag = st.checkbox(
        "Enable BM25 Retrieval",
        value=st.session_state.ContextualBM25RAG,
        help="Use BM25 algorithm for document retrieval"
    )
    st.session_state.ContextualBM25RAG = bm25_rag

# Document Upload
st.header("Document Management")
uploaded_files = st.file_uploader(
    "Upload Documents",
    accept_multiple_files=True,
    type=['pdf'],
    help="Upload PDF documents to be processed"
)

if uploaded_files:
    st.session_state.docs = []  # Reset the docs list before adding new files
    for uploaded_file in uploaded_files:
        save_path = os.path.join(SOURCE_PATH, uploaded_file.name)
        with open(save_path, mode='wb') as w:
            w.write(uploaded_file.getvalue())
        st.session_state.docs.append(uploaded_file.name)  # Add the filename to docs list
    st.success(f"Uploaded {len(uploaded_files)} document(s)")

# Database Management
st.header("Database Management")
col1, col2 = st.columns(2)
with col1:
    if st.button('Update Database', disabled=(st.session_state.embeddingModel is None)):
        updateDB()
with col2:
    if st.button('Clear Database'):
        deleteDB()
        st.success("Database cleared successfully") 