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
import ollama

def deleteDB():
    """Deletes the current database and its contents."""
    try:
        # Remove source documents
        for file in os.listdir(SOURCE_PATH):
            os.remove(os.path.join(SOURCE_PATH, file))

        # Reset docs list
        st.session_state.docs = []
        st.session_state.processed_docs = set()

        # Reset ChromaDB
        if st.session_state.collection is not None:
            # Delete the collection
            st.session_state.chroma_client.delete_collection("rag_collection_demo")
            
            # Reset the client
            st.session_state.chroma_client.reset()
            
            # Reset session state variables
            st.session_state.collection = None
            st.session_state.db_ready = False
            st.session_state.embedding = None
            
            st.success("Database cleared successfully")
            st.rerun()
        else:
            st.warning("No active database collection found")
            
    except Exception as e:
        st.error(f"Error clearing database: {str(e)}")

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

    # Create progress containers
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    try:
        # Initialize context LLM if needed
        contextLLM = None
        if st.session_state.context_model and st.session_state.ContextualRAG:
            contextLLM = OllamaLLM(
                model=st.session_state.context_model,
                temperature=0.0,
                num_predict=1000,
            )

        # Process new documents
        all_splits = []
        aiSplits = []
        
        for doc in new_docs:
            progress_text.text(f"Processing: {doc}")
            
            # Process document
            doc_splits, doc_ai_splits = asyncio.run(load_documents(
                int(st.session_state.chunk_size),
                int(st.session_state.overlap),
                [doc],
                contextLLM
            ))
            
            if doc_splits:
                # Update progress with chunk count
                chunk_count = len(doc_splits)
                progress_text.text(f"Processing: {doc} ({chunk_count} chunks)")
                all_splits.extend(doc_splits)
                if doc_ai_splits:
                    aiSplits.extend(doc_ai_splits)
            else:
                progress_text.text(f"No content extracted from {doc}")

            # Update progress bar
            progress_bar.progress(len(all_splits) / (len(new_docs) * int(st.session_state.chunk_size)))

        if not all_splits:
            progress_text.empty()
            progress_bar.empty()
            st.error("No text content was extracted from any of the documents.")
            return

        # Setup BM25 retriever if enabled
        if st.session_state.ContextualBM25RAG and not hasattr(st.session_state, 'BM25retriver'):
            st.session_state.BM25retriver = get_bm25_retriever(all_splits)

        # Initialize or get existing collection
        if not st.session_state.db_ready:
            progress_text.text("Initializing database...")
            st.session_state.collection = get_collection(
                st.session_state.embeddingModel,
                st.session_state.chroma_client,
                "rag_collection_demo",
                "A collection for RAG with Ollama - Demo1"
            )

            if st.session_state.collection is None:
                progress_text.empty()
                progress_bar.empty()
                st.error(f"""Failed to create ChromaDB collection. Please check:
                1. Is Ollama running? Run 'ollama serve' in a terminal
                2. Try pulling the embedding model manually:
                   ```
                   ollama pull nomic-embed-text
                   ```
                3. If the above doesn't work, try restarting Ollama and the application
                """)
                return

        # Generate UUIDs and combine AI context
        uuids = [str(uuid4()) for _ in range(len(all_splits))]
        if len(aiSplits) > 0:
            for i in range(len(aiSplits)):
                all_splits[i] = all_splits[i] + "\n\n" + aiSplits[i]

        # Add documents to collection
        progress_text.text("Adding documents to database...")
        st.session_state.collection.add(documents=all_splits, ids=uuids)
        
        # Update processed documents list
        processed_docs.update(new_docs)
        
        # Update final status
        st.session_state.db_ready = True
        progress_text.empty()
        progress_bar.empty()
        st.success(f"Successfully added {len(new_docs)} document(s) to the database!")
        st.rerun()
        
    except Exception as e:
        progress_text.empty()
        progress_bar.empty()
        st.error(f"Error updating database: {str(e)}")

# Set page config
st.set_page_config(
    page_title="RAG Configuration - OllamaRAG",
    page_icon="🔗",
    initial_sidebar_state="expanded"
)

# Initialize session state variables if they don't exist
if 'ollama_model' not in st.session_state:
    st.session_state.ollama_model = None

if 'contextWindow' not in st.session_state:
    st.session_state.contextWindow = 2048

if 'newMaxTokens' not in st.session_state:
    st.session_state.newMaxTokens = 1024

if 'temperature' not in st.session_state:
    st.session_state.temperature = 0.7

if 'system_prompt' not in st.session_state:
    st.session_state.system_prompt = "You are a helpful AI assistant."

if 'chatReady' not in st.session_state:
    st.session_state.chatReady = False

if 'ContextualRAG' not in st.session_state:
    st.session_state.ContextualRAG = False

if 'ContextualBM25RAG' not in st.session_state:
    st.session_state.ContextualBM25RAG = False

if 'BM25retriver' not in st.session_state:
    st.session_state.BM25retriver = None

if 'dbRetrievalAmount' not in st.session_state:
    st.session_state.dbRetrievalAmount = 3

# Custom CSS for cooler color scheme
st.markdown("""
<style>
    /* Main background and text colors */
    .stApp {
        background-color: #a9b89e;
        color: #1a2234;
    }
    
    /* Main content width and layout */
    .block-container {
        max-width: 60% !important;
        padding-left: 1rem;
        padding-right: 1rem;
        background-color: #fff;
        border-radius: 6px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.15);
        margin: 0.5rem;
    }
    
    /* Headers */
    h1 {
        color: #2c3e50 !important;
        margin-bottom: 0.5rem !important;
        margin-top: 0.5rem !important;
        font-size: 2.2em !important;
        padding-bottom: 0.3rem !important;
        font-weight: 800 !important;
        border-bottom: 3px solid #3498db !important;
    }
    
    h2 {
        color: #2c3e50 !important;
        margin-bottom: 0.4rem !important;
        margin-top: 0.4rem !important;
        font-size: 1.8em !important;
        padding-bottom: 0.2rem !important;
        font-weight: 700 !important;
    }
    
    h3 {
        color: #2c3e50 !important;
        margin-bottom: 0.3rem !important;
        margin-top: 0.3rem !important;
        font-size: 1.4em !important;
        padding-bottom: 0.2rem !important;
        font-weight: 600 !important;
    }
    
    /* Reduce markdown spacing */
    .stMarkdown {
        margin-bottom: 0.2rem !important;
    }
    
    /* Buttons */
    .stButton button {
        background-color: #3498db;
        color: #fff;
        border: none;
        font-weight: bold;
        padding: 0.4rem 0.8rem;
        border-radius: 6px;
        min-height: 40px;
        margin: 0.3rem 0;
        font-size: 0.9em;
    }
    
    /* Messages */
    .stSuccess, .stError, .stInfo, .stWarning {
        padding: 0.3rem;
        border-radius: 6px;
        margin: 0.2rem 0;
        font-size: 0.9em;
    }
    
    /* Input fields */
    .stTextInput input, .stNumberInput input, .stTextArea textarea {
        background-color: #f8fafc;
        color: #2c3e50;
        border: 2px solid #3498db;
        border-radius: 6px;
        padding: 0.4rem;
        min-height: 40px;
        font-size: 0.9em;
        margin: 0.2rem 0;
    }
    
    /* Selectbox */
    .stSelectbox select {
        background-color: #f8fafc;
        color: #2c3e50;
        border: 2px solid #3498db;
        border-radius: 6px;
        padding: 0.4rem;
        min-height: 40px;
        font-size: 0.9em;
        margin: 0.2rem 0;
    }
    
    /* Checkbox */
    .stCheckbox {
        margin: 0.2rem 0;
    }
    .stCheckbox label {
        color: #2c3e50 !important;
        font-size: 0.9em;
        padding: 0.2rem 0;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: #f8fafc;
        border: 2px solid #3498db;
        padding: 0.4rem;
        margin: 0.3rem 0;
        border-radius: 6px;
    }
    
    /* Divider */
    hr {
        margin: 0.4rem 0;
        border-width: 1px;
    }

    /* Section spacing */
    .element-container {
        margin-bottom: 0.3rem !important;
    }

    /* Column gaps */
    .row-widget {
        gap: 0.3rem !important;
    }

    /* Help text */
    .stTextInput .help-text, .stNumberInput .help-text, .stSelectbox .help-text {
        font-size: 0.8em;
        margin-top: 0.1rem;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables if they don't exist
if 'embeddingModel' not in st.session_state:
    st.session_state.embeddingModel = ""

if 'chunk_size' not in st.session_state:
    st.session_state.chunk_size = 500

if 'overlap' not in st.session_state:
    st.session_state.overlap = 50

if 'dbRetrievalAmount' not in st.session_state:
    st.session_state.dbRetrievalAmount = 3

if 'context_model' not in st.session_state:
    st.session_state.context_model = ""

if 'db_ready' not in st.session_state:
    st.session_state.db_ready = False

if 'collection' not in st.session_state:
    st.session_state.collection = None

if 'chroma_client' not in st.session_state:
    st.session_state.chroma_client = get_client()

if 'dropDown_embeddingModel_list' not in st.session_state:
    st.session_state.dropDown_embeddingModel_list = []

if 'dropDown_model_list' not in st.session_state:
    st.session_state.dropDown_model_list = []

if 'embedding' not in st.session_state:
    st.session_state.embedding = None

# Update available models
def update_ollama_model():
    """Updates the list of available Ollama models."""
    try:
        ollama_models = ollama.list()
        st.session_state.dropDown_embeddingModel_list = []
        st.session_state.dropDown_model_list = []
        
        for model in ollama_models['models']:
            if 'embedding' in model['model'] or 'embed' in model['model']:
                st.session_state.dropDown_embeddingModel_list.append(model['model'])
            else:
                st.session_state.dropDown_model_list.append(model['model'])
    except Exception as e:
        st.error(f"Error updating model list: {str(e)}")
        st.session_state.dropDown_embeddingModel_list = []
        st.session_state.dropDown_model_list = []

# Update available models on page load
update_ollama_model()

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
db_status_col1, db_status_col2 = st.columns([3,2])
with db_status_col1:
    if st.session_state.db_ready and st.session_state.collection is not None:
        st.success("✅ Embedding Database is ready and connected")
    else:
        st.error("❌ Embedding Database is not ready")
with db_status_col2:
    if st.session_state.db_ready and st.session_state.collection is not None:
        st.info(f"Documents processed: {len(st.session_state.processed_docs) if hasattr(st.session_state, 'processed_docs') else 0}")

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