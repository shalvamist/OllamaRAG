import streamlit as st
import os
from database.rag_db import (
    get_client,
    SOURCE_PATH,
    get_collection
)
from database.RAG_util import (
    delete_db,
    update_db,
    update_ollama_models,
    get_db_documents,
    add_document_to_db,
    remove_document_from_db
)
import asyncio

# Set page config
st.set_page_config(
    page_title="RAG Configuration - OllamaRAG",
    page_icon="🔗",
    initial_sidebar_state="expanded",
    layout="wide"
)

# Initialize session state variables
if 'databases' not in st.session_state:
    st.session_state.databases = {}  # {db_name: {"ready": bool, "collection": None}}

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

if 'chroma_client' not in st.session_state:
    st.session_state.chroma_client = get_client()

if 'dropDown_embeddingModel_list' not in st.session_state:
    st.session_state.dropDown_embeddingModel_list = []

if 'dropDown_model_list' not in st.session_state:
    st.session_state.dropDown_model_list = []

# Update available models on page load
embedding_models, regular_models = update_ollama_models()
st.session_state.dropDown_embeddingModel_list = embedding_models
st.session_state.dropDown_model_list = regular_models

# Function to handle database creation
def create_database(db_name: str) -> tuple[bool, str]:
    try:
        if db_name in st.session_state.databases:
            return False, "A database with this name already exists!"
        
        # Initialize database entry
        st.session_state.databases[db_name] = {"ready": False, "collection": None}
        
        # Create directory for database
        db_path = os.path.join(SOURCE_PATH, db_name)
        if not os.path.exists(db_path):
            os.makedirs(db_path)
        
        return True, f"Database '{db_name}' created successfully!"
    except Exception as e:
        return False, f"Error creating database: {str(e)}"

# Function to handle document deletion
def delete_document(db_name: str, doc_name: str) -> tuple[bool, str]:
    try:
        success, message = remove_document_from_db(db_name, doc_name)
        if success:
            # Mark database as needing update
            st.session_state.databases[db_name]["ready"] = False
        return success, message
    except Exception as e:
        return False, f"Error deleting document: {str(e)}"

# Function to handle document upload
def upload_documents(db_name: str, files) -> list[tuple[bool, str]]:
    results = []
    try:
        for file in files:
            success, message = add_document_to_db(
                db_name,
                file.name,
                file.getvalue()
            )
            if success:
                # Mark database as needing update
                st.session_state.databases[db_name]["ready"] = False
            results.append((success, message))
        return results
    except Exception as e:
        return [(False, f"Error uploading documents: {str(e)}")]

# Function to handle database deletion
def delete_database(db_name: str) -> tuple[bool, str]:
    try:
        success, message = delete_db(
            db_name,
            st.session_state.databases[db_name]["collection"],
            st.session_state.chroma_client
        )
        if success:
            del st.session_state.databases[db_name]
        return success, message
    except Exception as e:
        return False, f"Error deleting database: {str(e)}"

# Sidebar - Embedding Configuration
with st.sidebar:
    st.title("⚙️ RAG Settings")
    
    # Embedding Model Selection
    st.header("Embedding Configuration")
    selected_model = st.selectbox(
        "Select Embedding Model",
        st.session_state.dropDown_embeddingModel_list,
        index=st.session_state.dropDown_embeddingModel_list.index(st.session_state.embeddingModel) if st.session_state.embeddingModel in st.session_state.dropDown_embeddingModel_list else None,
        placeholder="Select model...",
        help="Choose the model for generating document embeddings"
    )
    if selected_model:
        st.session_state.embeddingModel = selected_model
    
    # Document Processing Settings
    st.header("Document Processing")
    st.session_state.chunk_size = st.number_input(
        "Chunk Size",
        min_value=100,
        max_value=2000,
        value=st.session_state.chunk_size,
        help="Size of text chunks for processing"
    )
    
    st.session_state.overlap = st.number_input(
        "Chunk Overlap",
        min_value=0,
        max_value=500,
        value=st.session_state.overlap,
        help="Number of overlapping tokens between chunks"
    )
    
    st.session_state.dbRetrievalAmount = st.number_input(
        "Retrieved Documents",
        min_value=1,
        max_value=10,
        value=st.session_state.dbRetrievalAmount,
        help="Number of documents to retrieve for each query"
    )
    
    # Contextual RAG Settings
    st.header("Contextual RAG")
    st.session_state.ContextualRAG = st.checkbox(
        "Enable Context Generation",
        value=st.session_state.ContextualRAG,
        help="Use AI to generate additional context for chunks"
    )
    
    if st.session_state.ContextualRAG:
        selected_context_model = st.selectbox(
            "Context Generation Model",
            st.session_state.dropDown_model_list,
            index=st.session_state.dropDown_model_list.index(st.session_state.context_model) if st.session_state.context_model in st.session_state.dropDown_model_list else None,
            placeholder="Select model...",
            help="Model used for generating context"
        )
        if selected_context_model:
            st.session_state.context_model = selected_context_model
    
    st.session_state.ContextualBM25RAG = st.checkbox(
        "Enable BM25 Retrieval",
        value=st.session_state.ContextualBM25RAG,
        help="Use BM25 algorithm for document retrieval"
    )

# Main content - Database Management
st.title("🗃️ RAG Databases")

# Create New Database
with st.expander("➕ Create New Database", expanded=True):
    new_db_col1, new_db_col2 = st.columns([3, 1])
    with new_db_col1:
        new_db_name = st.text_input(
            "Database Name",
            placeholder="Enter a name for the new database",
            help="Choose a unique name for your new RAG database"
        )
    with new_db_col2:
        if st.button("Create Database", key="create_db", disabled=not new_db_name):
            success, message = create_database(new_db_name)
            if success:
                st.success(message)
                st.rerun()
            else:
                st.error(message)

# Existing Databases
for db_name in st.session_state.databases:
    with st.expander(f"📚 {db_name}", expanded=True):
        db_col1, db_col2 = st.columns([3, 1])
        
        with db_col1:
            st.markdown("### Documents")
            documents = get_db_documents(db_name)
            if documents:
                for doc in documents:
                    doc_col1, doc_col2 = st.columns([4, 1])
                    with doc_col1:
                        st.text(f"📄 {doc}")
                    with doc_col2:
                        if st.button("🗑️", key=f"delete_doc_{db_name}_{doc}"):
                            success, message = delete_document(db_name, doc)
                            if success:
                                st.success(message)
                                st.rerun()
                            else:
                                st.error(message)
            else:
                st.info("No documents in this database.")
            
            # Upload new documents
            uploaded_files = st.file_uploader(
                "Add Documents",
                accept_multiple_files=True,
                type=['pdf'],
                key=f"upload_{db_name}",
                help="Upload PDF documents to this database"
            )
            
            if uploaded_files:
                results = upload_documents(db_name, uploaded_files)
                for success, message in results:
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
                st.rerun()
        
        with db_col2:
            st.markdown("### Actions")
            
            # Database Status
            if st.session_state.databases[db_name]["ready"]:
                st.success("✅ Database Ready")
            else:
                st.warning("⚠️ Needs Update")
            
            # Update Database
            async def update_database(db_name: str):
                if not st.session_state.embeddingModel:
                    st.error("Please select an embedding model first.")
                    return
                
                try:
                    progress_text = st.empty()
                    progress_bar = st.progress(0)
                    
                    def progress_callback(message: str, progress: float):
                        progress_text.text(message)
                        progress_bar.progress(progress)
                    
                    success, message, processed = await update_db(
                        db_name,
                        st.session_state.embeddingModel,
                        st.session_state.chroma_client,
                        st.session_state.chunk_size,
                        st.session_state.overlap,
                        st.session_state.context_model if st.session_state.ContextualRAG else None,
                        st.session_state.ContextualRAG,
                        st.session_state.ContextualBM25RAG,
                        progress_callback
                    )
                    
                    progress_text.empty()
                    progress_bar.empty()
                    
                    if success:
                        # Get fresh collection after update
                        collection = get_collection(
                            st.session_state.embeddingModel,
                            st.session_state.chroma_client,
                            f"rag_collection_{db_name}",
                            f"RAG collection for {db_name}"
                        )
                        if collection:
                            st.session_state.databases[db_name] = {
                                "ready": True,
                                "collection": collection
                            }
                            st.success(message)
                            st.rerun()
                        else:
                            st.error("Failed to get collection after update")
                    else:
                        st.error(message)
                except Exception as e:
                    st.error(f"Error updating database: {str(e)}")
            
            if st.button("🔄 Update", key=f"update_db_{db_name}", disabled=not st.session_state.embeddingModel):
                asyncio.run(update_database(db_name))
            
            # Delete Database
            if st.button("🗑️ Delete", key=f"delete_db_{db_name}", type="primary"):
                success, message = delete_database(db_name)
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message) 