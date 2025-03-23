import streamlit as st
import os
from CommonUtils.rag_utils import (
    get_client,
    SOURCE_PATH,
    get_collection,
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

# Add these helper functions at the top of the file after imports
def show_success(message):
    """Display a success message in the Streamlit UI"""
    st.success(message)

def show_warning(message):
    """Display a warning message in the Streamlit UI"""
    st.warning(message)

def show_error(message):
    """Display an error message in the Streamlit UI"""
    st.error(message)

# Function to handle database creation
def create_database(db_name: str) -> tuple[bool, str]:
    try:
        if db_name in st.session_state.databases:
            return False, "A database with this name already exists!"
        
        # Sanitize database name for ChromaDB collection naming
        # Remove special characters that could cause issues
        if any(char in db_name for char in r'\/:*?"<>|'):
            return False, "Database name cannot contain special characters: \\ / : * ? \" < > |"
        
        # Initialize database entry
        st.session_state.databases[db_name] = {"ready": False, "collection": None}
        
        # Create directory for database
        db_path = os.path.join(SOURCE_PATH, db_name)
        if not os.path.exists(db_path):
            os.makedirs(db_path)
        
        return True, f"Database '{db_name}' created successfully! Next steps:\n1. Upload PDF documents to the database\n2. Select an embedding model in the sidebar\n3. Click 'Update' to process the documents"
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
        # Get the collection object from session state
        collection = st.session_state.databases[db_name].get("collection")
        
        # Call the delete_db function with proper error handling
        try:
            success, message = delete_db(
                db_name,
                collection,
                st.session_state.chroma_client
            )
            
            if success:
                # Remove from session state
                if db_name in st.session_state.databases:
                    del st.session_state.databases[db_name]
                    
                # Force reset the chroma client to ensure clean state
                try:
                    st.session_state.chroma_client.reset()
                    # Recreate the client to ensure it's completely fresh
                    st.session_state.chroma_client = get_client()
                except Exception as reset_error:
                    print(f"Warning when resetting client: {str(reset_error)}")
                
                return True, message
            else:
                return False, f"Failed to delete database: {message}"
        except Exception as delete_error:
            return False, f"Error in delete_db function: {str(delete_error)}"
    except Exception as e:
        return False, f"Error accessing database {db_name}: {str(e)}"

# Sidebar - Embedding Configuration
with st.sidebar:
    # Smaller title with more subtle styling - exactly matching Model Settings page
    st.markdown("""
    <h2 style="font-size: 1.5em; color: #0D47A1; margin-bottom: 15px; padding-bottom: 5px; border-bottom: 1px solid #e0e0e0;">
    ⚙️ RAG Settings
    </h2>
    """, unsafe_allow_html=True)
    
    # Emergency reset button for ChromaDB
    if st.button("🔄 Reset ChromaDB", key="reset_chromadb", help="Emergency reset of ChromaDB client"):
        try:
            # Reset client
            st.session_state.chroma_client.reset()
            # Recreate client
            st.session_state.chroma_client = get_client()
            st.success("ChromaDB client reset successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"Error resetting ChromaDB: {str(e)}")
    
    # Embedding Model Selection - Collapsable
    with st.expander("🔤 Embedding Model", expanded=False):
        # Show a clear warning if no embedding model is selected
        if not st.session_state.embeddingModel:
            st.warning("⚠️ Please select an embedding model below to enable database updates")
        
        # Check if embedding models are available
        if not st.session_state.dropDown_embeddingModel_list:
            st.error("No embedding models available!")
            st.info("""
            Please pull an embedding model with Ollama. Run this in your terminal:
            ```
            ollama pull nomic-embed-text
            ```
            or another embedding model like:
            ```
            ollama pull all-minilm:embedding
            ```
            """
            )
            
            if st.button("🔄 Refresh Models", key="refresh_models"):
                embedding_models, regular_models = update_ollama_models()
                st.session_state.dropDown_embeddingModel_list = embedding_models
                st.session_state.dropDown_model_list = regular_models
                st.rerun()
        
        selected_model = st.selectbox(
            "Select Embedding Model",
            st.session_state.dropDown_embeddingModel_list,
            index=st.session_state.dropDown_embeddingModel_list.index(st.session_state.embeddingModel) if st.session_state.embeddingModel in st.session_state.dropDown_embeddingModel_list else None,
            placeholder="Select model...",
            help="Choose the model for generating document embeddings"
        )
        if selected_model:
            st.session_state.embeddingModel = selected_model
            st.success(f"Selected model: {selected_model}")
    
    # Document Processing Settings - Collapsable
    with st.expander("📄 Document Processing", expanded=False):
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
    
    # Contextual RAG Settings - Collapsable
    with st.expander("🔍 Advanced RAG Options", expanded=False):
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
st.markdown("""
<h1 style="text-align: center; color: #0D47A1; margin-bottom: 30px;">🗃️ RAG Databases</h1>
""", unsafe_allow_html=True)

# Create New Database
with st.expander("", expanded=True):
    # Add a prominent header for creating a new database
    st.markdown("""
    <h2 style="text-align: center; color: #1E88E5; margin-bottom: 20px;">➕ Create New Database</h2>
    """, unsafe_allow_html=True)
    
    create_container = st.container(border=True)
    with create_container:
        st.markdown('<div style="color: #555; font-weight: 600; font-size: 0.9rem;">New Database Setup</div>', unsafe_allow_html=True)
        
        new_db_name = st.text_input(
            "Database Name",
            placeholder="Enter a name for the new database",
            help="Choose a unique name for your new RAG database"
        )
        
        if st.button(
            "➕ Create Database", 
            key="create_db", 
            disabled=not new_db_name,
            type="primary",
            use_container_width=True
        ):
            success, message = create_database(new_db_name)
            if success:
                st.success(message)
                st.rerun()
            else:
                st.error(message)

# Existing Databases
for db_name in st.session_state.databases:
    with st.expander(f"", expanded=True):
        # Add a prominent header for the database name
        st.markdown(f"""
        <h2 style="text-align: center; color: #1E88E5; margin-bottom: 20px;">📚 {db_name}</h2>
        """, unsafe_allow_html=True)
        
        db_col1, db_col2 = st.columns([2, 1])  # Adjusted ratio to give more space to the Actions column
        
        with db_col1:
            st.markdown('<div style="color: #555; font-size: 1rem; font-weight: 600; border-bottom: 1px solid #eee; padding-bottom: 0.5rem; margin-bottom: 1rem;">Documents</div>', unsafe_allow_html=True)
            documents = get_db_documents(db_name)
            if documents:
                docs_container = st.container(border=True)
                with docs_container:
                    st.markdown('<div style="color: #388E3C; font-weight: 500; font-size: 0.9rem;">Step 1: ✅ Documents Uploaded</div>', unsafe_allow_html=True)
                    for doc in documents:
                        doc_col1, doc_col2 = st.columns([4, 1])
                        with doc_col1:
                            st.text(f"📄 {doc}")
                        with doc_col2:
                            if st.button("🗑️", key=f"delete_doc_{db_name}_{doc}", help=f"Delete {doc}"):
                                with st.spinner(f"Deleting {doc}..."):
                                    success, message = delete_document(db_name, doc)
                                    if success:
                                        st.success(f"Deleted {doc}")
                                        # Mark database as needing update if it was previously ready
                                        if st.session_state.databases[db_name]["ready"]:
                                            st.session_state.databases[db_name]["ready"] = False
                                            st.warning("Database needs to be updated after document removal")
                                        st.rerun()
                                    else:
                                        st.error(message)
                    
                    # Show document count
                    st.caption(f"Total: {len(documents)} document(s)")
                
                # If documents are uploaded but database is not ready, show guidance
                if not st.session_state.databases[db_name]["ready"]:
                    guidance_container = st.container()
                    with guidance_container:
                        if st.session_state.embeddingModel:
                            st.info("##### Step 2: ⏳ Click 'Process Documents Now' button to complete setup")
                        else:
                            st.warning("##### Step 2: ⚠️ Select an embedding model in the sidebar to enable processing")
            else:
                st.markdown('<div style="color: #F57C00; font-weight: 500; font-size: 0.9rem;">Step 1: ⏳ Upload PDF Documents</div>', unsafe_allow_html=True)
                st.info("No documents in this database. Upload PDFs to get started.")
            
            # Upload new documents
            upload_container = st.container(border=True)
            with upload_container:
                st.markdown('<div style="color: #555; font-weight: 600; font-size: 0.9rem;">Upload Documents</div>', unsafe_allow_html=True)
                uploaded_files = st.file_uploader(
                    "Select PDF Files",
                    accept_multiple_files=True,
                    type=['pdf'],
                    key=f"upload_{db_name}",
                    help="Upload PDF documents to this database"
                )
                
                # Show upload button only when files are selected
                if uploaded_files:
                    if st.button("📤 Upload Selected Files", key=f"upload_btn_{db_name}", type="primary", use_container_width=True):
                        with st.spinner("Uploading documents..."):
                            results = upload_documents(db_name, uploaded_files)
                            
                            # Count successes and failures
                            successes = sum(1 for success, _ in results if success)
                            failures = len(results) - successes
                            
                            if successes > 0:
                                st.success(f"✅ Successfully uploaded {successes} document(s)")
                            
                            if failures > 0:
                                st.error(f"❌ Failed to upload {failures} document(s)")
                            
                            # Only reload if at least one document was successfully uploaded
                            if successes > 0:
                                st.rerun()
        
        with db_col2:
            st.markdown('<div style="color: #555; font-size: 1rem; font-weight: 600; border-bottom: 1px solid #eee; padding-bottom: 0.5rem; margin-bottom: 1rem;">Actions</div>', unsafe_allow_html=True)
            
            # Database Status in a more visible card-like container
            status_container = st.container(border=True)
            with status_container:
                st.markdown('<div style="color: #555; font-weight: 600; font-size: 0.9rem;">Status</div>', unsafe_allow_html=True)
                if st.session_state.databases[db_name]["ready"]:
                    st.success("✅ Database Ready for RAG")
                else:
                    st.warning("⚠️ Database Needs Update")
            
            # Update Database
            async def update_database(db_name: str) -> None:
                """
                Updates a database by processing PDF files in database folder.
                Triggered by the 'Update Database' button.
                """
                if not st.session_state.embeddingModel:
                    st.error("Please select an embedding model first.")
                    return
                
                embedding_model = st.session_state.embeddingModel
                
                # Log key parameters for debugging
                print(f"Starting update for database: {db_name}")
                print(f"Using embedding model: {embedding_model}")
                
                # Initialize progress bar in the UI
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(message, progress):
                    status_text.text(message)
                    progress_bar.progress(progress)
                
                try:
                    # Call the async update_db function
                    success, message, processed_docs = await update_db(
                        db_name=db_name,
                        embedding_model=embedding_model,
                        chroma_client=st.session_state.chroma_client,
                        chunk_size=int(st.session_state.chunk_size),
                        overlap=int(st.session_state.overlap),
                        context_model=st.session_state.context_model if st.session_state.ContextualRAG else None,
                        use_contextual_rag=st.session_state.ContextualRAG,
                        progress_callback=update_progress
                    )
                    
                    # Update UI based on result
                    if success:
                        show_success(message)
                        try:
                            # Try to get a fresh collection
                            collection = get_collection(
                                embedding_model, 
                                st.session_state.chroma_client,
                                f"rag_collection_{db_name}",
                                f"RAG collection for {db_name}"
                            )
                            
                            if collection:
                                st.session_state.databases[db_name]["collection"] = collection
                                st.session_state.databases[db_name]["ready"] = True
                                print(f"Successfully updated collection for {db_name}")
                            else:
                                show_warning(f"Could not retrieve the updated collection for {db_name}")
                                print(f"Failed to get collection for {db_name} after update")
                        except Exception as e:
                            show_error(f"Error getting updated collection: {str(e)}")
                            import traceback
                            print(f"Collection retrieval error: {traceback.format_exc()}")
                    else:
                        show_error(message)
                except Exception as e:
                    error_msg = f"Error updating database: {str(e)}"
                    show_error(error_msg)
                    import traceback
                    print(traceback.format_exc())
                    
                    # Try to reset the client to recover from errors
                    try:
                        st.session_state.chroma_client.reset()
                        st.session_state.chroma_client = get_client()
                        print("Reset ChromaDB client after error")
                    except Exception as reset_error:
                        print(f"Failed to reset client: {str(reset_error)}")
                finally:
                    # Clean up UI elements
                    progress_bar.empty()
                    status_text.empty()
                    # Force a page refresh
                    st.rerun()
            
            # Show update button with helpful guidance in a card-like container
            processing_container = st.container(border=True)
            with processing_container:
                st.markdown('<div style="color: #555; font-weight: 600; font-size: 0.9rem;">Process Documents</div>', unsafe_allow_html=True)
                documents = get_db_documents(db_name)
                
                if st.session_state.embeddingModel and documents:
                    # Show prominent update button when everything is ready
                    update_button = st.button(
                        "🔄 Process Documents Now",
                        key=f"update_db_{db_name}",
                        type="primary",
                        use_container_width=True
                    )
                    if update_button:
                        # Add safety check to verify Ollama is running
                        try:
                            # Try to ping Ollama service
                            models = update_ollama_models()
                            if not models[0] and not models[1]:
                                st.error("⚠️ Cannot connect to Ollama service. Please make sure Ollama is running.")
                                st.info("Tip: Run 'ollama serve' in a terminal to start the Ollama service")
                            else:
                                # Proceed with database update
                                asyncio.run(update_database(db_name))
                        except Exception as e:
                            st.error(f"Error connecting to Ollama: {str(e)}")
                            st.info("Please make sure the Ollama service is running before processing documents")
                else:
                    # Show disabled button with explanation
                    update_button = st.button(
                        "🔄 Update", 
                        key=f"update_db_{db_name}", 
                        disabled=True,
                        help="Upload documents and select an embedding model to enable updates",
                        use_container_width=True
                    )
                    
                    missing_items = []
                    if not documents:
                        missing_items.append("• Upload documents")
                    
                    if not st.session_state.embeddingModel:
                        missing_items.append("• Select an embedding model")
                    
                    if missing_items:
                        st.info("To enable processing:\n" + "\n".join(missing_items))
            
            # Delete Database in a card-like container
            delete_container = st.container(border=True)
            with delete_container:
                st.markdown('<div style="color: #555; font-weight: 600; font-size: 0.9rem;">Database Management</div>', unsafe_allow_html=True)
                
                delete_confirm = st.checkbox(
                    f"Confirm permanent deletion",
                    key=f"confirm_delete_{db_name}",
                    help="This will permanently delete all source documents and embeddings"
                )
                
                if st.button(
                    f"🗑️ Delete '{db_name}' Database", 
                    key=f"delete_db_{db_name}", 
                    type="primary" if delete_confirm else "secondary",
                    disabled=not delete_confirm,
                    help="Delete this database, its documents, and associated embeddings",
                    use_container_width=True
                ):
                    with st.spinner(f"Deleting database '{db_name}'..."):
                        success, message = delete_database(db_name)
                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message) 