import streamlit as st
import os
from uuid import uuid4
import asyncio
from langchain_ollama import OllamaLLM
from dbAPI import loadDocuments, getClient, getCollection, getBM25retriver, SOURCE_PATH

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

def updateDB():
    """Updates the database with new documents and configurations."""
    st.session_state.db_ready = False
    
    # Check if embedding model is selected
    if st.session_state.embeddingModel == "":
        st.error("Please select an embedding model")
        return
    
    # Check for new files
    if os.listdir(SOURCE_PATH) == st.session_state.docs:
        st.error("No new sources found. Please upload new sources for DB update.")
        return    

    with st.spinner("Rebuilding Database - Please wait..."):    
        # Initialize context LLM if needed
        contextLLM = None
        if st.session_state.context_model and st.session_state.ContextualRAG:
            contextLLM = OllamaLLM(
                model=st.session_state.context_model,
                temperature=0.0,
                num_predict=1000,
            )

        # Process documents
        all_splits, aiSplits = asyncio.run(loadDocuments(
            int(st.session_state.chunk_size),
            int(st.session_state.overlap),
            st.session_state.docs,
            contextLLM
        ))

        # Setup BM25 retriever if enabled
        if st.session_state.ContextualBM25RAG:
            st.session_state.BM25retriver = getBM25retriver(all_splits)

        # Initialize collection
        st.session_state.collection = getCollection(
            st.session_state.embeddingModel,
            st.session_state.chroma_client,
            "rag_collection_demo",
            "A collection for RAG with Ollama - Demo1"
        )

        # Generate UUIDs for documents
        uuids = [str(uuid4()) for _ in range(len(all_splits))]

        # Combine AI-generated context if available
        if len(aiSplits) > 0:
            for i in range(len(aiSplits)):
                all_splits[i] = all_splits[i] + "\n\n" + aiSplits[i]

        # Add documents to collection
        if len(st.session_state.docs) > 0:
            st.session_state.collection.add(documents=all_splits, ids=uuids)
            st.session_state.db_ready = True
            st.success("Database updated successfully!")
        else:
            st.error("No PDF files found in the source folder. Please upload PDF files to proceed.")
            st.session_state.db_ready = False

# Set page config
st.set_page_config(
    page_title="RAG Configuration - OllamaRAG",
    page_icon="🔗",
)

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
st.session_state.embeddingModel = st.selectbox(
    "Select Embedding Model",
    st.session_state.dropDown_embeddingModel_list,
    index=None,
    placeholder="Select model...",
    help="Choose the model for generating document embeddings"
)

# Document Processing Settings
st.header("Document Processing")
col1, col2 = st.columns(2)
with col1:
    st.session_state.chunk_size = st.number_input(
        "Chunk Size",
        min_value=100,
        max_value=2000,
        value=int(st.session_state.chunk_size),
        help="Size of text chunks for processing"
    )
    
    st.session_state.dbRetrievalAmount = st.number_input(
        "Number of Retrieved Documents",
        min_value=1,
        max_value=10,
        value=int(st.session_state.dbRetrievalAmount),
        help="Number of documents to retrieve for each query"
    )

with col2:
    st.session_state.overlap = st.number_input(
        "Chunk Overlap",
        min_value=0,
        max_value=500,
        value=int(st.session_state.overlap),
        help="Number of overlapping tokens between chunks"
    )

# Contextual RAG Settings
st.header("Contextual RAG Settings")
col1, col2 = st.columns(2)
with col1:
    st.session_state.ContextualRAG = st.checkbox(
        "Enable Contextual RAG",
        value=st.session_state.ContextualRAG,
        help="Use AI to generate additional context for chunks"
    )
    
    if st.session_state.ContextualRAG:
        st.session_state.context_model = st.selectbox(
            "Context Generation Model",
            st.session_state.dropDown_model_list,
            index=None,
            placeholder="Select model...",
            help="Model used for generating context"
        )

with col2:
    st.session_state.ContextualBM25RAG = st.checkbox(
        "Enable BM25 Retrieval",
        value=st.session_state.ContextualBM25RAG,
        help="Use BM25 algorithm for document retrieval"
    )

# Document Upload
st.header("Document Management")
uploaded_files = st.file_uploader(
    "Upload Documents",
    accept_multiple_files=True,
    type=['pdf'],
    help="Upload PDF documents to be processed"
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        save_path = os.path.join(SOURCE_PATH, uploaded_file.name)
        with open(save_path, mode='wb') as w:
            w.write(uploaded_file.getvalue())
    st.success(f"Uploaded {len(uploaded_files)} document(s)")

# Database Management
st.header("Database Management")
col1, col2 = st.columns(2)
with col1:
    if st.button('Rebuild Database', disabled=(st.session_state.embeddingModel is None)):
        updateDB()
with col2:
    if st.button('Clear Database'):
        deleteDB()
        st.success("Database cleared successfully") 