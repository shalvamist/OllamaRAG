import streamlit as st
import os
from database.rag_db import SOURCE_PATH, DB_PATH

def initApp():
    # Create the directories if it does not exist    
    if not os.path.isdir(SOURCE_PATH):
        os.mkdir(SOURCE_PATH)
    else:
        for file in os.listdir(SOURCE_PATH):
            os.remove(os.path.join(SOURCE_PATH, file))

    if not os.path.isdir(DB_PATH):
        os.mkdir(DB_PATH)

    if 'ollama_model' not in st.session_state:
        st.session_state.ollama_model = None
    if 'chatReady' not in st.session_state:
        st.session_state.chatReady = False
    if 'dropDown_model_list' not in st.session_state:
        st.session_state.dropDown_model_list = []
    if 'dropDown_embeddingModel_list' not in st.session_state:
        st.session_state.dropDown_embeddingModel_list = []
    if 'loaded_model_list' not in st.session_state:
        st.session_state.loaded_model_list = []
    if 'llm' not in st.session_state:
        st.session_state.llm = None
    if 'embedding' not in st.session_state:
        st.session_state.embedding = None
    if 'context_model' not in st.session_state:
        st.session_state.context_model = ""
    if 'embeddingModel' not in st.session_state:
        st.session_state.embeddingModel = ""
    if 'ollama_embedding_model' not in st.session_state:
        st.session_state.ollama_embedding_model = None
    if 'collection' not in st.session_state:
        st.session_state.collection = None
    if 'chroma_client' not in st.session_state:
        st.session_state.chroma_client = None
    if 'docs' not in st.session_state:
        st.session_state.docs = []
    if 'newMaxTokens' not in st.session_state:
        st.session_state.newMaxTokens = 128
    if 'CRAG_iterations' not in st.session_state:
        st.session_state.CRAG_iterations = 5
    if 'overlap' not in st.session_state:
        st.session_state.overlap = 200
    if 'chunk_size' not in st.session_state:
        st.session_state.chunk_size = 1000
    if 'database_ready' not in st.session_state:
        st.session_state.database_ready = False
    if 'contextWindow' not in st.session_state:
        st.session_state.contextWindow = 2048
    if 'db_ready' not in st.session_state:
        st.session_state.db_ready = False
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    if "ContextualRAG" not in st.session_state:
        st.session_state.ContextualRAG = False
    if "ContextualBM25RAG" not in st.session_state:
        st.session_state.ContextualBM25RAG = False
    if "BM25retriver" not in st.session_state:
        st.session_state.BM25retriver = None
    if "dbRetrievalAmount" not in st.session_state:
        st.session_state.dbRetrievalAmount = 3
    if "temperature" not in st.session_state:
        st.session_state.temperature = 1.0
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = "You are a helpful assistant."

# Set page config
st.set_page_config(
    page_title="OllamaRAG",
    page_icon="🦙",
    layout="wide"
)

# Initialize app
initApp()

# Main page content
st.title("🦙🦜🔗 OllamaRAG 🔗🦜🦙")

st.markdown("""
## Welcome to OllamaRAG! 👋

OllamaRAG is a powerful tool that combines the capabilities of Ollama's local LLMs with RAG (Retrieval-Augmented Generation) for enhanced conversational AI.

### What is OllamaRAG?
OllamaRAG is an advanced chat interface that leverages local Language Models (LLMs) through Ollama and enhances them with RAG capabilities. This combination allows for more accurate and contextually relevant responses based on your documents.

### Key Features:
- 🤖 **Local LLM Support**: Run AI models locally on your machine
- 📚 **RAG Integration**: Enhance responses with relevant document context
- 🎯 **Contextual Retrieval**: Smart document chunk retrieval for better context
- 📊 **BM25 Search**: Advanced search algorithm for improved document matching
- 🔄 **Flexible Configuration**: Customize model and RAG parameters
""")

# Quick Start Guide
st.header("🚀 Quick Start Guide")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 1️⃣ Model Setup
    
    Go to **🦙 Model Settings** to:
    - Select your Ollama model
    - Configure model parameters
    - Set system prompt
    - Apply your settings
    """)

with col2:
    st.markdown("""
    ### 2️⃣ RAG Configuration
    
    Visit **🔗 RAG Config** to:
    - Choose embedding model
    - Upload your documents
    - Configure chunk settings
    - Enable advanced features
    """)

with col3:
    st.markdown("""
    ### 3️⃣ Start Chatting
    
    Head to **💬 Chat** to:
    - Interact with the model
    - Ask questions about your documents
    - View retrieved context
    - Get AI-powered responses
    """)

# System Status
st.header("📊 System Status")

col1, col2, col3 = st.columns(3)

with col1:
    if st.session_state.chatReady and st.session_state.ollama_model:
        st.success(f"Model: {st.session_state.ollama_model}")
    else:
        st.error("Model: Not Configured")

with col2:
    if st.session_state.db_ready:
        st.success("RAG Database: Ready")
    else:
        st.warning("RAG Database: Not Configured")

with col3:
    if len(st.session_state.docs) > 0:
        st.info(f"Documents Loaded: {len(st.session_state.docs)}")
    else:
        st.warning("No Documents Loaded")

# Additional Information
st.header("ℹ️ Additional Information")

st.markdown("""
### About RAG
Retrieval-Augmented Generation (RAG) is a technique that enhances Large Language Models by providing them with relevant context from your documents. This results in:
- More accurate responses
- Better factual grounding
- Reduced hallucinations
- Domain-specific knowledge

### About Ollama
Ollama is a framework for running large language models locally. Benefits include:
- Privacy and security
- No API costs
- Customizable models
- Local processing

### Tips for Best Results
1. Choose the right model for your needs
2. Configure appropriate chunk sizes for your documents
3. Experiment with retrieval settings
4. Use specific questions for better context retrieval
""") 