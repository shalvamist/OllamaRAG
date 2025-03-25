import streamlit as st
import os
from CommonUtils.rag_utils import SOURCE_PATH, DB_PATH, get_client

def initApp():
    # Create the directories if they do not exist    
    os.makedirs(SOURCE_PATH, exist_ok=True)
    os.makedirs(DB_PATH, exist_ok=True)

    # Clean up source documents directory
    for file in os.listdir(SOURCE_PATH):
        file_path = os.path.join(SOURCE_PATH, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Initialize session state variables
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
        st.session_state.chroma_client = get_client()
    if 'docs' not in st.session_state:
        st.session_state.docs = []
    if 'newMaxTokens' not in st.session_state:
        st.session_state.newMaxTokens = 1024
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

# Custom CSS styling
st.markdown("""
<style>
    /* Main background and text colors */
    .stApp {
        color: #1a2234;
    }
    
    /* Headers */
    h1 {
        color: #0D47A1 !important;
        margin-bottom: 1rem !important;
        font-size: 2.2em !important;
        font-weight: 800 !important;
    }
    
    h2 {
        color: #1E88E5 !important;
        margin-bottom: 0.8rem !important;
        font-size: 1.8em !important;
        font-weight: 700 !important;
    }
    
    h3 {
        color: #1976D2 !important;
        margin-bottom: 0.6rem !important;
        font-size: 1.4em !important;
        font-weight: 600 !important;
    }
    
    /* Card styling */
    [data-testid="stExpander"] {
        border: none !important;
        box-shadow: none !important;
    }
    
    /* Buttons */
    .stButton button {
        border-radius: 4px;
    }
    
    /* Container borders */
    [data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column"] > div[data-testid="stVerticalBlock"] {
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Success and warning messages */
    .stSuccess, .stWarning, .stError, .stInfo {
        border-radius: 4px;
    }
    
    /* Input fields */
    .stTextInput input, .stNumberInput input, .stTextArea textarea, .stSelectbox select {
        border-radius: 4px;
    }
    
    /* Feature card styling */
    .feature-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        border-left: 4px solid #1E88E5;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    /* Status indicators */
    .status-card {
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        background-color: #f8f9fa;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize app
initApp()

# Main page content
st.markdown("""
<h1 style="text-align: center; color: #0D47A1; margin-bottom: 20px;">
    🦙 OllamaRAG - Local LLM Assistant
</h1>
""", unsafe_allow_html=True)

# Introduction section with blue card styling
st.markdown("""
<div style="background-color: #f0f7ff; padding: 20px; border-radius: 10px; margin-bottom: 30px; border-left: 5px solid #1E88E5;">
<h2 style="color: #1E88E5; margin-top: 0;">Welcome to OllamaRAG</h2>

A powerful platform that combines Ollama's local LLMs with advanced Retrieval-Augmented Generation (RAG), 
deep research capabilities, and debate tools - all running locally on your machine.

- 🔒 **Privacy-focused**: All processing happens on your device
- 🚀 **No API costs**: Use your own local models without subscription fees
- 🔍 **Enhanced context**: RAG technology for more accurate responses
- 📚 **Document intelligence**: Process your documents for better answers
- 🔎 **Research capabilities**: Automated deep research across multiple sources
- 🗣️ **Debate simulation**: Generate balanced perspectives on any topic
</div>
""", unsafe_allow_html=True)

# Quick Start Guide
st.markdown('<h2 style="color: #1E88E5;">Getting Started</h2>', unsafe_allow_html=True)

st.markdown("""
<div style="background-color: #f0f7ff; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
<h3 style="color: #1E88E5; margin-top: 0;">📚 Quick Start Guide</h3>

<ol>
    <li><strong>Set up a model</strong> - First, visit the Model Settings page to select and configure your Ollama model</li>
    <li><strong>Configure RAG</strong> - Upload documents and set up your embedding model in the RAG Configuration page</li>
    <li><strong>Start chatting</strong> - Use the Chat interface to interact with your model with document context</li>
    <li><strong>Try advanced features</strong> - Explore Deep Research or Debate simulation capabilities</li>
</ol>

<p><strong>Need Help?</strong> Each page includes detailed instructions and tooltips to guide you through the process.</p>
</div>
""", unsafe_allow_html=True)

# Features section in a grid layout
st.markdown('<h2 style="color: #1E88E5;">Main Features</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="feature-card">
        <h3 style="color: #1E88E5; margin-top: 0;">🦙 Model Settings</h3>
        <p>Configure and manage your Ollama models:</p>
        <ul>
            <li>Select from locally installed models</li>
            <li>Configure temperature and token settings</li>
            <li>Customize context window size</li>
            <li>Define system prompts for specialized assistants</li>
        </ul>
        <a href="./🦙_Model_Settings">Configure your models →</a>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <h3 style="color: #1E88E5; margin-top: 0;">💬 Chat Interface</h3>
        <p>Interact with your models in a modern chat interface:</p>
        <ul>
            <li>Chat with context from your documents</li>
            <li>Access specific databases using <code>@database_name</code> mentions</li>
            <li>Save and load conversations</li>
            <li>Clear conversation history when needed</li>
        </ul>
        <a href="./💬_Chat">Start chatting →</a>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <h3 style="color: #1E88E5; margin-top: 0;">🔗 RAG Configuration</h3>
        <p>Set up your Retrieval-Augmented Generation system:</p>
        <ul>
            <li>Upload and process documents (PDF, DOCX, TXT)</li>
            <li>Configure embedding models for semantic search</li>
            <li>Customize document chunking parameters</li>
            <li>Enable contextual processing for better retrieval</li>
        </ul>
        <a href="./🔗_RAG_Config">Configure RAG →</a>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <h3 style="color: #1E88E5; margin-top: 0;">🔍 Deep Research</h3>
        <p>Perform comprehensive research on any topic:</p>
        <ul>
            <li>Break down topics into logical subtopics</li>
            <li>Search across multiple sources (Web, News, Wikipedia)</li>
            <li>Generate well-structured research reports</li>
            <li>Track research progress and sources</li>
        </ul>
        <a href="./🔍_DeepResearch">Start researching →</a>
    </div>
    """, unsafe_allow_html=True)

# Advanced Features Section
st.markdown('<h2 style="color: #1E88E5; margin-top: 30px;">Advanced Capabilities</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="feature-card">
        <h3 style="color: #1E88E5; margin-top: 0;">🗣️ Debate Simulation</h3>
        <p>Generate balanced perspectives on any topic:</p>
        <ul>
            <li>Configure two AI debaters with different viewpoints</li>
            <li>Set a neutral AI judge to evaluate arguments</li>
            <li>Customize debate parameters and depth</li>
            <li>Explore complex topics from multiple angles</li>
        </ul>
        <a href="./🗣️_Debate">Start a debate →</a>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <h3 style="color: #1E88E5; margin-top: 0;">⚙️ RAG Technology</h3>
        <p>Advanced Retrieval-Augmented Generation features:</p>
        <ul>
            <li>Context-aware document retrieval</li>
            <li>BM25 + semantic hybrid search options</li>
            <li>Custom chunking strategies for different document types</li>
            <li>Iterative context generation for complex queries</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# System Status
st.markdown('<h2 style="color: #1E88E5; margin-top: 30px;">System Status</h2>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    if st.session_state.chatReady and st.session_state.ollama_model:
        st.markdown("""
        <div class="status-card" style="border-left: 4px solid #4CAF50;">
            <h3 style="color: #4CAF50; margin-top: 0; font-size: 1.2em;">✅ Model Connected</h3>
            <p><strong>Active Model:</strong> {}</p>
            <p><strong>Temperature:</strong> {}</p>
            <p><strong>Max Tokens:</strong> {}</p>
        </div>
        """.format(
            st.session_state.ollama_model,
            st.session_state.temperature,
            st.session_state.newMaxTokens
        ), unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-card" style="border-left: 4px solid #F44336;">
            <h3 style="color: #F44336; margin-top: 0; font-size: 1.2em;">❌ No Model Connected</h3>
            <p>Please visit the Model Settings page to select and configure an Ollama model.</p>
            <a href="./🦙_Model_Settings">Configure model →</a>
        </div>
        """, unsafe_allow_html=True)

with col2:
    if st.session_state.db_ready:
        st.markdown("""
        <div class="status-card" style="border-left: 4px solid #4CAF50;">
            <h3 style="color: #4CAF50; margin-top: 0; font-size: 1.2em;">✅ RAG System Ready</h3>
            <p><strong>Embedding Model:</strong> {}</p>
            <p><strong>Chunk Size:</strong> {}</p>
            <p><strong>Retrieved Docs:</strong> {}</p>
        </div>
        """.format(
            st.session_state.embeddingModel,
            st.session_state.chunk_size,
            st.session_state.dbRetrievalAmount
        ), unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-card" style="border-left: 4px solid #FF9800;">
            <h3 style="color: #FF9800; margin-top: 0; font-size: 1.2em;">⚠️ RAG Not Configured</h3>
            <p>Visit the RAG Configuration page to set up your document database.</p>
            <a href="./🔗_RAG_Config">Configure RAG →</a>
        </div>
        """, unsafe_allow_html=True)

with col3:
    if len(st.session_state.docs) > 0:
        st.markdown("""
        <div class="status-card" style="border-left: 4px solid #2196F3;">
            <h3 style="color: #2196F3; margin-top: 0; font-size: 1.2em;">ℹ️ Documents Loaded</h3>
            <p><strong>Document Count:</strong> {}</p>
            <p>Your documents have been processed and are ready for use with RAG.</p>
        </div>
        """.format(len(st.session_state.docs)), unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-card" style="border-left: 4px solid #9E9E9E;">
            <h3 style="color: #9E9E9E; margin-top: 0; font-size: 1.2em;">📄 No Documents</h3>
            <p>Upload documents in the RAG Configuration page to enable document-based context.</p>
            <a href="./🔗_RAG_Config">Upload documents →</a>
        </div>
        """, unsafe_allow_html=True)
# Footer
st.markdown("""
<div style="margin-top: 50px; text-align: center; color: #666; font-size: 0.9em;">
<p>OllamaRAG runs entirely on your local machine. No data is sent to external servers.</p>
<p>For more information, visit the <a href="https://github.com/ollama/ollama" target="_blank">Ollama GitHub page</a>.</p>
</div>
""", unsafe_allow_html=True) 