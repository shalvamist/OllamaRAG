import streamlit as st
import ollama
from langchain_ollama import OllamaLLM

def update_ollama_model():
    """Updates the list of available Ollama models."""
    ollama_models = ollama.list()
    st.session_state.dropDown_model_list = []
    st.session_state.dropDown_embeddingModel_list = []
    
    for model in ollama_models['models']:
        if 'embedding' not in model['model'] and 'embed' not in model['model']:
            st.session_state.dropDown_model_list.append(model['model'])
        if 'embedding' in model['model'] or 'embed' in model['model']:
            st.session_state.dropDown_embeddingModel_list.append(model['model'])

def check_loaded_models():
    """Checks currently loaded Ollama models and updates the session state."""
    try:
        loaded_models = ollama.ps()
        if 'models' in loaded_models and loaded_models['models']:
            st.session_state.loaded_model_list = [model['model'] for model in loaded_models['models']]
            return True
        else:
            st.session_state.loaded_model_list = []
            return False
    except Exception as e:
        st.error(f"Error checking loaded models: {str(e)}")
        st.session_state.loaded_model_list = []
        return False

def update_main_ollama_model():
    """Updates the main Ollama model configuration."""
    if st.session_state.ollama_model is not None:
        try:
            st.session_state.chatReady = True
            st.session_state.llm = OllamaLLM(
                model=st.session_state.ollama_model,
                temperature=st.session_state.temperature,
                num_predict=int(st.session_state.newMaxTokens),
                num_ctx=int(st.session_state.contextWindow),
            )
            # Warmup with a simple prompt
            st.session_state.llm.invoke("Hello")
            st.success("Model settings applied successfully!")
        except Exception as e:
            st.error(f"Error initializing model: {str(e)}")
            st.session_state.chatReady = False
    else:
        st.session_state.chatReady = False
        st.error("Please select a model first")

# Set page config
st.set_page_config(
    page_title="Model Settings - OllamaRAG",
    page_icon="🦙",
    initial_sidebar_state="expanded"
)

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
        margin-bottom: 1rem !important;
        margin-top: 1rem !important;
        font-size: 2.2em !important;
        padding-bottom: 0.5rem !important;
        font-weight: 800 !important;
        border-bottom: 3px solid #3498db !important;
    }
    
    h2 {
        color: #2c3e50 !important;
        margin-bottom: 0.8rem !important;
        margin-top: 0.8rem !important;
        font-size: 1.8em !important;
        padding-bottom: 0.4rem !important;
        font-weight: 700 !important;
    }
    
    h3 {
        color: #2c3e50 !important;
        margin-bottom: 0.6rem !important;
        margin-top: 0.6rem !important;
        font-size: 1.4em !important;
        padding-bottom: 0.3rem !important;
        font-weight: 600 !important;
    }
    
    /* Reduce markdown spacing */
    .stMarkdown {
        margin-bottom: 0.3rem !important;
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
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stButton button:hover {
        background-color: #2980b9;
        box-shadow: 0 4px 8px rgba(52, 152, 219, 0.4);
        transform: translateY(-1px);
    }
    
    /* Messages */
    .stSuccess, .stError, .stInfo, .stWarning {
        padding: 0.5rem;
        border-radius: 6px;
        margin: 0.3rem 0;
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
    .stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus {
        border-color: #2980b9;
        box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.2);
        transform: translateY(-1px);
        background-color: #fff;
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
    .stSelectbox select:hover {
        border-color: #2980b9;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(52, 152, 219, 0.2);
    }
    
    /* Checkbox */
    .stCheckbox {
        margin: 0.2rem 0;
    }
    .stCheckbox label {
        color: #2c3e50 !important;
        font-weight: 600;
        font-size: 0.9em;
        padding: 0.2rem 0;
    }
    
    /* Slider */
    .stSlider {
        color: #2c3e50;
        padding: 0.8rem 0;
    }
    .stSlider [data-baseweb="slider"] {
        margin-top: 0.8rem;
    }
    .stSlider [data-baseweb="thumb"] {
        background-color: #3498db;
        border: 2px solid #fff;
        box-shadow: 0 2px 4px rgba(52, 152, 219, 0.3);
        width: 20px;
        height: 20px;
    }
    .stSlider [data-baseweb="track"] {
        background-color: #bdc3c7;
        height: 6px;
    }
    .stSlider [data-baseweb="track-fill"] {
        background-color: #3498db;
    }
    
    /* Divider */
    hr {
        margin: 0.8rem 0;
        border-width: 1px;
    }

    /* Section spacing */
    .element-container {
        margin-bottom: 0.5rem !important;
    }

    /* Column gaps */
    .row-widget {
        gap: 0.5rem !important;
    }

    /* Labels and help text */
    .stTextInput label, .stNumberInput label, .stSelectbox label, .stTextArea label {
        color: #2c3e50;
        font-weight: 700;
        font-size: 0.9rem;
        margin-bottom: 0.2rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stTextInput .help-text, .stNumberInput .help-text, .stSelectbox .help-text, .stTextArea .help-text {
        color: #666;
        font-size: 0.8rem;
        margin-top: 0.1rem;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

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

# Update available models
update_ollama_model()

# Sidebar - Model Status
with st.sidebar:
    st.header("🔄 Model Status")
    if st.button('Check Loaded Models', key='check_models'):
        has_loaded_models = check_loaded_models()
        if has_loaded_models:
            st.success(f"Currently loaded models: {', '.join(st.session_state.loaded_model_list)}")
        else:
            st.info("No models currently running")
    
    # Show current model status
    st.divider()
    if st.session_state.chatReady and st.session_state.ollama_model:
        st.success(f"Active Model: {st.session_state.ollama_model}")
    else:
        st.warning("No model currently active")

# Main content
st.title("🦙 Ollama Model Settings")

st.markdown("""
Configure your Ollama model settings here. These settings will affect how the model processes your queries and generates responses.
""")

# Model Selection
st.header("Model Selection")
selected_model = st.selectbox(
    "Select an Ollama model",
    st.session_state.dropDown_model_list,
    index=st.session_state.dropDown_model_list.index(st.session_state.ollama_model) if st.session_state.ollama_model in st.session_state.dropDown_model_list else None,
    placeholder="Select model...",
    help="Choose the main model for chat interactions"
)
if selected_model:
    st.session_state.ollama_model = selected_model

# Model Installation
st.header("Model Installation")
ollama_model_pull = st.text_input(
    "Install new Ollama model",
    placeholder="Enter model name (e.g., llama2)",
    help="Enter the name of a model to download from Ollama's model library"
)

if ollama_model_pull:
    with st.spinner(f"Downloading {ollama_model_pull}..."):
        try:
            ollama.pull(ollama_model_pull)
            update_ollama_model()
            st.success(f"Successfully downloaded {ollama_model_pull}")
        except Exception as e:
            st.error(f"Error downloading model: {str(e)}")

# Model Parameters
st.header("Model Parameters")

col1, col2 = st.columns(2)
with col1:
    context_window = st.number_input(
        "Context Window Size",
        min_value=512,
        max_value=8192,
        value=st.session_state.contextWindow,
        help="Maximum number of tokens the model can process at once"
    )
    st.session_state.contextWindow = context_window

with col2:
    max_tokens = st.number_input(
        "Maximum New Tokens",
        min_value=64,
        max_value=4096,
        value=st.session_state.newMaxTokens,
        help="Maximum number of tokens the model can generate in response"
    )
    st.session_state.newMaxTokens = max_tokens

temperature = st.slider(
    "Temperature",
    min_value=0.0,
    max_value=2.0,
    value=st.session_state.temperature,
    step=0.01,
    help="Controls randomness in the output (0 = deterministic, 2 = very random)"
)
st.session_state.temperature = temperature

system_prompt = st.text_area(
    "System Prompt",
    value=st.session_state.system_prompt,
    help="Sets the behavior and role of the assistant"
)
st.session_state.system_prompt = system_prompt

# Apply Settings Button
if st.button("Apply Settings", key='apply_settings'):
    update_main_ollama_model() 