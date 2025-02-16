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