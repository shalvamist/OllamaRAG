import streamlit as st
import ollama
from langchain_ollama import OllamaLLM

def updateOllamaModel():
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

def updateMainOllamaModel():
    """Updates the main Ollama model configuration."""
    if st.session_state.ollama_model is not None:
        try:
            st.session_state.chatReady = True
            st.session_state.llm = OllamaLLM(
                model=st.session_state.ollama_model,
                temperature=st.session_state.temperature,
                num_predict=st.session_state.newMaxTokens,
                num_ctx=st.session_state.contextWindow,
            )
            # Warmup with a simple prompt
            st.session_state.llm.invoke("Hello")
        except Exception as e:
            st.error(f"Error initializing model: {str(e)}")
            st.session_state.chatReady = False
    else:
        st.session_state.chatReady = False

# Set page config
st.set_page_config(
    page_title="Model Settings - OllamaRAG",
    page_icon="🦙",
)

# Update available models
updateOllamaModel()

st.title("🦙 Ollama Model Settings")

st.markdown("""
Configure your Ollama model settings here. These settings will affect how the model processes your queries and generates responses.
""")

# Model Selection
st.header("Model Selection")
st.session_state.ollama_model = st.selectbox(
    "Select an Ollama model",
    st.session_state.dropDown_model_list,
    index=None,
    placeholder="Select model...",
    help="Choose the main model for chat interactions"
)

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
            updateOllamaModel()
            st.success(f"Successfully downloaded {ollama_model_pull}")
        except Exception as e:
            st.error(f"Error downloading model: {str(e)}")

# Model Parameters
st.header("Model Parameters")

col1, col2 = st.columns(2)
with col1:
    st.session_state.contextWindow = st.number_input(
        "Context Window Size",
        min_value=512,
        max_value=8192,
        value=int(st.session_state.contextWindow),
        help="Maximum number of tokens the model can process at once"
    )

with col2:
    st.session_state.newMaxTokens = st.number_input(
        "Maximum New Tokens",
        min_value=64,
        max_value=4096,
        value=int(st.session_state.newMaxTokens),
        help="Maximum number of tokens the model can generate in response"
    )

st.session_state.temperature = st.slider(
    "Temperature",
    min_value=0.0,
    max_value=2.0,
    value=float(st.session_state.temperature) if 'temperature' in st.session_state else 1.0,
    step=0.01,
    help="Controls randomness in the output (0 = deterministic, 2 = very random)"
)

st.session_state.system_prompt = st.text_area(
    "System Prompt",
    value=st.session_state.system_prompt if 'system_prompt' in st.session_state else "You are a helpful assistant.",
    help="Sets the behavior and role of the assistant"
)

# Model Status
st.header("Model Status")
col1, col2 = st.columns(2)
with col1:
    if st.button('Check Loaded Models', key='check_models'):
        has_loaded_models = check_loaded_models()
        if has_loaded_models:
            st.success(f"Currently loaded models: {', '.join(st.session_state.loaded_model_list)}")
        else:
            st.info("No models currently running")

with col2:
    if st.button("Apply Settings", key='apply_settings'):
        updateMainOllamaModel()
        if st.session_state.chatReady:
            st.success("Model settings applied successfully!")
        else:
            st.error("Please select a model first") 