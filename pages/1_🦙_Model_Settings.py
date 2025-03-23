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
    initial_sidebar_state="expanded",
    layout="wide"
)

# Custom CSS for consistent styling with RAG Config page
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
    
    /* Card headings */
    .card-heading {
        color: #555;
        font-weight: 600;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    
    /* Success and warning messages */
    .stSuccess, .stWarning, .stError, .stInfo {
        border-radius: 4px;
    }
    
    /* Input fields */
    .stTextInput input, .stNumberInput input, .stTextArea textarea, .stSelectbox select {
        border-radius: 4px;
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
    # Smaller title with more subtle styling
    st.markdown("""
    <h2 style="font-size: 1.5em; color: #0D47A1; margin-bottom: 15px; padding-bottom: 5px; border-bottom: 1px solid #e0e0e0;">
    ⚙️ Model Settings
    </h2>
    """, unsafe_allow_html=True)
    
    # Emergency reset button for Ollama
    if st.button("🔄 Reset Ollama", key="reset_ollama", help="Emergency reset of Ollama connection"):
        try:
            update_ollama_model()
            st.success("Ollama connection refreshed successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"Error resetting Ollama connection: {str(e)}")
    
    # Status and loaded models in a collapsable section
    with st.expander("🤖 Model Status", expanded=False):
        # Show current status
        if st.session_state.chatReady and st.session_state.ollama_model:
            st.success(f"✅ {st.session_state.ollama_model} (Active)")
        else:
            st.warning("⚠️ No model active")
        
        # Check loaded models button
        if st.button('🔄 Check Loaded Models', key='check_models', use_container_width=True):
            with st.spinner("Checking running models..."):
                has_loaded_models = check_loaded_models()
                if has_loaded_models:
                    st.success(f"Running: {', '.join(st.session_state.loaded_model_list)}")
                else:
                    st.info("No models currently running")
    
    # Add Quick Actions section
    with st.expander("⚡ Quick Actions", expanded=False):
        if st.session_state.ollama_model:
            st.markdown(f"**Selected Model:** {st.session_state.ollama_model}")
            
            # Apply settings button in sidebar for quick access
            if st.button(
                "✅ Apply Current Settings",
                key='apply_settings_sidebar',
                type="primary",
                use_container_width=True
            ):
                with st.spinner(f"Applying settings to {st.session_state.ollama_model}..."):
                    update_main_ollama_model()
        else:
            st.info("Select a model in the main panel")
            st.button(
                "Apply Settings",
                key='apply_settings_sidebar',
                disabled=True,
                use_container_width=True
            )

# Main content
st.markdown("""
<h1 style="text-align: center; color: #0D47A1; margin-bottom: 20px;">🦙 Ollama Model Settings</h1>
""", unsafe_allow_html=True)

# Main content - using tabs for a more compact layout
tab1, tab2, tab3 = st.tabs(["📋 Model Selection", "⚙️ Parameters", "⬇️ Installation"])

# Tab 1: Model Selection
with tab1:
    st.subheader("Select Model")
    
    if not st.session_state.dropDown_model_list:
        st.warning("⚠️ No models available. Please install models first.")
    else:
        selected_model = st.selectbox(
            "Select an Ollama model",
            st.session_state.dropDown_model_list,
            index=st.session_state.dropDown_model_list.index(st.session_state.ollama_model) if st.session_state.ollama_model in st.session_state.dropDown_model_list else None,
            placeholder="Select model...",
            help="Choose the main model for chat interactions"
        )
        if selected_model:
            st.session_state.ollama_model = selected_model
    
    # Apply Settings Button - moved here for better workflow
    st.divider()
    
    # Check if a model is selected
    if not st.session_state.ollama_model:
        st.warning("⚠️ Please select a model first")
        apply_button_enabled = False
    else:
        apply_button_enabled = True
        st.info(f"Current model: {st.session_state.ollama_model}")
    
    if st.button(
        "✅ Apply Settings",
        key='apply_settings',
        disabled=not apply_button_enabled,
        type="primary",
        use_container_width=True
    ):
        with st.spinner(f"Applying settings to {st.session_state.ollama_model}..."):
            update_main_ollama_model()

# Tab 2: Parameters
with tab2:
    st.subheader("Model Parameters")
    
    # Two columns for window and max tokens
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
    
    # Temperature slider
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=st.session_state.temperature,
        step=0.01,
        help="Controls randomness in the output (0 = deterministic, 2 = very random)"
    )
    st.session_state.temperature = temperature
    
    # System prompt
    system_prompt = st.text_area(
        "System Prompt",
        value=st.session_state.system_prompt,
        help="Sets the behavior and role of the assistant",
        height=100
    )
    st.session_state.system_prompt = system_prompt

# Tab 3: Installation
with tab3:
    st.subheader("Install New Model")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        ollama_model_pull = st.text_input(
            "Model Name",
            placeholder="Enter model name (e.g., llama2)",
            help="Enter the name of a model to download from Ollama's model library"
        )
    
    with col2:
        pull_button = st.button(
            "📥 Install",
            key="pull_model",
            disabled=not ollama_model_pull,
            type="primary",
            use_container_width=True
        )
    
    if pull_button and ollama_model_pull:
        with st.spinner(f"Downloading {ollama_model_pull}..."):
            try:
                ollama.pull(ollama_model_pull)
                update_ollama_model()
                st.success(f"Successfully downloaded {ollama_model_pull}")
                st.rerun()
            except Exception as e:
                st.error(f"Error downloading model: {str(e)}")

    # Show available models
    if st.session_state.dropDown_model_list:
        with st.expander("Available Models"):
            st.write(", ".join(st.session_state.dropDown_model_list)) 