import streamlit as st
import time
from datetime import datetime
from pipelines.defaultRAG import generate_response
from database.chat_db import (
    create_conversation,
    add_message,
    get_conversation_history,
    get_recent_conversations,
    delete_conversation
)
import ollama

# Initialize session state variables
if 'chatReady' not in st.session_state:
    st.session_state.chatReady = False

if 'ollama_model' not in st.session_state:
    st.session_state.ollama_model = None

if 'llm' not in st.session_state:
    st.session_state.llm = None

if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

if 'db_ready' not in st.session_state:
    st.session_state.db_ready = False

if 'dropDown_model_list' not in st.session_state:
    st.session_state.dropDown_model_list = []

if 'BM25retriver' not in st.session_state:
    st.session_state.BM25retriver = None

if 'dbRetrievalAmount' not in st.session_state:
    st.session_state.dbRetrievalAmount = 3

if 'ContextualBM25RAG' not in st.session_state:
    st.session_state.ContextualBM25RAG = False

if 'temperature' not in st.session_state:
    st.session_state.temperature = 0.7

if 'system_prompt' not in st.session_state:
    st.session_state.system_prompt = "You are a helpful assistant."

if 'collection' not in st.session_state:
    st.session_state.collection = None

if 'ContextualRAG' not in st.session_state:
    st.session_state.ContextualRAG = False

# Initialize session state for save dialog
if 'show_save_dialog' not in st.session_state:
    st.session_state.show_save_dialog = False

# Update available models
def update_ollama_model():
    """Updates the list of available Ollama models."""
    try:
        ollama_models = ollama.list()
        st.session_state.dropDown_model_list = []
        
        for model in ollama_models['models']:
            if 'embedding' not in model['model'] and 'embed' not in model['model']:
                st.session_state.dropDown_model_list.append(model['model'])
    except Exception as e:
        st.error(f"Error updating model list: {str(e)}")
        st.session_state.dropDown_model_list = []

# Update available models on page load
update_ollama_model()

def stream_data(data):
    """
    Streams the input data word by word with a delay.
    
    Args:
        data (str): The input data to be streamed.
    """
    for word in data.split(" "):
        yield word + " "
        time.sleep(0.02)

def clear_chat_history():
    """Clears the chat history."""
    if 'conversation_id' in st.session_state:
        delete_conversation(st.session_state.conversation_id)
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    st.session_state.pop('conversation_id', None)
    st.rerun()

def load_conversation(conversation_id):
    """Loads a conversation from the database."""
    st.session_state.messages = get_conversation_history(conversation_id)
    st.session_state.conversation_id = conversation_id
    st.rerun()

def toggle_save_dialog():
    st.session_state.show_save_dialog = True

def save_current_chat(chat_name):
    """Saves the current chat with a user-provided name."""
    if chat_name:
        # Create new conversation in database
        conversation_id = create_conversation(
            st.session_state.ollama_model,
            st.session_state.system_prompt,
            chat_name
        )
        
        # Save all messages
        for message in st.session_state.messages:
            add_message(conversation_id, message["role"], message["content"])
        
        st.session_state.show_save_dialog = False
        st.rerun()

# Set page config
st.set_page_config(
    page_title="Chat - OllamaRAG",
    page_icon="💬",
)

# Custom CSS for styling
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
        padding: 1rem;
        background-color: #fff;
        border-radius: 6px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.15);
        margin: 0.5rem;
    }

    /* Chat container */
    [data-testid="stChatMessageContainer"] {
        overflow-y: auto !important;
        padding: 1rem 0;
        margin-bottom: 0.5rem;
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
        max-height: 600px;
    }

    /* Chat input container */
    .stChatInputContainer {
        position: sticky;
        bottom: 0;
        background: white;
        padding: 1rem 0;
        border-top: 1px solid #e2e8f0;
        z-index: 100;
    }
    
    /* Chat input */
    .stChatInput {
        margin: 0;
        padding: 0.4rem;
        border-radius: 6px;
    }
    
    /* Chat messages */
    [data-testid="stChatMessage"] {
        background-color: #f8fafc;
        padding: 0.8rem;
        margin: 0.3rem 0;
        border-radius: 6px;
        border: 1px solid #e2e8f0;
        max-width: 90%;
        animation: fadeIn 0.3s ease-in-out;
        opacity: 0;
        animation-fill-mode: forwards;
    }
    
    /* User message */
    [data-testid="stChatMessage"][data-testid="user"] {
        background-color: #ebf8ff;
        border-color: #3498db;
        margin-left: auto;
    }
    
    /* Assistant message */
    [data-testid="stChatMessage"][data-testid="assistant"] {
        background-color: #f0fff4;
        border-color: #2ecc71;
        margin-right: auto;
    }

    /* Message animation */
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Headers in chat */
    .main-header {
        position: sticky;
        top: 0;
        background: white;
        z-index: 100;
        padding: 1rem 0;
        border-bottom: 1px solid #e2e8f0;
    }

    /* Status bar */
    .status-bar {
        position: sticky;
        top: 80px;
        background: white;
        z-index: 99;
        padding: 0.5rem 0;
        border-bottom: 1px solid #e2e8f0;
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

    /* Help text */
    .stTextInput .help-text, .stNumberInput .help-text, .stSelectbox .help-text {
        font-size: 0.8em;
        margin-top: 0.1rem;
        color: #666;
    }
</style>

<script>
    function autoScroll() {
        const chatContainer = document.querySelector('[data-testid="stChatMessageContainer"]');
        if (chatContainer) {
            // Add a small delay to ensure content is rendered
            setTimeout(() => {
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }, 50);
        }
    }

    // Run on initial load
    window.addEventListener('load', () => {
        setTimeout(autoScroll, 100);
    });

    // Create and run MutationObserver
    const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            if (mutation.addedNodes.length || mutation.type === 'characterData') {
                autoScroll();
            }
        });
    });

    // Start observing with a delay to ensure container is ready
    setTimeout(() => {
        const chatContainer = document.querySelector('[data-testid="stChatMessageContainer"]');
        if (chatContainer) {
            observer.observe(chatContainer, {
                childList: true,
                subtree: true,
                characterData: true
            });
        }
    }, 100);
</script>
""", unsafe_allow_html=True)

# Initialize session state for messages if not exists
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

st.title("💬 Chat Interface")

# Check if model is configured
if not st.session_state.chatReady or st.session_state.ollama_model is None:
    st.error("Please configure your Ollama model in the Model Settings page before starting a chat.")
    st.markdown("""
    To get started:
    1. Go to the **🦙 Model Settings** page
    2. Select a model and configure its parameters
    3. Click "Apply Settings"
    4. Return here to start chatting
    """)
else:
    # Chat interface
    st.markdown(f"""
    Currently using model: **{st.session_state.ollama_model}**  
    """)
    
    # Status indicators
    if st.session_state.db_ready:
        st.success("RAG Database: Ready")
    else:
        st.warning("RAG Database: Not configured")

    st.divider()

    for message in st.session_state.messages:
        st.empty()
        with st.chat_message(message["role"]):
            # Adding this the chatbot messaging to workaround ghosting bug
            st.empty()
            # Display the message
            st.write(message["content"])

    # Save Chat Dialog
    if st.session_state.show_save_dialog:
        with st.form(key="save_chat_form"):
            st.subheader("Save Chat")
            chat_name = st.text_input("Enter a name for this chat:")
            col1, col2 = st.columns([1, 4])
            with col1:
                submitted = st.form_submit_button("Save")
            if submitted and chat_name:
                save_current_chat(chat_name)
                st.success(f"Chat saved as: {chat_name}")

    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        # Add user message to UI
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)

        # Generate response
        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = generate_response(
                        st.session_state.messages[-1]["content"],
                        st.session_state.collection,
                        st.session_state.db_ready,
                        st.session_state.system_prompt,
                        st.session_state.llm,
                        st.session_state.BM25retriver,
                        int(st.session_state.dbRetrievalAmount)
                    )
                    # Display the message
                    st.write_stream(stream_data(response))
                    
                    # Add assistant response to UI
                    st.session_state.messages.append({"role": "assistant", "content": response})

    # Display chat settings and history
    with st.sidebar:
        # Chat Controls Section
        st.header("💬 Chat Controls")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.button("Clear History", on_click=clear_chat_history)
        with col2:
            if st.button("New Chat"):
                clear_chat_history()
        with col3:
            if st.button("Save Chat", on_click=toggle_save_dialog, disabled=len(st.session_state.messages) <= 1):
                pass
        
        st.divider()
        
        # Chat Settings Section
        st.header("⚙️ Chat Settings")
        st.markdown("""
        **Model Parameters:**
        - Temperature: {:.2f}
        - Max Tokens: {}
        - Context Window: {}
        
        **RAG Settings:**
        - Documents Retrieved: {}
        - Contextual RAG: {}
        - BM25 Retrieval: {}
        """.format(
            st.session_state.temperature,
            st.session_state.newMaxTokens,
            st.session_state.contextWindow,
            st.session_state.dbRetrievalAmount,
            "Enabled" if st.session_state.ContextualRAG else "Disabled",
            "Enabled" if st.session_state.ContextualBM25RAG else "Disabled"
        ))

        # Saved Chats Section
        st.header("💾 Saved Chats")
        recent_conversations = get_recent_conversations()
        
        if recent_conversations:
            for conv in recent_conversations:
                col1, col2 = st.columns([4, 1])
                with col1:
                    # Display just the chat name without timestamp
                    chat_name = conv['first_message'].split(" (")[0]  # Get name without timestamp
                    if st.button(f"{chat_name}", key=f"conv_{conv['id']}"):
                        load_conversation(conv['id'])
                with col2:
                    if st.button("🗑️", key=f"del_{conv['id']}"):
                        delete_conversation(conv['id'])
                        if 'conversation_id' in st.session_state and st.session_state.conversation_id == conv['id']:
                            clear_chat_history()
                        st.rerun()
        else:
            st.info("No saved chats available") 