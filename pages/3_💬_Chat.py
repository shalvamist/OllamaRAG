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

# Set page config
st.set_page_config(
    page_title="Chat - OllamaRAG",
    page_icon="💬",
)

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
    System prompt: *{st.session_state.system_prompt}*
    """)
    
    # Status indicators and controls
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.session_state.db_ready:
            st.success("RAG Database: Ready")
        else:
            st.warning("RAG Database: Not configured")
    with col2:
        st.button("Clear Chat History", on_click=clear_chat_history)
    with col3:
        if st.button("New Chat"):
            clear_chat_history()

    st.divider()

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        # Create new conversation if not exists
        if 'conversation_id' not in st.session_state:
            st.session_state.conversation_id = create_conversation(
                st.session_state.ollama_model,
                st.session_state.system_prompt
            )
            # Save the initial assistant message
            add_message(
                st.session_state.conversation_id,
                st.session_state.messages[0]["role"],
                st.session_state.messages[0]["content"]
            )

        # Add user message to UI and database
        st.session_state.messages.append({"role": "user", "content": prompt})
        add_message(st.session_state.conversation_id, "user", prompt)
        
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
                    st.write_stream(stream_data(response))
                    
                    # Add assistant response to UI and database
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    add_message(st.session_state.conversation_id, "assistant", response)

    # Display chat settings and history
    with st.sidebar:
        # Chat Settings Section
        st.header("Chat Settings")
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

        # Chat History Section
        st.header("Chat History")
        recent_conversations = get_recent_conversations()
        
        if recent_conversations:
            for conv in recent_conversations:
                col1, col2 = st.columns([4, 1])
                with col1:
                    # Format the date
                    date = datetime.strptime(conv['created_at'], '%Y-%m-%d %H:%M:%S')
                    formatted_date = date.strftime('%Y-%m-%d %H:%M')
                    
                    # Create a button for each conversation
                    if st.button(
                        f"{formatted_date} - {conv['model']}\n{conv['first_message'][:50]}...",
                        key=f"conv_{conv['id']}"
                    ):
                        load_conversation(conv['id'])
                with col2:
                    # Add delete button for each conversation
                    if st.button("🗑️", key=f"del_{conv['id']}"):
                        delete_conversation(conv['id'])
                        if 'conversation_id' in st.session_state and st.session_state.conversation_id == conv['id']:
                            clear_chat_history()
                        st.rerun()
        else:
            st.info("No conversation history available") 