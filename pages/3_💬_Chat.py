import streamlit as st
import time
import re
import os
from langchain_community.tools import DuckDuckGoSearchResults
from pipelines.defaultRAG import generate_response
from CommonUtils.rag_utils import ( 
    SOURCE_PATH,
    get_collection,
    get_client
)
from CommonUtils.chat_db import ( 
    create_conversation,
    add_message,
    get_conversation_history,
    get_recent_conversations,
    delete_conversation
)

import ollama
from CommonUtils.research_utils import (
    format_thinking_content,
    parse_thinking_content,
    THINKING_SECTION_TEMPLATE,
    THINKING_CSS,
    perform_web_search,
    synthesize_search_results,
    WEBSEARCH_QUERY_TEMPLATE,
)

# Initialize session state variables
if 'chatReady' not in st.session_state:
    st.session_state.chatReady = False

if 'web_search_enabled' not in st.session_state:
    st.session_state.web_search_enabled = False

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

# Initialize session state for available databases
if 'available_databases' not in st.session_state:
    st.session_state.available_databases = []
    
# Initialize database collections dict
if 'database_collections' not in st.session_state:
    st.session_state.database_collections = {}

# Update available databases
def update_available_databases():
    """Updates the list of available databases from the source_documents directory."""
    try:
        if os.path.exists(SOURCE_PATH):
            st.session_state.available_databases = [d for d in os.listdir(SOURCE_PATH) 
                                                 if os.path.isdir(os.path.join(SOURCE_PATH, d))]
    except Exception as e:
        st.error(f"Error updating database list: {str(e)}")
        st.session_state.available_databases = []

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

# Update available models and databases on page load
update_ollama_model()
update_available_databases()

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

# Replace the duplicated functions with aliases for backward compatibility if needed
def parse_message_content(content):
    """Alias for parse_thinking_content to maintain backward compatibility."""
    return parse_thinking_content(content)

# Parse user prompt for @database mentions
def parse_database_mention(prompt):
    """
    Parses the prompt for @database mentions and returns the database name and cleaned prompt.
    
    Returns:
        tuple: (database_name, cleaned_prompt)
    """
    # Regular expression to match @database_name
    match = re.search(r'@(\w+[-]?\w*)', prompt)
    
    if match:
        db_name = match.group(1)
        # Remove the @database_name from the prompt
        cleaned_prompt = prompt.replace(match.group(0), '').strip()
        return db_name, cleaned_prompt
    
    return None, prompt

# Get or initialize a database collection
def get_db_collection(db_name, embedding_model):
    """
    Gets or initializes a database collection for the specified database name.
    
    Returns:
        The ChromaDB collection for the database
    """
    if db_name in st.session_state.database_collections:
        return st.session_state.database_collections[db_name]
    
    # Check if embedding model is set
    if not st.session_state.embeddingModel:
        return None
        
    # Get a ChromaDB client
    client = get_client()
    
    # Initialize the collection
    collection_name = f"rag_collection_{db_name}"
    collection = get_collection(
        st.session_state.embeddingModel,
        client,
        collection_name,
        f"RAG collection for {db_name}"
    )
    
    # Store in session state
    if collection:
        st.session_state.database_collections[db_name] = collection
    
    return collection

# Set page config
st.set_page_config(
    page_title="Chat - OllamaRAG",
    page_icon="💬",
    initial_sidebar_state="expanded",
    layout="wide"
)

# Custom CSS for styling - replace thinking-related CSS with the centralized version
st.markdown(f"""
<style>
    /* Main background and text colors */
    .stApp {{
        color: #1a2234;
    }}
    
    /* Headers */
    h1 {{
        color: #0D47A1 !important;
        margin-bottom: 1rem !important;
        font-size: 2.2em !important;
        font-weight: 800 !important;
    }}
    
    h2 {{
        color: #1E88E5 !important;
        margin-bottom: 0.8rem !important;
        font-size: 1.8em !important;
        font-weight: 700 !important;
    }}
    
    h3 {{
        color: #1E88E5 !important;
        margin-bottom: 0.6rem !important;
        font-size: 1.4em !important;
        font-weight: 600 !important;
    }}
    
    /* Card styling */
    [data-testid="stExpander"] {{
        border: none !important;
        box-shadow: none !important;
    }}
    
    /* Buttons */
    .stButton button {{
        border-radius: 4px;
    }}
    
    /* Container borders */
    [data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column"] > div[data-testid="stVerticalBlock"] {{
        border-radius: 10px;
        padding: 1rem;
    }}
    
    /* Card headings */
    .card-heading {{
        color: #555;
        font-weight: 600;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }}
    
    /* Success and warning messages */
    .stSuccess, .stWarning, .stError, .stInfo {{
        border-radius: 4px;
    }}
    
    /* Input fields */
    .stTextInput input, .stNumberInput input, .stTextArea textarea, .stSelectbox select {{
        border-radius: 4px;
    }}

    /* Chat container */
    [data-testid="stChatMessageContainer"] {{
        overflow-y: auto !important;
        padding: 1rem 0;
        margin-bottom: 0.5rem;
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
        max-height: 600px;
        background-color: #f9f9f9;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
    }}

    /* Chat input container */
    .stChatInputContainer {{
        position: sticky;
        bottom: 0;
        background: white;
        padding: 1rem 0;
        border-top: 1px solid #e2e8f0;
        z-index: 100;
    }}
    
    /* Chat input */
    .stChatInput {{
        margin: 0;
        padding: 0.4rem;
        border-radius: 6px;
    }}
    
    /* Chat messages */
    [data-testid="stChatMessage"] {{
        background-color: #f8fafc;
        padding: 0.8rem;
        margin: 0.3rem 0;
        border-radius: 6px;
        border: 1px solid #e2e8f0;
        max-width: 90%;
        animation: fadeIn 0.3s ease-in-out;
        opacity: 0;
        animation-fill-mode: forwards;
    }}
    
    /* User message */
    [data-testid="stChatMessage"][data-testid="user"] {{
        background-color: #e3f2fd;
        border-color: #1E88E5;
        margin-left: auto;
    }}
    
    /* Assistant message */
    [data-testid="stChatMessage"][data-testid="assistant"] {{
        background-color: #f0f7ff;
        border-color: #0D47A1;
        margin-right: auto;
    }}

    /* Message animation */
    @keyframes fadeIn {{
        from {{
            opacity: 0;
            transform: translateY(10px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}

    /* Import thinking-related CSS from research_utils */
    {THINKING_CSS}
</style>
""", unsafe_allow_html=True)

# Add JavaScript for auto-scrolling
st.markdown("""
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

# Title with centered styling to match other pages
st.markdown("""
<h1 style="text-align: center; color: #0D47A1; margin-bottom: 20px;">💬 Chat with Ollama</h1>
""", unsafe_allow_html=True)

# Check if model is configured
if not st.session_state.chatReady or st.session_state.ollama_model is None:
    # Display model configuration message in a card-like container
    st.markdown("""
    <div style="background-color: #ffebee; padding: 20px; border-radius: 10px; margin-bottom: 30px; border-left: 5px solid #f44336;">
    <h2 style="color: #f44336; margin-top: 0;">Model Not Configured</h2>
    
    <p>Please configure your Ollama model in the Model Settings page before starting a chat.</p>
    
    <p><strong>To get started:</strong></p>
    <ol>
      <li>Go to the <strong>🦙 Model Settings</strong> page</li>
      <li>Select a model and configure its parameters</li>
      <li>Click "Apply Settings"</li>
      <li>Return here to start chatting</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
else:
    # Display messages
    for message in st.session_state.messages:
        st.empty()
        with st.chat_message(message["role"]):
            # Adding this the chatbot messaging to workaround ghosting bug
            st.empty()
            # Display the message with parsed thinking sections
            st.markdown(parse_message_content(message["content"]), unsafe_allow_html=True)

    # Save Chat Dialog
    if st.session_state.show_save_dialog:
        st.markdown("""
        <div style="background-color: #f5f5f5; padding: 15px; border-radius: 10px; margin: 15px 0;">
        """, unsafe_allow_html=True)
        
        with st.form(key="save_chat_form"):
            st.subheader("Save Conversation")
            chat_name = st.text_input("Enter a name for this chat:", 
                                     placeholder="Enter descriptive name...",
                                     help="Choose a unique name that helps you remember this conversation")
            col1, col2 = st.columns([1, 1])
            with col1:
                submitted = st.form_submit_button("Save", use_container_width=True)
            with col2:
                cancel = st.form_submit_button("Cancel", use_container_width=True)
            
            if submitted and chat_name:
                save_current_chat(chat_name)
                st.success(f"Chat saved as: {chat_name}")
            elif cancel:
                st.session_state.show_save_dialog = False
                st.rerun()
        
        st.markdown("""
        </div>
        """, unsafe_allow_html=True)

    # Chat input
    if prompt := st.chat_input("Type @database to use a specific RAG database..."):
        # Add user message to UI
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)

        # Parse for @database mentions
        db_name, cleaned_prompt = parse_database_mention(prompt)
        
        # Use specific database if mentioned and available
        collection_to_use = None
        use_rag = st.session_state.db_ready
        db_indicator = ""
        
        if db_name:
            if db_name in st.session_state.available_databases:
                # Get collection for the database
                collection_to_use = get_db_collection(db_name, st.session_state.embeddingModel)
                
                # If collection could not be initialized, show warning
                if not collection_to_use:
                    with st.chat_message("assistant"):
                        st.warning(f"Database '{db_name}' could not be initialized. Using default RAG instead.")
                    collection_to_use = st.session_state.collection
                else:
                    use_rag = True
                    db_indicator = f"Using database: {db_name}"
            else:
                # Database not found
                with st.chat_message("assistant"):
                    available_dbs = ", ".join([f"`@{db}`" for db in st.session_state.available_databases]) if st.session_state.available_databases else "No databases available"
                    st.warning(f"Database '{db_name}' not found. Available databases: {available_dbs}")
                    # Continue with default RAG
                collection_to_use = st.session_state.collection
        else:
            # Use default collection
            collection_to_use = st.session_state.collection

        # Generate response
        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                thinking_text = ""
                is_thinking = False
                final_response = ""
                has_thinking = False
                
                # Add web search if enabled
                web_results = []
                if st.session_state.web_search_enabled:
                    async def process_web_search():
                        with st.spinner("Searching the web..."):
                            try:
                                # Generate search query
                                search_query = st.session_state.llm.invoke(WEBSEARCH_QUERY_TEMPLATE.format(topic=cleaned_prompt))
                                
                                # Perform web search
                                DDG_web_tool = DuckDuckGoSearchResults()
                                DDG_news_tool = DuckDuckGoSearchResults(backend="news")

                                web_results = []
                                web_results = await perform_web_search(DDG_web_tool, search_query, None)
                                web_results.extend(await perform_web_search(DDG_news_tool, search_query, None))

                                if web_results:
                                    synthesis = []
                                    # Synthesize and validate results
                                    synthesis = await synthesize_search_results(
                                        web_results,
                                        st.session_state.llm_websearch,
                                        cleaned_prompt,
                                        st.empty()
                                    )
                                    
                                    if synthesis:
                                        if synthesis["needs_more_sources"]:
                                            # Get additional sources if confidence is low
                                            st.warning("Web search results are not sufficient. Please provide additional sources.")
                                        
                                        # Add synthesis information to prompt
                                        web_context = f"""Web search results with confidence {synthesis['confidence']}/10:
                                        
                                        Corroborated facts:
                                        {chr(10).join('- ' + fact for fact in synthesis['corroborated_facts'])}
                                        
                                        From credible sources:
                                        {chr(10).join('- ' + source for source in synthesis['credible_sources'])}
                                        
                                        Raw search results:
                                        {web_results}"""
                                        
                                        return f"Web search results:\n{web_context}\n\nUser question: {cleaned_prompt}"
                                return None
                            except Exception as e:
                                st.warning(f"Web search failed: {str(e)}")
                                return None
                    
                    import asyncio
                    web_search_result = asyncio.run(process_web_search())
                    if web_search_result:
                        cleaned_prompt = web_search_result
                
                # Stream the response
                for chunk in generate_response(
                    cleaned_prompt,  # Use cleaned prompt without @mention
                    collection_to_use,  # Use specific collection if mentioned
                    use_rag,  # Enable RAG if collection available
                    st.session_state.system_prompt + ("\n\nWhen web search results are provided, use them to enhance your response and cite the sources when appropriate." if st.session_state.web_search_enabled else ""),
                    st.session_state.llm,
                    None,
                    int(st.session_state.dbRetrievalAmount)
                ):
                    full_response += chunk
                    
                    # Handle thinking process for DeepSeek models
                    if '<think>' in full_response and not is_thinking:
                        is_thinking = True
                        has_thinking = True
                        thinking_text = ""
                        # Get text before thinking
                        display_text = full_response.split('<think>')[0].strip()
                        if display_text:
                            if db_indicator:
                                message_placeholder.markdown(f"*{db_indicator}*\n\n{display_text} 💭")
                            else:
                                message_placeholder.markdown(display_text + " 💭")
                        continue
                    
                    if is_thinking and '</think>' not in full_response:
                        thinking_text += chunk
                        # Show thinking indicator and current thinking content
                        display_text = full_response.split('<think>')[0].strip()
                        thinking_section = THINKING_SECTION_TEMPLATE.format(
                            content=format_thinking_content(thinking_text)
                        )
                        
                        if display_text:
                            if db_indicator:
                                message_placeholder.markdown(f"*{db_indicator}*\n\n{display_text}\n\n{thinking_section}", unsafe_allow_html=True)
                            else:
                                message_placeholder.markdown(f"{display_text}\n\n{thinking_section}", unsafe_allow_html=True)
                        else:
                            if db_indicator:
                                message_placeholder.markdown(f"*{db_indicator}*\n\n{thinking_section}", unsafe_allow_html=True)
                            else:
                                message_placeholder.markdown(thinking_section, unsafe_allow_html=True)
                        continue
                        
                    if '</think>' in full_response and is_thinking:
                        is_thinking = False
                        # Use the parse_thinking_content function to handle the thinking section
                        final_response = parse_thinking_content(full_response)
                        
                        if db_indicator:
                            message_placeholder.markdown(f"*{db_indicator}*\n\n{final_response}", unsafe_allow_html=True)
                        else:
                            message_placeholder.markdown(final_response, unsafe_allow_html=True)
                        continue
                    
                    # Normal response handling for non-thinking parts
                    if not is_thinking:
                        if '<think>' not in full_response:
                            final_response = full_response
                            if db_indicator:
                                message_placeholder.markdown(f"*{db_indicator}*\n\n{final_response}")
                            else:
                                message_placeholder.markdown(final_response)
                
                # Update final response without cursor
                if not final_response:
                    final_response = full_response
                
                if '<think>' in final_response and '</think>' in final_response:
                    # Use the parse_thinking_content function to properly format any thinking sections
                    final_response = parse_thinking_content(final_response)
                
                # Add database indicator to stored message
                if db_indicator:
                    final_message = f"*{db_indicator}*\n\n{final_response}"
                    message_placeholder.markdown(final_message, unsafe_allow_html=True)
                    st.session_state.messages.append({"role": "assistant", "content": final_response})
                else:
                    message_placeholder.markdown(final_response, unsafe_allow_html=True)
                    st.session_state.messages.append({"role": "assistant", "content": final_response})

    # Sidebar content
    with st.sidebar:
        # Smaller title with more subtle styling - exactly matching Model Settings page
        st.markdown("""
        <h2 style="font-size: 1.5em; color: #0D47A1; margin-bottom: 15px; padding-bottom: 5px; border-bottom: 1px solid #e0e0e0;">
        ⚙️ Chat Settings
        </h2>
        """, unsafe_allow_html=True)
        
        # Web Search Toggle - Add this section first
        with st.expander("🔍 Web Search", expanded=False):
            st.session_state.web_search_enabled = st.toggle("Enable Web Search", 
                value=st.session_state.web_search_enabled,
                help="When enabled, the chat will search the web for relevant information")
            if st.session_state.web_search_enabled:
                st.success("Web search is enabled")
                st.info("The assistant will search the web for relevant information when answering questions")
            else:
                st.info("Web search is disabled")
        
        # Model Status Section - Collapsable
        with st.expander("🤖 Model Status", expanded=False):
            if hasattr(st.session_state, 'ollama_model') and st.session_state.ollama_model:
                st.success("Model Connected")
                st.info(f"**Model:** {st.session_state.ollama_model}")
                
                # Display model parameters in a more organized way
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Parameters:**")
                    st.markdown(f"- Temperature: {st.session_state.temperature}")
                with col2:
                    st.markdown("**Context:**")
                    st.markdown(f"- Window: {st.session_state.contextWindow}")
                    st.markdown(f"- Max Tokens: {st.session_state.newMaxTokens}")
            else:
                st.error("No Model Selected")
                st.warning("Please select a model in the Model Settings page")
        
        # Chat Controls Section - Collapsable
        with st.expander("💬 Chat Controls", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Clear History", use_container_width=True):
                    clear_chat_history()
            with col2:
                if st.button("✨ New Chat", use_container_width=True):
                    clear_chat_history()
            
            # Save chat button with better styling
            if st.button("💾 Save Chat", on_click=toggle_save_dialog, 
                        disabled=len(st.session_state.messages) <= 1,
                        use_container_width=True):
                pass
        
        # RAG Status display - Collapsable
        with st.expander("🔗 RAG Status", expanded=False):
            if st.session_state.db_ready:
                st.success("RAG System: Connected")
                
                # Display RAG settings in a cleaner format
                st.markdown("**Retrieval Settings:**")
                st.markdown(f"- Documents: {st.session_state.dbRetrievalAmount}")
                st.markdown(f"- Contextual: {'✅' if st.session_state.ContextualRAG else '❌'}")
                st.markdown(f"- BM25: {'✅' if st.session_state.ContextualBM25RAG else '❌'}")
            else:
                st.warning("RAG System: Not Configured")
                st.info("Visit the RAG Configuration page to set up document retrieval")
                
        # Available Databases Section - Shows databases that can be used with @mentions
        with st.expander("📚 Available Databases", expanded=True):
            if st.session_state.available_databases:
                st.markdown("### Using Specific Databases")
                st.markdown("You can access specific databases by using the `@` symbol followed by the database name:")
                st.markdown("```\n@database_name your question here\n```")
                
                st.markdown("### Available Databases")
                for db in st.session_state.available_databases:
                    st.markdown(f"- `@{db}`")
                
                if st.button("🔄 Refresh Databases"):
                    update_available_databases()
                    st.rerun()
            else:
                st.info("No databases available")
                st.markdown("Create databases in the RAG Configuration page, then use them by typing `@database_name` in the chat.")
                
                if st.button("🔄 Check for Databases"):
                    update_available_databases()
                    st.rerun()
        
        # Saved Chats Section - Collapsable
        with st.expander("💾 Saved Conversations", expanded=False):
            recent_conversations = get_recent_conversations()
            
            if recent_conversations:
                for idx, conv in enumerate(recent_conversations):
                    # Create a container for each conversation
                    st.markdown(f"""
                    <div style="border: 1px solid #e0e0e0; border-radius: 5px; padding: 8px; margin-bottom: 8px; background-color: {'#f5f5f5' if idx % 2 == 0 else 'white'}">
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        # Display just the chat name without timestamp
                        chat_name = conv['first_message'].split(" (")[0]  # Get name without timestamp
                        if st.button(f"{chat_name}", key=f"conv_{conv['id']}", use_container_width=True):
                            load_conversation(conv['id'])
                    with col2:
                        if st.button("🗑️", key=f"del_{conv['id']}", use_container_width=True):
                            delete_conversation(conv['id'])
                            if 'conversation_id' in st.session_state and st.session_state.conversation_id == conv['id']:
                                clear_chat_history()
                            st.rerun()
            else:
                st.info("No saved conversations available")
                st.markdown("""
                <div style="font-size: 0.9em; color: #666; margin-top: 10px;">
                Conversations will appear here after you save them using the 'Save Chat' button.
                </div>
                """, unsafe_allow_html=True) 