import streamlit as st
from langchain_ollama import OllamaLLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.retrievers import WikipediaRetriever
from langchain_community.utilities import GoogleSerperAPIWrapper

import os
import asyncio
import json
import re
import shutil  # Added for directory operations
from urllib.parse import urlparse
from bs4 import SoupStrainer

# Import utility functions from CommonUtils
from CommonUtils.research_utils import (
    sanitize_filename,
    create_research_directory,
    write_markdown_file,
    fetch_and_process_url,
    clean_json_string,
    # Import prompt templates
    SEARCH_QUERY_TEMPLATE,
    SUBTOPICS_TEMPLATE,
    SUBTOPIC_SUMMARY_TEMPLATE,
    FINAL_SYNTHESIS_TEMPLATE,
    SEARCH_RESULTS_EVALUATION_TEMPLATE
)

# Import necessary RAG utilities
from CommonUtils.rag_utils import SOURCE_PATH, get_client
import pypandoc

# Set page config
st.set_page_config(
    page_title="Deep Research - OllamaRAG",
    page_icon="🔍",
    initial_sidebar_state="expanded",
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
    
    /* Research specific styling */
    .research-summary {
        background-color: #f0f7ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1E88E5;
        margin: 1rem 0;
    }

    .source-citation {
        font-size: 0.8em;
        color: #666;
        border-top: 1px solid #eee;
        margin-top: 0.5rem;
        padding-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'research_summary' not in st.session_state:
    st.session_state.research_summary = ""
if 'sources' not in st.session_state:
    st.session_state.sources = []
if 'research_in_progress' not in st.session_state:
    st.session_state.research_in_progress = False
if 'iteration_count' not in st.session_state:
    st.session_state.iteration_count = 0
if 'max_search_attempts' not in st.session_state:
    st.session_state.max_search_attempts = 3
if 'num_subtopics' not in st.session_state:
    st.session_state.num_subtopics = 3
if 'serper_api_key' not in st.session_state:
    st.session_state.serper_api_key = ""
if 'stop_requested' not in st.session_state:
    st.session_state.stop_requested = False
if 'output_folder' not in st.session_state:
    st.session_state.output_folder = os.path.join(os.getcwd(), "research_results")
if 'current_research_topic' not in st.session_state:
    st.session_state.current_research_topic = ""

# Initialize RAG parameters if not present
if 'chunk_size' not in st.session_state:
    st.session_state.chunk_size = 1000
if 'overlap' not in st.session_state:
    st.session_state.overlap = 200
if 'chroma_client' not in st.session_state:
    st.session_state.chroma_client = get_client()
if 'databases' not in st.session_state:
    st.session_state.databases = {}
if 'available_databases' not in st.session_state:
    st.session_state.available_databases = []

# Title and description
st.markdown("""
<h1 style="text-align: center; color: #0D47A1; margin-bottom: 20px;">🔍 Deep Research Assistant</h1>
""", unsafe_allow_html=True)

# Introduction in a card-like container
st.markdown("""
<div style="background-color: #f0f7ff; padding: 20px; border-radius: 10px; margin-bottom: 30px; border-left: 5px solid #1E88E5;">

<h2 style="color: #1E88E5; margin-top: 0;">Research Automation System</h2>

This advanced research assistant can help you explore any topic in depth by:

- ✅ Breaking down complex topics into manageable subtopics
- ✅ Searching multiple sources simultaneously (Web, News, Wikipedia, Google)
- ✅ Evaluating content quality and relevance automatically
- ✅ Generating a comprehensive, well-structured research report
- ✅ Citing all sources for academic integrity

Configure your research parameters in the sidebar and let the AI handle the heavy lifting!
</div>
""", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    # Smaller title with more subtle styling - exactly matching other pages
    st.markdown("""
    <h2 style="font-size: 1.5em; color: #0D47A1; margin-bottom: 15px; padding-bottom: 5px; border-bottom: 1px solid #e0e0e0;">
    ⚙️ Research Settings
    </h2>
    """, unsafe_allow_html=True)
    
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
    
    # Research Configuration Section - Collapsable
    with st.expander("📁 Output Configuration", expanded=False):
        # Output folder configuration with better help text
        st.session_state.output_folder = st.text_input(
            "Output Location",
            value=st.session_state.output_folder,
            placeholder="Enter folder path...",
            help="Specify where to save all research documents and reports"
        )
        if not st.session_state.output_folder:
            st.error("⚠️ Output folder path is required")
    
    # Google Serper API Configuration - Collapsable
    with st.expander("🔑 API Keys", expanded=False):
        st.markdown("""
        <p style="color: #1E88E5; font-weight: 600; margin-bottom: 10px;">Google Search Integration</p>
        <p style="font-size: 0.9em; margin-bottom: 15px;">
        To use Google Search, you need a Serper API key. Serper provides access to Google search results and is free for limited use:
        <ol style="font-size: 0.9em;">
            <li>Sign up at <a href="https://serper.dev" target="_blank">serper.dev</a></li>
            <li>Create an API key in your dashboard</li>
            <li>Copy and paste the API key below</li>
        </ol>
        </p>
        """, unsafe_allow_html=True)
        
        st.session_state.serper_api_key = st.text_input(
            "Google Serper API Key",
            value=st.session_state.serper_api_key,
            type="password",
            help="Get your free API key from serper.dev"
        )
        
        if st.session_state.serper_api_key:
            st.success("✅ API key provided")
        else:
            st.info("ℹ️ Google Search will be disabled without an API key")
    
    # Search Provider Configuration - Collapsable
    with st.expander("🔎 Data Sources", expanded=False):
        st.markdown("""
        <p style="font-size: 0.9em; margin-bottom: 15px;">
        Select which search engines to use for your research. Using multiple sources improves research quality.
        </p>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            use_duckduckgo = st.checkbox("DuckDuckGo Web", value=True, 
                                        help="Search the general web for information via DuckDuckGo")
            use_wikipedia = st.checkbox("Wikipedia", value=True,
                                      help="Search Wikipedia for well-established information")
        
        with col2:
            use_duckduckgo_news = st.checkbox("DuckDuckGo News", value=True,
                                             help="Search recent news articles via DuckDuckGo")
            use_google_serper = st.checkbox("Google Search", value=False,
                                           help="Use Google Search for high-quality, current results (requires API key)")
        
        # Show Google Serper configuration options if enabled
        if use_google_serper:
            if not st.session_state.serper_api_key:
                st.warning("⚠️ Google Serper API key is required to use Google Search. Add your key in the API Keys section.")
            
            st.markdown("#### Google Search Options")
            google_serper_type = st.radio(
                "Search Type",
                options=["search", "news", "images"],
                horizontal=True,
                help="Select the type of Google search to perform"
            )
            
            # Explain the different search types
            if google_serper_type == "search":
                st.info("Standard Google search results, including web pages and knowledge graph")
            elif google_serper_type == "news":
                st.info("Recent news articles from Google News")
            elif google_serper_type == "images":
                st.info("Image search results with source webpage information")
        
        if not any([use_duckduckgo, use_duckduckgo_news, use_wikipedia, use_google_serper]):
            st.warning("⚠️ Please enable at least one search provider")
    
    # Research parameters - Collapsable
    with st.expander("🔬 Research Depth", expanded=False):
        st.session_state.max_iterations = st.slider(
            "Research Iterations",
            min_value=1,
            max_value=20,
            value=5,
            help="Higher values mean more thorough research but longer processing time"
        )

        st.session_state.num_subtopics = st.slider(
            "Subtopics to Generate",
            min_value=1,
            max_value=20,
            value=3,
            help="How many subtopics to break down your main topic into"
        )
    
    # Research Stats section when research is active - Collapsable
    if st.session_state.research_in_progress or st.session_state.iteration_count > 0:
        with st.expander("📊 Research Progress", expanded=False):
            st.markdown(f"**Current Iteration:** {st.session_state.iteration_count}")
            if len(st.session_state.sources) > 0:
                st.markdown(f"**Sources Found:** {len(st.session_state.sources)}")

# Note: All prompt templates have been moved to CommonUtils/research_utils.py

# Function to handle research stopping
def stop_research():
    st.session_state.stop_requested = True
    st.session_state.research_in_progress = False
    st.warning("Research process stopped by user.")
    # Force a rerun to immediately update the UI
    st.rerun()

# Function to clear research folder
def clear_research_folder():
    if st.session_state.output_folder and os.path.exists(st.session_state.output_folder):
        try:
            # List all items in the output folder
            items = os.listdir(st.session_state.output_folder)
            for item in items:
                item_path = os.path.join(st.session_state.output_folder, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
            st.success(f"Research folder '{st.session_state.output_folder}' has been cleared!")
        except Exception as e:
            st.error(f"Error clearing research folder: {str(e)}")
    else:
        st.error("Research folder does not exist or is not specified.")

# Function to create a RAG DB from research files
async def create_rag_db_from_research(research_dir, topic_name, progress_container, status_container):
    """
    Creates a RAG database from research files.
    
    Args:
        research_dir: Path to the research directory
        topic_name: Name of the research topic (used as DB name)
        progress_container: Streamlit container for progress bar
        status_container: Streamlit container for status messages
    
    Returns:
        tuple[bool, str]: Success status and message
    """
    from CommonUtils.rag_utils import SOURCE_PATH, add_document_to_db, update_db, get_client
    import tempfile
    
    # Check for conversion dependencies
    use_pdf_conversion = True
    conversion_warning = ""
    
    # Create progress trackers
    progress_bar = progress_container.progress(0)
    
    def update_status(message, progress=None):
        """Update status message and progress bar"""
        status_container.info(message)
        if progress is not None:
            progress_bar.progress(progress)

    # Check for wkhtmltopdf only if we're using pypandoc
    if use_pdf_conversion:
        try:
            import subprocess
            subprocess.run(['wkhtmltopdf', '-V'], capture_output=True, check=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            use_pdf_conversion = False
            conversion_warning = "wkhtmltopdf not found - using direct markdown files instead of PDF conversion."
    
    try:
        # Sanitize topic name for DB name
        db_name = sanitize_filename(topic_name)
        
        # Check if a database with this name already exists
        if os.path.exists(os.path.join(SOURCE_PATH, db_name)):
            return False, f"A RAG database named '{db_name}' already exists. Please delete it first."
        
        update_status("Step 1/3: Creating database directory...", 0.1)
        
        # Step 1: Create the database directory
        db_path = os.path.join(SOURCE_PATH, db_name)
        os.makedirs(db_path, exist_ok=True)
        
        # Find all markdown files in the research directory and subdirectories
        all_md_files = []
        for root, dirs, files in os.walk(research_dir):
            for file in files:
                if file.endswith('.md'):
                    all_md_files.append(os.path.join(root, file))
        
        if not all_md_files:
            return False, "No markdown files found in the research directory."
        
        # If we have conversion warnings, show them
        if conversion_warning:
            status_container.warning(conversion_warning)
        
        update_status(f"Step 2/3: Converting {len(all_md_files)} research files...", 0.2)
        
        # Process each markdown file
        for idx, md_file in enumerate(all_md_files):
            # Update progress based on file index
            current_progress = 0.2 + (0.3 * (idx / len(all_md_files)))
            progress_bar.progress(current_progress)
            
            try:
                if use_pdf_conversion:
                    # PDF conversion path
                    # Create a temporary PDF file
                    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
                        temp_pdf_path = temp_pdf.name
                    
                    # Update status with current file
                    file_name = os.path.basename(md_file)
                    update_status(f"Converting file {idx+1}/{len(all_md_files)}: {file_name}")
                    
                    # Convert markdown to PDF using pypandoc
                    pypandoc.convert_file(
                        md_file, 
                        'pdf', 
                        outputfile=temp_pdf_path,
                        extra_args=[
                            '--pdf-engine=wkhtmltopdf',
                            '-V', 'geometry:margin=1in'
                        ]
                    )
                    
                    # Read the PDF file
                    with open(temp_pdf_path, 'rb') as f:
                        file_content = f.read()
                    
                    # Get relative filename for the PDF
                    rel_filename = os.path.basename(md_file).replace('.md', '.pdf')
                    
                    # Clean up temporary file
                    os.unlink(temp_pdf_path)
                else:
                    # Direct markdown file path - read the markdown file directly
                    file_name = os.path.basename(md_file)
                    update_status(f"Processing file {idx+1}/{len(all_md_files)}: {file_name}")
                    
                    with open(md_file, 'rb') as f:
                        file_content = f.read()
                    
                    # Keep markdown extension
                    rel_filename = os.path.basename(md_file)
                
                # Add the file to the database
                success, message = add_document_to_db(db_name, rel_filename, file_content)
                if not success:
                    status_container.warning(f"Issue with file {rel_filename}: {message}")
                
            except Exception as e:
                status_container.warning(f"Error processing {md_file}: {str(e)}")
                continue
        
        # Update the database if embedding model is available
        update_status("Step 3/3: Updating database with embeddings...", 0.5)
        
        if hasattr(st.session_state, 'embeddingModel') and st.session_state.embeddingModel:
            embedding_model = st.session_state.embeddingModel
            
            try:
                # Call the update_db function to process the files and create embeddings
                success, message, processed_docs = await update_db(
                    db_name=db_name,
                    embedding_model=embedding_model,
                    chroma_client=st.session_state.chroma_client,
                    chunk_size=int(st.session_state.chunk_size if hasattr(st.session_state, 'chunk_size') else 1000),
                    overlap=int(st.session_state.overlap if hasattr(st.session_state, 'overlap') else 200),
                    context_model=st.session_state.context_model if hasattr(st.session_state, 'ContextualRAG') and st.session_state.ContextualRAG else None,
                    use_contextual_rag=st.session_state.ContextualRAG if hasattr(st.session_state, 'ContextualRAG') else False,
                    progress_callback=update_status
                )
                
                if success:
                    # Mark database as ready
                    if 'databases' in st.session_state:
                        st.session_state.databases[db_name] = {
                            "ready": True,
                            "collection": get_client().get_collection(f"rag_collection_{db_name}")
                        }
                    
                    # Update available databases list
                    if 'available_databases' in st.session_state:
                        if db_name not in st.session_state.available_databases:
                            st.session_state.available_databases.append(db_name)
                    
                    return True, f"""
                    ✅ RAG database '{db_name}' created and indexed successfully!
                    
                    Your research is now ready to use in the Chat page (💬).
                    """
                else:
                    # Database was created but update failed
                    if 'databases' in st.session_state:
                        st.session_state.databases[db_name] = {"ready": False, "collection": None}
                    
                    # Update available databases list
                    if 'available_databases' in st.session_state:
                        if db_name not in st.session_state.available_databases:
                            st.session_state.available_databases.append(db_name)
                    
                    return False, f"""
                    ⚠️ Database '{db_name}' was created but indexing failed: {message}
                    
                    You can try updating it manually from the RAG Config page (🔗).
                    """
            except Exception as e:
                status_container.error(f"Error updating database: {str(e)}")
                
                # Still update the available databases since files were created
                if 'available_databases' in st.session_state:
                    if db_name not in st.session_state.available_databases:
                        st.session_state.available_databases.append(db_name)
                
                if 'databases' in st.session_state:
                    st.session_state.databases[db_name] = {"ready": False, "collection": None}
                
                return False, f"""
                ⚠️ Database '{db_name}' was created but encountered an error during indexing.
                
                Please go to the RAG Config page (🔗) to update it manually.
                """
        else:
            # No embedding model available
            if 'available_databases' in st.session_state:
                if db_name not in st.session_state.available_databases:
                    st.session_state.available_databases.append(db_name)
            
            if 'databases' in st.session_state:
                st.session_state.databases[db_name] = {"ready": False, "collection": None}
                
            return True, f"""
            ✅ RAG database '{db_name}' created with research files!
            
            Next steps:
            1. Go to the Model Settings page (⚙️) to select an embedding model
            2. Go to the RAG Config page (🔗) to update the database
            3. Click on '{db_name}' in the database list
            4. Click the 'Update' button to process the files
            5. Return to the Chat page (💬) to use your research in conversations
            """
            
    except Exception as e:
        return False, f"Error creating RAG database: {str(e)}"

async def research_subtopic(subtopic, search_tools, synthesis_chain, main_topic, status_text):
    """Conduct research on a specific subtopic."""
    try:
        # Check if stop was requested
        if st.session_state.stop_requested:
            status_text.warning("Research stopped by user.")
            return None, None, None
            
        # Initialize LLM chains
        llm = OllamaLLM(
            model=st.session_state.ollama_model,
            temperature=st.session_state.temperature,
            num_ctx=st.session_state.contextWindow,
            num_predict=st.session_state.newMaxTokens
        )

        llm_jason = OllamaLLM(
            model=st.session_state.ollama_model,
            temperature=st.session_state.temperature,
            num_ctx=st.session_state.contextWindow,
            num_predict=st.session_state.newMaxTokens,
            format="json",
        )
        
        # Create search query chain
        search_query_chain = LLMChain(
            llm=llm,
            prompt=PromptTemplate(
                template=SEARCH_QUERY_TEMPLATE,
                input_variables=["main_topic", "topic"]
            )
        )
        
        # Create evaluation chain
        evaluation_chain = LLMChain(
            llm=llm_jason,
            prompt=PromptTemplate(
                template=SEARCH_RESULTS_EVALUATION_TEMPLATE,
                input_variables=["topic", "results"]
            )
        )
        
        max_search_attempts = st.session_state.max_search_attempts
        search_attempt = 0
        
        while search_attempt < max_search_attempts:
            # Check if stop was requested
            if st.session_state.stop_requested:
                status_text.warning("Research stopped by user.")
                return None, None, None
                
            search_attempt += 1
            
            # Generate contextual search query
            search_query = (await search_query_chain.ainvoke({"main_topic": main_topic, "topic": subtopic}))["text"]
            
            # Debug search query
            status_text.text(f"Search attempt {search_attempt}/{max_search_attempts} for: {search_query}")
            
            detailed_results = []
            
            # Perform web searches based on enabled providers
            if search_tools.get('web') and not st.session_state.stop_requested:
                status_text.text("Performing DuckDuckGo web search...")
                search_results = search_tools['web'].run(search_query)
                urls = re.findall(r'https?://[^\s<>"]+|www\.[^\s<>"]+', search_results)
                status_text.text(f"Found URLs from web search: {urls}")
                
                # Process web URLs
                for url in urls:
                    status_text.text(f"Processing URL: {url}")
                    content = await fetch_and_process_url(url, status_text)
                    if content:
                        detailed_results.append({
                            "url": url,
                            "content": content,
                            "source": "web"
                        })
            
            # Perform news search if enabled
            if search_tools.get('news') and not st.session_state.stop_requested:
                status_text.text("Performing DuckDuckGo news search...")
                news_results = search_tools['news'].run(search_query)
                news_urls = re.findall(r'https?://[^\s<>"]+|www\.[^\s<>"]+', news_results)
                status_text.text(f"Found URLs from news search: {news_urls}")
                
                # Process news URLs
                for url in news_urls:
                    status_text.text(f"Processing news URL: {url}")
                    content = await fetch_and_process_url(url, status_text)
                    if content:
                        detailed_results.append({
                            "url": url,
                            "content": content,
                            "source": "news"
                        })
            
            # Get Google Serper results if enabled
            if search_tools.get('google_serper') and not st.session_state.stop_requested:
                status_text.text(f"Performing Google Serper {search_tools['google_serper_type']} search...")
                
                # Get the search results with full metadata
                google_results = search_tools['google_serper'].results(search_query)
                
                # Extract URLs based on the search type
                google_urls = []
                
                if search_tools['google_serper_type'] == 'search':
                    # Extract URLs from organic search results
                    if 'organic' in google_results:
                        google_urls = [item['link'] for item in google_results['organic'] if 'link' in item]
                    
                    # Add knowledge graph URL if available
                    if 'knowledgeGraph' in google_results and 'descriptionLink' in google_results['knowledgeGraph']:
                        google_urls.append(google_results['knowledgeGraph']['descriptionLink'])
                
                elif search_tools['google_serper_type'] == 'news':
                    # Extract URLs from news results
                    if 'news' in google_results:
                        google_urls = [item['link'] for item in google_results['news'] if 'link' in item]
                
                elif search_tools['google_serper_type'] == 'images':
                    # Extract source URLs from image results
                    if 'images' in google_results:
                        google_urls = [item['link'] for item in google_results['images'] if 'link' in item]
                
                status_text.text(f"Found URLs from Google Serper search: {google_urls}")
                
                # Process Google URLs
                for url in google_urls:
                    status_text.text(f"Processing Google Serper URL: {url}")
                    content = await fetch_and_process_url(url, status_text)
                    if content:
                        detailed_results.append({
                            "url": url,
                            "content": content,
                            "source": f"google_{search_tools['google_serper_type']}"
                        })
            
            # Get Wikipedia results if enabled
            if search_tools.get('wikipedia') and not st.session_state.stop_requested:
                status_text.text("Retrieving Wikipedia results...")
                wikipedia_results = search_tools['wikipedia'].get_relevant_documents(search_query)
                
                # Process Wikipedia results
                for wiki_doc in wikipedia_results:
                    detailed_results.append({
                        "url": f"Wikipedia: {wiki_doc.metadata.get('title', 'Unknown Article')}",
                        "content": wiki_doc.page_content,
                        "source": "wikipedia"
                    })
            
            # Combine search results with detailed content
            combined_results = []
            summaries_subtopic = ""
            collected_urls = []  # Track sources for this subtopic

            for result in detailed_results:
                # Check if stop was requested
                if st.session_state.stop_requested:
                    status_text.warning("Research stopped by user.")
                    return None, None, None
                    
                # Add source to all_sources regardless of evaluation result
                if 'url' in result:
                    # Track this URL for the final sources list
                    collected_urls.append(result['url'])
                    
                # Evaluate combined results
                try:
                    source_type = result['source'].capitalize()
                    status_text.text(f"Evaluating {source_type} content from {result['url']}")
                    evaluation_result = (await evaluation_chain.ainvoke({
                        "topic": f"{main_topic} - {subtopic}",
                        "results": result['content']
                    }))["text"]

                    evaluation_result = json.loads(evaluation_result)
                    
                    # Check if results are sufficient
                    if evaluation_result["sufficient_data"]:
                        status_text.text(f"Found sufficient data in {result['url']} - adding to combined results")
                        combined_results.append(result)
                        # Add to summaries with source indication
                        source_prefix = f"[{source_type}] "
                        summaries_subtopic += f"{source_prefix}{subtopic}\n\n\n{result['content']}\n\n"

                except Exception as e:
                    st.error(f"Error evaluating search results: {str(e)}")
                    continue
                
        # Check if stop was requested
        if st.session_state.stop_requested:
            status_text.warning("Research stopped by user.")
            return None, None, None
            
        status_text.text(f"Synthesizing findings for {subtopic}")
        # Synthesize findings using combined results
        synthesis_result = (await synthesis_chain.ainvoke({
            "topic": subtopic,
            "results": summaries_subtopic
        }))["text"]
            
        # Clean up the debug container
        status_text.empty()
        
        return synthesis_result, combined_results, collected_urls  # Return synthesis, webpage contents, and all URLs
        
    except Exception as e:
        st.error(f"Error researching subtopic '{subtopic}': {str(e)}")
        return None, None, None

async def conduct_research(topic):
    """
    Conducts structured research on the given topic.
    The research flow is defined as follows:
    1. Generate subtopics
    2. Research each subtopic
    3. Synthesize findings
    4. Create final synthesis
    """
    try:
        # Store the research topic in session state for recovery after stop/restart
        st.session_state.current_research_topic = topic
        
        # Initialize progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        subtopic_status = st.empty()
        debug_container = st.empty()
        
        # Update UI and check if research should be stopped
        def researchUpdate(message="", progress_value=None):
            """Update UI and check if research should be stopped"""
            if message:
                status_text.text(message)
            if progress_value is not None:
                # Ensure progress value is within valid range [0.0, 1.0]
                capped_progress = min(1.0, max(0.0, progress_value))
                progress_bar.progress(capped_progress)
            # Check if stop has been requested - important for responsiveness
            if st.session_state.stop_requested:
                status_text.warning("Research process stopping - completing current operation...")
                # Don't set research_in_progress to False here, 
                # it will be set to False in the finally block of conduct_research
                return True
            return False
            
        # Initialize search tools based on user configuration
        search_tools = {}
        if use_duckduckgo:
            search_tools['web'] = DuckDuckGoSearchResults()
        if use_duckduckgo_news:
            search_tools['news'] = DuckDuckGoSearchResults(backend="news")
        if use_wikipedia:
            search_tools['wikipedia'] = WikipediaRetriever()
        
        # Add Google Serper if enabled and API key is provided
        if use_google_serper and st.session_state.serper_api_key:
            # Set the API key for the GoogleSerperAPIWrapper
            os.environ["SERPER_API_KEY"] = st.session_state.serper_api_key
            
            # Initialize the GoogleSerperAPIWrapper with the selected type
            search_tools['google_serper'] = GoogleSerperAPIWrapper(type=google_serper_type)
            search_tools['google_serper_type'] = google_serper_type
            
        if not search_tools:
            st.error("Please enable at least one search provider to proceed")
            return

        # Update UI and check if research should be stopped
        if researchUpdate("Initializing research process...", 0.05):
            return
            
        # Initialize the LLM and chains
        llm = OllamaLLM(
            model=st.session_state.ollama_model,
            temperature=st.session_state.temperature,
            num_ctx=st.session_state.contextWindow,
            num_predict=st.session_state.newMaxTokens
        )

        llm_jason = OllamaLLM(
            model=st.session_state.ollama_model,
            temperature=st.session_state.temperature,
            num_ctx=st.session_state.contextWindow,
            num_predict=st.session_state.newMaxTokens,
            format="json",
        )
        
        subtopics_chain = LLMChain(
            llm=llm_jason,
            prompt=PromptTemplate(
                template=SUBTOPICS_TEMPLATE,
                input_variables=["topic"]
            )
        )
        
        synthesis_chain = LLMChain(
            llm=llm,
            prompt=PromptTemplate(
                template=SUBTOPIC_SUMMARY_TEMPLATE,
                input_variables=["topic", "results"]
            )
        )
        
        final_synthesis_chain = LLMChain(
            llm=llm,
            prompt=PromptTemplate(
                template=FINAL_SYNTHESIS_TEMPLATE,
                input_variables=["topic", "summaries"]
            )
        )
            
        # Step 1: Create research directory structure
        if not st.session_state.output_folder:
            st.error("Please specify an output folder path")
            return
            
        if researchUpdate("Creating research directory...", 0.07):
            return
            
        research_dir, subtopics_dir = create_research_directory(st.session_state.output_folder, topic)
        if researchUpdate("Research directory created", 0.1):
            return
        
        # Step 2: Generate subtopics
        if researchUpdate("Generating research subtopics...", 0.15):
            return
        
        # Check if stop was requested
        if st.session_state.stop_requested:
            status_text.warning("Research stopped by user.")
            return
            
        try:
            subtopics_result = (await subtopics_chain.ainvoke({"topic": topic, "num_subtopics": st.session_state.num_subtopics}))["text"]
            # Clean up the JSON string
            subtopics_result = clean_json_string(subtopics_result)
            
            if researchUpdate(f"Processing subtopics result...", 0.18):
                return
                
            subtopics_data = json.loads(subtopics_result)
            if not isinstance(subtopics_data, dict) or 'subtopics' not in subtopics_data:
                raise ValueError("Invalid subtopics format. Expected a dictionary with 'subtopics' key.")
            
            subtopics = subtopics_data["subtopics"]
            if not isinstance(subtopics, list) or len(subtopics) == 0:
                raise ValueError("No valid subtopics generated.")
            
            if researchUpdate(f"Generated {len(subtopics)} subtopics", 0.2):
                return
            
        except Exception as e:
            st.error(f"Error generating subtopics: {str(e)}")
            st.error("Raw response: " + subtopics_result)
            return
        
        # Write main topic file
        main_topic_file = os.path.join(research_dir, f"{sanitize_filename(topic)}.md")
        main_content = f"# Research: {topic}\n\n## Subtopics\n\n"
        for idx, subtopic in enumerate(subtopics, 1):
            main_content += f"{idx}. {subtopic}\n"
        write_markdown_file(main_topic_file, main_content)
        
        if researchUpdate("Starting subtopic research...", 0.25):
            return
        
        # Step 3: Research each subtopic
        all_summaries = []
        all_sources = []  # Track all sources across subtopics
        
        # Calculate how much progress to allocate for each subtopic (making sure total doesn't exceed 1.0)
        total_subtopic_progress = 0.55  # Total progress allocation for all subtopics
        subtopic_start = 0.25  # Starting progress value before subtopics
        subtopic_count = min(len(subtopics), st.session_state.num_subtopics)
        
        for idx, subtopic in enumerate(subtopics[:st.session_state.num_subtopics], 1):
            # Calculate exact position in the progress bar range safely
            current_subtopic_progress = subtopic_start + (total_subtopic_progress * ((idx-1) / subtopic_count))
            
            # Check after each subtopic - critical for responsiveness
            if researchUpdate(f"Researching subtopic {idx}/{st.session_state.num_subtopics}: {subtopic}", 
                              current_subtopic_progress):
                return
            
            # Research the subtopic with main topic context
            result, webpage_contents, collected_urls = await research_subtopic(subtopic, search_tools, synthesis_chain, topic, subtopic_status)

            # Calculate progress after completion
            completed_progress = subtopic_start + (total_subtopic_progress * (idx / subtopic_count))
            
            # Check after each subtopic is complete
            if researchUpdate(f"Completed research on subtopic {idx}", completed_progress):
                return

            # Add any collected URLs to all_sources list
            if collected_urls:
                all_sources.extend(collected_urls)
            
            if result:
                # Create subtopic file
                subtopic_file = os.path.join(subtopics_dir, f"{sanitize_filename(subtopic)}.md")
                content = f"# {subtopic}\n\n"
                
                # Add detailed webpage content first
                content += "## Webpage Contents\n\n"
                for webpage in webpage_contents:
                    content += f"### URL - {webpage['url']}\n\n"
                    content += "#### Content\n"
                    content += f"```\n{webpage['content']}\n```\n\n"
                
                # Add subtopic summary
                content += "## Subtopic Summary\n\n"
                content += f"{result}\n\n"
                
                # Add sources in a more structured format
                content += "## Sources\n\n"
                # Group by types for cleaner display
                web_sources = []
                wiki_sources = []
                other_sources = []
                
                if collected_urls:
                    for source in collected_urls:
                        if source.startswith(('http://', 'https://')):
                            web_sources.append(source)
                        elif source.startswith('Wikipedia:'):
                            wiki_sources.append(source)
                        else:
                            other_sources.append(source)
                
                # Format and display sources by type
                if web_sources:
                    content += "### Web Sources\n"
                    for idx, url in enumerate(web_sources, 1):
                        content += f"{idx}. {url}\n"
                    content += "\n"
                    
                if wiki_sources:
                    content += "### Wikipedia Sources\n"
                    for idx, wiki in enumerate(wiki_sources, 1):
                        content += f"{idx}. {wiki}\n"
                    content += "\n"
                    
                if other_sources:
                    content += "### Other Sources\n"
                    for idx, other in enumerate(other_sources, 1):
                        content += f"{idx}. {other}\n"
                    content += "\n"
                
                write_markdown_file(subtopic_file, content)
                all_summaries.append({
                    "subtopic": subtopic,
                    "summary": result,
                })
            
            # Update UI after writing files
            file_saved_progress = min(0.8, completed_progress + 0.01)
            if researchUpdate(f"Saved research for subtopic {idx}", file_saved_progress):
                return
        
        # Step 4: Create final synthesis
        if researchUpdate("Creating final research synthesis...", 0.85):
            return
            
        try:
            final_result = (await final_synthesis_chain.ainvoke({
                "topic": topic,
                "summaries": json.dumps(all_summaries)  # Pass complete summary objects
            }))["text"]
                
            # Write final overview
            overview_file = os.path.join(research_dir, "research_overview.md")
            write_markdown_file(overview_file, final_result)
                
            # Update session state with results
            st.session_state.research_summary = final_result
            st.session_state.sources = list(dict.fromkeys(all_sources)) 
                
                
        except Exception as e:
            st.error(f"Error creating final synthesis: {str(e)}")
            return
        
        if researchUpdate("Research complete!", 0.95):
            return
            
        debug_container.empty()
        
        # Ensure the progress bar is exactly at 100%
        progress_bar.progress(1.0)
        
        # Show success message with directory location
        st.success(f"Research completed successfully! Results saved to: {research_dir}")
        
    except Exception as e:
        st.error(f"Research Error: {str(e)}")
    finally:
        st.session_state.research_in_progress = False

# Main research interface in a card-like container
st.markdown("""
<h2 style="color: #1E88E5; margin-top: 0;">Research Topic</h2>
</div>
""", unsafe_allow_html=True)

research_topic = st.text_input(
    "Enter your research query",
    placeholder="Enter any topic you want to research in depth...",
    disabled=st.session_state.research_in_progress,
    help="Be specific enough to get targeted results, but broad enough to explore the topic"
)

# Button row with Start, Stop, and Clear buttons
col1, col2, col3 = st.columns([3, 1, 1])
with col1:
    start_button = st.button(
        "🚀 Start Research", 
        disabled=st.session_state.research_in_progress,
        use_container_width=True,
        type="primary"
    )
    
    if start_button:
        if not research_topic:
            st.warning("Please enter a research topic.")
        elif not hasattr(st.session_state, 'ollama_model') or not st.session_state.ollama_model:
            st.error("Please select an Ollama model in the Model Settings page first.")
        elif not st.session_state.output_folder:
            st.error("Please specify an output folder path for saving research results.")
        else:
            # Reset any previous stop_requested flag
            st.session_state.stop_requested = False
            st.session_state.research_in_progress = True
            st.session_state.research_summary = ""
            st.session_state.sources = []
            st.session_state.iteration_count = 0
            # Store research topic for potential recovery
            st.session_state.current_research_topic = research_topic 
            asyncio.run(conduct_research(research_topic))
            
            # Check if research was stopped - provide feedback
            if st.session_state.stop_requested:
                st.warning(f"Research on '{research_topic}' was stopped by user request.")
            else:
                st.success(f"Research on '{research_topic}' completed successfully!")

with col2:
    stop_button = st.button(
        "⏹️ Stop", 
        disabled=not st.session_state.research_in_progress,
        use_container_width=True,
        type="secondary"
    )
    
    if stop_button:
        stop_research()

with col3:
    if st.button(
        "🗑️ Clear", 
        disabled=st.session_state.research_in_progress,
        use_container_width=True
    ):
        st.session_state.research_summary = ""
        st.session_state.sources = []
        st.session_state.iteration_count = 0
        # Show modal to confirm clearing research folder
        clear_folder = st.checkbox("Also clear research folder?", value=False)
        if clear_folder:
            clear_research_folder()
        st.rerun()

# Display research results
if st.session_state.research_summary:   
    tabs = st.tabs(["📑 Summary", "🔗 Sources", "📂 Files"])
    
    with tabs[0]:
        st.markdown("""
        <div class="research-summary">
        """, unsafe_allow_html=True)
        st.markdown(st.session_state.research_summary)
        st.markdown("""
        </div>
        """, unsafe_allow_html=True)
    
    with tabs[1]:
        if st.session_state.sources:
            st.markdown("### Sources Used in Research")
            
            # Group sources by type
            web_sources = []
            wiki_sources = []
            other_sources = []
            
            for source in st.session_state.sources:
                source = source.strip().rstrip('.')
                if source.startswith(('http://', 'https://')):
                    web_sources.append(source)
                elif source.startswith('Wikipedia:'):
                    wiki_sources.append(source)
                else:
                    other_sources.append(source)
            
            # Display web sources
            if web_sources:
                st.markdown("#### Web Sources")
                for idx, source in enumerate(web_sources, 1):
                    st.markdown(f"{idx}. [{source}]({source})")
            
            # Display Wikipedia sources
            if wiki_sources:
                st.markdown("#### Wikipedia Sources")
                for idx, source in enumerate(wiki_sources, 1):
                    # Extract article title for better display
                    article_title = source.replace('Wikipedia:', '').strip()
                    search_query = article_title.replace(' ', '+')
                    wiki_url = f"https://en.wikipedia.org/wiki/Special:Search?search={search_query}"
                    st.markdown(f"{idx}. [{article_title}]({wiki_url})")
            
            # Display other sources
            if other_sources:
                st.markdown("#### Other Sources")
                for idx, source in enumerate(other_sources, 1):
                    st.markdown(f"{idx}. {source}")
                    
            if not (web_sources or wiki_sources or other_sources):
                st.info("No sources were found in the search results.")
        else:
            st.info("No sources are available for this research.")
    
    with tabs[2]:
        if st.session_state.output_folder and os.path.exists(st.session_state.output_folder):
            st.markdown("### Research Files")
            st.markdown(f"All research files are saved to: `{st.session_state.output_folder}`")
            
            # Get list of all research directories
            research_folders = [d for d in os.listdir(st.session_state.output_folder) 
                              if os.path.isdir(os.path.join(st.session_state.output_folder, d))]
            
            if research_folders:
                # Either use current topic or let user select from all available research folders
                if research_topic and os.path.exists(os.path.join(st.session_state.output_folder, sanitize_filename(research_topic))):
                    # If current research topic exists, default to it
                    current_research_dir = sanitize_filename(research_topic)
                    st.success(f"Showing files for current research topic: {research_topic}")
                else:
                    # Let user select from available research folders
                    current_research_dir = st.selectbox(
                        "Select Research Project",
                        options=research_folders,
                        format_func=lambda x: x.replace("_", " ").title(),
                        help="Choose a completed research project to view its files"
                    )
                    
                if current_research_dir:
                    research_dir = os.path.join(st.session_state.output_folder, current_research_dir)
                    
                    # List all files in the research directory
                    all_files = []
                    for root, dirs, files in os.walk(research_dir):
                        for file in files:
                            if file.endswith('.md'):
                                rel_path = os.path.relpath(os.path.join(root, file), research_dir)
                                all_files.append((rel_path, os.path.join(root, file)))
                    
                    if all_files:
                        st.markdown("#### Available Files")
                        for rel_path, full_path in all_files:
                            # Create columns for file name and download button
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(f"📄 **{rel_path}**")
                            with col2:
                                st.download_button(
                                    label="Download",
                                    data=open(full_path, "r", encoding="utf-8").read(),
                                    file_name=os.path.basename(full_path),
                                    mime="text/markdown",
                                    key=f"download_{rel_path}"
                                )
                            
                            # Preview option for main files
                            if rel_path == "research_overview.md" or rel_path.count('/') == 0:
                                with st.expander("Preview Content", expanded=False):
                                    try:
                                        file_content = open(full_path, "r", encoding="utf-8").read()
                                        # Only show first 500 characters of content as preview
                                        preview = file_content[:1000] + "..." if len(file_content) > 1000 else file_content
                                        st.markdown(preview)
                                    except Exception as e:
                                        st.error(f"Error reading file: {str(e)}")
                        
                        # Add Create RAG DB button
                        st.markdown("---")
                        st.markdown("### Create RAG Database")
                        st.markdown("Convert research files into a RAG database for chat and Q&A.")
                        
                        # Button should be enabled only when research is completed (not in progress)
                        # and there's actual research data available
                        button_disabled = (
                            st.session_state.research_in_progress or
                            not os.path.exists(research_dir) or
                            not os.path.isdir(research_dir) or
                            len(all_files) < 2  # Need at least overview and one other file
                        )
                        
                        create_rag_db_button = st.button(
                            "🔄 Create RAG DB",
                            disabled=button_disabled,
                            type="primary",
                            use_container_width=True,
                            help="Convert research files into a RAG database for use in the Chat page"
                        )
                        
                        if button_disabled and not st.session_state.research_in_progress:
                            st.info("Research data is required to create a RAG database. Run a research query to generate files.")
                        
                        if create_rag_db_button:
                            # Get the cleaned topic name (without timestamp)
                            topic_name = current_research_dir.split('_')[0] if '_' in current_research_dir else current_research_dir
                            
                            # Create a progress bar for the conversion process
                            rag_progress = st.empty()
                            rag_status = st.empty()
                            
                            # Show status
                            rag_status.info("Starting conversion of research files to RAG database...")
                            
                            # Use asyncio.run to run the async function
                            import asyncio
                            success, message = asyncio.run(create_rag_db_from_research(research_dir, topic_name, rag_progress, rag_status))
                            
                            # Update UI based on result
                            if success:
                                rag_status.success(message)
                            else:
                                rag_status.error(message)
                    else:
                        st.warning("No markdown files found in the research directory.")
            else:
                st.info("No research folders found. Run a research query to generate files.")
        else:
            st.warning("Output folder not found or not specified.") 