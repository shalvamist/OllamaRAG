import streamlit as st
from langchain_ollama import OllamaLLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.tools import DuckDuckGoSearchRun
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.tools import Tool
import os
import asyncio
import json

# Set page config
st.set_page_config(
    page_title="Deep Research - OllamaRAG",
    page_icon="🔍",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .stApp {
        background-color: #a9b89e;
        color: #1a2234;
    }
    
    .block-container {
        max-width: 80% !important;
        padding: 2rem;
        background-color: #fff;
        border-radius: 10px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.15);
        margin: 1rem auto;
    }
    
    h1 {
        color: #2c3e50 !important;
        border-bottom: 3px solid #3498db !important;
        padding-bottom: 0.5rem !important;
    }

    .research-summary {
        background-color: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3498db;
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

# Title and description
st.title("🔍 Deep Research")
st.markdown("""
This tool performs comprehensive web research on any topic you provide. It will:
1. Generate targeted search queries
2. Gather information from multiple sources
3. Synthesize findings into a coherent summary
4. Identify knowledge gaps and perform follow-up research
5. Provide a final report with citations
""")

# Sidebar configuration
with st.sidebar:
    # Model Status Section
    st.header("🤖 Model Status")
    if hasattr(st.session_state, 'ollama_model') and st.session_state.ollama_model:
        st.success("Model Connected")
        st.info(f"**Model:** {st.session_state.ollama_model}")
        
        # Display model parameters
        st.markdown("**Model Parameters:**")
        st.markdown(f"- Context Window: {st.session_state.contextWindow}")
        st.markdown(f"- Temperature: {st.session_state.temperature}")
        st.markdown(f"- Max Tokens: {st.session_state.newMaxTokens}")
    else:
        st.error("No Model Selected")
        st.warning("Please select a model in the RAG Configuration page")
    
    st.divider()
    
    # Research Configuration Section
    st.header("🔍 Research Configuration")
    
    # Output folder configuration
    default_output = os.path.join(os.getcwd(), "research_results")
    output_folder = st.text_input(
        "Research Output Folder Path",
        value=default_output,
        placeholder="Enter the full path to save results...",
        help="Required: Specify the folder path where research results will be saved"
    )
    if not output_folder:
        st.error("⚠️ Output folder path is required to proceed with research")
    
    # Search Provider Selection
    search_provider = st.selectbox(
        "Search Provider",
        ["DuckDuckGo", "Google", "Brave"],
        help="Select the web search provider"
    )
    
    # API Key Configuration
    if search_provider in ["Google", "Brave"]:
        with st.expander("API Configuration", expanded=True):
            if search_provider == "Google":
                google_api_key = st.text_input("Google API Key", type="password", help="Enter your Google Custom Search API Key")
                google_cse_id = st.text_input("Google CSE ID", type="password", help="Enter your Google Custom Search Engine ID")
                if google_api_key and google_cse_id:
                    os.environ["GOOGLE_API_KEY"] = google_api_key
                    os.environ["GOOGLE_CSE_ID"] = google_cse_id
            elif search_provider == "Brave":
                brave_api_key = st.text_input("Brave API Key", type="password", help="Enter your Brave Search API Key")
                if brave_api_key:
                    os.environ["BRAVE_API_KEY"] = brave_api_key
    
    max_iterations = st.slider(
        "Maximum Research Iterations",
        min_value=1,
        max_value=20,
        value=5,
        help="Maximum number of research cycles to perform"
    )
    
    st.divider()
    
    # Research Stats (when research is in progress or completed)
    if st.session_state.research_in_progress or st.session_state.iteration_count > 0:
        st.header("📊 Research Stats")
        st.markdown(f"**Current Iteration:** {st.session_state.iteration_count}")
        if len(st.session_state.sources) > 0:
            st.markdown(f"**Sources Found:** {len(st.session_state.sources)}")

# Research prompt templates
SEARCH_QUERY_TEMPLATE = """Generate a search query for web research.

TOPIC: {topic}
ITERATION: {iteration}
GAPS: {gaps}

RULES:
1. Return ONLY the search query as plain text
2. NO JSON, NO explanations, NO formatting
3. Keep it under 10 words
4. Make it specific and focused
5. Do not use quotes or special characters

BAD EXAMPLES:
- {{\"query\": \"quantum computing basics\"}}
- "quantum computing applications"
- <thinking>Let me generate...</thinking>

GOOD EXAMPLES:
quantum computing applications in cryptography
natural remedies for reducing inflammation
python async programming best practices

YOUR QUERY:"""

SYNTHESIS_TEMPLATE = """You are a research synthesis AI. Create a structured research summary in JSON format.

TOPIC: {topic}
SEARCH RESULTS: {results}
PREVIOUS_SUMMARY: {previous_summary}

INSTRUCTIONS:
1. Analyze the search results thoroughly
2. Create a comprehensive summary with these sections:
   - Introduction/Overview
   - Key Findings
   - Detailed Analysis
   - Conclusions
3. Use proper markdown headings and formatting
4. Include specific facts and data points
5. Cite sources for important claims
6. IMPORTANT: Extract and include ONLY valid web URLs from the search results in the sources list

YOUR RESPONSE MUST BE A SINGLE JSON OBJECT WITH THIS EXACT STRUCTURE:
{
    "summary": "## Research Summary\\n\\n### Overview\\nBrief introduction to the topic\\n\\n### Key Findings\\n* Important point 1\\n* Important point 2\\n* Important point 3\\n\\n### Detailed Analysis\\nIn-depth discussion of findings\\n\\n### Conclusions\\nMain takeaways and implications",
    "gaps": [
        "Specific unanswered question 1",
        "Specific unanswered question 2"
    ],
    "sources": [
        "https://example.com/article1",
        "https://example.com/article2"
    ]
}"""

SUBTOPICS_TEMPLATE = """Generate a list of comprehensive subtopics for detailed research on the given topic.

TOPIC: {topic}

INSTRUCTIONS:
1. Analyze the topic and break it down into logical subtopics
2. Each subtopic should be specific and focused
3. Include both fundamental and advanced aspects
4. Ensure coverage is comprehensive
5. Return ONLY a JSON object with a "subtopics" array
6. Each subtopic should be a clear, concise phrase

EXAMPLE RESPONSE:
{{
    "subtopics": [
        "Historical Development and Origins",
        "Core Principles and Fundamentals",
        "Modern Applications and Use Cases",
        "Future Trends and Developments",
        "Challenges and Limitations"
    ]
}}

YOUR RESPONSE MUST BE A VALID JSON OBJECT WITH THE EXACT STRUCTURE SHOWN ABOVE.
DO NOT include any additional text, formatting, or explanations.
DO NOT use newlines within the JSON structure.

RESPONSE:"""

SUBTOPIC_SUMMARY_TEMPLATE = """Analyze and summarize the research findings for this topic.

TOPIC: {topic}
SEARCH_RESULTS: {results}

INSTRUCTIONS:
1. Analyze the search results thoroughly
2. Extract key information and insights
3. Organize findings logically
4. Include specific data points and facts
5. Cite sources for important claims

YOU MUST RESPOND WITH A VALID JSON OBJECT USING THIS EXACT STRUCTURE:
{{
    "summary": "Write your detailed summary here using markdown formatting",
    "sources": [
        "https://example1.com",
        "https://example2.com"
    ],
    "key_points": [
        "First key point goes here",
        "Second key point goes here"
    ]
}}

REQUIREMENTS:
1. The response must be ONLY the JSON object
2. No other text before or after the JSON
3. No markdown code blocks or formatting
4. All fields must be present
5. Sources must be valid URLs
6. Use proper JSON formatting with escaped quotes

EXAMPLE RESPONSE:
{{
    "summary": "## Topic Overview\\n\\nThis is a detailed summary of the findings...\\n\\n### Key Insights\\n1. First insight\\n2. Second insight",
    "sources": ["https://example.com/article1", "https://example.com/article2"],
    "key_points": ["Major finding 1", "Major finding 2"]
}}"""

FINAL_SYNTHESIS_TEMPLATE = """Create a comprehensive research overview based on all subtopic summaries.

TOPIC: {topic}
SUBTOPIC_SUMMARIES: {summaries}

INSTRUCTIONS:
1. Synthesize all subtopic findings into a cohesive overview
2. Identify common themes and patterns
3. Note significant discoveries
4. Address gaps and limitations
5. Draw meaningful conclusions

YOUR RESPONSE MUST BE A VALID JSON OBJECT WITH THIS EXACT STRUCTURE:
{{
    "overview": "## Research Overview\\n\\nComprehensive overview text here...\\n\\n### Key Themes\\n* Theme 1\\n* Theme 2\\n\\n### Significant Findings\\nDetailed findings here...",
    "key_findings": [
        "Major finding 1",
        "Major finding 2",
        "Major finding 3"
    ],
    "conclusions": "Final conclusions and implications of the research..."
}}

REQUIREMENTS:
1. The response must be ONLY the JSON object
2. No other text before or after the JSON
3. No markdown code blocks or formatting
4. All three fields must be present
5. Use proper JSON formatting with escaped quotes
6. The overview field should use markdown formatting

RESPONSE:"""

def sanitize_filename(name):
    """Convert a string to a valid filename."""
    import re
    # Replace invalid characters with underscore
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    # Remove leading/trailing spaces and dots
    name = name.strip('. ')
    # Limit length
    if len(name) > 100:
        name = name[:97] + '...'
    return name

def create_research_directory(base_path, topic):
    """Create and return the research directory structure."""
    from datetime import datetime
    
    # Create main research directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    topic_name = sanitize_filename(topic)
    research_dir = os.path.join(base_path, f"{topic_name}_{timestamp}")
    os.makedirs(research_dir, exist_ok=True)
    
    # Create subdirectories
    subtopics_dir = os.path.join(research_dir, "subtopics")
    os.makedirs(subtopics_dir, exist_ok=True)
    
    return research_dir, subtopics_dir

def write_markdown_file(filepath, content):
    """Write content to a markdown file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

async def research_subtopic(subtopic, search_tool, synthesis_chain):
    """Conduct research on a specific subtopic."""
    try:
        # Perform web search
        search_results = search_tool.run(subtopic)
        
        # Debug search results
        debug_container = st.empty()
        debug_container.text(f"Searching for: {subtopic}")
        
        # Synthesize findings
        synthesis_result = await synthesis_chain.arun(
            topic=subtopic,
            results=search_results
        )
        
        # Clean up the JSON string
        synthesis_result = synthesis_result.strip()
        if synthesis_result.startswith('```json'):
            synthesis_result = synthesis_result[7:]
        if synthesis_result.endswith('```'):
            synthesis_result = synthesis_result[:-3]
        synthesis_result = synthesis_result.strip()
        
        try:
            # Parse and validate results
            result_dict = json.loads(synthesis_result)
            
            # Validate required keys and types
            if not isinstance(result_dict, dict):
                raise ValueError("Response is not a JSON object")
            
            required_keys = {
                'summary': str,
                'sources': list,
                'key_points': list
            }
            
            for key, expected_type in required_keys.items():
                if key not in result_dict:
                    raise ValueError(f"Missing required key: {key}")
                if not isinstance(result_dict[key], expected_type):
                    raise ValueError(f"Invalid type for {key}: expected {expected_type.__name__}, got {type(result_dict[key]).__name__}")
            
            # Validate sources are URLs
            for source in result_dict['sources']:
                if not isinstance(source, str) or not (source.startswith('http://') or source.startswith('https://')):
                    raise ValueError(f"Invalid source URL: {source}")
            
            # Clean up the debug container
            debug_container.empty()
            
            return result_dict
            
        except json.JSONDecodeError as e:
            st.error(f"Error parsing JSON response for subtopic '{subtopic}': {str(e)}")
            st.error("Raw response:")
            st.code(synthesis_result)
            return None
        except ValueError as e:
            st.error(f"Invalid response format for subtopic '{subtopic}': {str(e)}")
            st.error("Raw response:")
            st.code(synthesis_result)
            return None
        
    except Exception as e:
        st.error(f"Error researching subtopic '{subtopic}': {str(e)}")
        return None

async def conduct_research(topic):
    """Conducts structured research on the given topic."""
    try:
        # Initialize progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        debug_container = st.empty()
        
        # Initialize LLM chains
        llm = OllamaLLM(
            model=st.session_state.ollama_model,
            temperature=0.1
        )
        
        subtopics_chain = LLMChain(
            llm=llm,
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
        
        # Initialize search tool
        if search_provider == "DuckDuckGo":
            search_tool = DuckDuckGoSearchRun()
        elif search_provider == "Google":
            if not os.getenv("GOOGLE_API_KEY") or not os.getenv("GOOGLE_CSE_ID"):
                st.error("Google Search requires API Key and Custom Search Engine ID")
                return
            google = GoogleSearchAPIWrapper()
            search_tool = Tool(
                name="Google Search",
                description="Search Google for recent results.",
                func=google.run
            )
        elif search_provider == "Brave":
            if not os.getenv("BRAVE_API_KEY"):
                st.error("Brave Search requires an API key")
                return
            async def brave_search(query):
                import aiohttp
                headers = {
                    "X-Subscription-Token": os.getenv("BRAVE_API_KEY"),
                    "Accept": "application/json",
                }
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"https://api.search.brave.com/res/v1/web/search",
                        headers=headers,
                        params={"q": query}
                    ) as response:
                        if response.status != 200:
                            raise Exception(f"Brave Search API error: {response.status}")
                        data = await response.json()
                        results = []
                        for web in data.get("web", {}).get("results", []):
                            results.append(f"Title: {web['title']}\nURL: {web['url']}\nDescription: {web['description']}\n")
                        return "\n".join(results)
            search_tool = Tool(
                name="Brave Search",
                description="Search Brave for recent results.",
                func=lambda x: asyncio.get_event_loop().run_until_complete(brave_search(x))
            )
        
        # Step 1: Create research directory structure
        if not output_folder:
            st.error("Please specify an output folder path")
            return
            
        research_dir, subtopics_dir = create_research_directory(output_folder, topic)
        status_text.text("Created research directory structure...")
        progress_bar.progress(0.1)
        
        # Step 2: Generate subtopics
        status_text.text("Generating research subtopics...")
        try:
            subtopics_result = await subtopics_chain.arun(topic=topic)
            # Clean up the JSON string
            subtopics_result = subtopics_result.strip()
            if subtopics_result.startswith('```json'):
                subtopics_result = subtopics_result[7:]
            if subtopics_result.endswith('```'):
                subtopics_result = subtopics_result[:-3]
            subtopics_result = subtopics_result.strip()
            
            subtopics_data = json.loads(subtopics_result)
            if not isinstance(subtopics_data, dict) or 'subtopics' not in subtopics_data:
                raise ValueError("Invalid subtopics format. Expected a dictionary with 'subtopics' key.")
            
            subtopics = subtopics_data["subtopics"]
            if not isinstance(subtopics, list) or len(subtopics) == 0:
                raise ValueError("No valid subtopics generated.")
            
            debug_container.text(f"Generated {len(subtopics)} subtopics")
            
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
        
        progress_bar.progress(0.2)
        
        # Step 3: Research each subtopic
        all_summaries = []
        for idx, subtopic in enumerate(subtopics, 1):
            status_text.text(f"Researching subtopic {idx}/{len(subtopics)}: {subtopic}")
            
            # Research the subtopic
            result = await research_subtopic(subtopic, search_tool, synthesis_chain)
            if result:
                # Create subtopic file
                subtopic_file = os.path.join(subtopics_dir, f"{sanitize_filename(subtopic)}.md")
                content = f"# {subtopic}\n\n{result['summary']}\n\n## Sources\n"
                for source in result['sources']:
                    content += f"- {source}\n"
                content += "\n## Key Points\n"
                for point in result['key_points']:
                    content += f"- {point}\n"
                
                write_markdown_file(subtopic_file, content)
                all_summaries.append(result)
            
            progress = 0.2 + (0.6 * (idx / len(subtopics)))
            progress_bar.progress(progress)
        
        # Step 4: Create final synthesis
        status_text.text("Creating final research synthesis...")
        try:
            final_result = await final_synthesis_chain.arun(
                topic=topic,
                summaries=json.dumps([s['summary'] for s in all_summaries])
            )
            
            # Clean up the JSON string
            final_result = final_result.strip()
            if final_result.startswith('```json'):
                final_result = final_result[7:]
            if final_result.endswith('```'):
                final_result = final_result[:-3]
            final_result = final_result.strip()
            
            try:
                final_data = json.loads(final_result)
                
                # Validate required keys and types
                required_keys = {
                    'overview': str,
                    'key_findings': list,
                    'conclusions': str
                }
                
                for key, expected_type in required_keys.items():
                    if key not in final_data:
                        raise ValueError(f"Missing required key: {key}")
                    if not isinstance(final_data[key], expected_type):
                        raise ValueError(f"Invalid type for {key}: expected {expected_type.__name__}, got {type(final_data[key]).__name__}")
                
                # Write final overview
                overview_file = os.path.join(research_dir, "research_overview.md")
                overview_content = f"# Research Overview: {topic}\n\n"
                overview_content += final_data['overview'] + "\n\n"
                overview_content += "## Key Findings\n\n"
                for finding in final_data['key_findings']:
                    overview_content += f"- {finding}\n"
                overview_content += "\n## Conclusions\n\n"
                overview_content += final_data['conclusions']
                
                write_markdown_file(overview_file, overview_content)
                
                # Update session state with results
                st.session_state.research_summary = overview_content
                st.session_state.sources = [source for result in all_summaries for source in result['sources']]
                
            except json.JSONDecodeError as e:
                st.error(f"Error parsing final synthesis JSON: {str(e)}")
                st.error("Raw response:")
                st.code(final_result)
                return
            except ValueError as e:
                st.error(f"Invalid final synthesis format: {str(e)}")
                st.error("Raw response:")
                st.code(final_result)
                return
                
        except Exception as e:
            st.error(f"Error creating final synthesis: {str(e)}")
            return
        
        progress_bar.progress(1.0)
        status_text.text("Research complete!")
        debug_container.empty()
        
        # Show success message with directory location
        st.success(f"Research completed successfully! Results saved to: {research_dir}")
        
    except Exception as e:
        st.error(f"Research error: {str(e)}")
    finally:
        st.session_state.research_in_progress = False

# Main research interface
research_topic = st.text_input(
    "Enter Research Topic",
    placeholder="Enter any topic you want to research...",
    disabled=st.session_state.research_in_progress
)

if st.button("Start Research", disabled=st.session_state.research_in_progress):
    if not research_topic:
        st.warning("Please enter a research topic.")
    elif not hasattr(st.session_state, 'ollama_model') or not st.session_state.ollama_model:
        st.error("Please select an Ollama model in the RAG Configuration page first.")
    elif not output_folder:
        st.error("Please specify an output folder path for saving research results.")
    else:
        st.session_state.research_in_progress = True
        st.session_state.research_summary = ""
        st.session_state.sources = []
        st.session_state.iteration_count = 0
        asyncio.run(conduct_research(research_topic))

# Display research results
if st.session_state.research_summary:
    st.header("Research Results")
    st.markdown(f"**Completed {st.session_state.iteration_count} research iterations**")
    
    # Save results if enabled
    if output_folder and output_folder:
        try:
            import os
            from datetime import datetime
            
            # Create output folder if it doesn't exist
            os.makedirs(output_folder, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"research_results_{timestamp}.md"
            filepath = os.path.join(output_folder, filename)
            
            # Prepare content
            content = f"# Research Results: {research_topic}\n"
            content += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            content += st.session_state.research_summary
            
            # Add sources
            content += "\n\n## Sources\n"
            web_sources = [source for source in st.session_state.sources if source.startswith(('http://', 'https://'))]
            for idx, source in enumerate(web_sources, 1):
                content += f"{idx}. {source}\n"
            
            # Write to file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            st.success(f"Research results saved to: {filepath}")
            
        except Exception as e:
            st.error(f"Error saving research results: {str(e)}")
    
    with st.expander("View Research Summary", expanded=True):
        st.markdown(st.session_state.research_summary)
    
    if st.session_state.sources:
        with st.expander("View Sources", expanded=False):
            st.markdown("### Sources Used")
            # Filter and display only valid web URLs
            web_sources = [source for source in st.session_state.sources if source.startswith(('http://', 'https://'))]
            if web_sources:
                for idx, source in enumerate(web_sources, 1):
                    # Clean the URL and ensure it's properly formatted
                    clean_url = source.strip().rstrip('.')
                    st.markdown(f"{idx}. [{clean_url}]({clean_url})")
            else:
                st.info("No web sources were found in the search results. This might happen if:\n- The search didn't return any valid URLs\n- The sources weren't properly extracted from the search results\n- The search API is temporarily unavailable") 