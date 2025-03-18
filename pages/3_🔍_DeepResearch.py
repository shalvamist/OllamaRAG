import streamlit as st
from langchain_ollama import OllamaLLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.tools import DuckDuckGoSearchRun
import asyncio

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
    
    # Search Provider Selection
    search_provider = st.selectbox(
        "Search Provider",
        ["DuckDuckGo", "Google", "Brave"],
        help="Select the web search provider"
    )
    
    max_iterations = st.slider(
        "Maximum Research Iterations",
        min_value=1,
        max_value=5,
        value=3,
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
PREVIOUS SUMMARY: {previous_summary}

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

YOUR RESPONSE MUST BE A SINGLE JSON OBJECT WITH THIS EXACT STRUCTURE:
{{
    "summary": "## Research Summary\\n\\n### Overview\\nBrief introduction to the topic\\n\\n### Key Findings\\n* Important point 1\\n* Important point 2\\n* Important point 3\\n\\n### Detailed Analysis\\nIn-depth discussion of findings\\n\\n### Conclusions\\nMain takeaways and implications",
    "gaps": [
        "Specific unanswered question 1",
        "Specific unanswered question 2"
    ],
    "sources": [
        "Source 1 with URL or reference",
        "Source 2 with URL or reference"
    ]
}}

CRITICAL FORMATTING RULES:
1. The response must be a SINGLE valid JSON object
2. The "summary" field must be a STRING containing markdown text
3. All newlines must use \\n escape sequence
4. All quotes must be escaped with backslash: \\"
5. Do not use single quotes
6. Do not include any text before or after the JSON object
7. Do not include any markdown code block markers

EXAMPLE OF CORRECT JSON RESPONSE:
{{
    "summary": "## Research Summary\\n\\n### Overview\\nQuantum computing represents a revolutionary approach to computation.\\n\\n### Key Findings\\n* Current quantum computers maintain coherence for up to 100 microseconds\\n* IBM's latest processor achieves quantum volume of 128\\n\\n### Detailed Analysis\\nRecent advances in quantum computing...\\n\\n### Conclusions\\nSignificant progress but challenges remain",
    "gaps": ["What is the impact on current cryptography?", "How scalable are current approaches?"],
    "sources": ["IBM Quantum Computing Report 2024", "Nature Physics Journal Article"]
}}

REMEMBER:
1. The summary must be a properly escaped string
2. All sections must be present and properly formatted
3. Use only the specified markdown formatting
4. Write in an objective, academic tone
5. Include specific facts and figures
6. Cite sources for major claims

Generate your JSON response:"""

async def conduct_research(topic):
    """Conducts iterative web research on the given topic."""
    try:
        # Initialize search tool based on selected provider
        if search_provider == "DuckDuckGo":
            search_tool = DuckDuckGoSearchRun()
        elif search_provider == "Google":
            st.error("Google Search API requires API key configuration")
            return
        elif search_provider == "Brave":
            st.error("Brave Search API requires API key configuration")
            return
        
        # Initialize LLM with different parameters for query and synthesis
        query_llm = OllamaLLM(
            model=st.session_state.ollama_model,
            temperature=0.3,  # Higher temperature for more creative queries
            stop=["\n", "```"]  # Stop at newlines or code blocks
        )
        
        synthesis_llm = OllamaLLM(
            model=st.session_state.ollama_model,
            temperature=0.1,  # Lower temperature for consistent JSON
            stop=["}}}"],
            format="json"
        )
        
        query_chain = LLMChain(
            llm=query_llm,
            prompt=PromptTemplate(
                template=SEARCH_QUERY_TEMPLATE,
                input_variables=["topic", "iteration", "gaps"]
            )
        )
        
        synthesis_chain = LLMChain(
            llm=synthesis_llm,
            prompt=PromptTemplate(
                template=SYNTHESIS_TEMPLATE,
                input_variables=["topic", "results", "previous_summary"]
            )
        )
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        debug_container = st.empty()
        
        gaps = []
        previous_summary = ""
        
        def clean_json_string(text):
            """Clean and validate JSON string."""
            # Remove any non-JSON content
            text = text.strip()
            start_idx = text.find('{')
            end_idx = text.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                text = text[start_idx:end_idx]
            
            # Remove markdown code blocks
            if text.endswith('```'):
                text = text[:-3]
            if text.startswith('```json'):
                text = text[7:]
            if text.startswith('```'):
                text = text[3:]
            
            return text.strip()

        def validate_research_output(result_dict):
            """Validate the research output structure and types."""
            # Check required keys
            required_keys = ['summary', 'gaps', 'sources']
            if not all(key in result_dict for key in required_keys):
                raise ValueError("Missing required keys in JSON response")
            
            # Validate types
            if not isinstance(result_dict['summary'], str):
                raise ValueError(f"Summary must be a string, got {type(result_dict['summary'])}")
            if not isinstance(result_dict['gaps'], list):
                raise ValueError(f"Gaps must be a list, got {type(result_dict['gaps'])}")
            if not isinstance(result_dict['sources'], list):
                raise ValueError(f"Sources must be a list, got {type(result_dict['sources'])}")
            
            # Validate summary structure
            required_sections = [
                "## Research Summary",
                "### Overview",
                "### Key Findings",
                "### Detailed Analysis",
                "### Conclusions"
            ]
            for section in required_sections:
                if section not in result_dict['summary']:
                    raise ValueError(f"Summary missing required section: {section}")
            
            return True

        for iteration in range(max_iterations):
            st.session_state.iteration_count = iteration + 1
            progress = (iteration) / max_iterations
            progress_bar.progress(progress)
            
            try:
                # Generate search query
                status_text.text(f"Iteration {iteration + 1}: Generating search query...")
                query = await query_chain.arun(
                    topic=topic,
                    iteration=iteration,
                    gaps=str(gaps) if gaps else "none"
                )
                
                # Clean up the query
                query = query.strip().strip('"').strip("'").strip("{").strip("}")
                if len(query) > 150:
                    query = query[:150]
                
                # Validate query
                if not query or len(query.strip()) < 3 or "{" in query or "}" in query:
                    raise ValueError("Invalid search query generated")
                
                # Show query for debugging
                debug_container.text(f"Search query: {query}")
                
                # Perform web search with error handling
                status_text.text(f"Iteration {iteration + 1}: Searching the web...")
                try:
                    search_results = search_tool.run(query)
                    if not search_results or len(search_results.strip()) < 50:  # Require more substantial results
                        raise Exception("Insufficient search results returned")
                    
                    # Show first bit of results for debugging
                    debug_container.text(f"Results preview: {search_results[:200]}...")
                    
                except Exception as search_error:
                    st.warning(f"Search failed for query: {query}")
                    st.error(f"Search error: {str(search_error)}")
                    continue
                
                # Synthesize results
                status_text.text(f"Iteration {iteration + 1}: Synthesizing findings...")
                synthesis_result = await synthesis_chain.arun(
                    topic=topic,
                    results=search_results,
                    previous_summary=previous_summary
                )
                
                # Parse results and update state
                try:
                    # Clean up the JSON string
                    cleaned_json = clean_json_string(synthesis_result)
                    debug_container.code(cleaned_json, language="json")
                    
                    # Parse JSON
                    import json
                    result_dict = json.loads(cleaned_json)
                    
                    # Validate structure and types
                    validate_research_output(result_dict)
                    
                    # Update state
                    previous_summary = result_dict['summary']
                    gaps = result_dict['gaps']
                    st.session_state.sources.extend(result_dict['sources'])
                    st.session_state.research_summary = previous_summary
                    
                except (json.JSONDecodeError, ValueError) as json_error:
                    st.error(f"Error parsing research results: {str(json_error)}")
                    debug_container.code(synthesis_result)  # Show raw output for debugging
                    if iteration == 0:
                        # Create a properly structured initial summary
                        st.session_state.research_summary = f"""## Research Summary

### Overview
Initial research on {topic}...

### Key Findings
* More research needed to establish key findings
* Initial search results being analyzed

### Detailed Analysis
Preliminary analysis in progress...

### Conclusions
Early investigation phase, conclusions pending further research."""
                    continue
                    
            except Exception as iteration_error:
                st.error(f"Error in iteration {iteration + 1}: {str(iteration_error)}")
                continue
        
        progress_bar.progress(1.0)
        status_text.text("Research complete!")
        debug_container.empty()
        
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
    
    with st.expander("View Research Summary", expanded=True):
        st.markdown(st.session_state.research_summary)
    
    if st.session_state.sources:
        with st.expander("View Sources", expanded=False):
            for idx, source in enumerate(st.session_state.sources, 1):
                st.markdown(f"{idx}. {source}") 