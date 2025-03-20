import streamlit as st
from langchain_ollama import OllamaLLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.retrievers import WikipediaRetriever

import os
import asyncio
import json
import re
from urllib.parse import urlparse
from bs4 import SoupStrainer

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
if 'max_search_attempts' not in st.session_state:
    st.session_state.max_search_attempts = 3
if 'num_subtopics' not in st.session_state:
    st.session_state.num_subtopics = 3

# Title and description
st.title("🧠 Your AI Research Assistant with ADHD ")
st.markdown("""
Ever wished you could clone yourself to research multiple topics at once? Well, this is *almost* as good! 

This turbocharged research buddy will:
1. 🎯 Break down your topic into bite-sized subtopics (because who doesn't love a good outline?)
2. 🌐 Hunt for information across multiple sources:
   - 🦆 DuckDuckGo (for when you need the whole web)
   - 📰 News Search (for the fresh hot takes)
   - 📚 Wikipedia (for those "actually..." moments)
3. 🤖 Read and analyze EVERYTHING (so you don't have to)
4. 🧪 Filter out the fluff (no more cat videos... unless you're researching cats)
5. ✍️ Write a comprehensive report that would make your professor proud
6. 📝 Cite all sources (because we're not savages)

Just pick your research sources, set the number of subtopics, and watch as your AI assistant goes down multiple research rabbit holes simultaneously! 🕳️🐇

*Note: No AI assistants were harmed in the making of this research tool, though some did get slightly overwhelmed with excitement!* 
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
    
    # Search Provider Configuration
    st.subheader("Search Providers")
    use_duckduckgo = st.checkbox("DuckDuckGo Web Search", value=True)
    use_duckduckgo_news = st.checkbox("DuckDuckGo News Search", value=True)
    use_wikipedia = st.checkbox("Wikipedia", value=True)
    
    if not any([use_duckduckgo, use_duckduckgo_news, use_wikipedia]):
        st.warning("⚠️ Please enable at least one search provider")
    
    st.session_state.max_iterations = st.slider(
        "Maximum Research Iterations",
        min_value=1,
        max_value=20,
        value=5,
        help="Maximum number of research cycles to perform"
    )

    st.session_state.num_subtopics = st.slider(
        "Number of Subtopics",
        min_value=1,
        max_value=20,
        value=3,
        help="Number of subtopics to generate"
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

MAIN TOPIC: {main_topic}
SUBTOPIC: {topic}

INSTRUCTIONS:
1. Create a search query that combines the main topic and subtopic context
2. Return ONLY the search query as plain text
3. NO JSON, NO explanations, NO formatting
4. Keep it under 15 words
5. Make it specific and focused
6. Include both main topic and subtopic aspects
7. Do not use quotes or special characters

BAD EXAMPLES:
- {{\"query\": \"quantum computing basics\"}}
- "quantum computing applications"
- <thinking>Let me generate...</thinking>

GOOD EXAMPLES:
- quantum computing cryptography applications in cybersecurity
- artificial intelligence machine learning applications healthcare diagnosis
- renewable energy solar power efficiency improvements residential systems

YOUR QUERY:"""

SUBTOPICS_TEMPLATE = """Generate a list of comprehensive subtopics for detailed research on the given topic.

TOPIC: {topic}

INSTRUCTIONS:
1. Analyze the topic and break it down into logical subtopics
2. Each subtopic should be specific and focused
3. Include both fundamental and advanced aspects
4. Ensure coverage is comprehensive
5. Each subtopic should be a clear, concise phrase

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

YOU MUST RESPOND WITH A THIS EXACT STRUCTURE:
"summary": 
"Write your detailed summary here using markdown formatting",
"key_points":
"First key point goes here"
        "Second key point goes here"

REQUIREMENTS:
1. No markdown code blocks or formatting
2. All fields must be present


EXAMPLE RESPONSE:
"summary": 
"## Topic Overview\\n\\nThis is a detailed summary of the findings...\\n\\n### Key Insights\\n1. First insight\\n2. Second insight",
"key_points":
"Major finding 1"
"Major finding 2"
"""

FINAL_SYNTHESIS_TEMPLATE = """Create a comprehensive research overview based on all subtopic summaries.

TOPIC: {topic}
SUBTOPIC_SUMMARIES: {summaries}

INSTRUCTIONS:
1. Create a well-structured research report that includes:
   - Executive summary of the entire research
   - Key findings and insights across all subtopics
   - Detailed subtopic summaries
   - Final conclusions and recommendations
2. Use clear markdown formatting for readability
3. Reference specific subtopics when discussing findings
4. Maintain clear connection between findings and their sources
5. Include practical implications and next steps

YOUR RESPONSE MUST follow this exact structure:
```markdown
# Research Report: [Topic]

## Executive Summary
[Provide a concise overview of the entire research, major themes, and key conclusions]

## Key Research Findings
[List 3-5 major findings that emerged across multiple subtopics]

## Detailed Subtopic Analysis
[For each subtopic, include:
- Summary of findings
- Key insights
- Notable data points
- Relevant sources]

## Research Implications
[Discuss the practical implications of the findings]

## Recommendations
[Provide actionable recommendations based on the research]

## Next Steps
[Suggest potential areas for further research or investigation]
```

REQUIREMENTS:
1. Use proper markdown formatting
2. Include ALL sections as shown above
3. Reference specific subtopics when discussing findings
4. Maintain clear connection between findings and their sources
5. Be concise but comprehensive
6. Focus on synthesis rather than repetition

RESPONSE:"""

SEARCH_RESULTS_EVALUATION_TEMPLATE = """Evaluate the quality and relevance of these search results.

TOPIC: {topic}
SEARCH_RESULTS: {results}

INSTRUCTIONS:
1. Analyze the provided content for:
   - Relevance to the topic
   - Information density
   - Quality of sources
   - Comprehensiveness
2. Return ONLY a JSON object with your evaluation
3. If the results are not relevant or comprehensive, return false for is_sufficient
4. If the results are relevant and comprehensive, return true for is_sufficient
5. Rate the quality of the results from 0-10
6. Provide 2 reasons for your assessment

YOUR RESPONSE MUST BE A VALID JSON OBJECT WITH THIS EXACT STRUCTURE:
{{
    "sufficient_data": True/False,
    "quality_score": 0-10,
    "reasons": [
        "Reason 1 for assessment",
        "Reason 2 for assessment"
    ],
    "missing_aspects": [
        "Important aspect not covered 1",
        "Important aspect not covered 2"
    ]
}}

REQUIREMENTS:
1. is_sufficient: Set to true only if results are both relevant and comprehensive
2. quality_score: Rate from 0-10 where:
   - 0-3: Poor quality/irrelevant
   - 4-6: Moderate quality but incomplete
   - 7-10: High quality and comprehensive
3. Include at least 2 reasons for your assessment
4. List missing aspects if quality_score < 7

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

async def fetch_and_process_url(url: str, debug_container) -> str:
    """Fetch and process content from a URL using WebBaseLoader with BeautifulSoup strainer."""
    try:
        # Validate URL
        parsed_url = urlparse(url)
        if not all([parsed_url.scheme, parsed_url.netloc]):
            debug_container.warning(f"Invalid URL: {url}")
            return ""

        bs4_strainer = SoupStrainer(class_=("content-area"))
        
        # Initialize WebBaseLoader with BeautifulSoup configuration
        loader = WebBaseLoader(
            [url]
        )

        try:
            # Use synchronous loading as aload() might not work well with bs_kwargs
            docs = loader.load()
            if not docs:
                # Fallback to loading without strainer if no content found
                loader = WebBaseLoader(
                    web_paths=[url],
                    verify_ssl=True,
                    continue_on_failure=True,
                    requests_per_second=2
                )
                docs = loader.load()
                if not docs:
                    debug_container.warning(f"No content retrieved from: {url}")
                    return ""

            # Combine all document content
            content = "\n\n".join(doc.page_content for doc in docs)
            
            # Clean up the content
            content = re.sub(r'\s+', ' ', content).strip()  # Remove excessive whitespace
            content = re.sub(r'[^\x00-\x7F]+', '', content)  # Remove non-ASCII characters

            return content

        except Exception as e:
            debug_container.error(f"Error loading content from {url}: {str(e)}")
            return ""

    except Exception as e:
        debug_container.error(f"Error processing URL {url}: {str(e)}")
        return ""

def clean_json_string(json_str: str) -> str:
    """Clean a JSON string by removing invalid control characters and normalizing whitespace."""
    try:
        # First try to parse it as is (in case it's already valid JSON)
        json.loads(json_str)
        return json_str
    except:
        # If parsing fails, clean up the string
        # Remove JSON code block markers
        json_str = re.sub(r'^```json\s*|\s*```$', '', json_str.strip())
        
        # Handle newlines in the content while preserving them in strings
        json_str = re.sub(r'\\n', '__NEWLINE__', json_str)  # Temporarily replace valid \n
        json_str = re.sub(r'\n\s*', ' ', json_str)  # Remove actual newlines
        json_str = re.sub(r'__NEWLINE__', '\\n', json_str)  # Restore valid \n
        
        # Replace invalid control characters
        json_str = re.sub(r'[\x00-\x09\x0b\x0c\x0e-\x1F\x7F-\x9F]', '', json_str)
        
        # Normalize whitespace (but not inside strings)
        parts = []
        in_string = False
        current = []
        
        for char in json_str:
            if char == '"' and (not current or current[-1] != '\\'):
                in_string = not in_string
            
            if not in_string and char.isspace():
                if current and not current[-1].isspace():
                    current.append(' ')
            else:
                current.append(char)
        
        json_str = ''.join(current)
        
        # Ensure proper escaping of quotes
        json_str = json_str.replace('\\"', '__QUOTE__')
        json_str = json_str.replace('"', '\\"')
        json_str = json_str.replace('__QUOTE__', '\\"')
        
        # Remove any trailing commas before closing braces/brackets
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        return json_str.strip()

async def research_subtopic(subtopic, search_tools, synthesis_chain, main_topic, status_text):
    """Conduct research on a specific subtopic."""
    try:
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
            search_attempt += 1
            
            # Generate contextual search query
            search_query = (await search_query_chain.ainvoke({"main_topic": main_topic, "topic": subtopic}))["text"]
            
            # Debug search query
            status_text.text(f"Search attempt {search_attempt}/{max_search_attempts} for: {search_query}")
            
            detailed_results = []
            
            # Perform web searches based on enabled providers
            if search_tools.get('web'):
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
            if search_tools.get('news'):
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
            
            # Get Wikipedia results if enabled
            if search_tools.get('wikipedia'):
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

            for result in detailed_results:
                # Evaluate combined results
                try:
                    source_type = result['source'].capitalize()
                    status_text.text(f"Evaluating {source_type} content from {result['url']}")
                    evaluation_result = (await evaluation_chain.ainvoke({
                        "topic": subtopic,
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
                
        status_text.text(f"Synthesizing findings for {subtopic}")
        # Synthesize findings using combined results
        synthesis_result = (await synthesis_chain.ainvoke({
            "topic": subtopic,
            "results": summaries_subtopic
        }))["text"]
            
            # Clean up the debug container
        status_text.empty()
        
        return synthesis_result, combined_results  # Return both the synthesis and webpage contents
        
    except Exception as e:
        st.error(f"Error researching subtopic '{subtopic}': {str(e)}")
        return None, None

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
        # Initialize progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        subtopic_status = st.empty()
        debug_container = st.empty()
        
        # Initialize search tools based on user configuration
        search_tools = {}
        if use_duckduckgo:
            search_tools['web'] = DuckDuckGoSearchResults()
        if use_duckduckgo_news:
            search_tools['news'] = DuckDuckGoSearchResults(backend="news")
        if use_wikipedia:
            search_tools['wikipedia'] = WikipediaRetriever()
            
        if not search_tools:
            st.error("Please enable at least one search provider to proceed")
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
        if not output_folder:
            st.error("Please specify an output folder path")
            return
            
        research_dir, subtopics_dir = create_research_directory(output_folder, topic)
        status_text.text("Created research directory structure...")
        progress_bar.progress(0.1)
        
        # Step 2: Generate subtopics
        status_text.text("Generating research subtopics...")
        try:
            subtopics_result = (await subtopics_chain.ainvoke({"topic": topic, "num_subtopics": st.session_state.num_subtopics}))["text"]
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
            
            status_text.text(f"Generated {len(subtopics)} subtopics")
            
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
        all_sources = []  # Track all sources across subtopics
        for idx, subtopic in enumerate(subtopics[:st.session_state.num_subtopics], 1):
            status_text.text(f"Researching subtopic {idx}/{len(subtopics)}: {subtopic}")
            
            # Research the subtopic with main topic context
            result, webpage_contents = await research_subtopic(subtopic, search_tools, synthesis_chain, topic, subtopic_status)

            if result:
                # Create subtopic file
                subtopic_file = os.path.join(subtopics_dir, f"{sanitize_filename(subtopic)}.md")
                content = f"# {subtopic}\n\n"
                
                # Add detailed webpage content first
                content += "## Webpage Contents\n\n"
                for webpage in webpage_contents:
                    content += f"### URL - {webpage['url']}\n\n"
                    all_sources.append(webpage['url'])
                    content += "#### Content\n"
                    content += f"```\n{webpage['content']}\n```\n\n"
                
                # Add subtopic summary
                content += "## Subtopic Summary\n\n"
                content += f"{result}\n\n"
                
                # Add sources and key points
                content += "## Sources\n"
                content += f"{all_sources}\n\n"
                
                write_markdown_file(subtopic_file, content)
                all_summaries.append({
                    "subtopic": subtopic,
                    "summary": result,
                })
            
            progress = 0.2 + (0.6 * (idx / len(subtopics)))
            progress_bar.progress(progress)
        
        # Step 4: Create final synthesis
        status_text.text("Creating final research synthesis...")
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
        
        progress_bar.progress(1.0)
        status_text.text("Research complete!")
        debug_container.empty()
        
        # Show success message with directory location
        st.success(f"Research completed successfully! Results saved to: {research_dir}")
        
    except Exception as e:
        st.error(f"Research Error: {str(e)}")
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