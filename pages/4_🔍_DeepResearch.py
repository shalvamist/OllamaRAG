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

# Import utility functions
from CommonUtils.research_utils import (
    sanitize_filename,
    create_research_directory,
    write_markdown_file,
    fetch_and_process_url,
    clean_json_string
)

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
- ✅ Searching multiple sources simultaneously (Web, News, Wikipedia)
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
        default_output = os.path.join(os.getcwd(), "research_results")
        output_folder = st.text_input(
            "Output Location",
            value=default_output,
            placeholder="Enter folder path...",
            help="Specify where to save all research documents and reports"
        )
        if not output_folder:
            st.error("⚠️ Output folder path is required")
    
    # Search Provider Configuration - Collapsable
    with st.expander("🔎 Data Sources", expanded=False):
        use_duckduckgo = st.checkbox("Web Search (DuckDuckGo)", value=True, 
                                    help="Search the general web for information")
        use_duckduckgo_news = st.checkbox("News Search", value=True,
                                         help="Search recent news articles for timely information")
        use_wikipedia = st.checkbox("Wikipedia", value=True,
                                   help="Search Wikipedia for well-established information")
        
        if not any([use_duckduckgo, use_duckduckgo_news, use_wikipedia]):
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
            subtopics_result = clean_json_string(subtopics_result)
            
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

col1, col2 = st.columns([3, 1])
with col1:
    if st.button(
        "🚀 Start Research", 
        disabled=st.session_state.research_in_progress,
        use_container_width=True,
        type="primary"
    ):
        if not research_topic:
            st.warning("Please enter a research topic.")
        elif not hasattr(st.session_state, 'ollama_model') or not st.session_state.ollama_model:
            st.error("Please select an Ollama model in the Model Settings page first.")
        elif not output_folder:
            st.error("Please specify an output folder path for saving research results.")
        else:
            st.session_state.research_in_progress = True
            st.session_state.research_summary = ""
            st.session_state.sources = []
            st.session_state.iteration_count = 0
            asyncio.run(conduct_research(research_topic))

with col2:
    if st.button(
        "📋 Clear Results", 
        disabled=st.session_state.research_in_progress,
        use_container_width=True
    ):
        st.session_state.research_summary = ""
        st.session_state.sources = []
        st.session_state.iteration_count = 0
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
            # Filter and display only valid web URLs
            web_sources = [source for source in st.session_state.sources if source.startswith(('http://', 'https://'))]
            if web_sources:
                for idx, source in enumerate(web_sources, 1):
                    # Clean the URL and ensure it's properly formatted
                    clean_url = source.strip().rstrip('.')
                    st.markdown(f"{idx}. [{clean_url}]({clean_url})")
            else:
                st.info("No web sources were found in the search results.") 
        else:
            st.info("No sources are available for this research.")
    
    with tabs[2]:
        if output_folder and os.path.exists(output_folder):
            st.markdown("### Research Files")
            st.markdown(f"All research files are saved to: `{output_folder}`")
            
            if os.path.exists(os.path.join(output_folder, sanitize_filename(research_topic))):
                research_dir = os.path.join(output_folder, sanitize_filename(research_topic))
                st.success(f"Research directory created: {research_dir}")
                
                overview_file = os.path.join(research_dir, "research_overview.md")
                if os.path.exists(overview_file):
                    st.download_button(
                        label="Download Full Research Report",
                        data=open(overview_file, "r", encoding="utf-8").read(),
                        file_name=f"{sanitize_filename(research_topic)}_report.md",
                        mime="text/markdown",
                    )
            else:
                st.warning("Research directory not found. Files may not have been saved properly.")
        else:
            st.warning("Output folder not found or not specified.") 