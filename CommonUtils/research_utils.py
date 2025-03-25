import os
import re
import json
from urllib.parse import urlparse
from bs4 import SoupStrainer
from langchain_community.document_loaders import WebBaseLoader

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

SUBTOPIC_SUMMARY_TEMPLATE = """
Analyze and summarize the research findings for this topic. 
This should be a detailed summary of the findings aim to be 4-5 paragraphs (4-5 sentences per paragraph).

TOPIC: {topic}
SEARCH_RESULTS: {results}

INSTRUCTIONS:
1. Analyze the search results thoroughly
2. Extract key information and insights
3. Organize findings logically
4. Include specific data points and facts
5. Cite sources for important claims

REQUIREMENTS:
1. No markdown code blocks or formatting
2. All fields must be present

EXAMPLE RESPONSE:
"
## <Topic> Overview
This is a detailed summary of the findings... - should be 4-5 paragraphs

### Key Insights
First insight - should be 4-5 sentences
Second insight - should be 4-5 sentences
...

### Important Insights:
"Major finding 1"
"Major finding 2"
...

### Sources:
- Source 1
- Source 2
...
"""

FINAL_SYNTHESIS_TEMPLATE = """
Analyze and summarize the research findings for this topic, Create a comprehensive research overview based on all subtopic summaries.
This should be a detailed summary of the findings aim to be 4-5 paragraphs (4-5 sentences per paragraph).

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
[Provide a concise overview of the entire research, major themes, and key conclusions - should be 4-5 paragraphs & 3-4 sentences per paragraph]

## Key Research Findings
[List 3-5 major findings that emerged across multiple subtopics - should be 4-5 paragraphs & 3-4 sentences per paragraph]

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
        # Check if research has been stopped (access session state)
        import streamlit as st
        if 'stop_requested' in st.session_state and st.session_state.stop_requested:
            debug_container.warning(f"Skipping URL processing - research stopped: {url}")
            return ""
            
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
            # Check again if research has been stopped before potentially lengthy operation
            if 'stop_requested' in st.session_state and st.session_state.stop_requested:
                debug_container.warning(f"Skipping content loading - research stopped: {url}")
                return ""
                
            # Use synchronous loading as aload() might not work well with bs_kwargs
            docs = loader.load()
            
            # Check again if research has been stopped after loading
            if 'stop_requested' in st.session_state and st.session_state.stop_requested:
                debug_container.warning(f"Discarding loaded content - research stopped: {url}")
                return ""
                
            if not docs:
                # Fallback to loading without strainer if no content found
                loader = WebBaseLoader(
                    web_paths=[url],
                    verify_ssl=True,
                    continue_on_failure=True,
                    requests_per_second=2
                )
                
                # Check again if research has been stopped before retrying load
                if 'stop_requested' in st.session_state and st.session_state.stop_requested:
                    debug_container.warning(f"Skipping content retry - research stopped: {url}")
                    return ""
                    
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