import os
import re
import json
from langchain_community.document_loaders import WebBaseLoader
import streamlit as st

# Research prompt templates
SEARCH_QUERY_TEMPLATE = """
Generate a search query for web research.

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

SUBTOPICS_TEMPLATE = """
Generate a list of comprehensive subtopics for detailed research on the given topic.

TOPIC: {topic}
NUMBER OF SUBTOPICS: {num_subtopics}

INSTRUCTIONS:
1. Analyze the topic and break it down into EXACTLY {num_subtopics} logical subtopics
2. Each subtopic should be specific and focused
3. Include both fundamental and advanced aspects
4. Ensure coverage is comprehensive
5. Each subtopic should be a clear, concise phrase
6. Generate EXACTLY {num_subtopics} subtopics - no more, no less

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
ENSURE the "subtopics" array contains EXACTLY {num_subtopics} items.

RESPONSE:"""

SUBTOPIC_SUMMARY_TEMPLATE = """
Analyze and summarize the research findings for this topic. 
This should be a detailed summary of the findings aim to be 4-5 paragraphs (4-5 sentences per paragraph, ~150 words per paragraph).

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
This should be a detailed summary of the findings aim to be 4-5 paragraphs (5-7 sentences per paragraph, ~250 words per paragraph).

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
```
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

WEBSEARCH_QUERY_TEMPLATE = """
Generate a search query for web research.

TOPIC: {topic}

INSTRUCTIONS:
1. Create a search query that is relevant to the conversation topic
2. Return ONLY the search query as plain text
3. NO JSON, NO explanations, NO formatting
4. Keep it under 15 words
5. Make it specific and focused

BAD EXAMPLES:
- {{\"query\": \"quantum computing basics\"}}
- "quantum computing applications"
- <thinking>Let me generate...</thinking>   

GOOD EXAMPLES:
- quantum computing cryptography applications in cybersecurity
- artificial intelligence machine learning applications healthcare diagnosis
- renewable energy solar power efficiency improvements residential systems

YOUR QUERY:"""

SEARCH_RESULTS_EVALUATION_TEMPLATE = """
You are a search results evaluator. Your task is to evaluate the quality and correlation of search results.

TOPIC: {topic}
SEARCH_RESULTS: {results}

INSTRUCTIONS:
1. Analyze the content carefully
2. Return ONLY a valid JSON object
3. Follow the exact structure below
4. Do not include any additional text or explanations
5. Ensure all JSON values are properly quoted
6. Use only double quotes for JSON properties

REQUIRED JSON STRUCTURE:
{{
    "sufficient_data": true,
    "quality_score": 8,
    "confidence_score": 7,
    "source_correlation": {{
        "high_correlation": [
            "Fact 1 that appears in multiple sources",
            "Fact 2 that appears in multiple sources"
        ],
        "partial_correlation": [
            "Fact with partial support"
        ],
        "uncorroborated": [
            "Single source fact"
        ]
    }},
    "source_credibility": {{
        "high_credibility": [
            "example.com"
        ],
        "medium_credibility": [
            "example.org"
        ],
        "low_credibility": [
            "example.net"
        ]
    }},
    "reasons": [
        "Reason 1 for assessment",
        "Reason 2 for assessment"
    ],
    "missing_aspects": [
        "Missing aspect 1",
        "Missing aspect 2"
    ],
    "answer": "Answer to the query based on the search results"
}}

RESPONSE (VALID JSON ONLY):"""

# Thinking Section Template - Used for displaying thinking sections in the UI
THINKING_SECTION_TEMPLATE = '''<details class="thinking-details">
    <summary>💭 Thinking Process</summary>
    <div class="thinking-content">
        {content}
    </div>
</details>'''

# CSS for thinking sections - Can be included in page styling
THINKING_CSS = '''
/* Thinking process container styling */
.thinking-details {
    margin: 0.75em 0;
    padding: 0.25em;
    border-radius: 4px;
    background-color: #f5f7f9;
    border: 1px solid #e0e0e0;
}

.thinking-content {
    margin: 0.5em 0;
    padding: 0.75em;
    background-color: #f8f9fa;
    border-radius: 4px;
    border-left: 3px solid #1E88E5;
}

.thinking-text {
    white-space: pre-wrap;
    word-wrap: break-word;
    overflow-wrap: break-word;
    font-family: ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, Liberation Mono, monospace;
    font-size: 0.9em;
    line-height: 1.4;
    color: #333;
}

details > summary {
    cursor: pointer;
    padding: 0.3em 0.5em;
    border-radius: 4px;
    font-weight: 500;
    color: #1E88E5;
    font-size: 0.9em;
}

details > summary:hover {
    background-color: #e9ecef;
}

details[open] > summary {
    margin-bottom: 0.4em;
    border-bottom: 1px solid #eaecef;
}'''

def format_thinking_content(content):
    """Format thinking content with markdown code block and monospace styling."""
    # Clean up the content and ensure proper formatting
    content = content.strip()
    # Wrap in monospace styling without explicit code blocks
    return f'<div class="thinking-text">{content}</div>'

def clean_thinking_tags(content):
    """
    Clean <think> tags from content when we need just the final output.
    Returns the content without any thinking tags or thinking content.
    """
    if not content:
        return ""
        
    if '<think>' in content:
        # If thinking tags are present, extract only the final answer after thinking
        if '</think>' in content:
            # Get everything after the </think> tag
            content = content.split('</think>')[-1].strip()
        else:
            # If closing tag is missing, just take the part before the thinking started
            content = content.split('<think>')[0].strip()
    return content

def parse_thinking_content(content):
    """
    Parse content to handle thinking sections from model output.
    Returns formatted content with thinking sections displayed as collapsible details.
    """
    if not content or ('<think>' not in content or '</think>' not in content):
        return content
        
    try:
        parts = content.split('</think>')
        if len(parts) > 1:
            pre_think = parts[0].split('<think>')[0].strip()
            post_think = parts[1].strip()
            thinking_content = parts[0].split('<think>')[1].strip()
            
            # Format thinking section
            thinking_section = THINKING_SECTION_TEMPLATE.format(
                content=format_thinking_content(thinking_content)
            )

            # Combine parts with proper spacing
            if pre_think and post_think:
                return f"{pre_think}\n\n{thinking_section}\n\n{post_think}"
            elif pre_think:
                return f"{pre_think}\n\n{thinking_section}"
            elif post_think:
                return f"{thinking_section}\n\n{post_think}"
            else:
                return thinking_section
    except Exception as e:
        # If parsing fails, return original content
        import streamlit as st
        st.error(f"Error parsing thinking content: {str(e)}")
        return content
        
    return content

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

# Function to write markdown file - wrapper to handle thinking sections correctly
def write_markdown_file_with_thinking(file_path, content):
    """Wrapper around write_markdown_with_thinking to maintain backward compatibility."""
    return write_markdown_with_thinking(file_path, content)

def write_markdown_with_thinking(file_path, content):
    """
    Writes markdown content to a file, preserving thinking sections.
    This version ensures thinking sections are written properly to files.
    """
    try:
        # Make sure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Write the content to the file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
            
        return True
    except Exception as e:
        import streamlit as st
        st.error(f"Error writing to file {file_path}: {str(e)}")
        return False

async def fetch_and_process_url(url: str, debug_container) -> str:
    """Fetch and process content from a URL using WebBaseLoader with BeautifulSoup strainer."""
    try:              
        # Initialize WebBaseLoader with BeautifulSoup configuration
        loader = WebBaseLoader(
            [url]
        )

        try:                
            # Use synchronous loading as aload() might not work well with bs_kwargs
            docs = loader.load()
                
            if not docs:
                # Try loading without BeautifulSoup
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
            content = "\n\n".join(doc.page_content.strip() for doc in docs)
            
            # Clean up the content
            content = re.sub(r'\s+', ' ', content).strip()  # Remove excessive whitespace
            content = re.sub(r'[^\x00-\x7F]+', '', content)  # Remove non-ASCII characters

            # Display success message
            if debug_container!=None:
                debug_container.success(f"Content retrieved from: {url}")
                debug_container.empty()
            
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
        # Remove any leading/trailing whitespace
        json_str = json_str.strip()
        
        # Remove any markdown code block markers
        json_str = re.sub(r'^```json\s*|\s*```$', '', json_str)
        json_str = re.sub(r'^```\s*|\s*```$', '', json_str)
        
        # Replace any single quotes with double quotes
        json_str = re.sub(r"(?<!\\)'", '"', json_str)
        
        # Handle escaped characters
        json_str = json_str.encode('utf-8').decode('unicode-escape')
        
        # Remove any non-printable characters except valid whitespace
        json_str = ''.join(char for char in json_str if char.isprintable() or char.isspace())
        
        # Normalize whitespace between JSON structural elements
        json_str = re.sub(r'\s+(?=[,\{\}\[\]])', '', json_str)
        json_str = re.sub(r'(?<=[,\{\}\[\]])\s+', ' ', json_str)
        
        # Remove trailing commas
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        return json_str.strip()
        
    except Exception as e:
        raise ValueError(f"Error cleaning JSON string: {str(e)}")

async def perform_wikipedia_search(search_tool, query, status_text=None):
    """
    Performs a Wikipedia search using the provided search tool and processes the results.
    
    Args:
        search_tool: The search tool to use (e.g., DuckDuckGo)
        query: The search query
        status_text: Streamlit text element for status updates
        
    Returns:
        list: List of dictionaries containing processed search results
    """
    content = search_tool.run(query)
    detailed_results = []
    try:
        if content!=None:
            status_text.text(f"Processing Wikipedia Search Results")

        if content:
            detailed_results.append({
                "url": "None",
                "content": content,
                "source": "wikipedia"
            })

                
    except Exception as e:
        if status_text!=None:
            status_text.warning(f"Wikipedia search error: {str(e)}")
        
    return detailed_results

async def perform_web_search(search_tool, query, status_text=None):
    """
    Performs a web search using the provided search tool and processes the results.
    
    Args:
        search_tool: The search tool to use (e.g., DuckDuckGo)
        query: The search query
        status_text: Streamlit text element for status updates
        
    Returns:
        list: List of dictionaries containing processed search results
    """
    detailed_results = []
    
    try:
        search_results = search_tool.run(query)
        urls = re.findall(r'https?://[^\s<>"]+|www\.[^\s<>"]+', search_results)
        if status_text!=None:
            status_text.text(f"Found URLs from search: {urls}")
        
        for url in urls:
            try:
                if status_text!=None:
                    status_text.text(f"Processing URL: {url}")
                content = await fetch_and_process_url(url, status_text)
                if content:
                    detailed_results.append({
                        "url": url,
                        "content": content,
                        "source": "web"
                    })
            except Exception as e:
                if status_text!=None:
                    status_text.warning(f"Error processing URL {url}: {str(e)}")
                
    except Exception as e:
        if status_text!=None:
            status_text.warning(f"Web search error: {str(e)}")
        
    return detailed_results

async def perform_google_search(search_tool, query, search_type, status_text):
    """
    Performs a Google search using Serper API and processes the results.
    
    Args:
        search_tool: The GoogleSerperAPIWrapper instance
        query: The search query
        search_type: Type of search ('search', 'news', or 'images')
        status_text: Streamlit text element for status updates
        
    Returns:
        list: List of dictionaries containing processed search results
    """
    detailed_results = []
    try:
        google_results = search_tool.results(query)
        google_urls = []
        
        if search_type == 'search':
            if 'organic' in google_results:
                google_urls = [item['link'] for item in google_results['organic'] if 'link' in item]
            if 'knowledgeGraph' in google_results and 'descriptionLink' in google_results['knowledgeGraph']:
                google_urls.append(google_results['knowledgeGraph']['descriptionLink'])
        elif search_type == 'news':
            if 'news' in google_results:
                google_urls = [item['link'] for item in google_results['news'] if 'link' in item]
        elif search_type == 'images':
            if 'images' in google_results:
                google_urls = [item['link'] for item in google_results['images'] if 'link' in item]
        
        status_text.text(f"Found URLs from Google Serper search: {google_urls}")
        
        for url in google_urls:
            try:
                status_text.text(f"Processing Google Serper URL: {url}")
                content = await fetch_and_process_url(url, status_text)
                if content:
                    detailed_results.append({
                        "url": url,
                        "content": content,
                        "source": f"google_{search_type}"
                    })
            except Exception as e:
                status_text.warning(f"Error processing Google URL {url}: {str(e)}")
                
    except Exception as e:
        status_text.warning(f"Google Serper search error: {str(e)}")
        
    return detailed_results

async def perform_arxiv_search(search_tool, query, status_text):
    """
    Performs an arXiv search and processes the results.
    
    Args:
        search_tool: The ArxivAPIWrapper instance
        query: The search query
        status_text: Streamlit text element for status updates
        
    Returns:
        list: List of dictionaries containing processed search results
    """
    detailed_results = []
    try:
        status_text.text("Searching arXiv papers...")
        papers = search_tool.run(query)
        
        for paper in papers.split('\n'):
            if paper.strip() and 'arxiv.org' in paper:
                try:
                    url = paper.strip()
                    status_text.text(f"Processing arXiv paper: {url}")
                    content = await fetch_and_process_url(url, status_text)
                    if content:
                        detailed_results.append({
                            "url": url,
                            "content": content,
                            "source": "arxiv"
                        })
                except Exception as e:
                    status_text.warning(f"Error processing arXiv paper {url}: {str(e)}")
                    
    except Exception as e:
        status_text.warning(f"arXiv search error: {str(e)}")
        
    return detailed_results

async def synthesize_search_results(results, llm, topic, status_text):
    """
    Synthesizes and validates web search results using the LLM.
    
    Args:
        results: List of search results
        llm: The language model to use for evaluation
        topic: The search topic
        status_text: Streamlit text element for status updates
        
    Returns:
        dict: Synthesized results with confidence scores and correlations
    """
    synthesis_results = []
    try:
        for result in results:
            # Format results for evaluation
            formatted_results = "\n\n".join([
                f"Source: {result['url']}\nContent: {result['content']}"  # Truncate for reasonable context
            ])
        
            status_text.text("Evaluating search results correlation and confidence...")
            evaluation = llm.invoke(SEARCH_RESULTS_EVALUATION_TEMPLATE.format(
                topic=topic,
                results=formatted_results
            ))
        
            try:
                # Clean and parse JSON response
                cleaned_json = clean_json_string(evaluation)
                # Add fallback if JSON is not properly formatted
                if not cleaned_json.startswith('{'):
                    # Try to extract JSON from the response
                    json_match = re.search(r'\{[\s\S]*\}', cleaned_json)
                    if json_match:
                        cleaned_json = json_match.group(0)
                    else:
                        raise ValueError("No valid JSON found in response")
                
                evaluation_data = json.loads(cleaned_json)
                
                # Validate required fields
                required_fields = ['sufficient_data', 'quality_score', 'confidence_score', 
                                'source_correlation', 'source_credibility']
                missing_fields = [field for field in required_fields if field not in evaluation_data]
                
                if missing_fields:
                    raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")
                
                # If confidence is too low, try to get additional context
                if evaluation_data['confidence_score'] < 6:
                    if status_text!=None:
                        status_text.warning("Confidence score low - gathering additional sources...")
                    continue
                
                # Format the synthesized response
                synthesis_results.append({
                    "confidence": evaluation_data['confidence_score'],
                    "quality": evaluation_data['quality_score'],
                    "corroborated_facts": evaluation_data.get('source_correlation', {}).get('high_correlation', []),
                    "credible_sources": evaluation_data.get('source_credibility', {}).get('high_credibility', []),
                    "evaluation": evaluation_data,
                    "needs_more_sources": False,
                    "answer": evaluation_data['answer']
                })
                
                
            except json.JSONDecodeError as e:
                status_text.error(f"Error parsing evaluation results: {str(e)}\nResponse: {evaluation[:200]}...")
                # Provide a fallback response
                return {
                    "confidence": 5,
                    "quality": 5,
                    "corroborated_facts": ["Unable to parse detailed results"],
                    "credible_sources": [],
                    "evaluation": {
                        "sufficient_data": False,
                        "quality_score": 5,
                        "confidence_score": 5,
                        "reasons": ["Error parsing LLM response"]
                    },
                    "needs_more_sources": True
                }

        report = ""
        for result in synthesis_results:
            if result["needs_more_sources"]:
                continue
            else:
                report+=f"Answer: {result['answer']}\n\nConfidence: {result['confidence']}\n\nQuality: {result['quality']}\n\nCorroborated Facts: {result['corroborated_facts']}\n\nCredible Sources: {result['credible_sources']}\n\n"

        evaluation = llm.invoke(SEARCH_RESULTS_EVALUATION_TEMPLATE.format(
                topic=topic,
                results=report
            ))

        return evaluation

    except Exception as e:
        status_text.error(f"Error synthesizing results: {str(e)}")
        return None

