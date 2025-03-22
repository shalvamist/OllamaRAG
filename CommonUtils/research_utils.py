import os
import re
import json
from urllib.parse import urlparse
from bs4 import SoupStrainer
from langchain_community.document_loaders import WebBaseLoader

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