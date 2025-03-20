<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/shalvamist/OllamaRAG/">
  </a>
  <h1 align="center">🦙🦜🔗 OllamaRAG 🔗🦜🦙</h1>
  <p align="center">
    <strong>Supercharge your local LLMs with powerful RAG capabilities! 🚀</strong>
    <br />
    A modern, user-friendly interface for document-enhanced conversations
    <br />
    <a href="#features"><strong>Explore Features »</strong></a>
    &nbsp;·&nbsp;
    <a href="#installation"><strong>Quick Start »</strong></a>
    &nbsp;·&nbsp;
    <a href="#setup-guide"><strong>Setup Guide »</strong></a>
  </p>
</div>

<!-- ABOUT THE PROJECT -->
## 🌟 About The Project

OllamaRAG brings the power of Retrieval-Augmented Generation (RAG) to your local environment, seamlessly integrating with Ollama's local LLMs. It's designed to make document-enhanced AI conversations accessible and efficient.

### What Makes OllamaRAG Special?

- 🏠 **Fully Local Operation**: All processing happens on your machine, ensuring data privacy
- 🔄 **Hybrid Search**: Combines vector similarity and keyword matching for better context retrieval
- 💾 **Persistent Memory**: Save and manage conversation history across sessions
- 📊 **Smart Document Processing**: Intelligent chunking and embedding of your documents
- 🎯 **Context-Aware Responses**: LLM outputs enhanced with relevant document snippets
- 🔍 **Deep Research Engine**: Automated, multi-source research with Wikipedia integration
- 📝 **Structured Analysis**: AI-powered topic decomposition and synthesis
- 🌐 **Web Intelligence**: Smart web content extraction and evaluation

### Built For Everyone

- 🔬 **Researchers**: Quickly query, analyze, and synthesize information from multiple sources
- 💼 **Professionals**: Enhance productivity with AI-powered research and document analysis
- 🎓 **Students**: Learn and explore topics with comprehensive research capabilities
- 🧪 **Developers**: Experiment with RAG and research automation implementations

### Built With

OllamaRAG leverages the best open-source technologies:
- 🦙 [Ollama](https://ollama.ai/) - Run LLMs locally
- 🔗 [LangChain](https://www.langchain.com/) - Build LLM applications
- 🌊 [Streamlit](https://streamlit.io/) - Create web interfaces
- 🎨 [ChromaDB](https://www.trychroma.com/) - Embed and store documents

## Try it on Google Colab! 🚀

Want to test OllamaRAG without any local setup? We've got you covered! 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/shalvamist/OllamaRAG/blob/main/OllamaRAG.ipynb)

Our Colab notebook offers:
- 🔥 **Instant Setup**: Get started in minutes with zero local installation
- 📚 **Interactive Tutorial**: Step-by-step guide to all features
- 📖 Learning about RAG implementations
- 🧪 Quick prototyping and testing

Click the badge above to launch the notebook and start exploring! 

> Note: The Colab notebook includes detailed installation steps and usage examples to help you get the most out of OllamaRAG.

## Prerequisites

1. **Python 3.8+** installed on your system
2. **Ollama** installed and running on your system
   - Install from [Ollama's official website](https://ollama.ai)
   - Make sure the Ollama service is running before starting the app

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd OllamaRAG
```

2. Create and activate a virtual environment (recommended):
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```
## Running the Application

1. Make sure Ollama is running on your system
2. Start the Streamlit application:
```bash
streamlit run Home.py
```

## Application Structure

The application consists of five main pages:

1. **Home** (`Home.py`): Landing page with system status and quick start guide
2. **Model Settings** (`pages/1_🦙_Model_Settings.py`): Configure Ollama models and parameters
3. **RAG Config** (`pages/2_🔗_RAG_Config.py`): Set up document processing and retrieval settings
4. **Chat** (`pages/3_💬_Chat.py`): Interactive chat interface with RAG capabilities
5. **Deep Research** (`pages/4_🔍_DeepResearch.py`): Comprehensive research tool for in-depth topic exploration

## Setup Guide

### 1. Model Setup

1. Go to the "Model Settings" page
2. Select or download an Ollama model
   - For general use, we recommend starting with `llama2`
   - For embedding, we recommend `nomic-embed-text`
3. Configure model parameters:
   - Temperature (default: 1.0)
   - Context Window (default: 2048)
   - Max Tokens (default: 2048)
4. Set your system prompt
5. Click "Apply Settings"

### 2. RAG Configuration

1. Navigate to the "RAG Config" page
2. Select an embedding model
3. Configure document processing:
   - Chunk size (default: 1000)
   - Chunk overlap (default: 200)
   - Retrieved documents count (default: 3)
4. Upload PDF documents
5. Click "Rebuild Database"

### 3. Start Chatting

1. Go to the "Chat" page
2. Start asking questions
   - The system will retrieve relevant context from your documents
   - Responses are generated using your selected Ollama model
   - Chat history is automatically saved

## Features in Detail

### Deep Research Capabilities
- Automated topic decomposition into logical subtopics
- Multi-source research combining:
  - DuckDuckGo web search
  - Wikipedia articles
  - Web page content extraction
- Intelligent content evaluation and filtering
- Structured research output with:
  - Main topic overview
  - Subtopic analysis
  - Source citations
  - Key findings synthesis
- Configurable research parameters:
  - Number of subtopics (1-20)
  - Maximum research iterations
  - Search attempts per subtopic
- Markdown-formatted research reports
- Automatic source tracking and citation
- Progress tracking and status updates

### Chat History
- Conversations are automatically saved in SQLite database
- Access previous conversations from the sidebar
- Delete individual conversations
- Clear current chat history

### Document Processing
- PDF document support
- Automatic text chunking
- Configurable chunk size and overlap
- Vector embeddings for semantic search
- Optional BM25 retrieval for keyword matching

### RAG Capabilities
- Hybrid search combining vector and keyword matching
- Contextual retrieval with AI-generated context
- Configurable number of retrieved documents
- Document reference tracking

## Acknowledgments

This project is built upon the amazing work of several open-source projects:

- [Ollama](https://ollama.ai/) - For providing powerful local LLM capabilities
- [Streamlit](https://streamlit.io/) - For their excellent web application framework
- [LangChain](https://www.langchain.com/) - For comprehensive LLM tools and utilities
- [ChromaDB](https://www.trychroma.com/) - For the efficient vector database implementation
- [Langchain-Ollama](https://python.langchain.com/docs/integrations/llms/ollama) - For seamless Ollama integration with LangChain

## Troubleshooting

1. **Ollama Connection Issues**
   - Ensure Ollama is running (`ollama serve`)
   - Check if the selected model is downloaded
   - Verify no firewall blocking localhost connections

2. **Database Issues**
   - Check if `chroma_db` directory exists and is writable
   - Rebuild database if encountering embedding errors
   - Clear database and reupload documents if issues persist

3. **Memory Issues**
   - Reduce chunk size in RAG configuration
   - Lower the number of retrieved documents
   - Use a smaller model if available

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

Copyright (c) 2024 OllamaRAG

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
