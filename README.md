<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/shalvamist/OllamaRAG/">
  </a>
  <h1 align="center">🦙🦜🔗OllamaRAG🔗🦜🦙</h1>
  <p align="center">
    An awesome Streamlit app to empower your Ollama RAG usage
  </p>
</div>

<!-- ABOUT THE PROJECT -->
## About The Project
OllamaRAG is a Streamlit app leveraging Ollama for local RAG applications

I am working enabling custom langchain RAG pipes but that work is TBD - at the moment only a simple RAG pipeline is supported 

### Built With  
[Langchain](https://github.com/langchain-ai/langchain)

[Ollama](https://github.com/ollama/ollama)

[Chroma](https://github.com/chroma-core/chroma)

> [!IMPORTANT - Requirments]
> 
> You will need Ollama running on your machine - you can find the installation steps here [Ollama download](https://ollama.com/download)
> Check out the colab link - Installation and run steps are detailed here 👇

## Colab
[💡 Google Colab Notebook](https://github.com/shalvamist/OllamaRAG/blob/main/OllamaRAG.ipynb)

## Installation
1. Clone the repo
   
   ```bash git clone https://github.com/shalvamist/OllamaRAG.git```
3. Install requirments

   ```bash pip install -r OllamaRAG/requirments.txt```
5. Run the Streamlit app
  
   ```bash streamlit run .\webApp.py```

# OllamaRAG 🦙🦜🔗

A powerful local RAG (Retrieval-Augmented Generation) application that combines Ollama's local LLMs with document retrieval capabilities.

## Features

- 🤖 Local LLM support through Ollama
- 📚 RAG (Retrieval-Augmented Generation) capabilities
- 💾 Persistent chat history with SQLite
- 🔍 Hybrid search with vector embeddings and BM25
- 📄 PDF document processing and chunking
- 🎯 Contextual retrieval options
- 🎨 Modern Streamlit interface

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

## Required Python Packages

Create a `requirements.txt` file with the following dependencies:
```
streamlit
langchain
langchain-ollama
chromadb
pypdf
ollama
python-dotenv
```

## Running the Application

1. Make sure Ollama is running on your system
2. Start the Streamlit application:
```bash
streamlit run Home.py
```

## Application Structure

The application consists of four main pages:

1. **Home** (`Home.py`): Landing page with system status and quick start guide
2. **Model Settings** (`pages/1_🦙_Model_Settings.py`): Configure Ollama models and parameters
3. **RAG Config** (`pages/2_🔗_RAG_Config.py`): Set up document processing and retrieval settings
4. **Chat** (`pages/3_💬_Chat.py`): Interactive chat interface with RAG capabilities

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
