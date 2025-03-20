<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>

# 🦙🦜🔗 OllamaRAG

<div align="center">

[![Made for Local LLMs](https://img.shields.io/badge/Made%20for-Local%20LLMs-blue?style=for-the-badge&logo=robot&logoColor=white)](https://ollama.ai)
[![Built with Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)

**Your Personal AI Research & Debate Assistant**  
_Local. Private. Powerful._

---

🔍 **Research** anything with AI-powered analysis  
🤖 **Debate** topics with multiple AI personalities  
📚 **Learn** from AI-enhanced document interactions  
🔒 **Privacy** guaranteed with 100% local processing

---

[🚀 Quick Start](#-quick-start) •
[✨ Features](#-key-features) •
[🛠️ Setup Guide](#-quick-start) •
[📘 Documentation](https://github.com/yourusername/OllamaRAG/wiki)

</div>

> Supercharge your local LLMs with powerful RAG capabilities! Transform your research, spark engaging AI debates, and unlock the full potential of your documents - all while keeping your data private.

OllamaRAG is your all-in-one solution for document-enhanced AI conversations, research automation, and interactive AI debates - all running locally on your machine!

## ✨ Key Features

- 🏠 **100% Local Processing** - Complete data privacy with local LLM integration
- 🤖 **AI Debate Arena** - Watch AI models debate topics with fact-based arguments
- 🔍 **Smart Research Engine** - Automated multi-source research with topic analysis
- 💬 **RAG-Enhanced Chat** - Context-aware conversations using your documents
- 📊 **Intelligent Processing** - Advanced document chunking and hybrid search
- 💾 **Persistent Memory** - Save and manage conversations across sessions

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- [Ollama](https://ollama.ai) installed and running

### Setup
```bash
# Clone and enter directory
git clone [repository-url]
cd OllamaRAG

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run Home.py
```

## 🎯 Main Components

1. **Model Settings** - Configure your Ollama models
2. **RAG Config** - Set up document processing
3. **Chat** - Interactive RAG-enhanced conversations
4. **Deep Research** - Automated research and analysis
5. **AI Debate** - Dynamic debates between AI models

## 🛠️ Built With Amazing Tech

- [Ollama](https://ollama.ai) - Local LLM runtime
- [LangChain](https://www.langchain.com) - LLM application framework
- [Streamlit](https://streamlit.io) - Web interface
- [ChromaDB](https://www.trychroma.com) - Vector database

## 📝 License

MIT License - Free to use, modify, and distribute. No liability or warranty provided.

---
[Explore Features](https://github.com/yourusername/OllamaRAG#features) · 
[Report Bug](https://github.com/yourusername/OllamaRAG/issues) · 
[Request Feature](https://github.com/yourusername/OllamaRAG/issues)

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

### AI Debate Arena
- Dynamic debates between two AI models on any topic
- Customizable model selection for each debater and judge
- Configurable debate parameters:
  - Number of turns (1-10)
  - Response length
  - Model parameters (temperature, context window, max tokens)
- Personalized bot stances with customizable system prompts
- Real-time web research integration for fact-based arguments
- Unbiased debate analysis and winner determination
- Markdown-formatted responses for better readability
- Progress tracking and status updates
- Beautiful UI with distinct styling for each debater

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
