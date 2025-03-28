<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>

 <h1 align="center">🦙🦜🔗 OllamaRAG 🔗🦜🦙</h1>

<div align="center">

[![Made for Local LLMs](https://img.shields.io/badge/Made%20for-Local%20LLMs-blue?style=for-the-badge&logo=robot&logoColor=white)](https://ollama.ai)
[![Built with Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)

> Ever wanted to chat with multiple AI personalities about your documents? Well, now you can! And the best part? It's all happening on your machine. No data leaves your computer! 🔒

## 🎯 What Can It Do?

🤖 **Chat with Your Docs**: Upload PDFs, and your AI buddy will read them for you
🎭 **AI Debate Club**: Watch AI models argue (respectfully) about any topic
🔍 **Research Assistant**: Let AI do the heavy lifting in your research
🧠 **Smart Memory**: Remembers your chats for later (because we all forget sometimes)

## 🚀 Get Started in 5 Minutes!

### 1. First Things First
Install [Python 3.8+](https://python.org)
Get [Ollama](https://ollama.ai) running on your machine
Download a model (we recommend): `ollama pull mistral`

### 2. Clone & Setup
```bash
# Clone this bad boy
git clone [repository-url]
cd OllamaRAG

# Create your virtual playground
python -m venv venv

# Activate the magic
# For Windows folks:
.\venv\Scripts\activate
# For Unix cool kids:
source venv/bin/activate

# Install the goodies
pip install -r requirements.txt
```

### 3. Launch! 🚀
```bash
streamlit run Home.py
```

## 🎮 Quick Tips

- 💡 **First Time?** Start with the Chat tab - it's the friendliest!
- 🎯 **Pro Move:** Try the AI Debate feature with controversial topics (like "Is a hotdog a sandwich?")
- 📚 **Document Tips:** Smaller PDFs work better (we're working on handling the big ones)

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

## ✨ Key Features

- 🏠 **100% Local Processing** - Complete data privacy with local LLM integration
- 🤖 **AI Debate Arena** - Watch AI models debate topics with fact-based arguments
- 🔍 **Smart Research Engine** - Automated multi-source research with topic analysis
- 💬 **RAG-Enhanced Chat** - Context-aware conversations using your documents
- 📊 **Intelligent Processing** - Advanced document chunking and hybrid search
- 💾 **Persistent Memory** - Save and manage conversations across sessions

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
