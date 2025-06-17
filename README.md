# 🔍 DocuMind Pro - RAG with Cross-Encoder Re-ranking

> A powerful document analysis and Q&A application using Retrieval-Augmented Generation (RAG) with advanced cross-encoder re-ranking for enhanced accuracy.

## 🎯 Overview

DocuMind Pro is an intelligent document analysis platform that combines the power of:
- **RAG (Retrieval-Augmented Generation)** for contextual document understanding
- **Cross-encoder re-ranking** for improved retrieval accuracy  
- **Multi-format support** for PDF and DOCX documents
- **Local inference** using Ollama for privacy and control
- **Professional dark-themed UI** built with Streamlit

Perfect for researchers, analysts, and professionals who need accurate answers from their document collections.

## ✨ Features

- 📄 **Multi-format Document Support**: Upload and analyze PDF and DOCX files
- 🔍 **Advanced RAG Pipeline**: Intelligent document chunking and vector storage
- 🎯 **Cross-encoder Re-ranking**: Enhanced retrieval accuracy using sentence transformers
- 🤖 **Local LLM Integration**: Powered by Ollama for complete privacy
- 💾 **Persistent Vector Storage**: ChromaDB for efficient document indexing
- 🎨 **Professional UI**: Dark-themed, responsive interface
- ⚡ **Real-time Processing**: Fast document analysis and query responses
- 🔒 **Privacy-focused**: All processing happens locally

## 🚨 Requirements

- **Python**: Version 3.10 or higher
- **SQLite**: Version 3.35 or higher
- **Ollama**: For local LLM inference

## 🔨 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/DocuMind_RAG_LLM.git
cd DocuMind_RAG_LLM
```

### 2. Set Up Virtual Environment
```bash
python -m venv .venv

# On Windows
.venv\Scripts\activate

# On macOS/Linux
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements/requirements.txt
```

### 4. Install and Setup Ollama
Download and install Ollama from [ollama.ai](https://ollama.ai/download)

Pull required models:
```bash
ollama pull llama3.1:8b  # Or your preferred model
ollama pull nomic-embed-text  # For embeddings
```

## 🚀 Usage

### Starting the Application
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

### Using DocuMind Pro

1. **Upload Documents**: Drag and drop PDF or DOCX files in the sidebar
2. **Process Documents**: Click "Process Document" to analyze and index
3. **Ask Questions**: Enter your questions in the main interface
4. **Get Answers**: Receive detailed, context-aware responses

### Key Features Walkthrough

- **Document Upload**: Supports batch processing of multiple documents
- **Intelligent Chunking**: Automatically splits documents for optimal retrieval
- **Vector Search**: Find relevant document sections using semantic similarity
- **Re-ranking**: Improves result quality using cross-encoder models
- **Contextual Answers**: Get comprehensive responses based on document content

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Document      │    │   Text Splitter  │    │   ChromaDB      │
│   Loader        │───▶│   & Chunking     │───▶│   Vector Store  │
│   (PDF/DOCX)    │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Final         │    │   Cross-Encoder  │    │   Vector        │
│   Response      │◀───│   Re-ranking     │◀───│   Retrieval     │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         ▲                                               │
         │              ┌──────────────────┐             │
         └──────────────│   Ollama LLM     │◀────────────┘
                        │   Generation     │
                        └──────────────────┘
```

## 📁 Project Structure

```
DocuMind_RAG_LLM/
├── app.py                 # Main Streamlit application
├── requirements/
│   └── requirements.txt   # Python dependencies
├── README.md             # Project documentation
├── .gitignore           # Git ignore rules
└── .venv/               # Virtual environment (local)
```

## 🛠️ Dependencies

- `streamlit` - Web application framework
- `ollama` - Local LLM inference
- `chromadb` - Vector database for embeddings
- `sentence-transformers` - Cross-encoder re-ranking
- `PyMuPDF` - PDF document processing
- `docx2txt` - DOCX document processing
- `langchain-community` - Text processing utilities

## 🔧 Configuration

### Ollama Models
The application uses Ollama for local inference. Ensure you have appropriate models installed:

```bash
# Large language model for generation
ollama pull llama3.1:8b

# Embedding model for vector search
ollama pull nomic-embed-text
```

### Environment Variables
No additional environment variables required - the application runs entirely locally.

## 🐛 Troubleshooting

### Common Issues

**ChromaDB/SQLite Compatibility**
If you encounter SQLite version issues:
```bash
pip install --upgrade chromadb
```
Refer to [ChromaDB troubleshooting guide](https://docs.trychroma.com/troubleshooting#sqlite)

**Missing Dependencies**
If you see docx2txt import errors:
```bash
pip install docx2txt==0.8
```

**Ollama Connection Issues**
Ensure Ollama is running:
```bash
ollama serve
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Ollama](https://ollama.ai/) for local LLM inference
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [Streamlit](https://streamlit.io/) for the web framework
- [Sentence Transformers](https://www.sbert.net/) for re-ranking models
