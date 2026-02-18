# QMS Document Assistant

An AI-powered chatbot for medical device Quality Management System (QMS) teams to answer queries about regulatory and quality documents using Retrieval-Augmented Generation (RAG).

## Overview

This chatbot uses RAG techniques to ingest regulatory documents (PDFs, DOCX, TXT) and provide contextual answers with source citations. 
Built for medical device engineering teams to quickly access information from regulatory frameworks and internal QMS procedures.

## Features

- **Multi-format document support** - PDF, DOCX, TXT  
- **Source citation** - Every answer includes document references  
- **Local LLM** - Privacy-first design using LM Studio  
- **Clean web interface** - Professional UI with chat history  
- **Error handling** - Robust validation and user-friendly error messages  
- **Streaming responses** - Real-time answer generation  

## Architecture

User Query → FastAPI → RAG Logic → Vector DB (ChromaDB) → LLM (Llama 3.2) → Response

**Components:**

- **Frontend**: HTML/CSS/JavaScript chat interface
- **Backend**: FastAPI server
- **Vector Database**: ChromaDB for semantic search
- **Embeddings**: Sentence Transformers (e5-large-v2)
- **LLM**: Qwen-3.2-3B-Instruct via LM Studio (local inference)

## Usage

### Start the Server
```bash
uvicorn app:app --reload
```
The interface will be available at: **http://localhost:8000**

### CLI Mode (Optional)
Test queries directly in terminal:
```bash
python rag.py
```

## Project Structure
```
qms-document-assistant/
├── app.py                 # FastAPI server
├── rag.py                 # RAG logic and query processing
├── ingest.py              # Document ingestion pipeline
├── chat_ui.html           # Web interface
├── requirements.txt       # Python dependencies
├── docs/                  # Your regulatory documents (not tracked)
├── chroma_db/             # Vector database (generated)
└── README.md
```

## Configuration

### Adjust Chunk Size
In `ingest.py`, modify chunking parameters:
```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,      # Increase for longer context
    chunk_overlap=150    # Overlap between chunks
)
```

### Change Response Length
In `rag.py`, adjust max tokens:
```python
response = client.chat.completions.create(
    ...
    max_tokens=1024,  # Increase for longer answers
    ...
)
```

### Retrieval Settings
Change number of documents retrieved:
```python
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5  # Increase for more context
)
```

## Acknowledgments

- Built with [LM Studio](https://lmstudio.ai/)
- Powered by [Llama 3.2](https://qwenlm.github.io/) language model
- Uses [ChromaDB](https://www.trychroma.com/) vector database
- Embeddings by [Sentence Transformers](https://www.sbert.net/)

---

**⚠️ Disclaimer**: This is an AI assistant for informational purposes only. Always verify critical regulatory information with official source documents before making compliance decisions.
