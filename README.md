# QMS Document Assistant

An AI-powered chatbot for medical device Quality Management System (QMS) teams to answer queries about regulatory and quality documents using Retrieval-Augmented Generation (RAG). Built for medical device engineering teams to quickly access information from regulatory frameworks and internal QMS procedures.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation & Setup](#installation--setup)
- [Running the Application](#running-the-application)
- [Project Structure](#project-structure)
- [File Descriptions](#file-descriptions)
- [Adding Documents](#adding-documents)
- [Configuration](#configuration)
- [How It Works](#how-it-works)
- [Troubleshooting](#troubleshooting)
- [Development Notes](#development-notes)
- [Acknowledgments](#acknowledgments)

## Overview

This RAG-based chatbot system processes regulatory and quality documents, splits them into searchable chunks, and uses semantic embeddings to retrieve relevant context for LLM-powered answering. All processing is local and does not send data to external APIs.

**Use Case:** Replace manual document searching with intelligent Q&A for faster regulatory compliance and quality procedure lookups.

## Features

- **Multi-format document support** - PDF, DOCX, and TXT files
- **Source citations** - Every answer includes links to source documents
- **Local LLM inference** - Privacy-first design using LM Studio (no cloud API calls)
- **Professional web interface** - Chat UI with streaming responses and clear history
- **Robust error handling** - Graceful handling of missing documents, connection errors, and malformed input
- **Semantic search** - Uses embeddings for contextual document retrieval rather than keyword matching
- **Real-time streaming** - Responses appear character-by-character as they're generated

## Architecture

```
User Query
    ↓
FastAPI Server (/chat endpoint)
    ↓
RAG Pipeline (rag.py)
  ├─ Embed query using Sentence Transformers
  ├─ Retrieve top 3 similar documents from ChromaDB
  ├─ Build context prompt with retrieved documents
    ↓
LLM (Llama 3.2 via LM Studio)
  └─ Generate response with streaming
    ↓
Post-processing (strip think blocks)
    ↓
Format sources & return to frontend
    ↓
Web Interface displays response with clickable source links
```

**Key Components:**

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Frontend** | HTML/CSS/JavaScript | Chat interface and message display |
| **Backend API** | FastAPI | HTTP server and WebSocket streaming |
| **Vector DB** | ChromaDB | Persistent semantic search storage |
| **Embeddings** | Sentence Transformers (e5-large-v2) | Convert text to semantic vectors |
| **LLM** | Llama 3.2 (3B instruct) | Text generation via LM Studio local inference |
| **Document Processing** | LangChain TextSplitter, PyPDF, python-docx | Parse and chunk documents |

## Prerequisites

- **Python 3.8+** (tested with Python 3.x)
- **LM Studio** installed and running with llama-3.2-3b-instruct model loaded on `http://localhost:1234/v1`
- **pip** package manager
- **4GB+ RAM** (for embeddings model and local LLM)

**Important:** The LLM server must be running before starting the chatbot. LM Studio should have the model loaded and listening on port 1234.

## Installation & Setup

### 1. Clone and Navigate
```bash
cd /path/to/AIChatbotTest
```

### 2. Create Python Virtual Environment (Recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

This installs all required packages:
- **fastapi** - Web framework
- **uvicorn** - ASGI server
- **openai** - LLM client (works with local LM Studio)
- **sentence-transformers** - Embedding model
- **chromadb** - Vector database
- **langchain-text-splitters** - Document chunking
- **pypdf** - PDF parsing
- **python-docx** - DOCX parsing
- **pydantic** - Data validation

### 4. Set Up Document Folder
```bash
mkdir -p docs
```
Place your regulatory documents (PDF, DOCX, TXT) in the `docs/` folder.

### 5. Prepare Documents (One-time setup)
Before running the server, ingest your documents into the vector database:
```bash
python ingest.py
```

This will:
- Scan the `docs/` folder for all PDF, DOCX, and TXT files
- Extract text from each document
- Split documents into semantic chunks (800 tokens, 150 overlap)
- Generate embeddings for each chunk
- Store embeddings and metadata in `chroma_db/`

**Output Example:**
```
Found 5 files in 'docs'
Loading document: ISO13485-2016.pdf
Loading document: QMS-Procedures.docx
...
Loaded 5 documents
Created 245 chunks
Ingestion complete: 245 chunks added.
```

## Running the Application

### Start the FastAPI Server
```bash
uvicorn app:app --reload
```

**Output:**
```
INFO:     Started server process [12345]
INFO:     Uvicorn running on http://127.0.0.1:8000
```

**Access the web interface:** Open browser to `http://localhost:8000`

### CLI Testing (Optional)
To test the RAG pipeline without the web interface:
```bash
python rag.py
```

Then type questions at the prompt. Type `quit` or `exit` to exit.

## Project Structure

```
AIChatbotTest/
├── app.py                      # FastAPI web server & endpoints
├── rag.py                       # RAG pipeline & LLM query logic
├── ingest.py                    # Document ingestion & vectorization
├── chat_ui.html                 # Frontend chat interface
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── docs/                        # User documents folder (create manually)
│   ├── your-document.pdf
│   ├── procedures.docx
│   └── guidelines.txt
└── chroma_db/                   # Vector database (auto-generated)
    └── embeddings & metadata
```

## File Descriptions

### `app.py` - FastAPI Server
**Purpose:** Serves the web interface and exposes API endpoints

**Key Endpoints:**
- `GET /` - Serves `chat_ui.html`
- `POST /chat` - Accepts JSON `{"query": "..."}`, returns streaming text response
- `GET /documents/{filename}` - Serves documents from `docs/` folder for source links
- `GET /health` - Health check endpoint

**Key Features:**
- CORS middleware (allows cross-origin requests)
- Streaming responses for real-time feedback
- Query validation (non-empty, max 2000 chars)
- Error handling with proper HTTP status codes

**Configuration:**
- Port: 8000
- Reload mode: Development convenience feature

### `rag.py` - RAG Pipeline
**Purpose:** Core logic for querying documents and generating responses

**Key Functions:**
- `rag_query(user_query)` - Main RAG function, yields response tokens
- `strip_think_blocks(text)` - Post-processing to remove LLM thinking blocks

**Process Flow:**
1. Check if vector DB has documents
2. Embed user query using Sentence Transformers
3. Query ChromaDB for top 3 most similar document chunks
4. Build system prompt with retrieved context
5. Stream response from LM Studio LLM
6. Post-process response (remove think blocks if present)
7. Extract and format source document links
8. Return final response with sources

**LLM Configuration:**
- Server: `http://localhost:1234/v1` (LM Studio)
- Model: `llama-3.2-3b-instruct`
- Temperature: 0.1 (deterministic, factual responses)
- Max tokens: 768 (controls response length)
- Timeout: 120 seconds

### `ingest.py` - Document Ingestion Pipeline
**Purpose:** Convert documents into searchable vector embeddings

**Process:**
1. Scan `docs/` folder for `.txt`, `.pdf`, `.docx` files
2. Extract text from each file type using appropriate parsers
3. Split text into chunks using RecursiveCharacterTextSplitter
4. Generate embeddings for each chunk using Sentence Transformers
5. Store chunks + embeddings + metadata in ChromaDB
6. Clear old collection on re-run (safe to run multiple times)

**Chunking Settings:**
- Chunk size: 800 tokens (adjust for more/less context per chunk)
- Chunk overlap: 150 tokens (gives context continuity)

**Document Metadata Stored:**
- `source`: Document filename and extension
- `filepath`: Full file path for internal use

### `chat_ui.html` - Frontend Interface
**Purpose:** User-facing chat interface

**Features:**
- Clean, professional design (white theme with blue accents)
- Real-time message streaming
- Chat history display
- "Clear Chat" button to reset conversation
- Markdown link parsing (converts `[text](url)` to clickable links)
- Responsive layout
- Disclaimer footer

**How Streaming Works:**
- JavaScript `fetch()` gets response stream from `/chat` endpoint
- Response is read event-by-event and appended to chat bubble
- Links in responses are automatically converted to HTML `<a>` tags

### `requirements.txt` - Dependencies
Lists all Python packages with pinned versions for reproducibility:
- FastAPI/Uvicorn for web server
- OpenAI client for LLM communication
- Sentence-Transformers for embeddings
- ChromaDB for vector storage
- LangChain for text splitting
- PyPDF for PDF parsing
- python-docx for DOCX parsing
- Pydantic for data validation

## Adding Documents

### Step 1: Prepare Documents
Place your documents in the `docs/` folder:
```bash
docs/
├── ISO13485-2016.pdf
├── QMS-Procedures.docx
└── Design-Controls.txt
```

**Supported Formats:**
- `.pdf` - Extracts all text from all pages
- `.docx` - Extracts paragraph text
- `.txt` - Plain text files

### Step 2: Re-ingest Documents
```bash
python ingest.py
```

**Important Notes:**
- Running `ingest.py` clears the old vector database and re-ingests all documents
- Each document is parsed, chunked, and embedded from scratch
- This is a one-time setup before running the server
- If you add new documents, you must re-run ingest before querying

### Step 3: Verify Ingestion
Check the terminal output for:
```
Found X files in 'docs'
Loading document: [filename]
...
Loaded X documents
Created Y chunks
Ingestion complete: Y chunks added.
```

## Configuration

### Optimal Response Length
In `rag.py` (~line 100), adjust the LLM response length:
```python
response = client.chat.completions.create(
    ...
    max_tokens=768,  # Increase to 1024+ for longer responses
    ...
)
```

### Document Retrieval Sensitivity
In `rag.py` (~line 53), change how many documents are retrieved:
```python
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3  # Increase to 5-7 for more context, increases hallucination risk
)
```

**Trade-off:** More documents = more context but slower responses and potential confusion.

### Chunk Size
In `ingest.py` (~line 105), adjust text chunking:
```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,      # Increase for longer context per chunk (1000-1500 recommended for dense docs)
    chunk_overlap=150    # Increase for better continuity
)
```

**Impact:**
- Smaller chunks: Better precision, may miss context
- Larger chunks: Better context, slower inference, more tokens

### LLM Temperature
In `rag.py` (~line 99), adjust response creativity:
```python
response = client.chat.completions.create(
    ...
    temperature=0.1,  # 0 = deterministic, 1 = creative. Keep low for QMS (factual)
    ...
)
```

### CORS Settings
In `app.py` (~line 50), modify allowed origins if needed:
```python
allow_origins=["*"]  # Change to ["http://localhost:3000"] to restrict
```

## How It Works

### Embedding & Vector Search

1. **Document Ingestion:** Each document chunk is converted to a numeric vector (embedding) using Sentence Transformers. These embeddings capture semantic meaning.

2. **Query Processing:** User queries are converted to embeddings using the same model, ensuring compatibility.

3. **Semantic Search:** ChromaDB finds the 3 document chunks with embeddings closest to the query embedding (cosine similarity).

4. **Context Building:** The retrieved chunks are formatted into the LLM prompt as context.

5. **LLM Generation:** The LLM reads the context and user query, then generates an answer grounded in the documents.

### Response Pipeline

```
Raw LLM Output
    ↓
Strip <think>...</think> blocks (if model uses reasoning)
    ↓
Extract unique source metadata from retrieved chunks
    ↓
Format sources as markdown links: [Filename](http://localhost:8000/documents/Filename)
    ↓
Append sources section to response
    ↓
Return to frontend as plain text
    ↓
Frontend converts markdown links to HTML and displays
```

## Troubleshooting

### Error: "Unable to connect to the LLM server"
**Cause:** LM Studio is not running or not listening on `http://localhost:1234`

**Fix:**
1. Open LM Studio
2. Load the `llama-3.2-3b-instruct` model
3. Start the local server (should show "Server started at http://localhost:1234")
4. Verify: `curl http://localhost:1234/v1/models`

### Error: "No documents found in the vector database"
**Cause:** `ingest.py` was never run, or documents folder is empty

**Fix:**
1. Add documents to `docs/` folder
2. Run `python ingest.py`
3. Wait for "Ingestion complete" message
4. Try again

### Error: "No relevant documents found for the query"
**Cause:** Query doesn't match any documents, or similarity threshold is too high

**Fix:**
1. Verify documents are relevant to the query topic
2. Re-run `python ingest.py` to verify ingestion
3. Increase `n_results` in `rag.py` to retrieve more candidates
4. Try different query wording

### Port 8000 Already in Use
**Fix:**
```bash
# Use different port
uvicorn app:app --port 9000
# Then access http://localhost:9000
```

### Slow Response Generation
**Causes:**
- Computer doesn't have enough RAM
- LLM model is too large for hardware
- Document chunks are too large

**Fixes:**
1. Reduce `max_tokens` in `rag.py`
2. Reduce `chunk_size` in `ingest.py` then re-run ingestion
3. Close other applications to free memory
4. Reduce `n_results` to retrieve fewer documents

### PDF Text Not Extracting
**Cause:** PDF is image-based or has permission restrictions

**Fix:**
1. Try opening the PDF in Adobe Reader to verify it has extractable text
2. If PDF is scanned images, you'll need OCR (not supported currently)
3. Export PDF to text or convert to DOCX format

## Development Notes

### Adding New File Formats
To support additional file types (e.g., `.xlsx`, `.html`):

1. Add parsing function in `ingest.py`:
```python
def load_xlsx(path):
    # Parse your format
    return extracted_text
```

2. Update `load_document()` to handle new extension:
```python
elif path.endswith(".xlsx"):
    return load_xlsx(path)
```

3. Add required package to `requirements.txt`

### Customizing the System Prompt
In `rag.py` (~line 72), modify the system prompt for different response styles:
```python
prompt = f"""
You are a [CUSTOM ROLE]. Answer questions based on the provided documents.

INSTRUCTIONS:
- [Custom rules]
- [Custom formatting]

Context:
{context}

User question:
{user_query}

Answer:
"""
```

### Logging & Debugging
Enable debug output:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Enable verbose LLM output:
```python
# In rag.py, comment out response buffering to see streaming
for chunk in response:
    print(chunk.choices[0].delta.content, end="", flush=True)
```

### Testing
- **Unit test** RAG pipeline: `python rag.py` (CLI mode)
- **Integration test:** Run full server with `uvicorn app:app --reload`
- **Check embeddings:** Query Chroma directly in Python:
```python
import chromadb
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("my_docs")
print(collection.count())  # Should show number of chunks
```

## Acknowledgments

- Built with [LM Studio](https://lmstudio.ai/) for local LLM inference
- Powered by [Llama 3.2](https://qwenlm.github.io/) language model
- Uses [ChromaDB](https://www.trychroma.com/) for vector database
- Embeddings by [Sentence Transformers](https://www.sbert.net/)
- Document parsing by [PyPDF](https://github.com/py-pdf/pypdf) and [python-docx](https://python-docx.readthedocs.io/)
- Text chunking by [LangChain](https://github.com/langchain-ai/langchain)

---

**⚠️ Disclaimer:** This is an AI assistant for informational purposes only. Always verify critical regulatory information with official source documents before making compliance decisions.

**🔒 Privacy Note:** All processing happens locally. No data is sent to external APIs or cloud services.
