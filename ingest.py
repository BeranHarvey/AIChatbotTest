import os
import glob
import chromadb
import urllib.parse
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from docx import Document
from pypdf import PdfReader

# config 
docs_path = "docs"
chroma_path = "chroma_db"
collection_name = "my_docs"

# Check for folder existence
if not os.path.exists(docs_path):
    print(f"Documents folder '{docs_path}' does not exist. Please create it and add documents.")
    exit(1)

# Embedding model
try:
    embedder = SentenceTransformer("intfloat/e5-large-v2")
except Exception as e:
    print("Error loading embedding model:", e)
    exit(1)

# Load text documents
def load_txt(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error loading text file {path}: {e}")
        return None
    
def load_pdf(path):
    try:
        reader = PdfReader(path)
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        if not pages:
            print(f"No extractable text found in PDF file {path}.")
            return None
        return "\n".join(pages)
    except Exception as e:
        print(f"Error loading PDF file {path}: {e}")
        return None

def load_docx(path):
    try:
        doc = Document(path)
        text = "\n".join(p.text for p in doc.paragraphs if p.text)
        if not text.strip():
            print(f"No extractable text found in DOCX file {path}.")
            return None
        return text
    except Exception as e:
        print(f"Error loading DOCX file {path}: {e}")
        return None
            
def load_document(path):
    if path.endswith(".txt"):
        return load_txt(path)
    elif path.endswith(".pdf"):
        return load_pdf(path)
    elif path.endswith(".docx"):
        return load_docx(path)
    else:
        print(f"Unsupported file format for file {path}. Skipping.")
        return None

documents = []
file_paths = glob.glob(os.path.join(docs_path, "*"))

if not file_paths:
    print(f"No documents found in folder '{docs_path}'. Please add .txt, .pdf, or .docx files.")
    exit(1)

print(f"Found {len(file_paths)} files in '{docs_path}'")

for path in file_paths:
    filename = os.path.basename(path)
    print(f"Loading document: {filename}")
    text = load_document(path)
    if text:
        documents.append({
            "text": text,
            "source": urllib.parse.unquote(filename)
        })

if not documents:
    print("No valid documents were loaded. Exiting.")
    exit(1)

print(f"Loaded {len(documents)} documents")

# Chunking
try:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = []
    for doc in documents:
        for chunk in splitter.split_text(doc["text"]):
            chunks.append({
                "text": chunk,
                "source": doc["source"]
            })
    print(f"Created {len(chunks)} chunks")
except Exception as e:
    print("Error during text chunking:", e)
    exit(1)

# Vector DB
try:
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection(collection_name)

    # Clear existing collection
    existing_count = collection.count()
    if existing_count > 0:
        client.delete(collection_name)
        collection = client.create_collection(collection_name)
except Exception as e:
    print("Error connecting to Chroma DB:", e)
    exit(1)

# Insert embeddings
try: 
    for i, chunk in enumerate(chunks):
        emb = embedder.encode(f"passage: {chunk['text']}").tolist()
        collection.add(
            ids=[f"chunk_{i}"],
            documents=[chunk["text"]],
            embeddings=[emb],
            metadatas=[{"source": chunk["source"]}]
        )
        print(f"Ingestion complete: {len(chunks)} chunks added.")
except Exception as e:
    print("Error during embedding insertion:", e)
    exit(1)
