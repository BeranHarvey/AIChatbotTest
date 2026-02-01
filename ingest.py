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

# Embedding model
embedder = SentenceTransformer("intfloat/e5-large-v2")

# Load text documents
def load_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
    
def load_pdf(path):
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)

def load_docx(path):
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text)

def load_document(path):
    if path.endswith(".txt"):
        return load_txt(path)
    elif path.endswith(".pdf"):
        return load_pdf(path)
    elif path.endswith(".docx"):
        return load_docx(path)
    else:
        return None

documents = []

for path in glob.glob(os.path.join(docs_path, "*")):
    text = load_document(path)
    if text:
        documents.append({
            "text": text,
            "source": urllib.parse.unquote(os.path.basename(path))
        })

print(f"Loaded {len(documents)} documents")

# Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
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

# Vector DB
client = chromadb.PersistentClient(path=chroma_path)
collection = client.get_or_create_collection(collection_name)

# Insert embeddings
for i, chunk in enumerate(chunks):
    emb = embedder.encode(f"passage: {chunk['text']}").tolist()
    collection.add(
        ids=[f"chunk_{i}"],
        documents=[chunk["text"]],
        embeddings=[emb],
        metadatas=[{"source": chunk["source"]}]
    )

print(f"Ingestion complete: {len(chunks)} chunks added.")
