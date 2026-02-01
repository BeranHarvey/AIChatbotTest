# Import modules
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import chromadb
import re

def strip_think_blocks(text: str) -> str:
    """
    Removes <think>...</think> blocks from model output.
    """
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

# Connect to LM Studio LLM
client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)

# Load embedding model
embedder = SentenceTransformer("intfloat/e5-large-v2")

# Connect to Chroma vector DB
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection("my_docs")

# RAG function
def rag_query(user_query):
    # Embed query
    query_embedding = embedder.encode(f"query: {user_query}").tolist()
    
    # Retrieve top-k documents
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )
    retrieved_docs = results["documents"][0]
    sources = results.get("metadatas", [[]])[0]
    
    # Build context block
    context = "\n\n".join([f"[Source: {meta.get('source', 'Unknown')}]\n{doc}" 
                        for doc, meta in zip(retrieved_docs, sources)])
    
    # System prompt + context
    prompt = f"""
You are a helpful medical device QMS assistant. Answer questions based on the provided regulatory documents.

IMPORTANT RULES:
- Write in clear, natural prose - NO bullet points, numbered lists, or excessive formatting
- Explain things clearly and concisely
- Do NOT include document reference markers like (a), (b), Article numbers, or asterisks
- Do NOT mention source document names in your answer (sources will be added automatically)
- If you don't know based on the documents, say so clearly

Context:
{context}

User question: 
{user_query}

Answer:
"""
    
    # Call LM Studio
    response = client.chat.completions.create(
        model="qwen3-14b-mlx",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        stream=True
    )
    
    # Buffer the full response first
    full_response = ""
    for chunk in response:
        delta = chunk.choices[0].delta
        if hasattr(delta, "content") and delta.content:
            full_response += delta.content
    
    # Strip think blocks from complete response
    cleaned_response = strip_think_blocks(full_response)

    # Extract unique source documents
    unique_sources = list(set([meta.get('source', 'Unknown') for meta in sources]))
    source_list = "\n".join([f"- {src}" for src in unique_sources])
    final_output = f"{cleaned_response}\n\Sources:\n{source_list}"
    
    # Yield the cleaned result
    yield final_output

# Simple CLI test
if __name__ == "__main__":
    print("RAG Chatbot Ready! Ask a question:")
    while True:
        q = input("\n You: ")
        if q.lower() in ("quit", "exit"):
            break
        print("\n Bot: ", end="", flush=True)
        for token in rag_query(q):
            print(token, end="", flush=True)
        print()