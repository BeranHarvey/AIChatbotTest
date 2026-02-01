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
    
    # Build context block
    context = "\n\n".join(retrieved_docs)
    
    # System prompt + context
    prompt = f"""
You are an expert AI assistant in answering questions based on only the information provided in the following context.
Provide only the final answer to the user.
Remain factual and concise, do not try to guess or add external information.
If the answer is not explicitly stated in the context, say "I don't know based on the documents provided."

Context:
{context}

User question: {user_query}

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
    
    # Yield the cleaned result
    yield cleaned_response

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