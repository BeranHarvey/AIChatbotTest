# Import modules
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import chromadb
import re
import urllib.parse

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
try: 
    chroma_client = chromadb.PersistentClient(path="chroma_db")
    collection = chroma_client.get_or_create_collection("my_docs")
except Exception as e:
    print("Error connecting to Chroma DB:", e)
    raise

# RAG function
def rag_query(user_query):
    try: 
        # Check collection has docs
        collection_count = collection.count()
        if collection_count == 0:
            yield "No documents found in the vector database. Please add documents first."
            return

        # Embed query
        try:
            query_embedding = embedder.encode(f"query: {user_query}").tolist()
        except Exception as e:
            yield f"Error generating embedding for the query: {e}"
            return
        
        # Retrieve top-k documents
        try: 
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=3
            )
        except Exception as e:
            yield f"Error querying the vector database: {e}"
            return

        retrieved_docs = results["documents"][0]
        sources = results.get("metadatas", [[]])[0]

        # Check if any results were found
        if not retrieved_docs or all(not doc.strip() for doc in retrieved_docs):
            yield "No relevant documents found for the query."
            return
        
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
        try:
            response = client.chat.completions.create(
                model="llama-3.2-3b-instruct",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=768,
                stream=True,
                timeout=120
            )
        except Exception as e:
            error_msg = str(e).lower()
            if "connection" in error_msg or "refused" in error_msg:
                yield "Error: Unable to connect to the LLM server. Please ensure it is running."
            elif "timeout" in error_msg:
                yield "Error: The request to the LLM server timed out. Please try again."
            else:
                yield f"Error during LLM request: {e}"
            return
        
        # Buffer the full response first
        full_response = ""
        try: 
            for chunk in response:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    full_response += delta.content
        except Exception as e:
            if full_response:
                yield f"\n\n[Partial response received before error]\n{full_response}\n\nError during streaming: {e}"
            else:
                yield f"Error during streaming response: {e}"
            return
        
        # Strip think blocks from complete response
        cleaned_response = strip_think_blocks(full_response)

        # Extract unique source documents with file paths
        source_info = []
        seen = set()
        for meta in sources:
            source_name = meta.get('source', 'Unknown')
            
            if source_name not in seen:
                seen.add(source_name)
                
                # Use server URL
                encoded_name = urllib.parse.quote(source_name)
                doc_url = f"http://localhost:8000/documents/{encoded_name}"
                
                # Display name can be decoded for readability
                display_name = urllib.parse.unquote(source_name)
                source_info.append(f"- [{display_name}]({doc_url})")

        source_list = "\n".join(source_info)

        # Append sources to response
        final_output = f"{cleaned_response}\n\nSources:\n{source_list}"
        
        # Yield the cleaned result
        yield final_output

    except Exception as e:
        yield f"An unexpected error occurred: {e}"

# Simple CLI test
if __name__ == "__main__":
    print("RAG Chatbot Ready! Ask a question:")
    while True:
        q = input("\n You: ")
        if q.lower() in ("quit", "exit"):
            break
        print("\n Bot: ", end="", flush=True)
        try:
            for token in rag_query(q):
                print(token, end="", flush=True)
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            break
        except Exception as e:
            print(f"\nError: {e}")
        print()