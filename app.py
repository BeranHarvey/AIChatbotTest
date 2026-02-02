from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag import rag_query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
import logging

# config logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def index():
    try:
        with open("chat_ui.html") as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="chat_ui.html not found")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class ChatRequest(BaseModel):
    query: str

@app.post("/chat")
def chat(chat_request: ChatRequest):
    if not chat_request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    if len(chat_request.query) > 2000:
        raise HTTPException(status_code=400, detail="Query is too long (max. 2000 characters)")
    
    def event_generator():
        try:
            for token in rag_query(chat_request.query):
                yield token
        except Exception as e:
            logger.error(f"Error in rag_query: {e}")
            yield f"An error occurred processing your request: {e}"
            
    return StreamingResponse(event_generator(), media_type="text/plain")

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}