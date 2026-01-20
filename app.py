from fastapi import FastAPI
from pydantic import BaseModel
from rag import rag_query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.responses import HTMLResponse

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def index():
    with open("chat_ui.html") as f:
        return f.read()

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
    def event_generator():
        for token in rag_query(chat_request.query):
            yield token
            
    return StreamingResponse(event_generator(), media_type="text/plain")