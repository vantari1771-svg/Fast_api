import os
from dotenv import load_dotenv
from typing import Literal, List

from fastapi import FastAPI
from pydantic import BaseModel
from ollama import Client

load_dotenv()

ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

client = Client(host=ollama_host)
app = FastAPI()

Role = Literal["system", "user", "assistant"]

class Message(BaseModel):
    role: Role
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]

class ChatResponse(BaseModel):
    response: str
    model: str

@app.post("/messages", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        result = client.chat(
            model=request.model,
            messages=[m.model_dump() for m in request.messages],
            stream=False
        )
        return ChatResponse(
            response=result["message"]["content"],
            model=result.get("model", request.model),
        )
    except Exception as e:
        return ChatResponse(
            response=f"[error] {e}",
            model=request.model
        )
