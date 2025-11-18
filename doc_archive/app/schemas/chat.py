from typing import List, Dict, Optional
from pydantic import BaseModel

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None

class ChatResponse(BaseModel):
    response: str
    usage: Dict[str, int]
    model: str