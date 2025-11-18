from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.files import archive_service

router = APIRouter()

class Question(BaseModel):
    question: str

@router.post('/chat')
async def chat(question: Question):
    try:
        answer = archive_service.ask_question(question.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
