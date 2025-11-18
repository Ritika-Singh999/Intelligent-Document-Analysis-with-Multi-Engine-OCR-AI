from fastapi import APIRouter

router = APIRouter()

@router.get('/')
async def index():
    return {"service": "skor-lite - AI document archive (stub)"}
