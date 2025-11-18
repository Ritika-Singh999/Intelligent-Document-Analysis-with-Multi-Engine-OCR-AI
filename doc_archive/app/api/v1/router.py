from fastapi import APIRouter
from .endpoints import upload, health, chat, public, reports, document_verification

api_router = APIRouter(prefix="/api/v1")

# include routers from endpoints
api_router.include_router(health.router)
api_router.include_router(upload.router, prefix="/documents")
api_router.include_router(chat.router)
api_router.include_router(public.router)
api_router.include_router(reports.router)
api_router.include_router(document_verification.router)
