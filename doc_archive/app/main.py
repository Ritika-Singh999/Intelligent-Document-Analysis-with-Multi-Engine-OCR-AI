from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, SkipValidation
from typing import List, Optional, Dict, Any, Union
import os
from dotenv import load_dotenv
import logging
from pathlib import Path
import uvicorn
import asyncio

from app.api.v1.router import api_router
from app.core.donut import get_donut
from app.core.llm import LLMConfig

# Load environment variables
load_dotenv()

# Logging (minimal for production)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Setup Tesseract OCR ===
# Set TESSDATA_PREFIX to find language files (spa.traineddata, eng.traineddata, etc.)
if not os.getenv("TESSDATA_PREFIX"):
    # Windows: Tesseract-OCR typically installs at C:\Program Files\Tesseract-OCR
    # Data files are in C:\Program Files\Tesseract-OCR\tessdata
    possible_paths = [
        r"C:\Program Files\Tesseract-OCR\tessdata",
        r"C:\Program Files (x86)\Tesseract-OCR\tessdata",
        os.path.join(os.path.expanduser("~"), ".tesseract", "tessdata"),
    ]
    for path in possible_paths:
        if os.path.isdir(path):
            os.environ["TESSDATA_PREFIX"] = path
            logger.info(f"Set TESSDATA_PREFIX to {path}")
            break
    else:
        logger.warning("TESSDATA_PREFIX not found. Tesseract may fail for some languages.")

# Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-pro")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
DONUT_PRELOAD = os.getenv("DONUT_PRELOAD", "false").lower() in ("1", "true", "yes")

# --- Lifespan Events ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        logger.info("⏱️  Starting application (models load lazily on first request)...")
        
        # Initialize LLM config (non-blocking, lazy loads on first use)
        app.state.llm_config = LLMConfig()
        app.state.llm_config.initialize()  # This now returns quickly
        logger.info("✅ LLM configuration loaded (models will load on first request)")

        # SKIP preloading Donut model at startup - load on first request instead
        # This was causing startup delays
        if DONUT_PRELOAD:
            logger.warning("DONUT_PRELOAD=true but skipping at startup to prevent delays. Will load on first request.")

        await chat_memory.connect()
        logger.info("✅ Chat memory initialized")
        logger.info("✅ Application startup complete (Ready for requests)")
        
    except asyncio.CancelledError:
        logger.warning("Startup cancelled, shutting down gracefully")
        raise
    except Exception as e:
        logger.error(f"Startup warning: {e} (App will continue, attempting recovery on first request)")
        # Don't fail startup entirely - allow graceful degradation

    yield

    # Shutdown
    logger.info("Shutting down")

# FastAPI app initialization
app = FastAPI(
    title="Document Checker API - Minimal",
    lifespan=lifespan
)

# CORS 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Simple Vector Store (Fallbac
# k) 
class SimpleDocStore:
    def __init__(self):
        self.docs = []  # (id, text, metadata)

    def add(self, doc_id: str, text: str, metadata: dict = None):
        self.docs.append((doc_id, text, metadata or {}))

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        matches = []
        for doc_id, text, metadata in self.docs:
            score = text.lower().count(query.lower())
            if score > 0:
                matches.append((score, doc_id, text, metadata))
        matches.sort(reverse=True, key=lambda x: x[0])
        return [{"id": m[1], "text": m[2], "metadata": m[3], "score": m[0]} for m in matches[:top_k]]

vector_store = SimpleDocStore()

# Chat Memory Manager (Redis Fallback)
try:
    import aioredis
    redis_available = True
except Exception:
    aioredis = None
    redis_available = False

class MemoryManager:
    def __init__(self, redis_url: str = None, max_history: int = 5):
        self.max_history = max_history
        self.redis_url = redis_url
        self._store = {}
        self._redis = None

    async def connect(self):
        if aioredis and self.redis_url:
            try:
                self._redis = await aioredis.from_url(self.redis_url, encoding="utf-8", decode_responses=True)
                logger.info("Connected to Redis for chat memory")
            except Exception as e:
                logger.warning(f"Could not connect to Redis: {e}. Using in-memory memory.")
                self._redis = None
        else:
            logger.info("Redis not available; using in-memory chat memory")

    async def add_message(self, session_id: str, message: str):
        if self._redis:
            key = f"chat:{session_id}"
            await self._redis.rpush(key, message)
            await self._redis.ltrim(key, -self.max_history, -1)
        else:
            hist = self._store.setdefault(session_id, [])
            hist.append(message)
            self._store[session_id] = hist[-self.max_history:]

    async def get_history(self, session_id: str) -> List[str]:
        if self._redis:
            key = f"chat:{session_id}"
            return await self._redis.lrange(key, 0, -1)
        return list(self._store.get(session_id, []))

chat_memory = MemoryManager(redis_url=REDIS_URL)

# Pydantic Schemas
class IngestRequest(BaseModel):
    id: Optional[str] = None
    text: str
    metadata: Optional[Dict[str, Any]] = None

class QARequest(BaseModel):
    question: str
    doc_ids: Optional[List[str]] = None
    session_id: Optional[str] = "default"

class DocumentCheckRequest(BaseModel):
    documents: List[str]
    additionalData: Optional[Dict[str, SkipValidation[Any]]] = None

class DocumentVerificationResponse(BaseModel):
    status: str
    details: Dict[str, SkipValidation[Any]]
    issues: Optional[List[Dict[str, SkipValidation[Any]]]] = None



# Helper: PDF Text Extraction
async def extract_text_from_pdf(doc_path: str) -> str:
    try:
        from app.utils.pdf_forensics.core.pdf_loader import PDFLoader
        async with PDFLoader(doc_path) as loader:
            return loader.extract_text()
    except Exception as e:
        logger.error(f"Error extracting text from {doc_path}: {e}")
        return ""

# Health Check
@app.get("/health")
async def health():
    return {"status": "ok"}

# Ingest Endpoint
@app.post("/ingest")
async def ingest(req: IngestRequest):
    doc_id = req.id or f"doc_{len(vector_store.docs)+1}"
    vector_store.add(doc_id, req.text, req.metadata or {})
    return {"message": "ingested", "id": doc_id}

# Document Verification 
@app.post("/check-documents")
async def check_documents(req: DocumentCheckRequest):
    session_id = f"verify_{len(vector_store.docs)}"
    all_texts = []

    for doc_path in req.documents:
        text = await extract_text_from_pdf(doc_path)
        all_texts.append(text or f"Could not read text from {doc_path}")

    combined_text = "\n\n".join(all_texts)
    doc_id = f"batch_{len(vector_store.docs)}"
    metadata = {"type": "verification_batch", "documents": req.documents}
    if req.additionalData:
        metadata.update(req.additionalData)

    vector_store.add(doc_id, combined_text, metadata)

    verification_questions = [
        "Check for inconsistencies in these documents.",
        "Verify if documents follow the same structure.",
        "Check if any document seems suspicious.",
        "Is the user info consistent across documents?"
    ]

    issues = []
    details = {}

    for question in verification_questions:
        result = await qa(QARequest(
            question=question,
            doc_ids=[doc_id],
            session_id=session_id
        ))
        details[question] = result["answer"]
        if any(keyword in result["answer"].lower() for keyword in ["inconsistent", "irregular", "suspicious", "different", "error", "warning", "issue"]):
            issues.append({"type": "potential_issue", "description": result["answer"]})

    return DocumentVerificationResponse(
        status="completed",
        details=details,
        issues=issues if issues else None
    )

# --- Document Analysis (OCR → Chunk → LLM → Merge → Output) ---
from app.services.profile_report import generate_profile_report
from app.services.pipeline import run_pipeline

@app.post("/analyze-documents")
async def analyze_documents(req: DocumentCheckRequest):
    
    try:
        report = await generate_profile_report(req.documents)
        return report
    except Exception as e:
        logger.error(f"Document analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Document analysis failed: {str(e)}")


# --- Process Documents (OCR -> Chunk -> Donut -> Merge -> JSON) ---
@app.post("/process-documents")
async def process_documents(req: DocumentCheckRequest):
    
    try:
        import asyncio

        # Run the synchronous pipeline in a thread to avoid blocking the event loop
        report = await asyncio.to_thread(run_pipeline, req.documents)
        return report
    except Exception as e:
        logger.error(f"Pipeline processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline processing failed: {str(e)}")

# --- Process Payload (Download URLs -> Process Documents) ---
from app.schemas.document_schemas import DocumentVerificationRequest
from app.utils.download_utils import download_documents_from_urls

@app.post("/process-payload")
async def process_payload(req: DocumentVerificationRequest):
 
    try:
        # Download documents from URLs
        logger.info(f"Downloading {len(req.documents)} documents from URLs")
        downloaded_files = await download_documents_from_urls(req.documents)

        try:
            # Process downloaded documents using existing pipeline
            import asyncio
            logger.info(f"Processing {len(downloaded_files)} downloaded documents")
            report = await asyncio.to_thread(run_pipeline, downloaded_files)

            # Add additional data to the report
            if req.additionalData:
                report["additionalData"] = req.additionalData

            # Add verified documents info if present
            if hasattr(req, 'verifiedDocuments') and req.verifiedDocuments:
                report["verifiedDocuments"] = req.verifiedDocuments

            return report

        finally:
            # Clean up downloaded files
            import os
            for filepath in downloaded_files:
                try:
                    os.unlink(filepath)
                    logger.info(f"Cleaned up temporary file: {filepath}")
                except Exception as e:
                    logger.warning(f"Failed to clean up {filepath}: {e}")

    except Exception as e:
        logger.error(f"Payload processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Payload processing failed: {str(e)}")

#QA Endpoint
@app.post("/qa")
async def qa(req: QARequest):
    await chat_memory.add_message(req.session_id, f"Q: {req.question}")
    candidates = [d for d in vector_store.docs if not req.doc_ids or d[0] in req.doc_ids]
    if not candidates:
        return {"answer": "No relevant documents found.", "sources": []}

    context = "\n\n".join([c[1] for c in candidates])
    try:
        text = f"Simulated answer: Context analyzed for question '{req.question}'."  # simplified fallback
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        raise HTTPException(status_code=500, detail=f"LLM call failed: {e}")

    await chat_memory.add_message(req.session_id, f"A: {text}")
    return {"answer": text, "sources": [c[0] for c in candidates]}

# Include Routers
app.include_router(api_router)

# Include v2 routers (universal extraction)
try:
    from app.api.v2 import router as api_v2_router
    app.include_router(api_v2_router)
    logger.info("API v2 router loaded successfully")
except Exception as e:
    logger.warning(f"Could not load API v2 router: {e}")

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

