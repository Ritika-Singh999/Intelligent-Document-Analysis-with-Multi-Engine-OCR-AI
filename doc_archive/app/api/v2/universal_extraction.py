"""
Universal Document Extraction API Endpoint
Handles: PDF/JPG/PNG URLs → Full multilingual extraction with grouping
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List
import logging
from datetime import datetime
import asyncio

from app.core.universal_extractor import extract_documents_universal
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/extract", tags=["universal-extraction"])

class DocumentURL(BaseModel):
    """Document URL input model."""
    url: str = Field(..., description="Direct URL to PDF, JPG, or PNG document")
    metadata: dict = Field(default_factory=dict, description="Optional metadata")

class UniversalExtractionRequest(BaseModel):
    """Request model for universal extraction."""
    documents: List[DocumentURL] = Field(..., description="List of document URLs to extract")
    batchId: str = Field(default=None, description="Optional batch ID for tracking")
    options: dict = Field(default_factory=dict, description="Extraction options")

class UniversalExtractionResponse(BaseModel):
    """Response model matching exact JSON shape."""
    status: str
    batchId: str
    summary: dict
    groupedDocuments: dict
    keyFactors: dict
    processingSummary: dict

@router.post("/extract", response_model=dict)
async def universal_extract(
    request: UniversalExtractionRequest,
    background_tasks: BackgroundTasks
) -> dict:
    """
    Universal multilingual document extraction endpoint.
    
    Features:
    - Automatic OCR (PaddleOCR → ONNX → Tesseract fallback)
    - Language detection & translation to English
    - Document type auto-classification
    - Intelligent grouping by document type
    - Key field extraction with regex + NER
    - Passport detection on every document
    
    Request:
    {
        "documents": [
            {"url": "https://example.com/doc1.pdf"},
            {"url": "https://example.com/doc2.jpg"}
        ],
        "batchId": "optional",
        "options": {}
    }
    
    Response:
    {
        "status": "success|partial|failed",
        "batchId": "...",
        "summary": { ... },
        "groupedDocuments": { ... },
        "keyFactors": { ... },
        "processingSummary": { ... }
    }
    """
    try:
        if not request.documents:
            raise HTTPException(status_code=400, detail="No documents provided")
        
        # Extract URLs
        urls = [doc.url for doc in request.documents]
        
        logger.info(f"Starting universal extraction for {len(urls)} documents")
        
        # Process documents
        result = await extract_documents_universal(urls)
        
        # Log successful extraction
        logger.info(f"Extraction completed: {result['summary']['verifiedDocuments']}/{len(urls)} successful")
        
        return result
        
    except Exception as e:
        logger.error(f"Extraction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

@router.get("/extract/status/{batchId}")
async def get_extraction_status(batchId: str) -> dict:
    """
    Get status of batch extraction.
    (Placeholder for async job tracking)
    """
    return {
        "status": "completed",
        "batchId": batchId,
        "message": "Extraction completed. Use /extract endpoint for results."
    }

@router.post("/extract/async", response_model=dict)
async def universal_extract_async(
    request: UniversalExtractionRequest,
    background_tasks: BackgroundTasks
) -> dict:
    """
    Async universal extraction - returns job ID immediately.
    Use /extract/status/{batchId} to check progress.
    """
    try:
        if not request.documents:
            raise HTTPException(status_code=400, detail="No documents provided")
        
        batch_id = request.batchId or f"async_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        urls = [doc.url for doc in request.documents]
        
        # Queue background task
        background_tasks.add_task(extract_documents_universal, urls)
        
        logger.info(f"Queued async extraction: {batch_id} for {len(urls)} documents")
        
        return {
            "status": "queued",
            "batchId": batch_id,
            "documentCount": len(urls),
            "message": "Extraction queued. Check /extract/status/{batchId} for progress"
        }
        
    except Exception as e:
        logger.error(f"Async extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Async extraction failed: {str(e)}")
