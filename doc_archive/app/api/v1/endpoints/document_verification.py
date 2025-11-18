from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List
import os
import time
import logging
import re
from datetime import datetime
import asyncio

from app.schemas.document_schemas import (
    DocumentVerificationRequest,
    DocumentVerificationResponse,
    ProfileReportData,
    PayslipExtractionResponse,
    PayslipData,
    DocumentVerificationResult,
    EnhancedDocumentVerificationResponse,
    AdditionalData,
    DocumentSummaryResponse,
    Summary,
    KeyFactors,
    ProcessingSummary
)
from app.services.profile_report import (
    generate_profile_report,
    extract_payslip_data,
    detect_document_type,
    detect_sensitive_identifiers,
    generate_document_summary,
    extract_text_from_pdf_native
)
from app.services.forensic import analyze_document
from app.services.files import archive_service
from app.services.document_verification import verify_document
from app.core.config import settings
from app.schemas.document_schemas import DocumentType as SchemaDocumentType
from app.utils.download_utils import download_documents_from_urls
from app.core.optimized_extraction import extract_documents_fast

logger = logging.getLogger(__name__)
router = APIRouter()

# Request timeout configuration (in seconds)
REQUEST_TIMEOUT = 3600  # 1 hour timeout
DOCUMENT_BATCH_SIZE = 10  # Process in batches to reduce memory

@router.post("/verify", response_model=DocumentSummaryResponse)
async def verify_documents(
    request: DocumentVerificationRequest,
    background_tasks: BackgroundTasks
) -> DocumentSummaryResponse:
    """
    Verify and analyze uploaded documents to:
    1. Extract and validate document owners
    2. Classify document types
    3. Determine employment status
    4. Generate comprehensive profile report with summarized response
    
    Note: Request will timeout after 1 hour if documents exceed processing limits.
    """
    try:
        start_time = datetime.utcnow()

        # Wrap with timeout to prevent hanging
        try:
            # Download documents from URLs with timeout
            document_paths = await asyncio.wait_for(
                download_documents_from_urls(request.documents),
                timeout=600  # 10 minute timeout for downloads
            )
        except asyncio.TimeoutError:
            logger.error("Document download timeout")
            raise HTTPException(status_code=408, detail="Document download timeout - took too long")

        if not document_paths:
            raise HTTPException(status_code=400, detail="No documents could be downloaded")
        
        # WARN if too many documents (memory issue)
        if len(document_paths) > 50:
            logger.warning(f"Large batch detected: {len(document_paths)} documents. Processing may be slow. Recommended: ≤50 docs per request")

        # TRY FAST EXTRACTION FIRST (40s for 33 docs vs 778s old pipeline)
        logger.info(f"📄 Starting fast extraction on {len(document_paths)} documents...")
        documents_for_fast = []
        for doc_path in document_paths:
            try:
                text = extract_text_from_pdf_native(doc_path)
                if text:
                    documents_for_fast.append({
                        "fileName": os.path.basename(doc_path),
                        "content": text[:5000]  # Limit to 5K chars per doc
                    })
            except Exception as e:
                logger.warning(f"Could not extract text from {doc_path}: {e}")

        # Fast extraction (semantic type detection + spaCy NER + passport detection)
        fast_result = None
        if documents_for_fast:
            try:
                # Add timeout for fast extraction too
                logger.info(f"⚡ Fast extraction with 5min timeout...")
                fast_result = await asyncio.wait_for(
                    extract_documents_fast(documents_for_fast),
                    timeout=300  # 5 minute timeout for fast extraction
                )
                logger.info(f"✅ Fast extraction completed: {fast_result['summary']['processingTimeMs']}ms")
            except asyncio.TimeoutError:
                logger.warning("⏱️  Fast extraction timeout - continuing with profile report")
                fast_result = None
            except Exception as e:
                logger.warning(f"⚠️  Fast extraction fallback: {e}")

        # Generate profile report with real processing (backup/detailed extraction)
        logger.info(f"📊 Starting profile report generation (45min timeout)...")
        try:
            profile_report = await asyncio.wait_for(
                generate_profile_report(document_paths),
                timeout=2700  # 45 minute timeout for profile report
            )
        except asyncio.TimeoutError:
            logger.error("Profile report generation timeout")
            # Return partial response with what we have
            processing_time = round((datetime.utcnow() - start_time).total_seconds(), 2)
            batch_id = f"{datetime.now().strftime('%Y%m%d')}-{request.additionalData.userName.replace(' ', '-')}-{int(time.time())}"
            
            return DocumentSummaryResponse(
                status="partial_timeout",
                batchId=batch_id,
                summary=Summary(
                    ownerName=request.additionalData.userName,
                    documentCount=len(document_paths),
                    verifiedDocuments=0,
                    documentsWithSensitiveData=0,
                    documentsWithFormatErrors=len(document_paths),
                    averageConfidence=0.0,
                    dominantDocumentType="unknown",
                    languagesDetected=[],
                    fiscalResidency="Unknown",
                    employmentType=request.additionalData.employmentType,
                    hasPassport=False,
                    passportNumber=None,
                    passportVerified=False,
                    notes=f"⏱️ Processing timeout after {processing_time}s. Partial results available. Try with fewer documents."
                ),
                groupedDocuments={},
                keyFactors=KeyFactors(
                    ownerId=request.additionalData.userName.replace(" ", "_").upper(),
                    totalDocumentsAnalyzed=len(document_paths),
                    highConfidenceDocs=0,
                    primaryLanguage="unknown",
                    businessCategory="Unknown",
                    financialPeriodCoverage="Unknown",
                    incomeStability="Unknown",
                    riskSummary=f"⏱️ Request timeout after {processing_time}s"
                ),
                processingSummary=ProcessingSummary(
                    systemVersion="v2.3.4",
                    pipelinesUsed=["fast_extraction"] if fast_result else [],
                    modelsUsed=["donut", "spacy"],
                    processingTimeSeconds=processing_time,
                    detectedFormatWarnings=[f"Processing timeout - {len(document_paths)} documents could not be fully processed"],
                    qualityAssurance="Partial - timeout occurred"
                )
            )

        # Calculate processing time
        processing_time = round((datetime.utcnow() - start_time).total_seconds(), 2)

        # Generate batch ID
        batch_id = f"{datetime.now().strftime('%Y%m%d')}-{request.additionalData.userName.replace(' ', '-')}-{int(time.time())}"

        # Generate document summary response
        response = await generate_document_summary(profile_report, processing_time, batch_id)

        # Log fast extraction performance if available
        if fast_result:
            logger.info(f"Fast extraction: {fast_result['summary']['successCount']}/{fast_result['summary']['totalProcessed']} documents, "
                       f"avg confidence: {fast_result['summary']['avgConfidence']}, "
                       f"time: {fast_result['summary']['processingTimeMs']}ms")

        # Schedule cleanup
        background_tasks.add_task(archive_service.cleanup_old_files)

        logger.info(f"Document verification completed: {len(document_paths)} documents processed in {processing_time}s")
        return response

    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}", exc_info=True)
        # Fallback to basic error response
        return DocumentSummaryResponse(
            status="error",
            batchId=f"error-{int(time.time())}",
            summary=Summary(
                ownerName=request.additionalData.userName,
                documentCount=0,
                verifiedDocuments=0,
                documentsWithSensitiveData=0,
                documentsWithFormatErrors=0,
                averageConfidence=0.0,
                dominantDocumentType="unknown",
                languagesDetected=[],
                fiscalResidency="Unknown",
                employmentType="Unknown",
                hasPassport=False,
                passportNumber=None,
                passportVerified=False,
                notes=f"Processing failed: {str(e)}"
            ),
            groupedDocuments={},
            keyFactors=KeyFactors(
                ownerId=request.additionalData.userName.replace(" ", "_").upper(),
                totalDocumentsAnalyzed=0,
                highConfidenceDocs=0,
                primaryLanguage="unknown",
                businessCategory="Unknown",
                financialPeriodCoverage="Unknown",
                incomeStability="Unknown",
                riskSummary=f"Error: {str(e)}"
            ),
            processingSummary=ProcessingSummary(
                systemVersion="v2.3.4",
                pipelinesUsed=[],
                modelsUsed=[],
                processingTimeSeconds=0.0,
                detectedFormatWarnings=[],
                qualityAssurance="Processing failed"
            )
        )
@router.get("/status/{report_id}")
async def get_verification_status(report_id: str):
    """Get the status of a document verification process"""
    try:
        # Check if archive_service has the method before calling
        if hasattr(archive_service, "get_report_status"):
            status = await archive_service.get_report_status(report_id)
        else:
            status = "unknown"
        return {
            "status": status,
            "report_id": report_id,
            "last_updated": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting report status: {str(e)}", exc_info=True)
        raise HTTPException(status_code=404, detail=f"Report not found: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check endpoint - returns system status and recommendations"""
    import psutil
    
    try:
        # Get system stats
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Determine system load
        if cpu_percent > 80 or memory.percent > 85:
            health_status = "degraded"
            recommendation = "System under load. Reduce batch size or wait before sending new requests."
        elif cpu_percent > 60 or memory.percent > 70:
            health_status = "ok"
            recommendation = "System OK. Recommended batch size: 10-20 documents."
        else:
            health_status = "healthy"
            recommendation = "System healthy. Can process up to 50 documents per request."
        
        return {
            "status": health_status,
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_available_gb": round(memory.available / (1024**3), 2),
            "recommendation": recommendation,
            "suggested_batch_size": "10-20" if cpu_percent > 60 else "50",
            "max_request_timeout_seconds": 3600
        }
    except Exception as e:
        logger.warning(f"Health check error: {e}")
        return {
            "status": "unknown",
            "error": str(e),
            "recommendation": "System status unknown. Proceed with caution."
        }


async def process_single_doc(doc_url: str, request_id_suffix: int):
    """Helper function to process one document."""
    doc_path = await archive_service.download_document(doc_url)
    doc_size = doc_path.stat().st_size
    doc_path_str = str(doc_path)

    # Detect document type
    doc_type, type_confidence = await detect_document_type(doc_path_str)

    # Detect sensitive identifiers
    sensitive_ids = await detect_sensitive_identifiers(doc_path_str)

    # Run document verification
    verification_result = await verify_document(
        doc_path_str,
        {
            "request_id": f"req_{int(time.time())}_{request_id_suffix}",
            "timestamp": datetime.utcnow().isoformat()
        }
    )

    # Extract data based on document type
    extracted_data = None
    if doc_type == "payslip":
        extracted_data = await extract_payslip_data(doc_path_str)

    result = DocumentVerificationResult(
        document_url=doc_url,
        document_type=doc_type,
        extracted_data=extracted_data,
        sensitive_identifiers=sensitive_ids,
        verification_result=verification_result.dict() if verification_result else None,
        confidence=type_confidence
    )
    return result, doc_size


@router.post("/verify/enhanced", response_model=EnhancedDocumentVerificationResponse)
async def verify_documents_enhanced(
    request: DocumentVerificationRequest,
    background_tasks: BackgroundTasks
) -> EnhancedDocumentVerificationResponse:
    """Enhanced verification for multiple documents in parallel."""
    try:
        start_time = time.time()

        # Process documents in parallel
        tasks = [process_single_doc(doc_url, i) for i, doc_url in enumerate(request.documents)]
        results = await asyncio.gather(*tasks)

        verification_results = [res[0] for res in results]
        total_size = sum(res[1] for res in results)

        end_time = time.time()
        time_taken = round(end_time - start_time, 2)

        # Schedule cleanup
        background_tasks.add_task(archive_service.cleanup_old_files)

        response = EnhancedDocumentVerificationResponse(
            requestData=request,
            verificationResults=verification_results,
            reportStartedAt=start_time,
            reportGeneratedAt=end_time,
            timeTaken=time_taken,
            mergedFileSizeInKbs=round(total_size / 1024, 2)
        )

        logger.info(f"Enhanced document verification completed: {len(results)} documents processed")
        return response

    except Exception as e:
        logger.error(f"Error in enhanced document verification: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error in enhanced document verification: {str(e)}"
        )
