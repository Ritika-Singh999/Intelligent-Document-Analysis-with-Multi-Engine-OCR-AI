from fastapi import APIRouter, HTTPException
from typing import List
import time

from schemas.document_schemas import (
    DocumentVerificationRequest,
    DocumentVerificationResponse,
    ProfileReportData
)
from services.profile_report import ProfileReportService
from services.forensic import ForensicService

router = APIRouter()
profile_report_service = ProfileReportService()
forensic_service = ForensicService()

@router.post("/verify", response_model=DocumentVerificationResponse)
async def verify_documents(request: DocumentVerificationRequest) -> DocumentVerificationResponse:
    """
    Verify and analyze uploaded documents to:
    1. Extract and validate document owners
    2. Classify document types
    3. Determine employment status
    4. Generate comprehensive profile report
    """
    try:
        # Start timing
        start_time = time.time()
        
        # Download and process documents
        downloaded_files = await forensic_service.download_documents(request.documents)
        
        # Analyze documents and generate profile reports
        profile_reports = await profile_report_service.analyze_documents(
            documents=downloaded_files,
            additional_data=request.additionalData
        )
        
        # Calculate timing and size metrics
        end_time = time.time()
        time_taken = round(end_time - start_time, 2)
        merged_size = await forensic_service.calculate_merged_size(downloaded_files)
        
        # Prepare response
        response = DocumentVerificationResponse(
            requestData=request,
            profileReports=profile_reports,
            reportStartedAt=start_time,
            reportGeneratedAt=end_time,
            timeTaken=time_taken,
            mergedFileSizeInKbs=merged_size
        )
        
        return response

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing documents: {str(e)}"
        )

@router.get("/status/{report_id}")
async def get_verification_status(report_id: str):
    """
    Get the status of a document verification process
    """
    try:
        status = await profile_report_service.get_report_status(report_id)
        return {"status": status}
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"Report not found: {str(e)}"
        )