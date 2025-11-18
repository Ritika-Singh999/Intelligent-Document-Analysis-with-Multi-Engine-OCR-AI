from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from enum import Enum

class VerificationStatus(str, Enum):
    TRUSTED = "trusted"
    SUSPICIOUS = "suspicious"
    FRAUDULENT = "fraudulent"
    
class VerificationConfidence(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    reasons: List[str]

class DocumentAuthenticityResult(BaseModel):
    status: VerificationStatus
    confidence: VerificationConfidence
    metadata: Dict[str, Any] = {}
    forensic_flags: List[str] = []
    content_consistency: bool
    timestamp: str

class VerificationDetail(BaseModel):
    check_name: str
    result: bool
    confidence: float
    details: str

class DocumentVerificationResult(BaseModel):
    document_id: str
    overall_status: VerificationStatus
    authenticity: DocumentAuthenticityResult
    verification_details: List[VerificationDetail]
    risk_score: float = Field(..., ge=0.0, le=1.0)
    verification_timestamp: str