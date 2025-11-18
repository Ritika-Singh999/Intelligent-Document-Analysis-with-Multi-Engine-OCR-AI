from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel, ConfigDict
from datetime import datetime

class ReportBase(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    report_id: str
    created_at: datetime
    status: str

class DocumentReport(ReportBase):
    document_id: str
    analysis_results: Dict[str, Any]
    metadata: Dict[str, Any]

class ForensicReport(ReportBase):
    confidence_score: float
    authenticity_markers: List[str]
    anomalies: List[str]
    recommendations: List[str]

class ProfileReport(ReportBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    profile_id: str
    employment_type: str
    key_findings: List[str]
    document_summaries: List[Dict[str, Any]]

class ReportRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    report_type: str
    document_ids: List[str]
    parameters: Optional[Dict[str, Any]] = None

class ReportResponse(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    status: str
    reports: List[Union[DocumentReport, ForensicReport, ProfileReport]]
    metadata: Dict[str, any]