"""Schemas for document extraction and processing."""
from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field

class SensitiveInformation(BaseModel):
    type: str
    value: str
    confidence: float = Field(ge=0.0, le=1.0)

class ContentSchema(BaseModel):
    extracted_fields: Dict[str, Any]
    models_used: List[str]
    confidence: float = Field(ge=0.0, le=1.0)
    sensitive_information: Dict[str, tuple[str, float]]

class DocumentMetadata(BaseModel):
    uuid: str
    document_type: str
    owner_name: str
    has_passport: bool
    processing_stats: Dict[str, Any]
    timestamp: float

class ExtractedDocument(BaseModel):
    content_schema: ContentSchema
    metadata: DocumentMetadata
    raw_text_snapshot: Optional[str] = None

class ExtractedField(BaseModel):
    value: Any
    confidence: float = Field(ge=0.0, le=1.0)
    source: str  # Which model extracted this (donut/llm/regex)
    extraction_time: str

class KeyFactors(BaseModel):
    detected_fields: Dict[str, ExtractedField] = Field(default_factory=dict)
    field_types: List[str] = Field(default_factory=list)  # List of detected field types
    important_fields: List[str] = Field(default_factory=list)  # Fields marked as important by LLM
    salary_range: Optional[str] = None
    employment_status: Optional[str] = None
    has_passport: bool = False
    passport_number: Optional[str] = None
    employment_type: Optional[str] = None

class ConfidenceSummary(BaseModel):
    overall: float = Field(ge=0.0, le=1.0)
    by_document_type: Dict[str, float] = Field(default_factory=dict)
    by_field_type: Dict[str, float] = Field(default_factory=dict)
    document_count: int = Field(ge=0)

class ProcessedOwner(BaseModel):
    owner_name: str
    owner_id: str
    documents: List[ExtractedDocument]
    key_factors: KeyFactors
    confidence_summary: ConfidenceSummary

class ProcessingSummary(BaseModel):
    total_documents: int = Field(ge=0)
    total_owners: int = Field(ge=0)
    processing_time: float = Field(ge=0.0)
    success_rate: float = Field(ge=0.0, le=1.0)
    errors: List[str] = Field(default_factory=list)
    summary_details: Optional[Dict[str, Any]] = None

class ProfileReport(BaseModel):
    batch_id: str
    timestamp: str
    owners: List[ProcessedOwner]
    processing_summary: ProcessingSummary