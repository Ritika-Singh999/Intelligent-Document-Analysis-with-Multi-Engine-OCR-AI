from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel, ConfigDict, Field
from enum import Enum

class DocumentType(str, Enum):
    PAYSLIP = "payslip"
    BANK_STATEMENT = "bank_statement"
    DNI = "dni"
    SSN = "ssn"
    PASSPORT = "passport"
    JOB_CONTRACT = "job_contract"
    JOB_OFFER = "job_offer"
    GOVERNMENT_FORM = "government_form"
    TAX_DOCUMENT = "tax_document"
    RESERVATION = "reservation"
    INVOICE = "invoice"
    RECEIPT = "receipt"
    CERTIFICATE = "certificate"
    LICENSE = "license"
    PERMIT = "permit"
    APPLICATION = "application"
    AGREEMENT = "agreement"
    CONTRACT = "contract"
    STATEMENT = "statement"
    REPORT = "report"
    FORM = "form"
    UNKNOWN = "unknown"

class SensitiveIdentifier(BaseModel):
    type: str  # e.g., "passport_number", "ssn", "dni", "pan"
    value: str
    confidence: float = Field(ge=0.0, le=1.0)
    method: Optional[str] = None  # detection method (regex, llm, etc)
    location: Optional[str] = None  # page/section where found
    status: Optional[str] = "detected"  # status of the detection

class AdditionalData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    employmentType: str
    userName: str
    documentNo: str

class DocumentVerificationRequest(BaseModel):
    additionalData: AdditionalData
    documents: List[str]

class DocumentMetadata(BaseModel):
    name: str
    documentNameTranslated: Dict[str, str]
    url: str
    filepath: str
    mime_type: str
    filename: str
    document_id: str

class DocumentInfo(BaseModel):
    documentName: str
    documentNameTranslated: Dict[str, str]
    documentId: str
    data: DocumentMetadata

class ClassifiedFile(BaseModel):
    filename: str
    documents: List[Dict[str, str]]

class ProfileData(BaseModel):
    owner: str
    is_uploader: bool
    dni: str
    passportNo: str
    remarks: str
    documents: List[str]
    classified_files: List[ClassifiedFile]
    employment_type: str

class ProfileReport(BaseModel):
    allDocuments: List[DocumentInfo]
    matchedDocuments: List[DocumentInfo]
    otherDocuments: List[DocumentInfo]
    documentsRequired: List[str]
    employmentType: str
    employmentTypeFound: str
    employmentTypeInput: str

class ProfileReportData(BaseModel):
    owner: str
    data: ProfileData
    report: ProfileReport

class Earnings(BaseModel):
    Basic_Pay: Dict[str, Union[float, int, str]]
    Dearness_Allowance: Dict[str, Union[float, int, str]]
    Conveyance_Allowance: Dict[str, Union[float, int, str]]
    Medical_Allowance: Dict[str, Union[float, int, str]]
    House_Rent_Allowance: Dict[str, Union[float, int, str]]
    Food_Allowance: Dict[str, Union[float, int, str]]

class Deductions(BaseModel):
    Professional_Tax: Dict[str, Union[float, int, str]]
    Provident_Fund: Dict[str, Union[float, int, str]]
    Employee_State_Insurance: Dict[str, Union[float, int, str]]

class ExtractedFields(BaseModel):
    model_config = ConfigDict(extra='allow')  # Allow any additional fields from LLM extraction

class DocumentData(BaseModel):
    document_name: str
    document_type: str
    passport_found: bool
    fields: ExtractedFields
    confidence_score: float

class KeyFactors(BaseModel):
    identity_data: Dict[str, Any]
    income_data: Dict[str, Any]
    employment_history: List[Dict[str, Any]]
    bank_details: Optional[Dict[str, Any]]
    tax_returns_history: List[Dict[str, Any]]
    passport: Optional[str]

class ConfidenceSummary(BaseModel):
    document_type_confidence: float
    field_extraction_confidence: float

class ProcessedOwnerData(BaseModel):
    owner_name: str
    documents: List[DocumentData]
    key_factors: KeyFactors
    confidence_summary: ConfidenceSummary
    timestamp: str

class PayslipData(BaseModel):
    model_config = ConfigDict(extra='allow')  # Allow any additional fields

class GroupedDocuments(BaseModel):
    owner_name: str
    documents: List[Dict[str, Any]]
    document_count: int
    document_types: List[str]

class PayslipExtractionResponse(BaseModel):
    additionalData: AdditionalData
    documents: List[Dict[str, Any]]
    verifiedDocuments: Optional[List[Dict[str, Any]]] = None
    groupedDocuments: Optional[List[GroupedDocuments]] = None
    status: str
    message: str

class DocumentVerificationResponse(BaseModel):
    requestData: DocumentVerificationRequest
    profileReports: List[ProfileReportData]
    reportStartedAt: float
    reportGeneratedAt: float
    timeTaken: float
    mergedFileSizeInKbs: float

class DocumentProcessingResponse(BaseModel):
    
    owners: List[ProcessedOwnerData]
    status: str
    processing_stats: Dict[str, Any]  # Includes timing, performance metrics
    message: str

class DocumentVerificationResult(BaseModel):
    document_url: str
    document_type: DocumentType
    extracted_data: Optional[Dict[str, Any]] = None  # Raw extracted data from Donut/LLM
    sensitive_identifiers: List[SensitiveIdentifier]
    verification_result: Optional[Dict[str, Any]] = None  # From document_verification service
    confidence: float

class EnhancedDocumentVerificationResponse(BaseModel):
    requestData: DocumentVerificationRequest
    verificationResults: List[DocumentVerificationResult]
    reportStartedAt: float
    reportGeneratedAt: float
    timeTaken: float
    mergedFileSizeInKbs: float

class Summary(BaseModel):
    ownerName: str
    documentCount: int
    verifiedDocuments: int
    documentsWithSensitiveData: int
    documentsWithFormatErrors: int
    averageConfidence: float
    dominantDocumentType: str
    languagesDetected: List[str]
    fiscalResidency: str
    employmentType: str
    hasPassport: bool
    passportNumber: Optional[str] = None
    passportVerified: bool
    notes: str

# ========== Per-Document Extraction Models with Source Tracking ==========

class ExtractionSource(BaseModel):
    """Track where each field was extracted from"""
    fileName: str
    documentType: str
    sourceUrl: Optional[str] = None
    extractionMethod: str = "llm"  # llm, ocr, regex, manual
    confidence: float = Field(ge=0.0, le=1.0, default=0.9)
    pageNumber: Optional[int] = None
    extractedAt: Optional[str] = None  # ISO timestamp

class TaxDocumentExtract(BaseModel):
    fileName: str
    rfc: Optional[str] = None
    fiscalYear: Optional[str] = None
    totalGrossIncome: Optional[str] = None
    passportDetected: bool = False
    passportNumber: Optional[str] = None
    documentType: Optional[str] = None
    submissionDate: Optional[str] = None
    operationNumber: Optional[str] = None
    fiscalPeriod: Optional[str] = None
    
    # NEW: Source tracking for audit trail
    _source: Optional[ExtractionSource] = None
    _fieldSources: Optional[Dict[str, ExtractionSource]] = None  # per-field tracking

class InvoiceExtract(BaseModel):
    fileName: str
    issuerRfc: Optional[str] = None
    invoiceId: Optional[str] = None
    totalAmount: Optional[str] = None
    passportDetected: bool = False
    passportNumber: Optional[str] = None
    issuerName: Optional[str] = None
    serviceDescription: Optional[str] = None
    
    # NEW: Source tracking
    _source: Optional[ExtractionSource] = None
    _fieldSources: Optional[Dict[str, ExtractionSource]] = None

class ReceiptExtract(BaseModel):
    fileName: str
    receiptId: Optional[str] = None
    issuerName: Optional[str] = None
    totalAmount: Optional[str] = None
    passportDetected: bool = False
    passportNumber: Optional[str] = None
    issuerRfc: Optional[str] = None
    
    # NEW: Source tracking
    _source: Optional[ExtractionSource] = None
    _fieldSources: Optional[Dict[str, ExtractionSource]] = None

class PayslipExtract(BaseModel):
    fileName: str
    employerName: Optional[str] = None
    paymentPeriod: Optional[str] = None
    netPay: Optional[str] = None
    employeeName: Optional[str] = None
    passportDetected: bool = False
    passportNumber: Optional[str] = None
    grossPay: Optional[str] = None
    
    # NEW: Source tracking
    _source: Optional[ExtractionSource] = None
    _fieldSources: Optional[Dict[str, ExtractionSource]] = None

class BankStatementExtract(BaseModel):
    fileName: str
    accountHolder: Optional[str] = None
    clabeId: Optional[str] = None
    statementPeriod: Optional[str] = None
    closingBalance: Optional[str] = None
    passportDetected: bool = False
    passportNumber: Optional[str] = None
    bankName: Optional[str] = None
    
    # NEW: Source tracking
    _source: Optional[ExtractionSource] = None
    _fieldSources: Optional[Dict[str, ExtractionSource]] = None

class EmploymentContractExtract(BaseModel):
    fileName: str
    contractingParty: Optional[str] = None
    passportDetected: bool = False
    passportNumber: Optional[str] = None
    contractType: Optional[str] = None
    representativeName: Optional[str] = None
    
    # NEW: Source tracking
    _source: Optional[ExtractionSource] = None
    _fieldSources: Optional[Dict[str, ExtractionSource]] = None

class PassportExtract(BaseModel):
    fileName: str
    mrz: Optional[str] = None
    name: Optional[str] = None
    nationality: Optional[str] = None
    passportId: str
    dob: Optional[str] = None
    expiryDate: Optional[str] = None
    passportDetected: bool = True
    
    # NEW: Source tracking
    _source: Optional[ExtractionSource] = None
    _fieldSources: Optional[Dict[str, ExtractionSource]] = None

class GroupedDocument(BaseModel):
    count: int
    avgConfidence: float
    samplePeriodRange: Optional[str] = None
    avgNetPay: Optional[str] = None
    commonEmployers: Optional[List[str]] = None
    commonAccounts: Optional[List[str]] = None
    balanceRange: Optional[str] = None
    summaryInsight: Optional[str] = None
    totalValueRange: Optional[str] = None
    avgAmount: Optional[str] = None
    passportNumber: Optional[str] = None
    nationality: Optional[str] = None
    verified: Optional[bool] = None
    fiscalYearsCovered: Optional[List[str]] = None

class KeyFactors(BaseModel):
    ownerId: str
    totalDocumentsAnalyzed: int
    highConfidenceDocs: int
    taxId: Optional[str] = None
    passportNumber: Optional[str] = None
    primaryLanguage: str
    businessCategory: str
    financialPeriodCoverage: str
    incomeStability: str
    riskSummary: str

class ProcessingSummary(BaseModel):
    systemVersion: str
    pipelinesUsed: List[str]
    modelsUsed: List[str]
    processingTimeSeconds: float
    detectedFormatWarnings: List[str]
    qualityAssurance: str
    verificationSummary: Optional[Dict[str, Any]] = None

class FieldAuditTrail(BaseModel):
    """Track where specific field values came from across documents"""
    fieldName: str  # e.g., "rfc", "passportNumber", "totalIncome"
    fieldValue: Any
    sourceDocuments: List[Dict[str, Any]]  # List of {fileName, documentType, extractionMethod, confidence}
    isPrimary: bool = False  # If true, this is the main/verified source
    crossReferences: Optional[List[str]] = None  # Other documents that mention this value
    verificationStatus: str = "unverified"  # unverified, verified, flagged, cross-validated

class DocumentAuditLog(BaseModel):
    """Comprehensive audit log for all documents processed"""
    ownerName: str
    totalDocuments: int
    fieldAuditTrails: List[FieldAuditTrail]  # Track for each unique field
    crossDocumentValidations: Dict[str, List[str]]  # field -> [sources where found]
    documentProcessingOrder: List[Dict[str, str]]  # {fileName, documentType, processingSeq}

class DocumentSummaryResponse(BaseModel):
    status: str
    batchId: str
    summary: Summary
    groupedDocuments: Dict[str, Union[List[TaxDocumentExtract], List[InvoiceExtract], List[ReceiptExtract], 
                                       List[PayslipExtract], List[BankStatementExtract], 
                                       List[EmploymentContractExtract], PassportExtract, List[Dict[str, Any]]]]
    keyFactors: KeyFactors
    processingSummary: ProcessingSummary
    
    # NEW: Audit trail for traceability
    auditLog: Optional[DocumentAuditLog] = None
