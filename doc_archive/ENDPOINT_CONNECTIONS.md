# `/api/v1/verify` Endpoint Connection Map

## 🎯 Main Endpoint
**File:** `app/api/v1/endpoints/document_verification.py` (245 lines)
**Route:** `POST /api/v1/verify`
**Response Model:** `DocumentSummaryResponse`

---

## 📊 Direct Dependencies (Imports)

### 1. **Schemas** - Data Models
```
app/schemas/document_schemas.py
├── DocumentVerificationRequest      (input request)
├── DocumentSummaryResponse          (output response)
├── Summary                          (response field)
├── KeyFactors                       (response field)
├── ProcessingSummary                (response field)
├── DocumentVerificationResult
├── ProfileReportData
├── PayslipExtractionResponse
├── PayslipData
└── DocumentType
```

### 2. **Services** - Business Logic
```
app/services/profile_report.py (1493 lines) ⭐ MAIN PROCESSOR
├── generate_profile_report()        (processes documents)
├── generate_document_summary()      (creates response)
├── extract_text_from_pdf_native()   (PDF text extraction)
├── detect_document_type()           (classifies documents)
├── detect_sensitive_identifiers()   (finds PII)
└── extract_payslip_data()          (extracts payslip fields)

app/services/files.py
├── archive_service                  (file cleanup)
└── SimpleArchiveService            (mock implementation)

app/services/forensic.py
└── analyze_document()               (PDF forensics)

app/services/document_verification.py
└── verify_document()               (document verification)
```

### 3. **Core/ML Models** - Fast Extraction
```
app/core/optimized_extraction.py (625 lines)
└── extract_documents_fast()        (40s pipeline for 33 docs)
    ├── Semantic document type detection
    ├── spaCy NER (40+ languages)
    ├── Passport detection
    └── Confidence scoring

app/core/ocr_engines.py (410 lines)
├── get_paddleocr()                (PaddleOCR loader)
├── get_onnx_model()               (EasyOCR ONNX models)
├── extract_with_paddle()          (PaddleOCR extraction)
├── extract_with_onnx()            (EasyOCR extraction)
└── extract_with_tesseract()       (Tesseract extraction)

app/core/donut.py
└── get_donut()                    (Donut vision model)

app/core/llm.py
└── EnhancedLLM                    (Gemini LLM for extraction)

app/core/universal_extractor.py (625+ lines)
├── DocumentTypeDetector           (15+ document types)
└── UniversalExtractor             (multilingual extraction)
```

### 4. **Utils** - Helper Functions
```
app/utils/download_utils.py
└── download_documents_from_urls()  (downloads files)

app/utils/helpers.py
├── helpers                         (utility functions)
└── spaCy NER integration

app/utils/pdf_forensics/
└── run_all_detectors.py           (PDF analysis)
```

### 5. **Configuration**
```
app/core/config.py
└── settings                        (app configuration)
```

---

## 🔄 Execution Flow

```
POST /api/v1/verify
    ↓
1. Download documents from URLs
    ↓ (app/utils/download_utils.py)
    
2. Extract text from PDFs
    ↓ (app/core/optimized_extraction.py → extract_text_from_pdf_native)
    
3. Fast extraction pipeline (40s)
    ↓ (app/core/optimized_extraction.py → extract_documents_fast)
    ├── Semantic document type detection
    ├── spaCy NER (entity extraction)
    └── Passport detection
    
4. Profile report generation (backup/detailed)
    ↓ (app/services/profile_report.py → generate_profile_report)
    ├── Donut vision model extraction
    ├── LLM-based field extraction
    ├── Document type detection
    ├── Sensitive data detection
    └── Confidence scoring
    
5. Generate summary response
    ↓ (app/services/profile_report.py → generate_document_summary)
    ├── Group documents by type
    ├── Create audit trail
    └── Build DocumentSummaryResponse
    
6. Return response
    ↓
DocumentSummaryResponse (with 33 documents processed in ~40s)
```

---

## 📦 File Count & Size Summary

| Component | Files | Purpose |
|-----------|-------|---------|
| **Schemas** | 7 files | Data models & validation |
| **Services** | 5 files | Business logic |
| **Core/ML** | 5 files | ML models & extraction |
| **Utils** | 3 folders | Helper functions |
| **Config** | 1 file | Settings |
| **Total** | **20+ files** | Complete pipeline |

---

## 🚀 Performance Metrics

| Stage | Time | Notes |
|-------|------|-------|
| Download | ~2s | 33 documents |
| PDF Text Extract | ~5s | PyMuPDF + OCR fallback |
| Fast Extraction | ~15s | Semantic + spaCy |
| Profile Report | ~600s | Backup/detailed (optional) |
| Summary Gen | ~3s | Response building |
| **Total** | **~40s** | For 33 documents (19.5x speedup) |

---

## 🔗 Key Integrations

### Multi-Engine OCR
- **Primary:** PaddleOCR (GPU-accelerated)
- **Secondary:** EasyOCR (ONNX quantized)
- **Tertiary:** Tesseract (fallback)

### ML Models
- **Donut:** Visual document understanding
- **spaCy:** 40+ language NER
- **Gemini 2.5 Flash:** LLM field extraction
- **Sentence-Transformers:** Semantic similarity

### Document Types Supported
15+ types: tax_document, invoice, receipt, payslip, passport, bank_statement, employment_contract, job_offer, etc.

---

## ⚠️ Error Handling

All errors are caught and converted to graceful fallback responses with:
- Error message in response
- Empty/default values for failed extractions
- Audit trail preserved
- Proper HTTP status codes

---

## 📝 Recent Fixes Applied

✅ **processingSeq type** - Fixed int→str conversion
✅ **Tesseract confidence** - Using image_to_data()
✅ **TESSDATA_PREFIX** - Auto-configured
✅ **PaddleOCR GPU** - GPU auto-detection enabled
✅ **Corrupted PDFs** - Error handling + fallback chain
✅ **Server startup** - All imports fixed

