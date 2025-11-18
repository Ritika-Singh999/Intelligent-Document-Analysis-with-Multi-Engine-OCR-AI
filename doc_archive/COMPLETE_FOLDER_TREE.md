# 🗂️ PROJECT FOLDER TREE & VISUAL GUIDE

## Complete Directory Tree

```
E:\n\
│
├── doc_archive/                          ← MAIN APPLICATION ROOT
│   │
│   ├── app/                              ← APPLICATION PACKAGE
│   │   ├── __init__.py
│   │   ├── main.py                       ⭐ FastAPI entry point (349 lines)
│   │   │
│   │   ├── api/                          📡 REST API Layer
│   │   │   ├── v1/                       🔵 Production APIs
│   │   │   │   ├── router.py             ← Routes dispatcher
│   │   │   │   └── endpoints/            ← Individual endpoint handlers
│   │   │   │       ├── health.py         ← System health check
│   │   │   │       ├── chat.py           ← Chat endpoint
│   │   │   │       ├── documents.py      ← Document management
│   │   │   │       ├── document_verification.py  ⭐ MAIN endpoint (322 lines)
│   │   │   │       ├── upload.py         ← File upload
│   │   │   │       ├── reports.py        ← Report generation
│   │   │   │       ├── public.py         ← Public endpoints
│   │   │   │       └── key_factors.py    ← Key factors extraction
│   │   │   │
│   │   │   └── v2/                       🟢 Experimental APIs
│   │   │       ├── __init__.py
│   │   │       └── universal_extraction.py
│   │   │
│   │   ├── core/                         ⚙️  CORE PROCESSING ENGINES
│   │   │   ├── config.py                 ← Configuration management
│   │   │   ├── llm.py                    ⭐ Gemini LLM (127 lines)
│   │   │   ├── donut.py                  ← Vision model (Donut)
│   │   │   ├── document_types.py         ← Document classification
│   │   │   ├── ocr_engines.py            ⭐ Multi-engine OCR (412 lines)
│   │   │   ├── enhanced_models.py        ← Enhanced LLM wrapper
│   │   │   ├── fast_extraction_hybrid.py ← Hybrid extraction
│   │   │   ├── optimized_extraction.py   ← Fast extraction pipeline
│   │   │   └── universal_extractor.py    ← Universal extractor
│   │   │
│   │   ├── services/                     🔄 BUSINESS LOGIC LAYER
│   │   │   ├── pipeline.py               ⭐ Main processing pipeline (296 lines)
│   │   │   ├── profile_report.py         ← Comprehensive reporting (1500 lines)
│   │   │   ├── document_verification.py  ← Verification logic
│   │   │   ├── forensic.py               ← Forensic analysis
│   │   │   ├── files.py                  ← File management
│   │   │   ├── owner_processor.py        ← Owner processing
│   │   │   ├── parallel_processor.py     ← Parallel batch processing
│   │   │   ├── document_extractor.py     ← Field extraction
│   │   │   └── document_field_extractors.py ← Field-specific extractors
│   │   │
│   │   ├── schemas/                      📦 DATA MODELS (Pydantic)
│   │   │   ├── document_schemas.py       ← Document data models
│   │   │   ├── extraction_schemas.py     ← Extraction data models
│   │   │   ├── response_schemas.py       ← Response data models
│   │   │   ├── verification_schemas.py   ← Verification models
│   │   │   ├── chat.py                   ← Chat models
│   │   │   ├── key_factor_schemas.py     ← Key factor models
│   │   │   ├── reports.py                ← Report models
│   │   │   └── __init__.py
│   │   │
│   │   ├── prompts/                      📝 LLM PROMPT TEMPLATES
│   │   │   ├── document_prompts.py       ← Main prompt templates
│   │   │   ├── document-owners-and-types.txt
│   │   │   ├── document-owners.txt
│   │   │   ├── employment-type.txt
│   │   │   ├── forensic-report.txt
│   │   │   ├── key-factors/
│   │   │   │   └── dni.txt
│   │   │   └── tink-reports/
│   │   │       ├── expense.txt
│   │   │       └── income.txt
│   │   │
│   │   └── utils/                        🛠️  UTILITY FUNCTIONS
│   │       ├── download_utils.py         ← URL download helpers
│   │       ├── helpers.py                ← General utilities
│   │       ├── cancellable_task.py       ← Async task management
│   │       ├── highlight_pdf.py          ← PDF annotation
│   │       └── pdf_forensics/            ← PDF forensic analysis
│   │           ├── run_all_detectors.py
│   │           └── core/
│   │               ├── pdf_loader.py
│   │               └── pdf_loader_ocr.py
│   │
│   ├── postman/                          🧪 API TESTING
│   │   ├── document_check.postman_collection.json  ← API collection
│   │   └── local.environment.json        ← Environment setup
│   │
│   ├── scripts/                          📜 UTILITY SCRIPTS
│   │   └── cleanup_pyc.py                ← Clean compiled Python
│   │
│   ├── cache/                            💾 CACHING SYSTEM
│   │   ├── embeddings/                   ← Cached embeddings
│   │   ├── extract_text/                 ← Cached text
│   │   ├── files/                        ← Cached files
│   │   └── vectors/                      ← Cached vectors
│   │
│   ├── documents/                        📄 OUTPUT STORAGE
│   │   └── vector_store/                 ← Embeddings storage
│   │
│   ├── .env                              ⚙️  ENVIRONMENT CONFIG
│   ├── setup.py                          📦 Package setup
│   ├── requirements.txt                  📋 Dependencies
│   ├── README.md                         📖 Main documentation
│   ├── __init__.py
│   │
│   ├── 📚 DOCUMENTATION FILES
│   ├── COMPLETE_PROJECT_STRUCTURE.md     ← THIS FILE (Project overview)
│   ├── COMPLETE_CODEBASE_REVIEW.md       ← Full code review
│   ├── ENDPOINT_CONNECTIONS.md           ← API mapping
│   ├── INTEGRATION_FIXES_COMPLETED.md    ← Integration fixes
│   ├── FIXED_MODULE_REFERENCES.md        ← Reference fixes
│   ├── PERFORMANCE_OPTIMIZATION.md       ← Performance guide
│   ├── PERFORMANCE_CRITICAL_FIXES.md     ← 5 Critical fixes
│   ├── PIL_TO_NUMPY_CONVERSION_FIX.md    ← Image format fix
│   ├── SESSION_5_SUMMARY.md              ← Latest session
│   ├── QUICK_REFERENCE.md                ← Troubleshooting
│   ├── AUDIT_TRAIL_EXAMPLE.md
│   └── TODO.md                           ← Project tasks
│
├── documents/                            📄 SHARED OUTPUT (Outside app)
│   └── vector_store/
│
├── cache/                                💾 SHARED CACHE (Outside app)
│   ├── embeddings/
│   ├── extract_text/
│   ├── files/
│   └── vectors/
│
├── venv/                                 🐍 Python Virtual Environment
│
├── .vscode/                              ⚙️  VS Code Configuration
│
└── .pytest_cache/                        🧪 Test cache
```

---

## 📊 LAYER ARCHITECTURE

```
┌─────────────────────────────────────────────┐
│           EXTERNAL CLIENTS                  │
│  (Postman, Web UI, Python Scripts)          │
└──────────────────┬──────────────────────────┘
                   │ HTTP/REST
                   ↓
┌─────────────────────────────────────────────┐
│            API LAYER (app/api/)              │  📡
│  ├─ v1: Production endpoints                │
│  │   ├─ /verify (document verification)     │
│  │   ├─ /health (system monitoring)         │
│  │   ├─ /chat (conversational)              │
│  │   └─ /documents (management)             │
│  └─ v2: Experimental features               │
└──────────────────┬──────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────┐
│       SERVICES LAYER (app/services/)        │  🔄
│  ├─ pipeline.py (main orchestrator)         │
│  ├─ profile_report.py (comprehensive)       │
│  ├─ document_verification.py (validation)   │
│  ├─ forensic.py (analysis)                  │
│  └─ [Other specialized services]            │
└──────────────────┬──────────────────────────┘
                   │
        ┌──────────┼──────────┐
        ↓          ↓          ↓
┌──────────────────────────────────────────────┐
│        CORE LAYER (app/core/)                │  ⚙️
│  ├─ ocr_engines.py (multi-engine OCR)       │
│  │  ├─ PaddleOCR (GPU-accelerated)          │
│  │  ├─ EasyOCR (ONNX quantized)             │
│  │  └─ Tesseract (fallback)                 │
│  ├─ llm.py (Gemini integration)             │
│  ├─ donut.py (vision model)                 │
│  ├─ document_types.py (classification)      │
│  └─ [Other core processors]                 │
└──────────────────┬──────────────────────────┘
                   │
        ┌──────────┴──────────┐
        ↓                     ↓
┌─────────────────────────────────────────────┐
│  EXTERNAL MODELS & SERVICES                 │
│  ├─ Google Gemini API (LLM)                 │
│  ├─ PaddleOCR (GPU)                         │
│  ├─ EasyOCR (ONNX)                          │
│  ├─ Tesseract (System)                      │
│  ├─ Donut (PyTorch)                         │
│  ├─ spaCy (NER)                             │
│  └─ PDF2Image (Conversion)                  │
└─────────────────────────────────────────────┘
```

---

## 🔄 DATA FLOW DIAGRAM

```
USER INPUT (DocumentVerificationRequest)
    │
    ├─ documents: [URL1, URL2, ...] ← Document URLs
    ├─ userName: "John Doe"          ← Owner name
    └─ employmentType: "Full-time"   ← Additional data
    │
    ↓
┌─────────────────────────────────────────────┐
│         DOCUMENT DOWNLOAD PHASE             │
│  download_utils.py                          │
│  └─ Fetch PDFs from URLs (10min timeout)   │
└──────────┬──────────────────────────────────┘
           │
           ↓
┌─────────────────────────────────────────────┐
│         PDF TO IMAGES PHASE                 │
│  profile_report.pdf_to_images()             │
│  └─ Convert PDF pages to PIL images         │
└──────────┬──────────────────────────────────┘
           │
           ↓
┌─────────────────────────────────────────────┐
│       PARALLEL PROCESSING PHASE             │
│                                             │
│  For each image page:                       │
│  ├─ OCR Text Extraction                    │
│  │  ├─ PIL→numpy array conversion          │
│  │  ├─ PaddleOCR (GPU) ✓ or               │
│  │  ├─ EasyOCR (ONNX) ✓ or                │
│  │  └─ Tesseract (fallback) ✓             │
│  │                                          │
│  └─ Vision Model Extraction (Donut)        │
│     └─ Structured data extraction           │
│                                             │
│  Yield to event loop (non-blocking)        │
└──────────┬──────────────────────────────────┘
           │
           ↓
┌─────────────────────────────────────────────┐
│     DOCUMENT ANALYSIS PHASE                 │
│  document_types.py / profile_report.py      │
│                                             │
│  ├─ Document Type Detection (LLM)           │
│  ├─ Owner Name Extraction (spaCy + LLM)    │
│  ├─ Sensitive Data Detection                │
│  └─ Language Detection                      │
└──────────┬──────────────────────────────────┘
           │
           ↓
┌─────────────────────────────────────────────┐
│    FIELD EXTRACTION PHASE                   │
│  document_field_extractors.py               │
│                                             │
│  ├─ Normalize extracted fields              │
│  ├─ Validate format                         │
│  ├─ Calculate confidence scores             │
│  └─ Handle missing fields                   │
└──────────┬──────────────────────────────────┘
           │
           ↓
┌─────────────────────────────────────────────┐
│   CROSS-VALIDATION PHASE                    │
│  services/document_verification.py          │
│                                             │
│  ├─ Owner consistency check                 │
│  ├─ Date consistency check                  │
│  ├─ Format validation                       │
│  └─ Passport detection                      │
└──────────┬──────────────────────────────────┘
           │
           ↓
┌─────────────────────────────────────────────┐
│    AGGREGATION PHASE                        │
│  services/owner_processor.py                │
│                                             │
│  ├─ Group results by owner                  │
│  ├─ Calculate per-owner statistics          │
│  ├─ Determine dominant document type        │
│  └─ Calculate average confidence            │
└──────────┬──────────────────────────────────┘
           │
           ↓
┌─────────────────────────────────────────────┐
│    REPORT GENERATION PHASE                  │
│  profile_report.generate_profile_report()   │
│                                             │
│  ├─ Create summary                          │
│  ├─ Group documents by type                 │
│  ├─ Extract key factors                     │
│  ├─ Generate audit trail                    │
│  └─ Build final response                    │
└──────────┬──────────────────────────────────┘
           │
           ↓
RESPONSE (DocumentSummaryResponse)
├─ status: "success" / "partial_timeout"
├─ batchId: "UUID"
├─ summary: {ownerName, documentCount, ...}
├─ groupedDocuments: {type: [docs]}
├─ keyFactors: {employment, salary, ...}
├─ processingSummary: {time, errors, ...}
└─ auditLog: {trails, validations, order}
```

---

## 📦 FILE SIZE & COMPLEXITY REFERENCE

```
Size (LOC)  File                              Complexity
────────────────────────────────────────────────────────
1500        profile_report.py                 ⭐⭐⭐⭐⭐ (Highest)
412         ocr_engines.py                    ⭐⭐⭐⭐
349         main.py                           ⭐⭐⭐⭐
322         document_verification.py          ⭐⭐⭐⭐
296         pipeline.py                       ⭐⭐⭐⭐
300+        enhanced_models.py                ⭐⭐⭐⭐
300+        optimized_extraction.py           ⭐⭐⭐⭐
300+        fast_extraction_hybrid.py         ⭐⭐⭐
300+        universal_extractor.py            ⭐⭐⭐
300+        document_types.py                 ⭐⭐⭐
127         llm.py                            ⭐⭐⭐
100+        Various schemas                   ⭐⭐
100+        Various services                  ⭐⭐
50          config.py                         ⭐
40          donut.py                          ⭐
```

---

## 🎯 KEY ENDPOINTS REFERENCE

### Document Verification (Main)
```
POST /verify
├─ Input: DocumentVerificationRequest
│  ├─ documents: [URLs]
│  └─ additionalData: {userName, employmentType}
│
├─ Processing:
│  ├─ Download (10min timeout)
│  ├─ OCR extraction (5min timeout)
│  ├─ Profile report (45min timeout)
│  └─ Total (1hour timeout)
│
└─ Output: DocumentSummaryResponse
   ├─ Status: "success" or "partial_timeout"
   ├─ Summary: Document statistics
   ├─ Grouped documents: By type/owner
   ├─ Key factors: Important data
   ├─ Audit trail: Data lineage
   └─ Processing summary: Metrics
```

### Health Check
```
GET /health
├─ Returns: System health metrics
│  ├─ CPU utilization %
│  ├─ Memory utilization %
│  └─ Recommendations for batch size
│
└─ Purpose: Monitor system status
```

### Chat Endpoint
```
POST /chat
├─ Input: ChatRequest {message, documentId}
├─ Processing: LLM-based Q&A
└─ Output: ChatResponse {response}
```

---

## 💾 PERSISTENCE & STORAGE

### Cache System (app/cache/)
```
Cache Types:
├─ embeddings/     ← Cached text embeddings
├─ extract_text/   ← Cached OCR text results
├─ files/          ← Cached file metadata
└─ vectors/        ← Cached vector embeddings
```

### Output Storage (documents/)
```
Generated Files:
├─ vector_store/   ← Indexed embeddings
├─ *.json          ← Processing results
└─ *.pdf           ← Archived documents
```

### Environment Config (.env)
```
GOOGLE_API_KEY=xxx              ← Gemini API key
GEMINI_MODEL=gemini-pro         ← Model name
REDIS_URL=redis://...           ← Cache backend
TESSERACT_CMD=...               ← Tesseract path
DONUT_PRELOAD=false             ← Lazy loading flag
```

---

## ✅ DEPLOYMENT READINESS

### Pre-Deployment Checklist
- [ ] All files syntax verified
- [ ] Dependencies installed (pip install -r requirements.txt)
- [ ] Environment variables set (.env)
- [ ] GPU drivers installed (for PaddleOCR optimization)
- [ ] Tesseract installed and TESSDATA_PREFIX set
- [ ] PostgreSQL/Redis configured (if using external cache)
- [ ] API keys configured (Google Gemini)

### Startup Sequence
1. Load environment variables
2. Initialize FastAPI app
3. Setup CORS middleware
4. Register all routers (v1, v2)
5. Setup lifespan manager
6. Configure LLM (lazy loading)
7. Configure chat memory
8. Start Uvicorn server
9. Ready for requests

### Health Verification
1. `curl http://localhost:8000/health` ✓ Running
2. `curl http://localhost:8000/verify` ← Test with sample
3. Monitor logs for errors
4. Check GPU availability (nvidia-smi)

---

**Document Status:** ✅ Complete  
**Last Updated:** 2025-11-18  
**Scope:** Full project structure + architecture + data flow
