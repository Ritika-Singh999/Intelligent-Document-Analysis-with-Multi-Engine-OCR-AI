# 🚀 QUICK REFERENCE - SYSTEM PERFORMANCE GUIDE

## If You See These Errors...

### Error 1: "Not supported input data type! Only `numpy.ndarray` and `str` are supported!"
```
❌ PIL.PpmImagePlugin.PpmImageFile ... has been ignored
```
**Status:** ✅ FIXED in Session 5

**What it means:** OCR engine received PIL image instead of numpy array

**Solution:** Already applied in:
- `app/services/profile_report.py` (line 250-320)
- `app/services/pipeline.py` (line 61-102)
- `app/core/ocr_engines.py` (all engines)

**No action needed** - Should be working now

---

### Error 2: "Startup failed: Timeout or hanging"
```
❌ App doesn't respond for 30+ seconds
```
**Status:** ✅ FIXED in Session 5

**What it means:** Model loading at startup is blocking event loop

**Solution:** Already applied in:
- `app/core/llm.py` - Lazy loading
- `app/main.py` - Skip Donut preload

**No action needed** - App should start in <1 second

---

### Error 3: "Request timeout after 30 seconds"
```
❌ Postman: timeout waiting for response
```
**Status:** ✅ FIXED in Session 5

**What it means:** Event loop blocked by synchronous processing

**Solution:** Already applied in:
- `app/services/pipeline.py` - Event loop yields
- `app/services/profile_report.py` - PIL conversion
- `app/api/v1/endpoints/document_verification.py` - Granular timeouts

**No action needed** - Should handle requests properly

---

### Error 4: "loop.run_until_complete called inside async"
```
❌ RuntimeError: This event loop is already running
```
**Status:** ✅ FIXED in Session 5

**What it means:** Nested event loop deadlock

**Solution:** Already applied in:
- `app/services/pipeline.py` (line 61-70)

**No action needed** - Code updated

---

## System Health Checks

### Check 1: Verify Startup
```bash
# Should complete in <1 second
time python -m uvicorn app.main:app --reload

# Look for:
# ✅ "Startup complete"
# ✅ "Application startup complete"
# ❌ NO "Timeout" messages
```

### Check 2: Test Document Processing
```bash
# Should complete in 5-30 seconds depending on pages
curl -X POST "http://localhost:8000/verify" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": ["https://example.com/test.pdf"],
    "additionalData": {"userName": "Test", "employmentType": "Full-time"}
  }'

# Look for:
# ✅ "status": "success"
# ✅ "extraction_summary" with extracted fields
# ✅ "confidence_score" > 0.7
```

### Check 3: Check System Health
```bash
curl http://localhost:8000/health

# Response should show:
{
  "status": "healthy",
  "cpu_percent": <0-100>,
  "memory_percent": <0-100>,
  "recommendation": "..."
}
```

### Check 4: OCR Engine Status
```bash
# Check logs for:
✅ "PaddleOCR loaded successfully"
✅ "GPU detected for PaddleOCR: ..."
✅ "Extracted N lines"

# If you see:
❌ "PaddleOCR import failed"
   → Install: pip install paddleocr
```

---

## Performance Tuning

### For Fast Processing (Best Case)
```
Single 1-page document:
  - Download: 1s
  - OCR: 1-2s
  - Total: ~2-3s
```

### For Medium Processing (Normal)
```
10 pages of documents:
  - Download: 1-2s
  - PDF to images: 2-3s
  - OCR (multi-engine): 5-10s
  - Extraction: 5-10s
  - Total: ~15-25s
```

### For Heavy Processing (Batch)
```
33 documents (100+ pages):
  - Total processing: 3-5 minutes
  - Per-document: ~5-10s
  - Bottleneck: Donut model (500ms per page)
```

---

## Environment Configuration

### Required Environment Variables
```bash
# Google API for Gemini LLM
export GOOGLE_API_KEY="your-api-key-here"

# Optional: Tesseract location (if not in PATH)
export TESSERACT_CMD="C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# Optional: Skip Donut preload (faster startup)
export DONUT_PRELOAD="false"
```

### Recommended for Performance
```bash
# Use GPU if available
export CUDA_VISIBLE_DEVICES="0"

# Batch processing (number of documents per request)
# Recommended: ≤50 documents
```

---

## Common Issues & Quick Fixes

### Issue: "GPU not detected for PaddleOCR"
```
WARNING: No GPU detected for EasyOCR, using CPU
```
**Fix:** 
```bash
# Check NVIDIA GPU
nvidia-smi

# If GPU exists, install CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Restart app
```

### Issue: "Tesseract not found"
```
ERROR: tesseract is not installed
```
**Fix:**
```bash
# Windows: Download from GitHub
# https://github.com/UB-Mannheim/tesseract/wiki

# Or set path manually:
export TESSDATA_PREFIX="C:\\Program Files\\Tesseract-OCR\\tessdata"
```

### Issue: "Memory usage too high"
```
Recommendation: Process fewer documents per request (current: N docs)
```
**Fix:**
- Reduce batch size: Try ≤30 documents per request
- Monitor with: `curl http://localhost:8000/health`

### Issue: "Requests still timing out"
```
⏱️ Processing timeout after 60s
```
**Fix:**
1. Check system resources: `curl http://localhost:8000/health`
2. Reduce batch size or document complexity
3. Check GPU availability: `nvidia-smi`
4. Increase timeout in `document_verification.py` if needed

---

## Performance Monitoring

### Real-time Metrics
```python
# In application logs, look for:
# ✅ "Pipeline processing [1/N]"
# ✅ "PaddleOCR: 1523 chars"
# ✅ "✓ Processed test.pdf (2.34s)"
```

### Check Processing Speed
```bash
# Extract from logs
grep "Processed" app.log | tail -20

# Analyze performance
# Fast: <5s per document
# Normal: 5-10s per document  
# Slow: >10s per document (check resources)
```

---

## When to Escalate

If after applying fixes you still see:
1. ❌ Startup taking >5 seconds
2. ❌ "Not supported input data type" errors
3. ❌ "This event loop is already running"
4. ❌ Requests timing out consistently
5. ❌ Memory usage >80%

**Action:** 
- Collect logs from startup to timeout
- Check environment variables
- Verify GPU/CPU availability
- Contact support with logs

---

## Documentation Index

- **PERFORMANCE_CRITICAL_FIXES.md** - Technical details of all fixes
- **PIL_TO_NUMPY_CONVERSION_FIX.md** - PIL image format error details
- **SESSION_5_SUMMARY.md** - Complete session summary
- **This file** - Quick reference guide

---

**Last Updated:** 2025-11-17  
**Status:** ✅ All critical issues fixed and documented
