import os
import json
import logging
import time
import asyncio
from typing import List, Dict, Any

from PIL import Image

try:
    from app.core.ocr_engines import OcrEngineFactory
    _HAS_MULTI_OCR = True
except ImportError:
    _HAS_MULTI_OCR = False

try:
    import pytesseract
    _HAS_TESSERACT = True
except ImportError:
    pytesseract = None
    _HAS_TESSERACT = False

from app.services.profile_report import (
    pdf_to_images,
    extract_with_donut_image as extract_with_donut,
    merge_page_level_data as merge_json,
    semantic_detect_document_type as guess_document_type,
    detect_has_passport,
)

logger = logging.getLogger(__name__)

if not _HAS_MULTI_OCR:
    logger.warning("Multi-engine OCR not available, falling back to Tesseract")

class SimpleTextSplitter:
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def create_documents(self, texts: List[str]) -> List:
        documents = []
        for text in texts:
            words = text.split()
            for i in range(0, len(words), self.chunk_size - self.overlap):
                chunk = " ".join(words[i:i + self.chunk_size])
                if chunk.strip():
                    documents.append(type('Document', (), {
                        'page_content': chunk,
                        'metadata': {'chunk_index': i}
                    })())
        return documents

class DocumentPipeline:

    def __init__(self, output_dir: str = None):
        self.text_splitter = SimpleTextSplitter()
        self.output_dir = output_dir or os.path.join(os.getcwd(), "documents", "processed")
        os.makedirs(self.output_dir, exist_ok=True)

    def _ocr_image(self, img: Image.Image) -> str:
        """Extract text using fallback OCR (PaddleOCR > Tesseract synchronously)
        IMPORTANT: Converts PIL Image to numpy array for compatibility.
        This is sync because it's called from async and we can't nest event loops."""
        
        import numpy as np
        
        # Convert PIL Image to numpy array ONCE for all engines
        if isinstance(img, Image.Image):
            img_array = np.array(img)
            logger.debug(f"Converted PIL {img.format} ({img.size}) to numpy {img_array.shape}")
        else:
            img_array = img
        
        # Try PaddleOCR first (fastest)
        if _HAS_MULTI_OCR:
            try:
                from app.core.ocr_engines import extract_with_paddle
                # Pass numpy array directly
                text, confidence = extract_with_paddle(img_array)
                if text and confidence > 0.5:
                    logger.debug(f"✓ PaddleOCR: {len(text)} chars, confidence: {confidence:.2f}")
                    return text
                elif text:
                    logger.debug(f"PaddleOCR low confidence ({confidence:.2f}), trying fallback")
            except Exception as e:
                logger.debug(f"PaddleOCR failed: {e}, falling back to Tesseract")
        
        # Fallback to Tesseract (reliable)
        if _HAS_TESSERACT:
            try:
                from app.core.ocr_engines import extract_with_tesseract
                # Pass numpy array to tesseract
                text, confidence = extract_with_tesseract(img_array)
                if text:
                    logger.debug(f"✓ Tesseract: {len(text)} chars, confidence: {confidence:.2f}")
                    return text
            except Exception as e:
                logger.warning(f"Tesseract OCR failed: {e}")
        
        # If both fail, return empty
        logger.warning("All OCR engines failed for this image")
        return ""

    def _extract_fields(self, merged_data: Dict[str, Any], ocr_text: str) -> Dict[str, Any]:
        """Extract and normalize fields from merged data with confidence scoring."""
        fields = {}
        
        # Common field mappings
        field_mappings = {
            "name": ["name", "owner", "employee_name", "full_name", "applicant_name"],
            "email": ["email", "contact_email"],
            "phone": ["phone", "contact_phone", "mobile"],
            "address": ["address", "location", "residence"],
            "organization": ["organization", "company", "employer", "issuer"],
            "document_id": ["document_id", "id_number", "passport_number", "rfc", "dni"],
            "date": ["date", "issue_date", "created_at", "submission_date"],
            "amount": ["amount", "total", "gross_pay", "net_pay", "balance"],
            "type": ["type", "document_type"],
        }
        
        for field_name, possible_keys in field_mappings.items():
            for key in possible_keys:
                if key in merged_data and merged_data[key]:
                    value = merged_data[key]
                    # Determine confidence based on source
                    confidence = 0.85 if key == "document_type" else 0.75
                    fields[field_name] = {
                        "value": str(value),
                        "confidence": confidence,
                        "source_key": key,
                        "extracted_from": "donut" if key in merged_data else "ocr"
                    }
                    break  # Take first found key
        
        return fields

    def _calculate_confidence(self, page_results: List[Dict]) -> float:
        """Calculate average confidence from all page extractions."""
        confidences = []
        for page in page_results:
            if page.get("donut") and isinstance(page["donut"], dict):
                # Extract confidence from donut results
                if "confidence" in page["donut"]:
                    confidences.append(float(page["donut"]["confidence"]))
        
        return (sum(confidences) / len(confidences)) if confidences else 0.65

    async def process(self, document_paths: List[str]) -> Dict[str, Any]:
        """Process documents ASYNC - non-blocking to prevent system hang.
        Handles multiple owners - groups results by owner.
        Strategy: Yield to event loop frequently to prevent blocking other requests."""
        start_time = time.time()
        results = []
        owners_map = {}  # Track documents by owner

        for path_idx, path in enumerate(document_paths):
            abs_path = os.path.abspath(path)
            logger.info(f"Pipeline processing [{path_idx+1}/{len(document_paths)}]: {abs_path}")
            doc_start = time.time()

            try:
                pages = pdf_to_images(abs_path)
                page_results = []
                accumulated_text = ""

                for i, page in enumerate(pages):
                    # Multi-engine OCR (synchronous, but fast)
                    ocr_text = self._ocr_image(page)
                    if ocr_text:
                        accumulated_text += ocr_text + "\n"

                    chunks = self._chunk_text(ocr_text)

                    # Donut visual extraction (can be slow - 500ms per page)
                    # Run it but yield to event loop immediately after
                    donut_result = extract_with_donut(page)

                    page_results.append({
                        "page_index": i,
                        "ocr_text": ocr_text,
                        "chunks": chunks,
                        "donut": donut_result,
                    })
                    
                    # Yield to event loop after each page to prevent blocking
                    await asyncio.sleep(0.01)  # Small yield interval

                # Merge and classify
                merged = merge_json([p["donut"] for p in page_results if p.get("donut")])
                merged["document_type"] = guess_document_type(accumulated_text or json.dumps(merged))
                merged["owner"] = merged.get("owner") or merged.get("name") or "Unknown"
                merged["has_passport"] = detect_has_passport(accumulated_text or json.dumps(merged))
                
                # Extract and normalize fields with confidence
                extracted_fields = self._extract_fields(merged, accumulated_text)
                confidence_score = self._calculate_confidence(page_results)
                
                # Get owner name for grouping
                owner_name = merged["owner"]
                
                out = {
                    "document": os.path.basename(path),
                    "owner": owner_name,
                    "document_type": merged.get("document_type", "unknown"),
                    "confidence_score": round(confidence_score, 3),
                    "extraction_summary": {
                        "fields_extracted": len(extracted_fields),
                        "pages_processed": len(page_results),
                        "has_passport": merged.get("has_passport", False),
                    },
                    "extracted_fields": extracted_fields,
                    "pages": page_results,
                    "extracted_data": merged,
                    "processing_time_seconds": round(time.time() - doc_start, 2)
                }

                # Group results by owner
                if owner_name not in owners_map:
                    owners_map[owner_name] = []
                owners_map[owner_name].append(out)

                out_path = os.path.join(self.output_dir, os.path.basename(path) + ".json")
                try:
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(out, f, ensure_ascii=False, indent=2)
                    logger.info(f"✅ Processed {os.path.basename(path)} ({out['processing_time_seconds']}s) - Owner: {owner_name}")
                except Exception as e:
                    logger.error(f"Failed to save pipeline output for {path}: {e}")

                results.append(out)
                
            except Exception as doc_error:
                logger.error(f"❌ Error processing {path}: {doc_error}")
                results.append({
                    "document": os.path.basename(path),
                    "error": str(doc_error),
                    "processing_time_seconds": round(time.time() - doc_start, 2)
                })
            
            # Yield between documents to prevent blocking
            await asyncio.sleep(0.01)

        total_time = round(time.time() - start_time, 2)
        
        # Calculate statistics per owner
        owners_stats = {}
        for owner, docs in owners_map.items():
            owner_time = sum(d["processing_time_seconds"] for d in docs)
            owners_stats[owner] = {
                "document_count": len(docs),
                "total_time_seconds": owner_time,
                "avg_time_per_doc": round(owner_time / len(docs), 2),
                "documents": [d["document"] for d in docs]
            }
        
        report = {
            "status": "completed",
            "documents": results,
            "owners_summary": owners_stats,
            "total_owners": len(owners_map),
            "total_processing_time_seconds": total_time,
            "documents_count": len(results),
            "avg_time_per_document": round(total_time / len(results), 2) if results else 0,
            "ocr_engine": "Multi-engine (PaddleOCR > ONNX > Tesseract)" if _HAS_MULTI_OCR else "Tesseract only"
        }
        
        # Log summary by owner
        logger.info(f"Pipeline completed: {len(results)} documents from {len(owners_map)} owner(s) in {total_time}s")
        for owner, stats in owners_stats.items():
            logger.info(f"  Owner '{owner}': {stats['document_count']} documents, {stats['total_time_seconds']}s total")
        
        return report

def run_pipeline(document_paths: List[str], output_dir: str = None) -> Dict[str, Any]:
    """Sync wrapper for async pipeline - handles event loop."""
    p = DocumentPipeline(output_dir=output_dir)
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is already running (in async context), create task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, p.process(document_paths))
                return future.result(timeout=3600)  # 1 hour timeout
        else:
            # Run normally
            return asyncio.run(p.process(document_paths))
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "documents": []
        }
