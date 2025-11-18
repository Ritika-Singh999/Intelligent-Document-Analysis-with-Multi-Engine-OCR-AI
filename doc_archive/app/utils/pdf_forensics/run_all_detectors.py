import asyncio
from typing import List, Dict, Any
import logging
from datetime import datetime

from .core.pdf_loader import PDFLoader
from .core.utils import (
    ForensicResult,
    analyze_metadata_consistency,
    find_embedded_files,
    is_encrypted_content,
    calculate_entropy
)
from app.utils.helpers import run_in_thread
from app.utils.cancellable_task import cancellable

logger = logging.getLogger(__name__)

class PDFForensicsAnalyzer:
    """Comprehensive PDF forensics analyzer."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.loader = PDFLoader(file_path)
        
    async def __aenter__(self):
        """Async context manager entry."""
        await self.loader.__aenter__()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.loader.__aexit__(exc_type, exc_val, exc_tb)
        
    @cancellable(timeout=60)
    async def run_all_detectors(self) -> List[ForensicResult]:
        """Run all forensic detectors in parallel."""
        results = await asyncio.gather(
            self.check_metadata(),
            self.check_structure(),
            self.check_content(),
            self.check_images(),
            return_exceptions=True
        )
        
        # Filter out exceptions and log them
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Detector error: {result}")
            elif isinstance(result, list):
                valid_results.extend(result)
            else:
                valid_results.append(result)
                
        return valid_results
        
    async def check_metadata(self) -> ForensicResult:
        """Check PDF metadata for suspicious patterns."""
        metadata = await self.loader.metadata
        meta_dict = metadata.to_dict()
        
        # Check consistency
        issues = analyze_metadata_consistency(meta_dict)
        
        # Calculate risk level
        risk_level = "low"
        if len(issues) >= 3:
            risk_level = "high"
        elif len(issues) > 0:
            risk_level = "medium"
            
        return ForensicResult(
            detector_name="metadata_analyzer",
            confidence=0.8,
            findings={
                "issues": issues,
                "metadata": meta_dict
            },
            risk_level=risk_level
        )
        
    async def check_structure(self) -> ForensicResult:
        """Analyze PDF structure for anomalies."""
        doc = self.loader._doc
        
        # Analyze objects
        obj_types = {}
        stream_count = 0
        encrypted_streams = 0
        
        # Run in thread to avoid blocking
        def analyze_objects():
            nonlocal stream_count, encrypted_streams
            for xref in range(1, doc.xref_length()):
                try:
                    obj = doc.xref_object(xref)
                    if not obj:
                        continue
                        
                    # Track object types
                    obj_type = obj.get("/Type", "unknown")
                    obj_types[obj_type] = obj_types.get(obj_type, 0) + 1
                    
                    # Analyze streams
                    if hasattr(obj, 'get_stream'):
                        stream_count += 1
                        stream_data = obj.get_stream()
                        if is_encrypted_content(stream_data):
                            encrypted_streams += 1
                except Exception:
                    continue
                    
        await run_in_thread(analyze_objects)
        
        # Determine risk level
        risk_level = "low"
        if encrypted_streams > 0:
            risk_level = "high"
        elif stream_count > 100:  # Arbitrary threshold
            risk_level = "medium"
            
        # Safe access to PDF version: older/newer PyMuPDF versions may not expose pdf_version attribute
        try:
            pdf_version = doc.pdf_version
        except Exception:
            try:
                pdf_version = getattr(doc, "PDFVersion", None)
            except Exception:
                pdf_version = None

        return ForensicResult(
            detector_name="structure_analyzer",
            confidence=0.9,
            findings={
                "object_types": obj_types,
                "stream_count": stream_count,
                "encrypted_streams": encrypted_streams,
                "version": pdf_version
            },
            risk_level=risk_level
        )
        
    async def check_content(self) -> ForensicResult:
        """Analyze PDF content for suspicious patterns."""
        # Extract text in chunks to handle large files
        text = await run_in_thread(
            self.loader.extract_text,
            0,
            None
        )
        
        # Look for suspicious patterns
        suspicious_patterns = [
            "javascript:",
            "eval(",
            "function(",
            "base64,",
            "document.write"
        ]
        
        findings = {
            "suspicious_patterns": [],
            "hidden_content": False,
            "total_length": len(text)
        }
        
        for pattern in suspicious_patterns:
            if pattern in text.lower():
                findings["suspicious_patterns"].append(pattern)
                
        # Check for potential hidden content
        if text.count("\0") > 0:  # Null bytes
            findings["hidden_content"] = True
            
        risk_level = "low"
        if findings["hidden_content"]:
            risk_level = "high"
        elif findings["suspicious_patterns"]:
            risk_level = "medium"
            
        return ForensicResult(
            detector_name="content_analyzer",
            confidence=0.7,
            findings=findings,
            risk_level=risk_level
        )
        
    async def check_images(self) -> ForensicResult:
        """Analyze images in the PDF."""
        findings = {
            "total_images": 0,
            "suspicious_images": [],
            "image_types": {}
        }
        
        # Analyze each page's images
        for page_num in range(len(self.loader._doc)):
            images = await self.loader.get_page_images(page_num)
            findings["total_images"] += len(images)
            
            for img in images:
                # Track image types
                img_type = img["type"]
                findings["image_types"][img_type] = \
                    findings["image_types"].get(img_type, 0) + 1
                    
                # Check for suspicious characteristics
                if img["size"] < 100:  # Suspiciously small
                    findings["suspicious_images"].append({
                        "page": page_num,
                        "reason": "small_size",
                        "details": img
                    })
                    
        risk_level = "low"
        if findings["suspicious_images"]:
            risk_level = "medium"
            
        return ForensicResult(
            detector_name="image_analyzer",
            confidence=0.8,
            findings=findings,
            risk_level=risk_level
        )

async def analyze_pdf(file_path: str) -> Dict[str, Any]:
    """Analyze PDF file and return comprehensive results."""
    async with PDFForensicsAnalyzer(file_path) as analyzer:
        results = await analyzer.run_all_detectors()
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "file_path": file_path,
            "results": [r.to_dict() for r in results],
            "summary": {
                "risk_level": max((r.risk_level for r in results), key=lambda x: 
                    {"low": 0, "medium": 1, "high": 2}[x]),
                "total_detectors": len(results),
                "high_risk_findings": sum(1 for r in results if r.risk_level == "high")
            }
        }