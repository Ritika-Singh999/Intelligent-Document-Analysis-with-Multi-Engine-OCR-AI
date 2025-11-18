"""Enhanced document processing pipeline with parallel processing and multi-model analysis."""
import os
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from collections import defaultdict
from PIL import Image
import fitz  # PyMuPDF

from app.core.llm import LLMConfig
from app.core.document_types import DocumentType
from app.core.donut import get_donut
from app.services.document_verification import verify_document
from app.services.files import extract_text_from_file
from app.services.profile_report import extract_with_donut_image

logger = logging.getLogger(__name__)

@dataclass
class ProcessingResult:
    """Complete processing results for a document."""
    file_path: str
    doc_type: DocumentType
    features: Dict[str, Any]
    owner_name: str
    verification_result: Dict[str, Any]
    confidence: float
    processed_at: datetime = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "file_path": self.file_path,
            "doc_type": str(self.doc_type),
            "features": self.features,
            "owner_name": self.owner_name,
            "verification_result": self.verification_result,
            "confidence": self.confidence,
            "processed_at": self.processed_at.isoformat()
        }

    def to_json_schema(self) -> Dict[str, Any]:
        """Format as required for unified output JSON, including content_schema if present."""
        from app.core.document_types import normalize_document_type_str
        doc_type_str = normalize_document_type_str(str(self.doc_type))
        passport_found = False
        passport_val = None
        content_schema = None
        if self.features:
            pf = self.features.get("has_passport")
            if pf is not None:
                passport_found = bool(pf)
            passport_val = self.features.get("passport_number")
            content_schema = self.features.get("content_schema")
        doc_name = self.features.get("document_name") or os.path.basename(self.file_path)
        conf = self.confidence or 0.0
        fields = self.features.copy() if self.features else {}
        for k in ["has_passport", "document_type", "owner", "document_name", "content_schema"]:
            fields.pop(k, None)
        result = {
            "document_name": doc_name,
            "document_type": doc_type_str,
            "passport_found": passport_found,
            "fields": fields,
            "confidence_score": conf
        }
        if content_schema is not None:
            result["content_schema"] = content_schema
        return result

class ParallelProcessor:
    """AI-driven universal document processing controller."""
    
    def __init__(self,
                output_dir: Optional[str] = None,
                max_workers: int = 4,
                llm_config: Optional[LLMConfig] = None,
                cache_dir: Optional[str] = None):
        """Initialize processor with configuration.
        
        Args:
            output_dir: Directory to save processing results
            max_workers: Maximum number of parallel workers
            llm_config: LLM configuration for AI processing
            cache_dir: Directory for caching extracted text and features
        """
        self.output_dir = output_dir or os.path.join("documents", "processed")
        self.cache_dir = cache_dir or os.path.join("cache", "processing")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.llm_config = llm_config or LLMConfig()
        
        # Initialize Donut model
        try:
            self.donut_processor, self.donut_model = get_donut()
            logger.info("Initialized document processing pipeline with Donut model")
        except Exception as e:
            logger.warning(f"Failed to initialize Donut model: {e}, will use text-only processing")

    async def process_documents(self, documents: List[str]) -> Dict[str, Any]:
        """Process multiple documents in parallel with AI-based extraction.
        
        Args:
            documents: List of document paths to process
            
        Returns:
            Dictionary with processing results and statistics
        """
        try:
            # Step 1: Process all documents in parallel
            tasks = [self.process_single_document(doc) for doc in documents]
            results = await asyncio.gather(*tasks)
            
            # Step 2: Group by owner
            owners = defaultdict(list)
            for result in results:
                if result:  # Skip failed documents
                        owner = (result.owner_name or "Unknown").strip()
                        owners[owner].append(result)
            
            # Step 3: Extract key factors for each owner
            key_factors_tasks = []
            for owner, docs in owners.items():
                task = asyncio.create_task(self._extract_key_factors(owner, docs))
                key_factors_tasks.append(task)
            
            # Wait for all key factors extraction
            owner_factors = await asyncio.gather(*key_factors_tasks)
            
            # Step 4: Generate summary report
            summary = {
                "status": "completed",
                "processed_at": datetime.now().isoformat(),
                "total_documents": len(documents),
                "successful": len([r for r in results if r]),
                "failed": len([r for r in results if not r]),
                "owners": {},
                "documents": [r.to_dict() for r in results if r]
            }
            
            # Add owner summaries with key factors
            for owner, docs, factors in zip(owners.keys(), owners.values(), owner_factors):
                summary["owners"][owner] = {
                    "document_count": len(docs),
                    "document_types": [str(d.doc_type) for d in docs],
                    "average_confidence": sum(d.confidence for d in docs) / len(docs),
                    "key_factors": factors
                }
            
            # Save summary
            summary_path = os.path.join(self.output_dir, "processing_summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            return summary
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "processed_at": datetime.now().isoformat()
            }

    async def process_single_document(self, doc_path: str) -> Optional[ProcessingResult]:
        """Process single document using AI-driven analysis.
        
        Args:
            doc_path: Path to document file
            
        Returns:
            ProcessingResult if successful, None on failure
        """
        try:
            # Step 1: Start verification in background task
            verify_task = asyncio.create_task(
                verify_document(doc_path, {"request_id": os.path.basename(doc_path)})
            )
            
            # Step 2: Extract text with caching
            cache_key = os.path.basename(doc_path)
            text = await self._get_cached_text(doc_path, cache_key)
            if not text:
                logger.error(f"Could not extract text from {doc_path}")
                return None
            
            # Step 3: AI-based document analysis (parallel)
            doc_info_task = asyncio.create_task(self._analyze_document_info(doc_path, text))
            features_task = asyncio.create_task(self._extract_features(text))
            
            # Step 4: Wait for all tasks
            doc_info, features, verification_result = await asyncio.gather(
                doc_info_task, features_task, verify_task
            )
            
            # Combine features with visual analysis if available
            if doc_info.get("visual_analysis"):
                visual_features = self._extract_visual_features(doc_info["visual_analysis"])
                features = self._merge_features(features, visual_features)
            
            # Step 5: Extract mandatory fields
            owner_info = await self._extract_owner_info(text, features)
            
            return ProcessingResult(
                file_path=doc_path,
                doc_type=doc_info["type"],
                features={
                    **features,
                    "document_name": doc_info["name"],
                    "owner": owner_info,
                    "extracted_at": datetime.now().isoformat()
                },
                owner_name=owner_info["full_name"],
                verification_result=verification_result.dict(),
                confidence=features.get("confidence", 0.0)
            )
            
        except Exception as e:
            logger.error(f"Failed to process {doc_path}: {e}")
            return None

    async def _get_cached_text(self, doc_path: str, cache_key: str) -> Optional[str]:
        """Get text from cache or extract from document."""
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.txt")
        
        # Check cache first
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                return f.read()
                
        # Extract text using existing functionality
        from app.services.files import extract_text_from_file
        text = await extract_text_from_file(doc_path)
        
        # Cache the result
        if text:
            with open(cache_path, "w", encoding="utf-8") as f:
                f.write(text)
                
        return text

    async def _analyze_document_info(self, doc_path: str, text: str) -> Dict[str, Any]:
        """Analyze document using both Donut visual analysis and LLM text analysis."""
        # Start both analyses in parallel
        # Start the LLM analysis
        llm_task = asyncio.create_task(self._llm_analyze_type(text))
        
        # If Donut is available, extract image features
        donut_results = []
        if hasattr(self, 'donut_processor'):
            try:
                # Process document pages
                doc = fitz.open(doc_path)
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    pix = page.get_pixmap()
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.tobytes())
                    donut_result = await extract_with_donut_image(img)
                    if donut_result and not donut_result.get("donut_error"):
                        donut_results.append(donut_result)
            except Exception as e:
                logger.warning(f"Donut analysis failed: {e}")
        
        # Wait for LLM analysis
        llm_result = await llm_task
        
        # Combine results
        combined_result = {
            **llm_result,
            "visual_analysis": donut_results,
            "multi_model": bool(donut_results)
        }
        
        # Update type if Donut found it with high confidence
        if donut_results and "doc_type" in donut_results[0]:
            doc_type = donut_results[0]["doc_type"].lower()
            if doc_type and donut_results[0].get("confidence", 0) > 0.8:
                combined_result["type"] = doc_type
                combined_result["type_source"] = "donut"
        
        return combined_result

    async def _llm_analyze_type(self, text: str) -> Dict[str, str]:
        """Analyze document type using LLM."""
        prompt = """Analyze this document text and determine its type and official name.
        Return ONLY a JSON object with these fields:
        {
            "type": "exact document type (e.g., contract, payslip, tax_return, passport)",
            "name": "specific document name/title"
        }

        Document text:
        {text_sample}
        """
        
        return await self._execute_with_fallback(
            prompt=prompt.replace("{text_sample}", text[:2000]),
            system_message="You are a document analysis expert.",
            operation_name="document_type_analysis",
            model_preference="gemini"
        )

    async def _extract_features(self, text: str) -> Dict[str, Any]:
        """Extract all possible features from document using AI."""
        prompt = """Analyze this document and extract ALL relevant information.
        Return ONLY a JSON object containing all detected fields and values.
        Include any dates, amounts, names, numbers, or other important information.
        Format numbers and dates consistently.
        
        Document text:
        {text_sample}
        """
        
        return await self._execute_with_fallback(
            prompt=prompt.replace("{text_sample}", text[:3000]),  # Use more text for features
            system_message="You are a document information extraction expert.",
            operation_name="feature_extraction",
            model_preference="auto",  # No preference, use default order
            timeout=45.0  # Longer timeout for feature extraction
        )

    async def _extract_owner_info(self, text: str, features: Dict[str, Any]) -> Dict[str, Any]:
        """Extract owner information with mandatory passport number."""
        prompt = """Extract owner information from this document text.
        Return ONLY a JSON object with these mandatory fields:
        {
            "full_name": "complete name of the document owner",
            "passport_number": "passport or ID number if found",
            "nationality": "nationality if found",
            "date_of_birth": "DOB if found"
        }
        
        Look for passport/ID numbers in formats like:
        - Passport: A12345678
        - ID: 123-45-6789
        
        Document text:
        {text_sample}
        """
        
        try:
            # Attempt to extract owner info
            owner_info = await self._execute_with_fallback(
                prompt=prompt.replace("{text_sample}", text[:2000]),
                system_message="You are a document analysis expert specializing in personal information extraction.",
                operation_name="owner_info_extraction",
                model_preference="openai",  # Prefer OpenAI for sensitive info
                timeout=25.0
            )
            
            # Extract from features if available
            feature_owner = features.get("owner", {})
            if feature_owner and isinstance(feature_owner, dict):
                # Merge with feature data, preferring explicit owner info
                for key, value in feature_owner.items():
                    if not owner_info.get(key) and value:
                        owner_info[key] = value
                        
        except Exception as e:
            logger.error(f"Owner info extraction failed: {e}")
            owner_info = {}
        
        # Ensure mandatory fields
        owner_info.setdefault("full_name", "Unknown")
        owner_info.setdefault("passport_number", "Not found")
        
        # Add confidence score
        owner_info["confidence"] = 1.0 if owner_info["passport_number"] != "Not found" else 0.5
            
        return owner_info
        
    async def _extract_key_factors(self, owner: str, documents: List[ProcessingResult]) -> Dict[str, Any]:
        """Extract key factors from all owner's documents."""
        # Group documents by type for targeted analysis
        docs_by_type = {}
        for doc in documents:
            docs_by_type.setdefault(doc.doc_type, []).append(doc)
        
        prompt = f"""Analyze these documents for {owner} and extract key factors.
        
        Required key factors:
        - Identity information (from passports/IDs)
        - Income data (from payslips/tax returns)
        - Employment history (from contracts/resumes)
        - Tax returns history
        - Any other important information
        
        Return ONLY a JSON object with these categories and their specific details.
        
        Document summaries:
        {json.dumps([{
            "type": str(doc.doc_type),
            "features": doc.features
        } for doc in documents], indent=2)}
        """
        
        # Extract key factors with automatic fallback
        factors = await self._execute_with_fallback(
            prompt=prompt,
            system_message="You are an expert at analyzing personal documents and extracting key information.",
            operation_name="key_factors_extraction",
            model_preference="auto",
            max_retries=3,  # More retries for this critical operation
            timeout=60.0    # Longer timeout for complex analysis
        )
            
        # Add metadata
        factors["analysis_timestamp"] = datetime.now().isoformat()
        factors["document_count"] = len(documents)
        factors["document_types"] = list(docs_by_type.keys())
        
        return factors
        
    def _extract_visual_features(self, visual_analysis: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract structured features from Donut's visual analysis."""
        features = {}
        
        for page_analysis in visual_analysis:
            analysis_text = page_analysis.get("visual_analysis", "")
            if not analysis_text:
                continue
                
            # Extract key-value pairs
            lines = analysis_text.split("\n")
            for line in lines:
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip().lower().replace(" ", "_")
                    value = value.strip()
                    features[key] = value
            
            # Extract tables if present
            if "table" in analysis_text.lower():
                features["contains_tables"] = True
                # Add table extraction logic if needed
                
        return features
        
    def _merge_features(self, llm_features: Dict[str, Any], 
                      visual_features: Dict[str, Any]) -> Dict[str, Any]:
        """Merge features from LLM and visual analysis with conflict resolution."""
        merged = {**llm_features}  # Start with LLM features
        
        # Add visual features with confidence tracking
        for key, value in visual_features.items():
            if key not in merged:
                # New feature from visual analysis
                merged[key] = value
            else:
                # Feature exists - keep most confident or combine
                llm_conf = llm_features.get("confidence", 0.5)
                visual_conf = visual_features.get("confidence", 0.8)  # Trust visual more
                
                if visual_conf > llm_conf:
                    merged[key] = value
                elif key == "tables" and value:
                    # Combine table information
                    merged[key] = value
        
        # Add source tracking
        merged["analysis_sources"] = ["llm", "visual"]
        
        return merged

    async def _execute_with_fallback(self, 
                                  prompt: str,
                                  system_message: str,
                                  operation_name: str,
                                  model_preference: str = "auto",
                                  max_retries: int = 2,
                                  timeout: float = 30.0) -> Dict[str, Any]:
        """Execute AI operation with automatic fallback and retry logic.
        
        Args:
            prompt: The prompt to send to the AI model
            system_message: System message for context
            operation_name: Name of operation for logging
            model_preference: Preferred model ("gemini", "openai", or "auto")
            max_retries: Maximum number of retries per model
            timeout: Operation timeout in seconds
            
        Returns:
            Dictionary containing AI response and metadata
            
        Raises:
            RuntimeError: If all models fail after retries
        """
        errors = []
        result = None
        
        # Determine model order based on preference
        if model_preference == "gemini":
            models = ["gemini", "openai"]
        elif model_preference == "openai":
            models = ["openai", "gemini"]
        else:
            models = ["gemini", "openai"]  # Default order
        
        for model in models:
            retries = 0
            while retries < max_retries:
                try:
                    # Try current model
                    if model == "gemini" and self.llm_config.gemini:
                        response = await asyncio.wait_for(
                            self.llm_config.gemini.generate_content(prompt),
                            timeout=timeout
                        )
                        result = json.loads(response.text)
                        result["model"] = "gemini"
                        logger.info(f"{operation_name}: Successfully used Gemini")
                        return result
                        
                    elif model == "openai" and self.llm_config.openai:
                        response = await asyncio.wait_for(
                            self.llm_config.openai.chat.completions.create(
                                model=self.llm_config.openai_model,
                                messages=[
                                    {"role": "system", "content": system_message},
                                    {"role": "user", "content": prompt}
                                ]
                            ),
                            timeout=timeout
                        )
                        result = json.loads(response.choices[0].message.content)
                        result["model"] = "openai"
                        logger.info(f"{operation_name}: Successfully used OpenAI")
                        return result
                        
                except asyncio.TimeoutError:
                    error = f"Timeout error with {model} (retry {retries + 1}/{max_retries})"
                    logger.warning(f"{operation_name}: {error}")
                    errors.append(error)
                    
                except Exception as e:
                    error = f"Error with {model}: {str(e)} (retry {retries + 1}/{max_retries})"
                    logger.warning(f"{operation_name}: {error}")
                    errors.append(error)
                
                retries += 1
                if retries < max_retries:
                    await asyncio.sleep(1)  # Brief pause between retries
        
        # All models failed
        error_msg = f"{operation_name} failed with all models: {'; '.join(errors)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)