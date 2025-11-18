from typing import Dict, Any, Optional, List
import logging
import re
from PIL import Image
import json
from functools import lru_cache
import asyncio
import google.generativeai as genai
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
from app.core.llm import get_llm_client, LLMConfig
from app.core.config import settings

logger = logging.getLogger(__name__)

_donut_processor_cache: Optional[DonutProcessor] = None
_donut_model_cache: Optional[VisionEncoderDecoderModel] = None
_gemini_model_cache: Optional[Any] = None


def robust_json_parse(text: str, fallback: Dict[str, Any] = None) -> Dict[str, Any]:
    if not text or not isinstance(text, str):
        return fallback or {}
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    try:
        start_idx = text.find('{')
        end_idx = text.rfind('}') + 1
        if start_idx != -1 and end_idx > start_idx:
            json_str = text[start_idx:end_idx]
            return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    try:
        import re
        json_block = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_block:
            return json.loads(json_block.group(1))
    except json.JSONDecodeError:
        pass

    try:
        fixed_text = re.sub(r',\s*}', '}', text)
        fixed_text = re.sub(r',\s*]', ']', fixed_text)
        return json.loads(fixed_text)
    except json.JSONDecodeError:
        pass

    try:
        result = {}
        pairs = re.findall(r'"([^"]+)"\s*:\s*("([^"]*)"|\{[^}]*\}|\[[^\]]*\]|[^,}\]]+)', text)
        for key, value, quoted_value in pairs:
            if quoted_value:
                result[key] = quoted_value
            elif value.startswith('{'):
                try:
                    result[key] = json.loads(value)
                except:
                    result[key] = value
            elif value.startswith('['):
                try:
                    result[key] = json.loads(value)
                except:
                    result[key] = value
            else:
                if value.lower() in ['true', 'false']:
                    result[key] = value.lower() == 'true'
                elif value.replace('.', '').isdigit():
                    result[key] = float(value) if '.' in value else int(value)
                else:
                    result[key] = value.strip()

        if result:
            return result
    except Exception:
        pass

    logger.warning(f"Failed to parse JSON from text: {text[:200]}...")
    return fallback or {"error": "Failed to parse JSON response", "raw_text": text}


def normalize_document_type(doc_type: str) -> str:
    if not doc_type or not isinstance(doc_type, str):
        return "unknown"
    normalized = doc_type.lower().strip()

    type_mappings = {
        "payslip": "payslip", "pay slip": "payslip", "salary slip": "payslip",
        "payroll": "payslip", "salary statement": "payslip", "pay statement": "payslip",
        "earnings statement": "payslip", "wage slip": "payslip", "compensation statement": "payslip",
        "bank statement": "bank_statement", "bank_statement": "bank_statement",
        "account statement": "bank_statement", "banking statement": "bank_statement",
        "financial statement": "bank_statement", "passport": "passport",
        "travel document": "passport", "international passport": "passport",
        "utility bill": "utility_bill", "electricity bill": "utility_bill",
        "gas bill": "utility_bill", "water bill": "utility_bill",
        "phone bill": "utility_bill", "internet bill": "utility_bill",
        "service bill": "utility_bill", "tax document": "tax_document",
        "tax_document": "tax_document", "tax return": "tax_document",
        "tax form": "tax_document", "income tax": "tax_document",
        "tax statement": "tax_document", "employment contract": "employment_contract",
        "employment_contract": "employment_contract", "job contract": "employment_contract",
        "work contract": "employment_contract", "labor contract": "employment_contract",
        "id card": "id_card", "id_card": "id_card", "identification card": "id_card",
        "identity card": "id_card", "national id": "id_card",
        "drivers license": "id_card", "driver's license": "id_card",
        "medical record": "medical_record", "medical_record": "medical_record",
        "health record": "medical_record", "patient record": "medical_record",
        "medical report": "medical_record", "insurance": "insurance",
        "insurance policy": "insurance", "insurance document": "insurance",
        "coverage document": "insurance", "visa": "visa",
        "entry visa": "visa", "travel visa": "visa",
        "immigration visa": "visa", "invoice": "invoice",
        "receipt": "receipt", "certificate": "certificate",
        "resume": "resume", "cv": "resume", "contract": "contract",
        "agreement": "contract", "unknown": "unknown"
    }
    return type_mappings.get(normalized, "unknown")


class EnhancedDonut:
    def __init__(self):
        global _donut_processor_cache, _donut_model_cache
        if _donut_processor_cache is None:
            _donut_processor_cache = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
        if _donut_model_cache is None:
            _donut_model_cache = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
            if torch.cuda.is_available():
                _donut_model_cache.to("cuda")

        self.processor = _donut_processor_cache
        self.model = _donut_model_cache

    async def extract(self, image: Image.Image, task_prompt: str = None, doc_type: str = None) -> Dict[str, Any]:
        try:
            pixel_values = self.processor(image, return_tensors="pt").pixel_values
            if torch.cuda.is_available():
                pixel_values = pixel_values.to("cuda")

            if not task_prompt:
                task_prompt = "<extract_document>Extract all visible text and structured data into key-value pairs</extract_document>"
            if task_prompt:
                self.processor.tokenizer.add_tokens([task_prompt])

            outputs = self.model.generate(
                pixel_values,
                max_length=1024,
                return_dict_in_generate=True,
                output_scores=True
            )

            if hasattr(outputs, "sequences") and outputs.sequences is not None and len(outputs.sequences) > 0:
                decoded = self.processor.batch_decode(outputs.sequences)
                sequence = decoded[0] if decoded else ""
            else:
                sequence = ""
            if not sequence:
                raise ValueError("Empty Donut output sequence")

            sequence = sequence.replace(self.processor.tokenizer.eos_token, "").replace("<pad>", "").strip()
            confidence = torch.mean(torch.stack(outputs.scores)).item() if outputs.scores else 0.0

            try:
                extracted = json.loads(sequence)
                if isinstance(extracted, dict):
                    analysis = {
                        "extraction_confidence": float(confidence),
                        "detection_basis": [],
                        "key_findings": {}
                    }
                    for key, value in extracted.items():
                        if isinstance(value, str) and len(value) > 5:
                            analysis["detection_basis"].append(f"Found {key}: {value[:50]}...")
                    extracted["detailed_analysis"] = analysis
            except json.JSONDecodeError:
                extracted = {
                    "raw_text": sequence,
                    "detailed_analysis": {
                        "extraction_confidence": float(confidence),
                        "detection_basis": ["Raw text extraction performed"],
                        "key_findings": {"raw_content_length": len(sequence)}
                    }
                }

            return {
                "model": "donut",
                "extracted": extracted,
                "confidence": float(confidence),
                "raw_output": sequence
            }

        except Exception as e:
            logger.error(f"Donut extraction failed: {str(e)}")
            return {"error": str(e), "model": "donut"}


class EnhancedLLM:
    def __init__(self):
        self.gemini = genai.GenerativeModel('models/gemini-2.5-flash')

    async def batch_extract(self, texts: List[str], doc_types: List[str] = None) -> List[Dict[str, Any]]:
        """Batch process multiple texts for parallel LLM extraction."""
        if not texts:
            return []

        if doc_types is None:
            doc_types = ["unknown"] * len(texts)

        # Create batch tasks
        tasks = []
        for i, (text, doc_type) in enumerate(zip(texts, doc_types)):
            task = asyncio.create_task(self._extract_single(text, doc_type, batch_id=i))
            tasks.append(task)

        # Execute in parallel with concurrency limit
        semaphore = asyncio.Semaphore(5)  # Limit concurrent LLM calls
        async def limited_extract(task):
            async with semaphore:
                return await task

        limited_tasks = [limited_extract(task) for task in tasks]
        results = await asyncio.gather(*limited_tasks, return_exceptions=True)

        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch extraction failed for item {i}: {str(result)}")
                processed_results.append({
                    "model": "gemini+batch",
                    "extracted": {},
                    "confidence": 0.0,
                    "error": str(result)
                })
            else:
                processed_results.append(result)

        return processed_results

    async def _extract_single(self, text: str, doc_type: str, batch_id: int) -> Dict[str, Any]:
        """Extract from single text (used by batch processing)."""
        try:
            from app.core.document_types import detect_sensitive_patterns
            sensitive_info = detect_sensitive_patterns(text)

            chunks = self._chunk_text(text)
            accumulated_results = None
            successful_chunks = 0

            for i, chunk in enumerate(chunks):
                chunk_result = await self._process_chunk(chunk, is_first_chunk=(i == 0),
                                                         previous_results=accumulated_results)
                if not chunk_result.get("structured"):
                    continue

                if accumulated_results is None:
                    accumulated_results = chunk_result
                else:
                    self._merge_results(accumulated_results, chunk_result)

                successful_chunks += 1

            if accumulated_results:
                if "confidence" not in accumulated_results or not isinstance(accumulated_results["confidence"], dict):
                    accumulated_results["confidence"] = {"overall": float(accumulated_results.get("confidence", 0.0))}
                accumulated_results["confidence"]["overall"] = 0.9

            if not accumulated_results:
                logger.warning(f"All chunks failed to process for batch item {batch_id}, using fallback")
                # Fallback: basic extraction using semantic detection
                try:
                    from app.services.profile_report import semantic_owner_candidate_from_text
                    owner = semantic_owner_candidate_from_text(text, user_name_fallback=None)
                    accumulated_results = {
                        "owner_name": owner,
                        "document_summary": text[:800] if text else "",
                        "raw_text": text
                    }
                except Exception as e:
                    logger.error(f"Fallback extraction failed for batch item {batch_id}: {str(e)}")
                    accumulated_results = {"raw_text": text or ""}

            return {
                "model": "gemini+batch",
                "extracted": accumulated_results or {},
                "confidence": 0.9,
                "chunks_processed": len(chunks),
                "batch_id": batch_id
            }

        except Exception as e:
            logger.error(f"LLM extraction failed for batch item {batch_id}: {str(e)}")
            return {"error": str(e), "model": "llm", "batch_id": batch_id}

    def _chunk_text(self, text: str, chunk_size: int = 2000) -> List[str]:
        if not text:
            return []
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        for word in words:
            word_len = len(word) + 1
            if current_length + word_len > chunk_size:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [word]
                    current_length = word_len
                else:
                    chunks.append(word)
                    current_length = 0
            else:
                current_chunk.append(word)
                current_length += word_len
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        return chunks

    def _merge_results(self, base: Dict, new: Dict):
        for key, value in new.items():
            if key not in base:
                base[key] = value
            elif isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_results(base[key], value)
            elif isinstance(base[key], list) and isinstance(value, list):
                base[key].extend(value)
            # else keep base

    async def validate_document_type(self, text: str, initial_type: str) -> Dict[str, Any]:
        try:
            prompt = f"""
            Validate the document type for this text. Initial guess: {initial_type}
            Document text:
            {text[:1000]}
            Return JSON: {{"document_type": "corrected_type", "confidence": 0.8}}
            """
            response = await asyncio.to_thread(self.gemini.generate_content, prompt)
            result_text = response.text if hasattr(response, 'text') else (response.parts[0].text if response.parts else "")
            data = robust_json_parse(result_text, {"document_type": initial_type, "confidence": 0.5})
            return data
        except Exception as e:
            logger.warning(f"Validate document type failed: {str(e)}, using initial type")
            return {"document_type": initial_type, "confidence": 0.0}

    async def _process_chunk(self, chunk: str, is_first_chunk: bool = False, previous_results: Dict = None) -> Dict[str, Any]:
        """Process a single chunk of text with consistent return format."""
        try:
            prompt = f'''Analyze this document and extract information in valid JSON format.

Document text:
{chunk}

Return ONLY a valid JSON object:
{{"document_type": "type", "owner_name": "name", "document_date": "date", "identifiers": {{}}, "key_fields": {{}}, "confidence": 0.8}}'''

            response = await asyncio.to_thread(self.gemini.generate_content, prompt)
            text = response.text if hasattr(response, 'text') else (response.parts[0].text if response.parts else "")

            # ✅ FIX: Ensure empty responses still return structured dict
            if not text or not text.strip():
                return {"structured": False, "error": "Empty Gemini response"}

            result = robust_json_parse(text, None)
            if not result or result.get("error"):
                return {"structured": False, "error": "Invalid JSON from Gemini"}

            result["structured"] = True
            return result

        except Exception as e:
            logger.error(f"Chunk processing failed: {str(e)}")
            return {"structured": False, "error": str(e)}

    async def extract(self, text: str, doc_type: str = "unknown") -> Dict[str, Any]:
        try:
            from app.core.document_types import detect_sensitive_patterns
            sensitive_info = detect_sensitive_patterns(text)

            chunks = self._chunk_text(text)
            accumulated_results = None
            successful_chunks = 0

            for i, chunk in enumerate(chunks):
                chunk_result = await self._process_chunk(chunk, is_first_chunk=(i == 0),
                                                         previous_results=accumulated_results)
                if not chunk_result.get("structured"):
                    continue

                if accumulated_results is None:
                    accumulated_results = chunk_result
                else:
                    self._merge_results(accumulated_results, chunk_result)

                successful_chunks += 1

            if accumulated_results:
                if "confidence" not in accumulated_results or not isinstance(accumulated_results["confidence"], dict):
                    accumulated_results["confidence"] = {"overall": float(accumulated_results.get("confidence", 0.0))}
                accumulated_results["confidence"]["overall"] = 0.9

            if not accumulated_results:
                logger.error("All chunks failed to process")
                return {"model": "gemini", "extracted": {}, "confidence": 0.0, "error": "All chunk processing failed"}

            return {
                "model": "gemini+chunked",
                "extracted": accumulated_results or {},
                "confidence": 0.9,
                "chunks_processed": len(chunks)
            }

        except Exception as e:
            logger.error(f"LLM extraction failed: {str(e)}")
            return {"error": str(e), "model": "llm"}
