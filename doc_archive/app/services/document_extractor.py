import asyncio
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

from app.core.llm import get_llm
from app.core.donut import get_donut
from app.utils.helpers import normalize_text
from app.core.optimizations import cache_result

logger = logging.getLogger(__name__)

async def extract_with_donut(image) -> Dict[str, Any]:
    try:
        if image is None:
            logger.warning("No image provided for Donut extraction")
            return {"error": "No image provided", "model": "donut"}

        processor, model = get_donut()
        if processor is None or model is None:
            logger.warning("Donut model not available")
            return {"error": "Donut model not available", "model": "donut"}

        pixel_values = processor(image, return_tensors="pt").pixel_values
        outputs = model.generate(
            pixel_values,
            max_length=1024,
            return_dict_in_generate=True
        )

        sequence = processor.batch_decode(outputs.sequences)[0]
        extracted = sequence.replace(processor.tokenizer.eos_token or "", "").replace("<pad>", "").replace("<s>", "").strip()

        logger.info(f"Donut extraction completed")
        return {
            "model": "donut",
            "extracted_data": extracted,
            "confidence": outputs.sequences_scores[0] if hasattr(outputs, 'sequences_scores') else 0.8
        }
    except Exception as e:
        logger.error(f"Donut extraction failed: {str(e)}")
        return {"error": str(e), "model": "donut"}

async def extract_with_llm(text: str, doc_type: str) -> Dict[str, Any]:
    try:
        llm = get_llm()
        
        # Dynamic prompt based on document type
        prompt = f"""Extract all key information from this {doc_type} in JSON format.
        Include any dates, amounts, names, ID numbers, or other significant fields you find.
        If you find sensitive information like passport numbers, SSN, or bank details, mark them clearly.
        
        Document text:
        {text[:2000]}  # Limit text length for token efficiency
        
        Return in this format:
        {{
            "extracted_fields": {{all found fields}},
            "sensitive_data": {{any sensitive numbers/ids found}},
            "metadata": {{document type, dates, confidence}}
        }}
        """
        
        result = await llm.agenerate_json(prompt)
        logger.info(f"LLM extraction completed")
        return {
            "model": "llm",
            "extracted_data": result,
            "confidence": 0.9  # LLM confidence score
        }
    except Exception as e:
        logger.error(f"LLM extraction failed: {str(e)}")
        return {"error": str(e), "model": "llm"}

async def merge_extractions(donut_result: Dict[str, Any], llm_result: Dict[str, Any]) -> Dict[str, Any]:
    
    merged = {
        "models_used": ["donut", "llm"],
        "timestamp": datetime.utcnow().isoformat(),
        "extraction_success": True,
        "errors": []
    }
    
    # Track if either model failed
    if "error" in donut_result:
        merged["errors"].append({"model": "donut", "error": donut_result["error"]})
    if "error" in llm_result:
        merged["errors"].append({"model": "llm", "error": llm_result["error"]})
    
    # Combine extracted data
    merged["fields"] = {}
    
    # Add Donut extracted data if available
    if "extracted_data" in donut_result:
        if isinstance(donut_result["extracted_data"], dict):
            merged["fields"].update(donut_result["extracted_data"])
        elif isinstance(donut_result["extracted_data"], str):
            # If Donut returned raw text, let LLM parse it in next run
            merged["donut_raw_text"] = donut_result["extracted_data"]
    
    # Add LLM extracted data
    if "extracted_data" in llm_result:
        if isinstance(llm_result["extracted_data"], dict):
            extracted = llm_result["extracted_data"]
            merged["fields"].update(extracted.get("extracted_fields", {}))
            merged["sensitive_data"] = extracted.get("sensitive_data", {})
            merged["metadata"] = extracted.get("metadata", {})
    
    # Calculate confidence
    confidences = []
    if "confidence" in donut_result:
        confidences.append(donut_result["confidence"])
    if "confidence" in llm_result:
        confidences.append(llm_result["confidence"])
    
    merged["confidence"] = sum(confidences) / len(confidences) if confidences else 0.0
    
    return merged

async def extract_document_info(image, text: str, doc_type: str) -> Dict[str, Any]:
    
    # Run both models in parallel
    donut_task = asyncio.create_task(extract_with_donut(image))
    llm_task = asyncio.create_task(extract_with_llm(text, doc_type))
    
    # Wait for both to complete
    donut_result, llm_result = await asyncio.gather(
        donut_task,
        llm_task,
        return_exceptions=True
    )
    
    # Handle any exceptions
    if isinstance(donut_result, Exception):
        logger.error(f"Donut extraction failed: {str(donut_result)}")
        donut_result = {"error": str(donut_result), "model": "donut"}
    if isinstance(llm_result, Exception):
        logger.error(f"LLM extraction failed: {str(llm_result)}")
        llm_result = {"error": str(llm_result), "model": "llm"}
    
    # Merge results
    merged = await merge_extractions(donut_result, llm_result)
    
    # Always include mandatory fields
    merged["document_type"] = doc_type
    merged["content_schema"] = {
        "raw_text": text,
        "extracted_fields": merged["fields"]
    }
    merged["raw_text_snapshot"] = text[:20000]  # Store first 20K chars
    
    return merged