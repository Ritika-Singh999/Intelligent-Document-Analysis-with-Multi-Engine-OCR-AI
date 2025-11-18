import asyncio
from datetime import datetime
from typing import List, Dict, Any
import logging

from app.core.llm import get_llm
from app.schemas.document_schemas import ProcessedOwnerData, DocumentData, KeyFactors, ConfidenceSummary
from app.core.optimizations import cache_result
from app.utils.helpers import normalize_name

logger = logging.getLogger(__name__)

async def group_documents_by_owner(documents: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    owner_groups = {}
    
    for doc in documents:
        owner = normalize_name(doc.get("owner", "Unknown"))
        if owner not in owner_groups:
            owner_groups[owner] = []
        owner_groups[owner].append(doc)
    
    return owner_groups

async def extract_key_factors(owner: str, documents: List[Dict[str, Any]]) -> KeyFactors:
    llm = get_llm()
    
    try:
        # Prepare document texts for analysis
        doc_texts = []
        for doc in documents:
            doc_text = doc.get("raw_text_snapshot", "")
            doc_type = doc.get("document_type", "unknown")
            doc_texts.append(f"Document Type: {doc_type}\nContent: {doc_text}")
        
        combined_text = "\n\n---\n\n".join(doc_texts)
        
        # Extract key factors using LLM
        prompt = f'''Analyze these documents for {owner} and extract key factors in this JSON format:
        {{
            "identity_data": {{name, designation, department, etc}},
            "income_data": {{salary details, frequency, etc}},
            "employment_history": [{{company, role, dates}}],
            "bank_details": {{bank info}},
            "tax_returns_history": [{{year, details}}],
            "passport": passport number or null
        }}
        
        Documents:
        {combined_text}
        '''
        
        key_factors = await llm.agenerate_json(prompt)
        return KeyFactors(**key_factors)
    
    except Exception as e:
        logger.error(f"Error extracting key factors: {str(e)}")
        # Return empty key factors structure
        return KeyFactors(
            identity_data={},
            income_data={},
            employment_history=[],
            bank_details=None,
            tax_returns_history=[],
            passport=None
        )

async def process_owner_documents(owner: str, documents: List[Dict[str, Any]]) -> ProcessedOwnerData:
    
    # Process documents in parallel
    async def process_document(doc: Dict[str, Any]) -> DocumentData:
        extracted = doc.get("extracted_data", {})
        return DocumentData(
            document_name=doc.get("document_name", "Unknown Document"),
            document_type=doc.get("document_type", "unknown"),
            passport_found=doc.get("has_passport", False),
            fields=extracted.get("fields", {}),
            confidence_score=doc.get("confidence", 0.0)
        )
    
    processed_docs = await asyncio.gather(*[
        process_document(doc) for doc in documents
    ])
    
    # Extract key factors
    key_factors = await extract_key_factors(owner, documents)
    
    # Calculate confidence scores
    confidence = ConfidenceSummary(
        document_type_confidence=sum(d.confidence_score for d in processed_docs) / len(processed_docs),
        field_extraction_confidence=sum(d.confidence_score for d in processed_docs) / len(processed_docs)
    )
    
    return ProcessedOwnerData(
        owner_name=owner,
        documents=processed_docs,
        key_factors=key_factors,
        confidence_summary=confidence,
        timestamp=datetime.utcnow().isoformat()
    )

async def process_all_owners(documents: List[Dict[str, Any]]) -> List[ProcessedOwnerData]:
    """Process all documents grouped by owner in parallel."""
    
    # Group documents by owner
    owner_groups = await group_documents_by_owner(documents)
    
    # Process each owner's documents in parallel
    owner_results = await asyncio.gather(*[
        process_owner_documents(owner, docs)
        for owner, docs in owner_groups.items()
    ])
    
    return owner_results