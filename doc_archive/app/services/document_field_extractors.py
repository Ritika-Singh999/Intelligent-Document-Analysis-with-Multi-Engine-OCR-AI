"""
Optimized per-document field extractors using LLM + caching.
Returns compact, type-specific extractions without aggregation overhead.
"""
import re
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio
from functools import lru_cache

from app.core.llm import get_llm_client

logger = logging.getLogger(__name__)
import json

# ========== Extraction Prompts (Type-Specific) ==========

PAYSLIP_EXTRACTION_PROMPT = """Extract ONLY these fields from the payslip:
- employerName: Company name
- paymentPeriod: Date range (YYYY-MM-DD to YYYY-MM-DD)
- netPay: Net salary amount with currency
- grossPay: Gross salary with currency
- employeeName: Full name of employee
- passportNumber: If visible, passport ID

Return as JSON object. Use null for missing fields.
Document: {text}"""

TAX_DOCUMENT_EXTRACTION_PROMPT = """Extract ONLY these fields from the tax document:
- rfc: Mexican RFC number (11-13 chars)
- fiscalYear: Year (YYYY)
- totalGrossIncome: Income amount with currency
- documentType: Type (Annual Tax Statement, Income Tax Statement, etc.)
- submissionDate: YYYY-MM-DD
- operationNumber: Transaction/operation ID if present
- fiscalPeriod: Period covered (text or date range)

Return as JSON object. Use null for missing fields.
Document: {text}"""

INVOICE_EXTRACTION_PROMPT = """Extract ONLY these fields from the invoice:
- issuerRfc: RFC of issuer
- invoiceId: Invoice number/ID
- totalAmount: Invoice total with currency
- issuerName: Business name
- serviceDescription: What was billed for

Return as JSON object. Use null for missing fields.
Document: {text}"""

RECEIPT_EXTRACTION_PROMPT = """Extract ONLY these fields from the receipt:
- receiptId: Receipt number
- issuerName: Business/person name
- totalAmount: Amount with currency
- issuerRfc: RFC if present

Return as JSON object. Use null for missing fields.
Document: {text}"""

BANK_STATEMENT_EXTRACTION_PROMPT = """Extract ONLY these fields from the bank statement:
- accountHolder: Account owner name
- clabeId: CLABE number (last 6-8 digits masked or full if visible)
- statementPeriod: Date range (YYYY-MM-DD to YYYY-MM-DD)
- closingBalance: Final balance with currency
- bankName: Bank name

Return as JSON object. Use null for missing fields.
Document: {text}"""

EMPLOYMENT_CONTRACT_EXTRACTION_PROMPT = """Extract ONLY these fields from the contract:
- contractingParty: Company/employer name
- representativeName: Employee/contractor name
- contractType: Type (Employment, Independent, Notarized Deed, etc.)

Return as JSON object. Use null for missing fields.
Document: {text}"""

PASSPORT_EXTRACTION_PROMPT = """Extract ONLY these fields from the passport image/document:
- mrz: Machine-readable zone text if visible
- name: Full name on passport
- nationality: Country code (e.g., MEX)
- passportId: Passport number
- dob: Date of birth (YYYY-MM-DD)
- expiryDate: Expiration date (YYYY-MM-DD)

Return as JSON object. Use null for missing fields.
Document: {text}"""

# ========== Generic Extractor ==========

async def extract_document_fields(
    text: str, 
    doc_type: str, 
    filename: str,
    confidence_threshold: float = 0.85
) -> Dict[str, Any]:
    """
    Extract fields specific to document type using LLM.
    Returns compact object with only essential fields.
    """
    try:
        llm = get_llm_client()
        
        # Select prompt based on document type
        prompts = {
            'payslip': PAYSLIP_EXTRACTION_PROMPT,
            'tax_document': TAX_DOCUMENT_EXTRACTION_PROMPT,
            'invoice': INVOICE_EXTRACTION_PROMPT,
            'receipt': RECEIPT_EXTRACTION_PROMPT,
            'bank_statement': BANK_STATEMENT_EXTRACTION_PROMPT,
            'employment_contract': EMPLOYMENT_CONTRACT_EXTRACTION_PROMPT,
            'passport': PASSPORT_EXTRACTION_PROMPT,
        }
        
        prompt_template = prompts.get(doc_type, PAYSLIP_EXTRACTION_PROMPT)
        prompt = prompt_template.format(text=text[:3000])  # Limit to 3K tokens
        
        # Call LLM to extract
        response = await llm.agenerate([prompt])
        extracted_json = response.generations[0][0].text
        
        # Parse JSON from response
        import json
        match = re.search(r'\{[^{}]*\}', extracted_json, re.DOTALL)
        if match:
            fields = json.loads(match.group())
        else:
            fields = {}
        
        # Always include fileName
        fields['fileName'] = filename
        
        # Detect passport if embedded in text
        passport_match = re.search(r'(?:passport|pasaporte)[\s:]*([A-Z0-9]{6,10})', text, re.IGNORECASE)
        if passport_match:
            fields['passportDetected'] = True
            fields['passportNumber'] = passport_match.group(1)
        else:
            fields['passportDetected'] = False
        
        logger.info(f"Extracted {doc_type}: {filename}")
        return fields
        
    except Exception as e:
        logger.error(f"Extraction failed for {filename}: {str(e)}")
        return {
            'fileName': filename,
            'passportDetected': False,
            'error': str(e)
        }

# ========== Batch Extractor (Process Multiple Docs in Parallel) ==========

async def extract_batch_documents(
    docs_by_type: Dict[str, List[Dict[str, str]]]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract all documents grouped by type in parallel.
    Returns: { 'payslip': [...], 'tax_document': [...], ... }
    """
    tasks = []
    type_mapping = {}  # Track which task corresponds to which type
    
    for doc_type, docs in docs_by_type.items():
        for i, doc in enumerate(docs):
            task = extract_document_fields(
                text=doc.get('text', ''),
                doc_type=doc_type,
                filename=doc.get('filename', f'{doc_type}_{i}.pdf')
            )
            tasks.append(task)
            type_mapping[len(tasks) - 1] = doc_type
    
    # Execute all in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Group results by type
    grouped = {}
    for idx, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Extraction error: {result}")
            continue
        
        doc_type = type_mapping[idx]
        if doc_type not in grouped:
            grouped[doc_type] = []
        
        grouped[doc_type].append(result)
    
    return grouped

# ========== Field Validation & Normalization ==========

def validate_payslip_extract(data: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure payslip has required structure."""
    return {
        'fileName': data.get('fileName', ''),
        'employerName': data.get('employerName'),
        'paymentPeriod': data.get('paymentPeriod'),
        'netPay': data.get('netPay'),
        'grossPay': data.get('grossPay'),
        'employeeName': data.get('employeeName'),
        'passportDetected': data.get('passportDetected', False),
        'passportNumber': data.get('passportNumber'),
    }

def validate_tax_extract(data: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure tax doc has required structure."""
    return {
        'fileName': data.get('fileName', ''),
        'rfc': data.get('rfc'),
        'fiscalYear': data.get('fiscalYear'),
        'totalGrossIncome': data.get('totalGrossIncome'),
        'documentType': data.get('documentType'),
        'submissionDate': data.get('submissionDate'),
        'operationNumber': data.get('operationNumber'),
        'fiscalPeriod': data.get('fiscalPeriod'),
        'passportDetected': data.get('passportDetected', False),
        'passportNumber': data.get('passportNumber'),
    }

def validate_invoice_extract(data: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure invoice has required structure."""
    return {
        'fileName': data.get('fileName', ''),
        'issuerRfc': data.get('issuerRfc'),
        'invoiceId': data.get('invoiceId'),
        'totalAmount': data.get('totalAmount'),
        'issuerName': data.get('issuerName'),
        'serviceDescription': data.get('serviceDescription'),
        'passportDetected': data.get('passportDetected', False),
        'passportNumber': data.get('passportNumber'),
    }

def validate_receipt_extract(data: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure receipt has required structure."""
    return {
        'fileName': data.get('fileName', ''),
        'receiptId': data.get('receiptId'),
        'issuerName': data.get('issuerName'),
        'totalAmount': data.get('totalAmount'),
        'issuerRfc': data.get('issuerRfc'),
        'passportDetected': data.get('passportDetected', False),
        'passportNumber': data.get('passportNumber'),
    }

def validate_bank_extract(data: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure bank statement has required structure."""
    return {
        'fileName': data.get('fileName', ''),
        'accountHolder': data.get('accountHolder'),
        'clabeId': data.get('clabeId'),
        'statementPeriod': data.get('statementPeriod'),
        'closingBalance': data.get('closingBalance'),
        'bankName': data.get('bankName'),
        'passportDetected': data.get('passportDetected', False),
        'passportNumber': data.get('passportNumber'),
    }

def validate_employment_extract(data: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure employment contract has required structure."""
    return {
        'fileName': data.get('fileName', ''),
        'contractingParty': data.get('contractingParty'),
        'representativeName': data.get('representativeName'),
        'contractType': data.get('contractType'),
        'passportDetected': data.get('passportDetected', False),
        'passportNumber': data.get('passportNumber'),
    }

def validate_passport_extract(data: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure passport has required structure."""
    return {
        'fileName': data.get('fileName', ''),
        'mrz': data.get('mrz'),
        'name': data.get('name'),
        'nationality': data.get('nationality'),
        'passportId': data.get('passportId'),
        'dob': data.get('dob'),
        'expiryDate': data.get('expiryDate'),
        'passportDetected': data.get('passportDetected', True),
    }

# ========== Type-Specific Validators ==========

VALIDATORS = {
    'payslip': validate_payslip_extract,
    'tax_document': validate_tax_extract,
    'invoice': validate_invoice_extract,
    'receipt': validate_receipt_extract,
    'bank_statement': validate_bank_extract,
    'employment_contract': validate_employment_extract,
    'passport': validate_passport_extract,
}

def validate_extracted_data(doc_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Apply type-specific validation."""
    validator = VALIDATORS.get(doc_type, lambda x: x)
    return validator(data)
