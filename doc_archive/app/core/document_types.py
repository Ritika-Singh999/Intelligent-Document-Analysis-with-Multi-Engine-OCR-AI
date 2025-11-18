from enum import Enum, auto
from typing import Dict, Set, Tuple, List, Optional
import re
import uuid
from datetime import datetime

class DocumentType(Enum):
    PAYSLIP = auto()
    CONTRACT = auto()
    TAX_RETURN = auto()
    PASSPORT = auto()
    BANK_STATEMENT = auto()
    ID_CARD = auto()
    RESUME = auto()
    OFFER_LETTER = auto()
    CERTIFICATE = auto()
    INVOICE = auto()
    RECEIPT = auto()
    LICENSE = auto()
    PERMIT = auto()
    APPLICATION = auto()
    FORM = auto()
    UTILITY_BILL = auto()
    MEDICAL_RECORD = auto()
    INSURANCE = auto()
    VISA = auto()
    EMPLOYMENT_CONTRACT = auto()
    TAX_DOCUMENT = auto()
    UNKNOWN = auto()

    @classmethod
    def normalize(cls, doc_type: str) -> 'DocumentType':
        """Convert various document type strings to normalized enum."""
        if not doc_type:
            return cls.UNKNOWN
            
        doc_type = doc_type.lower().strip().replace('-', ' ').replace('_', ' ')
        TYPE_MAPPINGS = {
            # Payslip variations
            'payslip': cls.PAYSLIP,
            'pay slip': cls.PAYSLIP,
            'salary slip': cls.PAYSLIP,
            'wage slip': cls.PAYSLIP,
            'earnings statement': cls.PAYSLIP,
            'salary statement': cls.PAYSLIP,
            'pay statement': cls.PAYSLIP,
            'compensation statement': cls.PAYSLIP,
            # Contract variations
            'contract': cls.CONTRACT,
            'employment contract': cls.CONTRACT,
            'work contract': cls.CONTRACT,
            'job contract': cls.CONTRACT,
            'agreement': cls.CONTRACT,
            'service agreement': cls.CONTRACT,
            'employment agreement': cls.CONTRACT,
            # Tax document variations
            'tax return': cls.TAX_RETURN,
            'tax filing': cls.TAX_RETURN,
            'tax statement': cls.TAX_RETURN,
            'income tax return': cls.TAX_RETURN,
            'tax report': cls.TAX_RETURN,
            'tax form': cls.TAX_RETURN,
            # ID document variations
            'passport': cls.PASSPORT,
            'travel document': cls.PASSPORT,
            'bank statement': cls.BANK_STATEMENT,
            'account statement': cls.BANK_STATEMENT,
            'banking statement': cls.BANK_STATEMENT,
            'id card': cls.ID_CARD,
            'identification': cls.ID_CARD,
            'identity card': cls.ID_CARD,
            'drivers license': cls.ID_CARD,
            'resume': cls.RESUME,
            'cv': cls.RESUME,
            'curriculum vitae': cls.RESUME,
            'offer': cls.OFFER_LETTER,
            'job offer': cls.OFFER_LETTER,
            'offer letter': cls.OFFER_LETTER,
            'employment offer': cls.OFFER_LETTER,
            'certificate': cls.CERTIFICATE,
            'certification': cls.CERTIFICATE,
            'diploma': cls.CERTIFICATE,
            # Utility bill variations
            'utility bill': cls.UTILITY_BILL,
            'utility_bill': cls.UTILITY_BILL,
            'bill': cls.UTILITY_BILL,
            'electricity bill': cls.UTILITY_BILL,
            'gas bill': cls.UTILITY_BILL,
            'water bill': cls.UTILITY_BILL,
            'phone bill': cls.UTILITY_BILL,
            'internet bill': cls.UTILITY_BILL,
            'service bill': cls.UTILITY_BILL,
            # Medical record variations
            'medical record': cls.MEDICAL_RECORD,
            'medical_record': cls.MEDICAL_RECORD,
            'health record': cls.MEDICAL_RECORD,
            'patient record': cls.MEDICAL_RECORD,
            'diagnosis': cls.MEDICAL_RECORD,
            'treatment record': cls.MEDICAL_RECORD,
            # Insurance variations
            'insurance': cls.INSURANCE,
            'insurance policy': cls.INSURANCE,
            'insurance_policy': cls.INSURANCE,
            'coverage': cls.INSURANCE,
            'policy': cls.INSURANCE,
            # Visa variations
            'visa': cls.VISA,
            'entry permit': cls.VISA,
            'work permit': cls.VISA,
            'student visa': cls.VISA,
            'tourist visa': cls.VISA,
            'immigration status': cls.VISA,
            # Employment contract variations
            'employment contract': cls.EMPLOYMENT_CONTRACT,
            'employment_contract': cls.EMPLOYMENT_CONTRACT,
            'job contract': cls.EMPLOYMENT_CONTRACT,
            'work agreement': cls.EMPLOYMENT_CONTRACT,
            # Tax document variations
            'tax document': cls.TAX_DOCUMENT,
            'tax_document': cls.TAX_DOCUMENT,
            'tax form': cls.TAX_DOCUMENT,
            'tax filing': cls.TAX_DOCUMENT,
            'income tax': cls.TAX_DOCUMENT,
            'w-2': cls.TAX_DOCUMENT,
            '1099': cls.TAX_DOCUMENT
        }
        # Try direct match first
        if doc_type in TYPE_MAPPINGS:
            return TYPE_MAPPINGS[doc_type]
        # Try partial matches
        for key, value in TYPE_MAPPINGS.items():
            if key in doc_type:
                return value
        return cls.UNKNOWN

    @classmethod
    def normalize_str(cls, doc_type: str) -> str:
        """Return normalized string (snake_case) for any input."""
        norm = cls.normalize(doc_type)
        return norm.name.lower()

    @classmethod
    def to_string(cls, doc_type: 'DocumentType') -> str:
        """Convert DocumentType enum to string."""
        return doc_type.name.lower()

    def __str__(self) -> str:
        """Return lowercase string representation for consistent usage."""
        return self.name.lower()

def normalize_document_type_str(doc_type: str) -> str:
    """Utility to normalize to snake_case string."""
    return DocumentType.normalize_str(doc_type)

# UUID utility for document tracking
def generate_document_uuid(filename: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, filename + str(datetime.utcnow().date())))

def detect_sensitive_patterns(text: str) -> Dict[str, Tuple[str, float]]:
    
    patterns = {
        "passport_number": (
            r"(?i)passport[:\s]*([A-Z0-9]{6,9})|([A-Z]{1,3}[0-9]{6,7})",
            0.8
        ),
        "ssn": (
            r"(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}",
            0.9
        ),
        "id_number": (
            r"(?i)ID[:\s]*([A-Z0-9]{5,12})|([A-Z]{1,2}[0-9]{5,10})",
            0.7
        ),
        "bank_account": (
            r"(?i)account[:\s]*([0-9]{8,12})",
            0.6
        ),
        "tax_id": (
            r"(?i)tax\s*id[:\s]*([0-9]{2}-?[0-9]{7})",
            0.85
        )
    }
    
    results = {}
    for pattern_type, (regex, base_confidence) in patterns.items():
        matches = re.findall(regex, text)
        if matches:
            # Use first match and its length to adjust confidence
            match = matches[0] if isinstance(matches[0], str) else matches[0][0]
            # Longer matches get slightly higher confidence
            confidence = min(base_confidence + (len(match) / 100), 0.99)
            results[pattern_type] = (match, confidence)
            
    return results

KEY_FACTORS = {
    DocumentType.PAYSLIP: {
        'required': ['salary', 'date', 'employer', 'employee'],
        'optional': ['tax_deductions', 'benefits', 'net_pay', 'gross_pay']
    },
    DocumentType.CONTRACT: {
        'required': ['start_date', 'employer', 'employee', 'position'],
        'optional': ['end_date', 'salary', 'terms', 'benefits']
    },
    DocumentType.TAX_RETURN: {
        'required': ['tax_year', 'total_income', 'tax_paid'],
        'optional': ['deductions', 'credits', 'filing_status']
    },
    # ... add more document types
}

OWNER_KEY_FACTORS = {
    'identity': [
        'full_name',
        'date_of_birth',
        'nationality',
        'passport_number',
        'id_numbers'
    ],
    'employment': [
        'current_employer',
        'position',
        'employment_history',
        'total_experience'
    ],
    'financial': [
        'annual_income',
        'salary_history',
        'tax_records',
        'banking_info'
    ],
    'education': [
        'qualifications',
        'certificates',
        'skills'
    ]
}