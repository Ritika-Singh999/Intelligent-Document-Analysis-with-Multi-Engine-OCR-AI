from typing import List, Dict, Optional, Any
from pydantic import BaseModel

class KeyFactor(BaseModel):
    name: str
    value: str
    confidence: float
    source_document: str
    extraction_method: str

class KeyFactorGroup(BaseModel):
    group_name: str
    factors: List[KeyFactor]

class KeyFactorRequest(BaseModel):
    document_ids: List[str]
    factor_types: List[str]

class KeyFactorResponse(BaseModel):
    factor_groups: List[KeyFactorGroup]
    metadata: Dict[str, Any]
