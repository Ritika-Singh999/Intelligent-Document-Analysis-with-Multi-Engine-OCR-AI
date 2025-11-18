from typing import List, Dict, Optional, Any
from pydantic import BaseModel

class ApiResponse(BaseModel):
    status: str
    message: str
    data: Optional[Any] = None
    errors: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

class ErrorResponse(BaseModel):
    status: str = "error"
    message: str
    error_code: str
    details: Optional[Dict[str, Any]] = None

class SuccessResponse(BaseModel):
    status: str = "success"
    message: str
    data: Any
    metadata: Optional[Dict[str, Any]] = None