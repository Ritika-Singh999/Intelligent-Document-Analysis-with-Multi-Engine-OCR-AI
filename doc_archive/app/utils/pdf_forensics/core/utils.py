"""
PDF forensics utilities.
"""
from typing import Dict, Any, List
import hashlib
import os
import logging
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class ForensicResult:
    """Result from a forensic analysis."""
    detector_name: str
    confidence: float
    findings: Dict[str, Any]
    risk_level: str  # 'low', 'medium', 'high'
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)

def calculate_entropy(data: bytes) -> float:
    """Calculate Shannon entropy of data."""
    if not data:
        return 0.0
        
    entropy = 0.0
    for x in range(256):
        p_x = data.count(x) / len(data)
        if p_x > 0:
            entropy += -p_x * log2(p_x)
    return entropy

def is_encrypted_content(data: bytes) -> bool:
    """Check if content appears to be encrypted."""
    # High entropy often indicates encryption
    return calculate_entropy(data) > 7.9

def get_file_signatures() -> Dict[str, List[bytes]]:
    """Get known file signatures."""
    return {
        'pdf': [b'%PDF'],
        'jpg': [b'\xFF\xD8\xFF'],
        'png': [b'\x89PNG\r\n\x1a\n'],
        'gif': [b'GIF87a', b'GIF89a'],
        'zip': [b'PK\x03\x04', b'PK\x05\x06', b'PK\x07\x08'],
    }

def detect_file_type(data: bytes) -> str:
    """Detect file type from magic numbers."""
    signatures = get_file_signatures()
    
    for file_type, sigs in signatures.items():
        if any(data.startswith(sig) for sig in sigs):
            return file_type
    return 'unknown'

def find_embedded_files(data: bytes) -> List[Dict[str, Any]]:
    """Find potential embedded files in binary data."""
    signatures = get_file_signatures()
    embedded = []
    
    # Flatten all signatures
    all_sigs = []
    for file_type, sigs in signatures.items():
        for sig in sigs:
            all_sigs.append((file_type, sig))
            
    # Search for signatures
    for file_type, sig in all_sigs:
        offset = 0
        while True:
            pos = data.find(sig, offset)
            if pos == -1:
                break
                
            # Extract potential file data
            max_size = 1024 * 1024  # 1MB max
            chunk = data[pos:pos + max_size]
            
            embedded.append({
                'type': file_type,
                'offset': pos,
                'size': len(chunk),
                'entropy': calculate_entropy(chunk)
            })
            
            offset = pos + 1
            
    return embedded

def analyze_metadata_consistency(metadata: Dict) -> List[str]:
    """Check metadata for consistency issues."""
    issues = []
    
    # Check date consistency
    created = metadata.get('creation_date')
    modified = metadata.get('modification_date')
    
    if created and modified and modified < created:
        issues.append("Modification date is before creation date")
        
    # Check producer/creator consistency
    producer = metadata.get('producer', '').lower()
    creator = metadata.get('creator', '').lower()
    
    if producer and creator:
        # Known inconsistent combinations
        inconsistent_pairs = [
            ('microsoft', 'adobe'),
            ('openoffice', 'adobe'),
            ('libreoffice', 'adobe')
        ]
        
        for pair in inconsistent_pairs:
            if any(x in producer for x in pair) and \
               any(x in creator for x in pair):
                issues.append(f"Inconsistent producer/creator: {producer} vs {creator}")
                
    return issues

def get_object_characteristics(obj: Dict) -> Dict[str, Any]:
    """Get characteristics of a PDF object."""
    return {
        'type': obj.get('/Type', 'unknown'),
        'subtype': obj.get('/Subtype', 'unknown'),
        'filter': obj.get('/Filter', []),
        'length': obj.get('/Length', 0),
        'has_stream': hasattr(obj, 'get_stream'),
    }