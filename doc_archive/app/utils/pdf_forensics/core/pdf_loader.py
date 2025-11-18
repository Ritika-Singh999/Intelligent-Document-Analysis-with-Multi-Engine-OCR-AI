"""
Core PDF loading and analysis utilities.
"""
from typing import Dict, List, Any, Optional
import fitz  # PyMuPDF
import os
import logging
from dataclasses import dataclass
from datetime import datetime
import hashlib

from ...helpers import run_in_thread, memoize_to_disk
from ...cancellable_task import cancellable

logger = logging.getLogger(__name__)

@dataclass
class PDFMetadata:
    """Structured PDF metadata."""
    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    keywords: Optional[str] = None
    creator: Optional[str] = None
    producer: Optional[str] = None
    creation_date: Optional[datetime] = None
    modification_date: Optional[datetime] = None
    encryption: Optional[Dict] = None
    file_size: int = 0
    page_count: int = 0
    version: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            k: (v.isoformat() if isinstance(v, datetime) else v)
            for k, v in self.__dict__.items()
            if v is not None
        }

class PDFLoader:
    """Efficient PDF document loader with caching."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self._doc = None
        self._metadata = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        await self.load()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
        
    @cancellable(timeout=30)
    async def load(self):
        """Load PDF document asynchronously."""
        if not self._doc:
            self._doc = await run_in_thread(fitz.open, self.file_path)
            
    async def close(self):
        """Close PDF document."""
        if self._doc:
            await run_in_thread(self._doc.close)
            self._doc = None
            
    @property
    async def metadata(self) -> PDFMetadata:
        """Get document metadata with caching."""
        if not self._metadata:
            if not self._doc:
                await self.load()
                
            raw_metadata = await run_in_thread(self._get_raw_metadata)
            self._metadata = PDFMetadata(**raw_metadata)
            
        return self._metadata
        
    def _get_raw_metadata(self) -> Dict:
        """Get raw metadata (runs in thread)."""
        meta = self._doc.metadata
        return {
            "title": meta.get("title"),
            "author": meta.get("author"),
            "subject": meta.get("subject"),
            "keywords": meta.get("keywords"),
            "creator": meta.get("creator"),
            "producer": meta.get("producer"),
            "creation_date": self._parse_pdf_date(meta.get("creationDate")),
            "modification_date": self._parse_pdf_date(meta.get("modDate")),
            "encryption": getattr(self._doc, 'encryption_info', None),
            "file_size": os.path.getsize(self.file_path),
            "page_count": len(self._doc),
            "version": getattr(self._doc, 'pdf_version', None)
        }
        
    @staticmethod
    def _parse_pdf_date(date_str: Optional[str]) -> Optional[datetime]:
        """Parse PDF date string to datetime."""
        if not date_str:
            return None
            
        try:
            # Handle D: prefix and timezone
            if date_str.startswith("D:"):
                date_str = date_str[2:]
            # Basic ISO format
            return datetime.strptime(date_str[:14], "%Y%m%d%H%M%S")
        except Exception:
            return None
            
    @memoize_to_disk(ttl=3600)
    def extract_text(self, start_page: int = 0, end_page: Optional[int] = None) -> str:
        """Extract text from PDF with caching."""
        end = end_page or len(self._doc)
        text = ""
        for page_num in range(start_page, end):
            page = self._doc[page_num]
            text += page.get_text()
        return text
        
    async def get_page_images(self, page_num: int) -> List[Dict[str, Any]]:
        """Get images from specific page."""
        if not self._doc:
            await self.load()
            
        return await run_in_thread(self._extract_page_images, page_num)
        
    def _extract_page_images(self, page_num: int) -> List[Dict[str, Any]]:
        """Extract images from page (runs in thread)."""
        page = self._doc[page_num]
        image_list = page.get_images()
        
        images = []
        for img_info in image_list:
            try:
                xref = img_info[0]
                base_image = self._doc.extract_image(xref)
                
                if base_image:
                    image_data = {
                        "size": len(base_image["image"]),
                        "width": base_image["width"],
                        "height": base_image["height"],
                        "colorspace": base_image["colorspace"],
                        "bpc": base_image["bpc"],
                        "type": base_image["ext"],
                        "xref": xref
                    }
                    images.append(image_data)
            except Exception as e:
                logger.warning(f"Error extracting image: {e}")
                
        return images