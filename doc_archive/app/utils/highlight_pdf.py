"""
PDF highlighting and annotation utilities.
"""
from typing import List, Dict, Tuple, Optional
import fitz  # PyMuPDF
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor
import asyncio

from .helpers import run_in_thread, memoize_to_disk
from .cancellable_task import cancellable

logger = logging.getLogger(__name__)

@dataclass
class Highlight:
    """Represents a text highlight in a PDF."""
    page: int
    text: str
    rect: Tuple[float, float, float, float]
    color: Tuple[float, float, float] = (1, 1, 0)  # Yellow
    opacity: float = 0.3
    metadata: Optional[Dict] = None

class PDFHighlighter:
    """Handles PDF highlighting and annotation with caching."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self._doc = None
        self._highlights: List[Highlight] = []
        
    async def __aenter__(self):
        """Async context manager entry."""
        await self.open()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
        
    @cancellable(timeout=30)
    async def open(self):
        """Open PDF document asynchronously."""
        if not self._doc:
            self._doc = await run_in_thread(fitz.open, self.file_path)
            
    async def close(self):
        """Close PDF document."""
        if self._doc:
            await run_in_thread(self._doc.close)
            self._doc = None
            
    @cancellable(timeout=10)
    async def add_highlight(self, highlight: Highlight):
        """Add a highlight to the document."""
        if not self._doc:
            raise RuntimeError("Document not open")
            
        self._highlights.append(highlight)
        page = self._doc[highlight.page]
        
        await run_in_thread(
            self._apply_highlight,
            page,
            highlight
        )
        
    def _apply_highlight(self, page, highlight: Highlight):
        """Apply highlight to page (runs in thread)."""
        try:
            annot = page.add_highlight_annot(highlight.rect)
            annot.set_colors(stroke=highlight.color)
            annot.set_opacity(highlight.opacity)
            
            if highlight.metadata:
                annot.info.update(highlight.metadata)
                
            # Add popup annotation if metadata contains comments
            if highlight.metadata and "comment" in highlight.metadata:
                popup = page.add_popup_annot(annot)
                popup.set_info(content=highlight.metadata["comment"])
                
        except Exception as e:
            logger.error(f"Error applying highlight: {e}")
            
    @memoize_to_disk(ttl=3600)
    def find_text_instances(self, text: str) -> List[Dict]:
        """Find all instances of text in document with caching."""
        results = []
        for page_num in range(len(self._doc)):
            page = self._doc[page_num]
            instances = page.search_for(text)
            if instances:
                results.extend([{
                    "page": page_num,
                    "rect": tuple(rect),
                    "text": text
                } for rect in instances])
        return results
        
    async def highlight_text(
        self,
        text: str,
        color: Tuple[float, float, float] = (1, 1, 0),
        opacity: float = 0.3,
        metadata: Optional[Dict] = None
    ):
        """Find and highlight all instances of text."""
        instances = await run_in_thread(
            self.find_text_instances,
            text
        )
        
        for instance in instances:
            highlight = Highlight(
                page=instance["page"],
                text=instance["text"],
                rect=instance["rect"],
                color=color,
                opacity=opacity,
                metadata=metadata
            )
            await self.add_highlight(highlight)
            
    async def save(self, output_path: Optional[str] = None):
        """Save highlighted document."""
        if not self._doc:
            raise RuntimeError("Document not open")
            
        save_path = output_path or self.file_path.replace(
            ".pdf",
            "_highlighted.pdf"
        )
        
        await run_in_thread(self._doc.save, save_path)
        return save_path