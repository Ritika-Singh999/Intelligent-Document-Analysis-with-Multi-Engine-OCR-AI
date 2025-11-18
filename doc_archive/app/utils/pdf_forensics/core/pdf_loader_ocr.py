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

class PDFLoaderWithOCR:

    def __init__(self, file_path: str):
        self.file_path = file_path
        self._doc = None

    async def __aenter__(self):
        await self.load()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    @cancellable(timeout=30)
    async def load(self):
        if not self._doc:
            try:
                self._doc = await run_in_thread(fitz.open, self.file_path)
            except Exception as e:
                error_msg = str(e).lower()
                if "format error" in error_msg and "non-page object in page tree" in error_msg:
                    logger.error(f"PDF corruption detected: {self.file_path} - {e}")
                    raise ValueError(f"Corrupted PDF: invalid page tree structure in {self.file_path}")
                else:
                    logger.error(f"PDF load failed: {self.file_path} - {e}")
                    raise

    async def close(self):
        if self._doc:
            await run_in_thread(self._doc.close)
            self._doc = None

    @memoize_to_disk(ttl=3600)
    def extract_text(self, start_page: int = 0, end_page: Optional[int] = None, use_ocr: bool = True) -> str:
        end = end_page or len(self._doc)
        text = ""
        for page_num in range(start_page, end):
            page = self._doc[page_num]
            page_text = page.get_text()

            # If no text found and OCR is enabled, try OCR
            if not page_text.strip() and use_ocr:
                try:
                    page_text = self._extract_text_with_ocr(page)
                except Exception as e:
                    logger.warning(f"OCR failed for page {page_num}: {e}")

            text += page_text
        return text

    def _extract_text_with_ocr(self, page) -> str:
        try:
            import pytesseract
            from PIL import Image
            import io

            # Convert page to image
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x scaling for better OCR
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))

            # Perform OCR
            text = pytesseract.image_to_string(img)
            return text
        except ImportError:
            logger.warning("pytesseract not available, OCR disabled")
            return ""
        except Exception as e:
            logger.warning(f"OCR extraction failed: {e}")
            return ""
