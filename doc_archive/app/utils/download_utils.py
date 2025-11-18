"""
Utilities for downloading documents from URLs.
"""
import os
import hashlib
import logging
import tempfile
import aiohttp
import asyncio
from typing import List, Optional

logger = logging.getLogger(__name__)

async def download_file_from_url(url: str, temp_dir: Optional[str] = None) -> str:
    """Download file from URL to temporary location."""
    try:
        # Create temp directory if not provided
        if not temp_dir:
            temp_dir = tempfile.mkdtemp()

        # Extract filename from URL or create one
        filename = os.path.basename(url.split('?')[0])  # Remove query params
        if not filename:
            # Generate filename from URL hash if no filename in URL
            filename = hashlib.md5(url.encode()).hexdigest()[:16]

        # Add extension if missing (assume PDF for documents)
        if not os.path.splitext(filename)[1]:
            filename += '.pdf'

        filepath = os.path.join(temp_dir, filename)

        # Download file using aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                response.raise_for_status()
                content = await response.read()

        # Save to file
        with open(filepath, 'wb') as f:
            f.write(content)

        logger.info(f"Downloaded {url} to {filepath}")
        return filepath

    except Exception as e:
        logger.error(f"Failed to download {url}: {str(e)}")
        raise

async def download_documents_from_urls(urls: List[str]) -> List[str]:
    """Download multiple documents from URLs in parallel and return file paths."""
    temp_dir = tempfile.mkdtemp(prefix="doc_payload_")

    try:
        # Create tasks for parallel downloads
        tasks = [download_file_from_url(url, temp_dir) for url in urls]

        # Execute downloads in parallel
        downloaded_files = await asyncio.gather(*tasks, return_exceptions=True)

        # Check for exceptions and raise if any
        for i, result in enumerate(downloaded_files):
            if isinstance(result, Exception):
                logger.error(f"Failed to download {urls[i]}: {str(result)}")
                raise result

        return downloaded_files

    except Exception as e:
        # Clean up on error
        for filepath in downloaded_files:
            if isinstance(filepath, str):  # Only if it's a valid filepath
                try:
                    os.unlink(filepath)
                except:
                    pass
        try:
            os.rmdir(temp_dir)
        except:
            pass
        raise
