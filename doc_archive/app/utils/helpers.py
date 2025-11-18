"""
General utility helpers for document processing.
"""
import os
import hashlib
from typing import Any, Dict, List, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
from functools import lru_cache, wraps
import json
import time

logger = logging.getLogger(__name__)

# Global thread pool for CPU-bound operations
_thread_pool = ThreadPoolExecutor(
    max_workers=min(32, (os.cpu_count() or 1) + 4),
    thread_name_prefix="helper_pool"
)

def get_file_hash(file_path: str) -> str:
    """Calculate SHA-256 hash of a file efficiently."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

@lru_cache(maxsize=1000)
def get_mime_type(file_path: str) -> str:
    """Get MIME type of file with caching."""
    import mimetypes
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type or 'application/octet-stream'

async def run_in_thread(func, *args, **kwargs) -> Any:
    """Run CPU-bound function in thread pool."""
    return await asyncio.get_event_loop().run_in_executor(
        _thread_pool, 
        lambda: func(*args, **kwargs)
    )

def batch_process(items: List[Any], batch_size: int = 10):
    """Generator for batch processing items."""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

class Cache:
    """Simple file-based cache with TTL support."""
    
    def __init__(self, cache_dir: str, ttl: int = 3600):
        self.cache_dir = cache_dir
        self.ttl = ttl
        os.makedirs(cache_dir, exist_ok=True)
        
    def _get_cache_path(self, key: str) -> str:
        """Get cache file path for key."""
        return os.path.join(
            self.cache_dir,
            hashlib.sha256(key.encode()).hexdigest()
        )
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            path = self._get_cache_path(key)
            if not os.path.exists(path):
                return None
                
            # Check TTL
            if self.ttl:
                mtime = os.path.getmtime(path)
                if (time.time() - mtime) > self.ttl:
                    os.unlink(path)
                    return None
                    
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Cache get error for {key}: {e}")
            return None
            
    def set(self, key: str, value: Any):
        """Set value in cache."""
        try:
            path = self._get_cache_path(key)
            with open(path, 'w') as f:
                json.dump(value, f)
        except Exception as e:
            logger.warning(f"Cache set error for {key}: {e}")
            
    def clear(self):
        """Clear all cached items."""
        for file in os.listdir(self.cache_dir):
            try:
                os.unlink(os.path.join(self.cache_dir, file))
            except Exception:
                pass

# Initialize global cache
file_cache = Cache(os.path.join("cache", "files"))

def memoize_to_disk(ttl: int = 3600):
    """Decorator for disk-based memoization with TTL."""
    def decorator(func):
        cache = Cache(
            os.path.join("cache", func.__name__),
            ttl=ttl
        )
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            result = cache.get(key)
            
            if result is None:
                result = func(*args, **kwargs)
                cache.set(key, result)
                
            return result
        return wrapper
    return decorator