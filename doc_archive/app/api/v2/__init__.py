"""
API v2 Router Loader
"""
import logging
from fastapi import APIRouter

logger = logging.getLogger(__name__)

# main v2 router (must be named router)
router = APIRouter(prefix="/api/v2", tags=["v2"])

# Lazy load routers on first access to avoid circular imports
_routers_loaded = False

def _load_routers():
    global _routers_loaded
    if _routers_loaded:
        return
    
    try:
        # import actual endpoint router
        from app.api.v2.universal_extraction import router as universal_router
        
        # attach endpoint router to v2 router
        router.include_router(universal_router)
        _routers_loaded = True
        logger.info("Loaded universal extraction router successfully")

    except Exception as e:
        logger.warning(f"Could not load universal extraction router: {e}")
        _routers_loaded = True

# Try to load at import, but don't fail if there's an error
try:
    _load_routers()
except Exception as e:
    logger.debug(f"Deferred router loading: {e}")

__all__ = ["router"]
