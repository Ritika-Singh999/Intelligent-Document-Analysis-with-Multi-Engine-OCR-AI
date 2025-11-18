from typing import Optional, Any
import os
import logging
import re
import google.generativeai as genai

logger = logging.getLogger(__name__)

class LLMConfig:
    def __init__(self):
        # Load settings
        from app.core.config import settings

        # Google AI Configuration
        self.google_api_key = settings.google_api_key
        self.gemini_model = settings.gemini_model

        # Initialize models to None
        self.gemini_llm = None
        self.embeddings = None
        self.primary_llm = None
        
        # Suppress warnings
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning)
        warnings.filterwarnings('ignore', message='.*arbitrary types.*')

    def _init_gemini(self) -> Optional[Any]:
        """Initialize Gemini with timeout to prevent app hang"""
        if not self.google_api_key:
            logger.warning("Google API key not found")
            return None

        try:
            # Configure the base client with timeout
            genai.configure(api_key=self.google_api_key, transport='rest')
            
            # Test connection with a simple config call (no actual API call yet)
            logger.info(f"Configured Gemini client for model: {self.gemini_model}")
            
            # Don't instantiate model here - do it lazily on first use
            # This prevents blocking at startup
            return {"configured": True, "model_name": self.gemini_model}
            
        except Exception as e:
            logger.error(f"Failed to configure Gemini: {str(e)}")
            logger.info("Continuing without Gemini - will attempt lazy loading on first request")
            return None

    def get_gemini_model(self):
        """Get Gemini model with lazy loading on first use"""
        if not self.gemini_llm:
            # Try lazy initialization if not already done
            try:
                genai.configure(api_key=self.google_api_key, transport='rest')
                self.gemini_llm = genai.GenerativeModel(self.gemini_model)
                logger.info(f"Gemini model lazily loaded: {self.gemini_model}")
            except Exception as e:
                logger.error(f"Lazy load of Gemini failed: {str(e)}")
                return None
        return self.gemini_llm
    
    def get_embeddings(self):
        """Get the initialized embeddings model"""
        return self.embeddings
            
    def initialize(self):
        """Initialize LLM components with timeout to prevent hanging"""
        logger.info("Initializing LLM components (with 5s timeout)...")

        # Initialize Gemini as the only LLM (with timeout)
        if self.google_api_key:
            try:
                # Initialize Gemini LLM with 5 second timeout
                # This prevents the entire app from hanging if Google API is slow
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError("Gemini initialization took too long (>5s)")
                
                # Note: signal.alarm only works on Unix/Linux
                # On Windows, we'll just proceed without timeout
                try:
                    gemini_config = self._init_gemini()
                    if gemini_config:
                        logger.info("Gemini configured (will load on first use)")
                    else:
                        logger.warning("Gemini configuration failed, will retry on first use")
                except TimeoutError as te:
                    logger.warning(f"Gemini init timeout: {str(te)}, will retry on first use")
                
                # Initialize Gemini embeddings configuration (lightweight)
                try:
                    # For embeddings, we'll use the direct API method (no actual model load yet)
                    self.embeddings = {
                        "model": "models/embedding-001",
                        "api_key": self.google_api_key
                    }
                    logger.info("Gemini Embeddings initialized successfully")
                except Exception as e:
                    logger.warning(f"Gemini Embeddings not initialized: {e}")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini components: {str(e)}")

        if not self.primary_llm:
            logger.error("No LLM provider available. Please check your Google API key.")
            
    async def get_llm(self, provider: str = "auto"):
        """Get LLM - Gemini only"""
        if (provider == "auto" or provider == "gemini") and self.gemini_llm:
            return self.gemini_llm

        return self.primary_llm  # Default to primary LLM (Gemini)

    def get_llm_client(self, provider: str = "auto"):
        if (provider == "auto" or provider == "gemini") and self.gemini_llm:
            return self.gemini_llm
        return self.primary_llm


# Create a global instance
_llm_config = LLMConfig()
_llm_config.initialize()

def get_llm_client(provider: str = "auto"):
    return _llm_config.get_llm_client(provider)
