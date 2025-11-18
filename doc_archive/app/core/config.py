from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API Keys
    openai_api_key: str = None
    google_api_key: str = None
    
    # Model settings
    openai_model: str = "gpt-4o-mini"
    gemini_model: str = "gemini-pro"
    
    # System settings
    upload_dir: str = 'documents'
    
    # Environment configuration
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "allow"  # Allow extra fields in environment
    }
    
    # For backward compatibility
    @property
    def OPENAI_API_KEY(self): return self.openai_api_key
    
    @property
    def GOOGLE_API_KEY(self): return self.google_api_key
    
    @property
    def UPLOAD_DIR(self): return self.upload_dir

settings = Settings()
