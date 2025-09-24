# app/config.py - Configuration for different LLM options

import os
from dotenv import load_dotenv

load_dotenv()

class LLMConfig:
    """Configuration class for different LLM options."""
    
    # OpenAI Configuration (Best quality, requires API key)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = "gpt-3.5-turbo"  # or "gpt-4" for even better quality
    
    # Hugging Face Models (Free, local)
    HF_MODELS = [
        "google/flan-t5-large",      # Good for Q&A
        "google/flan-t5-xl",         # Better but larger
        "microsoft/DialoGPT-large",  # Good for conversations
        "google/flan-t5-base",       # Fallback option
    ]
    
    # Model settings
    MAX_LENGTH = 512
    TEMPERATURE = 0.7
    TEMPERATURE_PLAIN = 0.8  # Higher temperature for standalone responses
    
    # Preferred model selection
    PREFERRED_MODEL = "auto"  # Options: "openai", "huggingface", "auto"
    
    @classmethod
    def get_best_available_model(cls):
        """Determine the best available model based on configuration."""
        if cls.PREFERRED_MODEL == "openai" and cls.OPENAI_API_KEY:
            return "openai"
        elif cls.PREFERRED_MODEL == "huggingface":
            return "huggingface"
        elif cls.OPENAI_API_KEY:
            return "openai"  # Auto-select OpenAI if available
        else:
            return "huggingface"  # Fallback to HuggingFace

