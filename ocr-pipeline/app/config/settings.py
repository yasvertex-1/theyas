import os
from app.config.prompts import (
    ARABIC_EXTRACTION_PROMPT,
    ARABIC_EXTRACTION_PROMPT_STRICT,
    LAYOUT_CLASSIFICATION_PROMPT,
    TABLE_EXTRACTION_PROMPT,
    COLUMN_AWARE_EXTRACTION_PROMPT,
)




class Settings:
    # Gemini API Configuration
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "AIzaSyBr0CFutMOvOshHOnfjCoHc1p9G0dZdgC0")
    GOOGLE_API_KEY_SECONDARY: str = os.getenv("GOOGLE_API_KEY_SECONDARY", "AIzaSyCWocApu62Ai9TOs2rxTPK7a_V6GnD-ntA")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # Rate Limiting Configuration
    RATE_LIMIT_THRESHOLD: int = int(os.getenv("RATE_LIMIT_THRESHOLD", "950"))
    RATE_LIMIT_WINDOW: int = int(os.getenv("RATE_LIMIT_WINDOW", "60"))

    # PDF Processing Configuration
    PDF_DPI: int = int(os.getenv("PDF_DPI", "300"))
    MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", "8"))
    LLM_CONCURRENCY: int = int(os.getenv("LLM_CONCURRENCY", "100"))
    # Number of PDFs processed concurrently at the pipeline level
    DOC_CONCURRENCY: int = int(os.getenv("DOC_CONCURRENCY", "50"))

    # Arabic Text Extraction Configuration
    ARABIC_EXTRACTION_PROMPT: str = ARABIC_EXTRACTION_PROMPT
    ARABIC_EXTRACTION_PROMPT_STRICT: str = ARABIC_EXTRACTION_PROMPT_STRICT
    TABLE_EXTRACTION_PROMPT: str = TABLE_EXTRACTION_PROMPT
    LAYOUT_CLASSIFICATION_PROMPT: str = LAYOUT_CLASSIFICATION_PROMPT
    COLUMN_AWARE_EXTRACTION_PROMPT: str = COLUMN_AWARE_EXTRACTION_PROMPT
    LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "3"))
    LLM_RETRY_DELAY_SECONDS: float = float(os.getenv("LLM_RETRY_DELAY_SECONDS", "2.0"))
    LLM_RETRY_BACKOFF: float = float(os.getenv("LLM_RETRY_BACKOFF", "2.0"))

    # Output Configuration
    OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "./output_pdfs")
    EXTRACTED_TAG: str = "_extracted"

    # Split detection thresholds (tunable)
    SPLIT_VALLEY_RATIO: float = float(os.getenv("SPLIT_VALLEY_RATIO", "0.35"))
    SPLIT_GAP_MIN_FRAC: float = float(os.getenv("SPLIT_GAP_MIN_FRAC", "0.03"))
    SPLIT_BOTH_SIDES_CONTENT_FRAC: float = float(os.getenv("SPLIT_BOTH_SIDES_CONTENT_FRAC", "0.25"))
    SPLIT_TABLE_BANDS_MIN: int = int(os.getenv("SPLIT_TABLE_BANDS_MIN", "20"))
    SPLIT_MIN_SIDE_FRAC: float = float(os.getenv("SPLIT_MIN_SIDE_FRAC", "0.30"))

# Global settings instance
settings = Settings()
