import asyncio
import concurrent.futures
import logging
from google import genai
from google.genai import types
from app.config.settings import settings
from app.utils.retry import retry_sync
from app.utils.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

# Global clients and executors
PRIMARY_CLIENT = genai.Client(api_key=settings.GOOGLE_API_KEY)
SECONDARY_CLIENT = genai.Client(api_key=settings.GOOGLE_API_KEY_SECONDARY)

# Rate limiters for each key
# We track usage for each key independently to know when to switch
primary_rate_limiter = RateLimiter(settings.RATE_LIMIT_THRESHOLD, settings.RATE_LIMIT_WINDOW)
secondary_rate_limiter = RateLimiter(settings.RATE_LIMIT_THRESHOLD, settings.RATE_LIMIT_WINDOW)

executor = concurrent.futures.ThreadPoolExecutor(max_workers=settings.MAX_WORKERS)
LLM_SEMAPHORE = asyncio.Semaphore(settings.LLM_CONCURRENCY)

class RecitationError(Exception):
    """Exception raised when Gemini API returns RECITATION finish_reason"""
    pass

class QuotaExceededError(Exception):
    """Exception raised when API quota/limit is exceeded"""
    pass

def _is_quota_error(error: Exception) -> bool:
    """
    Check if an error is a quota/limit exceeded error.
    Detects: 429 (Too Many Requests), quota exceeded, daily limits, etc.
    """
    error_str = str(error).lower()
    error_message = getattr(error, 'message', '').lower() if hasattr(error, 'message') else ''
    
    quota_keywords = [
        'quota',
        'rate limit',
        '429',
        'too many requests',
        'daily limit',
        'resource exhausted',
        'exceeded',
        'limit exceeded'
    ]
    
    full_error_text = f"{error_str} {error_message}"
    return any(keyword in full_error_text for keyword in quota_keywords)

def _call_genai_sync(parts, force_secondary: bool = False) -> str:
    """
    Call Gemini API with automatic API key rotation on quota/limit errors.
    
    Args:
        parts: The content parts for the API call
        force_secondary: If True, force use of secondary key (used after quota error on primary)
    
    Returns:
        The generated text response
        
    Raises:
        QuotaExceededError: If both keys hit quota limits
        RecitationError: If safety filter is triggered
    """
    # Determine which client to use
    # Logic:
    # 1. Check Primary usage. If < Threshold, use Primary.
    # 2. If Primary >= Threshold, check Secondary usage.
    # 3. If Secondary < Threshold, use Secondary.
    # 4. If both are full, default to Primary (or handle error/wait - for now, we'll try Primary and let it rate limit if needed, or maybe Secondary if we want to toggle back)
    # User requirement: "vice versa when the calls to this api key is upto 950 calls in a minute it should switch back to the former"
    
    use_secondary = force_secondary
    
    if not force_secondary:
        # Check current rates
        primary_usage = primary_rate_limiter.get_current_rate()
        secondary_usage = secondary_rate_limiter.get_current_rate()
        
        if primary_usage >= settings.RATE_LIMIT_THRESHOLD:
            if secondary_usage < settings.RATE_LIMIT_THRESHOLD:
                use_secondary = True
                logger.info(f"Primary key rate limit reached ({primary_usage}). Switching to Secondary key.")
            else:
                # Both are full. 
                # Strategy: Stick to the one that is 'less' full or just default to Primary?
                # Or maybe toggle back to Primary as per "switch back to the former"?
                # If both are full, we are likely to get 429s anyway.
                # Let's default to Primary if both are full, or maybe the one with lower usage?
                if secondary_usage < primary_usage:
                    use_secondary = True
                logger.warning(f"Both keys are near rate limit (Primary: {primary_usage}, Secondary: {secondary_usage}).")

    client = SECONDARY_CLIENT if use_secondary else PRIMARY_CLIENT
    limiter = secondary_rate_limiter if use_secondary else primary_rate_limiter
    key_label = "Secondary" if use_secondary else "Primary"
    
    # Record request
    limiter.record_request()
    
    try:
        response = client.models.generate_content(model=settings.GEMINI_MODEL, contents=parts)
    except Exception as e:
        # Check if this is a quota/limit error
        if _is_quota_error(e):
            logger.warning(f"{key_label} API key hit quota/limit error: {str(e)}")
            
            # If we were using primary, try secondary
            if not use_secondary:
                logger.info(f"Quota error on Primary key. Attempting with Secondary key...")
                try:
                    return _call_genai_sync(parts, force_secondary=True)
                except Exception as secondary_error:
                    logger.error(f"Secondary key also failed with quota error: {str(secondary_error)}")
                    raise QuotaExceededError(f"Both API keys exceeded quota limits. Primary: {str(e)}, Secondary: {str(secondary_error)}")
            else:
                # Both keys have hit quota
                logger.error(f"Secondary API key also hit quota/limit error: {str(e)}")
                raise QuotaExceededError(f"Both API keys exceeded quota limits: {str(e)}")
        else:
            # Not a quota error, re-raise
            raise
    
    # Check for RECITATION error (safety filter triggered)
    if hasattr(response, 'candidates') and response.candidates:
        candidate = response.candidates[0]
        if hasattr(candidate, 'finish_reason'):
            finish_reason = candidate.finish_reason
            
            # Check if finish_reason is RECITATION (handle both enum and string)
            is_recitation = False
            if finish_reason == 'RECITATION':
                is_recitation = True
            elif hasattr(finish_reason, 'name') and finish_reason.name == 'RECITATION':
                is_recitation = True
            elif hasattr(finish_reason, 'value') and finish_reason.value == 'RECITATION':
                is_recitation = True
            elif str(finish_reason).upper().find('RECITATION') != -1:
                is_recitation = True
            
            if is_recitation:
                logger.warning(f"RECITATION error detected, will retry. Finish reason: {finish_reason}")
                raise RecitationError("Gemini API returned RECITATION finish_reason (safety filter), retrying...")
    
    text = getattr(response, "text", None)
    if text is not None and text.strip():
        return text
    
    # If no text and no RECITATION error, return string representation or empty string
    return str(response) if response else ""

def _parts_for_image(image_bytes: bytes, prompt: str):
    return [types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"), prompt]

def _parts_for_text(text: str, prompt: str):
    return [prompt, text]

def _parts_for_mixed(images: list[bytes], text: str | None, prompt: str):
    parts = [prompt]
    if text:
        parts.append(text)
    for img in images:
        parts.append(types.Part.from_bytes(data=img, mime_type="image/jpeg"))
    return parts

def process_image_sync_with_retry(image_bytes: bytes, prompt: str) -> str:
    return retry_sync(_call_genai_sync, retries=5, base_delay=2.0, allowed_exceptions=(RecitationError,), parts=_parts_for_image(image_bytes, prompt))

def process_text_sync_with_retry(text: str, prompt: str) -> str:
    return retry_sync(_call_genai_sync, retries=5, base_delay=2.0, allowed_exceptions=(RecitationError,), parts=_parts_for_text(text, prompt))

def process_mixed_sync_with_retry(images: list[bytes], text: str | None, prompt: str) -> str:
    return retry_sync(_call_genai_sync, retries=5, base_delay=2.0, allowed_exceptions=(RecitationError,), parts=_parts_for_mixed(images, text, prompt))

async def process_image_async(image_bytes: bytes, prompt: str) -> str:
    async with LLM_SEMAPHORE:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, process_image_sync_with_retry, image_bytes, prompt)

async def process_text_async(text: str, prompt: str) -> str:
    async with LLM_SEMAPHORE:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, process_text_sync_with_retry, text, prompt)

async def process_mixed_async(images: list[bytes], text: str | None, prompt: str) -> str:
    async with LLM_SEMAPHORE:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, process_mixed_sync_with_retry, images, text, prompt)

def shutdown_executor():
    """Shutdown the executor gracefully"""
    executor.shutdown(wait=False)
