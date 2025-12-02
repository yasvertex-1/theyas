import time
import random
from typing import Callable, Any, Tuple, Type

def retry_sync(
    func: Callable[..., Any],
    retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    allowed_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    *args: Any,
    **kwargs: Any
) -> Any:
    """
    Retry a synchronous function with exponential backoff.

    Args:
        func: The function to retry
        retries: Number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        backoff_factor: Exponential backoff multiplier
        allowed_exceptions: Tuple of exception types to retry on
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function

    Returns:
        The function result

    Raises:
        The last exception if all retries are exhausted
    """
    last_exception = None

    for attempt in range(retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e

            # Check if exception type is allowed for retry
            if not any(isinstance(e, exc_type) for exc_type in allowed_exceptions):
                raise e

            # Don't wait after the last attempt
            if attempt == retries:
                break

            # Calculate delay with exponential backoff and jitter
            delay = min(base_delay * (backoff_factor ** attempt), max_delay)
            jitter = random.uniform(0.5, 1.5) * delay
            time.sleep(jitter)

    # If we get here, all retries failed
    raise last_exception
