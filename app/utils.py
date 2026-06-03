import logging
import sys
from app.config import LOG_LEVEL

def get_logger(name: str) -> logging.Logger:
    """Configures and returns a logger with standard formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
        
        # Console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        
        # Formatter
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger

import time
from typing import Callable, Any

def retry_on_quota_limit(func: Callable[..., Any], *args, retries: int = 5, initial_delay: float = 6.0, **kwargs) -> Any:
    """Wrapper that retries a function if it encounters Gemini API rate limit or quota exceeded errors (429)."""
    delay = initial_delay
    last_exception = None
    
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            err_msg = str(e).lower()
            
            # Check for API 429 quota limits
            if "429" in err_msg or "quota" in err_msg or "rate limit" in err_msg or "resource_exhausted" in err_msg:
                logging.getLogger("rate_limiter").warning(
                    f"Gemini API rate limit hit. Sleeping for {delay} seconds (attempt {attempt + 1}/{retries})..."
                )
                time.sleep(delay)
                delay = min(delay * 2, 60.0)  # cap backoff at 60s
            else:
                # Re-raise other exceptions immediately (like network or config errors)
                raise e
                
    # If all retries failed, raise the last encountered error
    raise last_exception

