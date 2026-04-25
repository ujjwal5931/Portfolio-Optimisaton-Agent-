import logging
import os
from typing import Optional

from dotenv import load_dotenv
from groq import Groq


# Load environment variables from .env file (if present).
# This is safe in production as long as secrets are managed securely in the environment.
load_dotenv()


def get_logger(name: str) -> logging.Logger:
    """
    Create and configure a module-level logger.

    The logger is configured once with a simple, production-friendly format.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def get_groq_client(api_key: Optional[str] = None) -> Groq:
    """
    Initialise and return a Groq client using the GROQ_API_KEY environment variable.

    The key is never hard-coded; it is read from the environment, which can be
    populated via a .env file in local development or via your deployment platform.
    """
    key = api_key or os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Please configure it in your environment or .env file."
        )

    # The Groq client automatically reads GROQ_API_KEY from the environment,
    # but we pass it explicitly here to make the dependency obvious.
    client = Groq(api_key=key)
    return client


def sanitize_risk_level(level: str) -> str:
    """
    Normalise risk level strings into one of: 'low', 'moderate', 'high'.

    This keeps downstream logic simple and robust against LLM variation.
    """
    if not level:
        return "moderate"
    value = level.strip().lower()
    if value in {"low", "conservative"}:
        return "low"
    if value in {"high", "aggressive", "very high"}:
        return "high"
    return "moderate"


def sanitize_geography(geo: str) -> str:
    """
    Normalise geography into canonical buckets: 'india', 'us', 'global'.
    """
    if not geo:
        return "global"
    value = geo.strip().lower()
    if "india" in value or "indian" in value:
        return "india"
    if value in {"us", "usa", "united states", "american"}:
        return "us"
    return "global"

