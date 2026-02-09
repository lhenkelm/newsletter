"""Configuration module for newsletter generator."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

CUTOFF_DAYS = int(os.getenv("CUTOFF_DAYS", "7"))
MIN_ITEMS = int(os.getenv("MIN_ITEMS", "3"))
MAX_ITEMS = int(os.getenv("MAX_ITEMS", "30"))
SOURCE_URI = os.getenv(
    "SOURCE_URI",
    "hf://datasets/zenml/llmops-database/data/train-00000-of-00001.parquet",
)

# Scoring Agent Configuration
SCORING_MODEL = os.getenv("SCORING_MODEL", "openai:gpt-4o-mini")
AUDIENCE_PROFILE_PATH = os.getenv("AUDIENCE_PROFILE_PATH", "data/audience_profile.txt")

# Category Agent Configuration
CATEGORY_MODEL = os.getenv("CATEGORY_MODEL", "openai:gpt-4o-mini")

# Section Compiler Agent Configuration
COMPILER_MODEL = os.getenv("COMPILER_MODEL", "openai:gpt-4o-mini")
MAX_TOTAL_ITEMS = int(os.getenv("MAX_TOTAL_ITEMS", "10"))
MAX_CATEGORIES = int(os.getenv("MAX_CATEGORIES", "3"))

# Newsletter Writer Agent Configuration
WRITER_MODEL = os.getenv("WRITER_MODEL", "openai:gpt-4o-mini")
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "newsletter.md")

_VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
LOGGING_LEVEL = str(os.getenv("LOGGING_LEVEL", "WARNING")).upper()
if LOGGING_LEVEL not in _VALID_LOG_LEVELS:
    raise RuntimeError(
        "unexpected LOGGING_LEVEL env. variable value: "
        f"expected one of {_VALID_LOG_LEVELS!r}, "
        f"got {LOGGING_LEVEL}"
    )

_CACHE_DIRECTORY = os.getenv("CACHE_DIRECTORY", None)
CACHE_DIRECTORY: Path | None = None
if _CACHE_DIRECTORY is not None:
    CACHE_DIRECTORY = Path(_CACHE_DIRECTORY.strip())
