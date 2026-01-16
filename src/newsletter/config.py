"""Configuration module for newsletter generator."""

import os

CUTOFF_DAYS = int(os.getenv("CUTOFF_DAYS", "7"))
MIN_ITEMS = int(os.getenv("MIN_ITEMS", "3"))
MAX_ITEMS = int(os.getenv("MAX_ITEMS", "30"))
SOURCE_URI = os.getenv(
    "SOURCE_URI",
    "hf://datasets/zenml/llmops-database/data/train-00000-of-00001.parquet",
)
