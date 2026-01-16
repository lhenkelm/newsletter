"""Configuration module for newsletter generator."""

import os
from dataclasses import dataclass


@dataclass
class IngestConfig:
    """Configuration for data ingestion."""

    cutoff_days: int
    min_items: int
    max_items: int
    source_uri: str

    @classmethod
    def from_env(cls) -> "IngestConfig":
        """Load configuration from environment variables with defaults."""
        return cls(
            cutoff_days=int(os.getenv("CUTOFF_DAYS", "7")),
            min_items=int(os.getenv("MIN_ITEMS", "3")),
            max_items=int(os.getenv("MAX_ITEMS", "30")),
            source_uri=os.getenv(
                "SOURCE_URI",
                "hf://datasets/zenml/llmops-database/data/train-00000-of-00001.parquet",
            ),
        )
