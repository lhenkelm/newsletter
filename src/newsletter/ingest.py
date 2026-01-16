"""Weekly ingest module for loading recent items from the ZenML LLMOps dataset."""

import warnings
from datetime import datetime, timedelta, timezone

import polars as pl


def load_recent_items(
    cutoff_days: int = 7,
    min_items: int = 3,
    max_items: int = 30,
    source_uri: str = "hf://datasets/zenml/llmops-database/data/train-00000-of-00001.parquet",
) -> pl.DataFrame:
    """Load recent items from the ZenML LLMOps dataset.

    Args:
        cutoff_days: Number of days to look back from now.
        min_items: Minimum expected items; warns if fewer.
        max_items: Maximum expected items; warns if more.
        source_uri: URI of the parquet dataset to load.

    Returns:
        A Polars DataFrame with items from the last `cutoff_days` days.
    """
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=cutoff_days)

    df = (
        pl.scan_parquet(source_uri)
        .with_columns(
            pl.col("created_at").str.to_datetime(time_zone="UTC").alias("created_at")
        )
        .filter(pl.col("created_at") >= cutoff_date)
        .collect()
    )

    row_count = len(df)
    if row_count < min_items:
        warnings.warn(f"Only {row_count} items found (minimum expected: {min_items})")
    elif row_count > max_items:
        warnings.warn(f"{row_count} items found (maximum expected: {max_items})")

    return df
