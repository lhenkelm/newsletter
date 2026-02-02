"""Weekly ingest module for loading recent items from the ZenML LLMOps dataset."""

from logging import getLogger
from datetime import datetime, timedelta, timezone
from logfire import instrument

import polars as pl

_LOGGER = getLogger(__name__)


@instrument()
async def load_recent_items(
    cutoff_days: int,
    min_items: int,
    max_items: int,
    source_uri: str,
) -> pl.DataFrame:
    """Load recent items from the ZenML LLMOps dataset.

    Args:
        cutoff_days: Number of days to look back from now.
        min_items: Minimum expected items; warns if fewer.
        max_items: Maximum expected items; warns if more.
        source_uri: URI of the parquet dataset to load.

    Returns:
        A Polars DataFrame with items from the last `cutoff_days` days.

    Raises:
        RuntimeError: If no recent items are found.
        ValueError: If input parameters are invalid.
    """
    if min_items > max_items:
        raise ValueError(
            f"expect 'min_items' <= 'max_items', got {min_items=!r}, {max_items=!r} "
        )
    if cutoff_days < 1:
        raise ValueError(f"expect cutoff_days >= 1, got {cutoff_days=!r}")

    cutoff_date = datetime.now(timezone.utc) - timedelta(days=cutoff_days)

    df = await (
        pl.scan_parquet(source_uri)
        .with_columns(
            pl.col("created_at").str.to_datetime(time_zone="UTC").alias("created_at")
        )
        .filter(pl.col("created_at") >= cutoff_date)
        .collect_async()
    )

    row_count = len(df)
    if row_count < min_items:
        _LOGGER.warning(f"Only {row_count} items found (minimum expected: {min_items})")
    elif row_count > max_items:
        _LOGGER.warning(f"{row_count} items found (maximum expected: {max_items})")

    if row_count <= 0:
        raise RuntimeError("No recent items found, aborting")

    _LOGGER.info(f"Loaded {len(df)} recent items")
    return df
