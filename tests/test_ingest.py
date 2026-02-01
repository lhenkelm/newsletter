"""Integration tests for ingest module."""

import polars as pl
import pytest

from newsletter.ingest import load_recent_items


@pytest.mark.asyncio
async def test_load_recent_items_returns_dataframe():
    """Test that load_recent_items returns a Polars DataFrame with expected columns."""
    # disable warnings to avoid flaky warns
    df = await load_recent_items(
        cutoff_days=7,
        min_items=0,
        max_items=500,
        source_uri="hf://datasets/zenml/llmops-database/data/train-00000-of-00001.parquet",
    )

    assert isinstance(df, pl.DataFrame)
    assert "created_at" in df.columns
    assert "title" in df.columns
    assert "short_summary" in df.columns
