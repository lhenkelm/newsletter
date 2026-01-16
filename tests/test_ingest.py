"""Integration tests for ingest module."""

import polars as pl
import pytest

from newsletter.ingest import load_recent_items


def test_load_recent_items_returns_dataframe():
    """Test that load_recent_items returns a Polars DataFrame with expected columns."""
    # disable warnings to avoid flaky warns
    df = load_recent_items(cutoff_days=7, min_items=0, max_items=500)

    assert isinstance(df, pl.DataFrame)
    assert "created_at" in df.columns
    assert "title" in df.columns
    assert "summary" in df.columns

