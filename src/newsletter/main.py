"""Main entry point for the newsletter generator."""

import asyncio

import polars as pl

from newsletter import config
from newsletter.ingest import load_recent_items
from newsletter.scoring import ScoringAgent


async def score_items(df: pl.DataFrame) -> pl.DataFrame:
    """Score items for relevance to the target audience.

    Args:
        df: DataFrame with news items to score.

    Returns:
        DataFrame with added 'relevance_score' and 'score_reasoning' columns.
    """
    # Initialize and load the scoring agent
    agent = ScoringAgent(model=config.SCORING_MODEL)
    agent.load_audience_profile(profile_path=config.AUDIENCE_PROFILE_PATH)

    # Score each item
    scores = []
    reasonings = []

    for row in df.iter_rows(named=True):
        result = await agent.score_item(
            title=row.get("title", ""),
            short_summary=row.get("summary", ""),
            industry=row.get("industry", ""),
            company=row.get("company", ""),
        )
        scores.append(result.score)
        reasonings.append(result.reasoning)

    # Add scores as new columns
    return df.with_columns(
        [
            pl.Series("relevance_score", scores),
            pl.Series("score_reasoning", reasonings),
        ]
    )


def main() -> None:
    """Main function to run the newsletter generator."""
    # Step 1: Load recent items
    df = load_recent_items(
        cutoff_days=config.CUTOFF_DAYS,
        min_items=config.MIN_ITEMS,
        max_items=config.MAX_ITEMS,
        source_uri=config.SOURCE_URI,
    )
    print(f"Loaded {len(df)} recent items")

    # Step 2: Score items for relevance
    print("Scoring items for relevance...")
    df_scored = asyncio.run(score_items(df))
    print(f"Scored {len(df_scored)} items")
    print(
        f"Score distribution: {df_scored['relevance_score'].value_counts().sort('relevance_score')}"
    )


if __name__ == "__main__":
    main()
