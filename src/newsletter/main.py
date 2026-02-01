"""Main entry point for the newsletter generator."""

import asyncio
from logging import getLogger, basicConfig

import polars as pl

from newsletter import config
from newsletter.category import CategoryAgent
from newsletter.ingest import load_recent_items
from newsletter.scoring import ScoringAgent

_LOGGER = getLogger(__name__)


async def score_items(df: pl.DataFrame) -> pl.DataFrame:
    """Score items for relevance to the target audience.

    Args:
        df: DataFrame with news items to score.

    Returns:
        DataFrame with added 'relevance_score' and 'score_reasoning' columns.
    """
    agent = ScoringAgent.from_config(config)

    tasks = []
    async with asyncio.TaskGroup() as tg:
        for row in df.iter_rows(named=True):
            tasks.append(
                tg.create_task(
                    agent.score_item(
                        title=row["title"],
                        short_summary=row["short_summary"],
                        industry=row["industry"],
                        company=row["company"],
                    )
                )
            )
    results = ((task.result()) for task in tasks)
    scores, reasonings = zip(*((result.score, result.reasoning) for result in results))
    # Add scores as new columns
    return df.with_columns(
        [
            pl.Series("relevance_score", scores),
            pl.Series("score_reasoning", reasonings),
        ]
    )


async def categorize_items(df: pl.DataFrame) -> pl.DataFrame:
    """Assign interest categories to news items.

    Args:
        df: DataFrame with news items to categorize.

    Returns:
        DataFrame with added 'interest_categories' and 'category_reasoning' columns.
    """
    # Initialize and load the category agent
    agent = CategoryAgent(model=config.CATEGORY_MODEL)
    agent.load_audience_profile(profile_path=config.AUDIENCE_PROFILE_PATH)

    # Categorize each item
    categories_list = []
    reasonings = []

    for row in df.iter_rows(named=True):
        result = await agent.select_categories(
            title=row.get("title", ""),
            short_summary=row.get("short_summary", ""),
            application_tags=row.get("application_tags"),
            tools_tags=row.get("tools_tags"),
            techniques_tags=row.get("techniques_tags"),
        )
        categories_list.append(result.categories)
        reasonings.append(result.reasoning)

    # Add categories as new columns
    return df.with_columns(
        [
            pl.Series("interest_categories", categories_list),
            pl.Series("category_reasoning", reasonings),
        ]
    )


async def main() -> None:
    """Main function to run the newsletter generator."""
    basicConfig(level=config.LOGGING_LEVEL)

    # Step 1: Load recent items
    df = load_recent_items(
        cutoff_days=config.CUTOFF_DAYS,
        min_items=config.MIN_ITEMS,
        max_items=config.MAX_ITEMS,
        source_uri=config.SOURCE_URI,
    )
    _LOGGER.info(f"Loaded {len(df)} recent items")

    # Step 2: Score items for relevance
    _LOGGER.info("Scoring items for relevance...")
    df_scored = await score_items(df)
    _LOGGER.info(f"Scored {len(df_scored)} items")
    _LOGGER.debug(
        f"Score distribution: {df_scored['relevance_score'].value_counts().sort('relevance_score')}"
    )

    # Step 3: Categorize items
    _LOGGER.info("Categorizing items...")
    df_categorized = await categorize_items(df_scored)
    _LOGGER.info(f"Categorized {len(df_categorized)} items")

    # Show category distribution
    all_categories = (
        df_categorized.select(pl.col("interest_categories").explode())
        .to_series()
        .value_counts()
        .sort("count", descending=True)
    )
    _LOGGER.debug(f"Category distribution:\n{all_categories}")


if __name__ == "__main__":
    asyncio.run(main())
