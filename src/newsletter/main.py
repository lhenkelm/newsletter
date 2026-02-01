"""Main entry point for the newsletter generator."""

import asyncio
from logging import DEBUG, getLogger, basicConfig, getLevelNamesMapping

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
    _LOGGER.info("Scoring items for relevance...")
    agent = await ScoringAgent.from_config(config)

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

    df = df.with_columns(
        [
            pl.Series("relevance_score", scores),
            pl.Series("score_reasoning", reasonings),
        ]
    )
    _LOGGER.info(f"Scored {len(df)} items")
    if getLevelNamesMapping()[config.LOGGING_LEVEL] > DEBUG:
        return df
    _LOGGER.debug(
        f"Score distribution: {df['relevance_score'].value_counts().sort('relevance_score')}"
    )
    return df


async def categorize_items(df: pl.DataFrame) -> pl.DataFrame:
    """Assign interest categories to news items.

    Args:
        df: DataFrame with news items to categorize.

    Returns:
        DataFrame with added 'interest_categories' and 'category_reasoning' columns.
    """
    _LOGGER.info("Categorizing items...")
    agent = await CategoryAgent.from_config(config)

    tasks = []
    async with asyncio.TaskGroup() as tg:
        for row in df.iter_rows(named=True):
            tasks.append(
                tg.create_task(
                    agent.select_categories(
                        title=row["title"],
                        short_summary=row["short_summary"],
                        application_tags=row["application_tags"],
                        tools_tags=row["tools_tags"],
                        techniques_tags=row["techniques_tags"],
                    )
                )
            )
    results = ((task.result()) for task in tasks)
    categories, reasonings = zip(
        *((result.categories, result.reasoning) for result in results)
    )

    df = df.with_columns(
        [
            pl.Series("interest_categories", categories),
            pl.Series("category_reasoning", reasonings),
        ]
    )
    _LOGGER.info(f"Categorized {len(df)} items")
    if getLevelNamesMapping()[config.LOGGING_LEVEL] > DEBUG:
        return df
    all_categories = (
        df.select(pl.col("interest_categories").explode())
        .to_series()
        .value_counts()
        .sort("count", descending=True)
    )
    _LOGGER.debug(f"Category distribution:\n{all_categories}")
    return df


async def main() -> None:
    """Main function to run the newsletter generator."""
    basicConfig(level=config.LOGGING_LEVEL)

    df = await load_recent_items(
        cutoff_days=config.CUTOFF_DAYS,
        min_items=config.MIN_ITEMS,
        max_items=config.MAX_ITEMS,
        source_uri=config.SOURCE_URI,
    )
    _LOGGER.info(f"Loaded {len(df)} recent items")

    df = df.with_row_index("index")
    async with asyncio.TaskGroup() as tg:
        scoring_task = tg.create_task(score_items(df))
        categorisation_task = tg.create_task(categorize_items(df))

    df_scored = await scoring_task
    df_categorized = await categorisation_task
    df = df_scored.join(
        df_categorized.select("index", "interest_categories", "category_reasoning"),
        on="index",
    )


if __name__ == "__main__":
    asyncio.run(main())
