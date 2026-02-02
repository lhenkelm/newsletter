"""Section compiler agent for selecting final newsletter items."""

import asyncio
from logging import getLogger
from typing import Any, Self, Type

import polars as pl
from logfire import instrument
from pydantic import BaseModel, Field, field_validator
from pydantic_ai import Agent, RunContext, Tool

from newsletter.async_disk_cache import AsyncDiskCache
from newsletter.profile import load_audience_profile
from newsletter.config import MAX_TOTAL_ITEMS, MAX_CATEGORIES

_LOGGER = getLogger(__name__)


class SectionItem(BaseModel):
    """A single item in a newsletter section."""

    index: int = Field(..., description="The index of the item in the source DataFrame")
    title: str = Field(..., description="The title of the newsletter item")
    summary: str = Field(..., description="The summary text for the newsletter item")
    source_url: str = Field(..., description="The source URL for the item")


class SectionSelection(BaseModel):
    """Output schema for section compilation."""

    section_items: dict[str, list[SectionItem]] = Field(
        ...,
        description="Maps category name to list of section items with summary and source_url",
    )
    selected_categories: list[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_CATEGORIES,
        description="The 3 or fewer chosen section categories",
    )
    selection_reasoning: str = Field(
        ...,
        description="Brief explanation of selection choices",
    )

    @field_validator("section_items")
    @classmethod
    def validate_section_items(
        cls, v: dict[str, list[SectionItem]]
    ) -> dict[str, list[SectionItem]]:
        """Validate section items constraints."""
        if len(v) > MAX_CATEGORIES:
            raise ValueError(
                f"Maximum {MAX_CATEGORIES} categories allowed, got {len(v)}"
            )

        total_items = sum(len(items) for items in v.values())
        if total_items > MAX_TOTAL_ITEMS:
            raise ValueError(
                f"Maximum {MAX_TOTAL_ITEMS} total items allowed, got {total_items}"
            )

        # Ensure at least 1 item per category
        for category, items in v.items():
            if len(items) < 1:
                raise ValueError(f"Category '{category}' must have at least 1 item")

        return v


class SectionCompilerDeps(BaseModel):
    """Dependencies for the section compiler agent tools."""

    model_config = {"arbitrary_types_allowed": True}

    df: pl.DataFrame = Field(..., description="Full DataFrame with all item details")


class ItemDetail(BaseModel):
    """Details of a single news item."""

    index: int
    title: str
    summary: str
    source_url: str
    industry: str
    company: str
    relevance_score: int
    score_reasoning: str
    interest_categories: list[str]
    category_reasoning: str


class SectionCompilerAgent:
    """Async agent for compiling newsletter sections from scored/categorized items.

    The agent selects up to 10 items across up to 3 categories to maximize
    newsletter value for the target audience.

    Attributes:
        agent: Underlying pydantic-ai Agent instance with tools.
        profile: Textual audience profile for selection priorities.
        cache: Optional disk cache for faster development iteration.

    Recommended usage:
        import newsletter.config as config
        agent = await SectionCompilerAgent.from_config(config)
        result = await agent.compile_sections(df_with_scores_and_categories)
    """

    def __init__(self, agent: Agent, profile: str, cache: AsyncDiskCache | None = None):
        """Initialize the section compiler agent."""
        if not isinstance(agent, Agent):
            raise TypeError(
                f"{self.__class__.__name__} expected 'agent' to be an instance of 'Agent'"
                f" got {agent!r} of type {type(agent)!r}"
            )
        self.agent = agent
        if not isinstance(profile, str):
            raise TypeError(
                f"{self.__class__.__name__} expected 'profile' to be of type 'str'"
                f" got {profile!r} of type {type(profile)!r}"
            )
        if not profile:
            raise ValueError(f"Audience profile cannot be empty, got {profile!r}")
        self.profile = profile
        self.cache = cache
        _LOGGER.debug(f"initialized {self!r}")

    def __repr__(self):
        if len(self.profile) > 180:
            profile = f"{self.profile[:87]} [...] {self.profile[-87:]}"
        else:
            profile = self.profile
        return (
            f"{self.__class__.__qualname__}("
            f"agent={self.agent!r}, "
            f"profile={profile!r}, "
            f"cache={self.cache!r})"
        )

    @classmethod
    @instrument()
    async def from_config(cls: Type[Self], config: Any) -> Self:
        """Create a SectionCompilerAgent instance from configuration.

        Args:
            config: Configuration module with attributes
                COMPILER_MODEL (defaults to CATEGORY_MODEL) and
                AUDIENCE_PROFILE_PATH and
                CACHE_DIRECTORY.
        """
        # Use COMPILER_MODEL if available, otherwise fall back to CATEGORY_MODEL
        model = getattr(config, "COMPILER_MODEL", None) or config.CATEGORY_MODEL

        # Create agent with tools
        agent = Agent(
            model,
            name=cls.__qualname__,
            deps_type=SectionCompilerDeps,
            tools=[
                Tool(get_item_details, takes_ctx=True),
                Tool(get_items_by_category, takes_ctx=True),
                Tool(get_items_by_company, takes_ctx=True),
                Tool(get_high_score_items, takes_ctx=True),
            ],
        )

        async with asyncio.TaskGroup() as tg:
            load_profile_task = tg.create_task(
                load_audience_profile(config.AUDIENCE_PROFILE_PATH)
            )
            if config.CACHE_DIRECTORY:
                init_cache_task = tg.create_task(
                    AsyncDiskCache.from_cache_dir_path(
                        config.CACHE_DIRECTORY / cls.__qualname__
                    )
                )

        profile = await load_profile_task
        cache = None
        if config.CACHE_DIRECTORY:
            cache = await init_cache_task

        return cls(agent, profile, cache)

    @instrument()
    async def compile_sections(self, df: pl.DataFrame) -> SectionSelection:
        """Select final newsletter items from scored and categorized dataset.

        Args:
            df: DataFrame with columns: title, summary, source_url, industry,
                company, relevance_score, score_reasoning, interest_categories,
                category_reasoning. Must have an 'index' column.

        Returns:
            SectionSelection with section_items mapping, selected_categories,
            and selection_reasoning.
        """
        # Validate required columns
        required_cols = {
            "index",
            "title",
            "summary",
            "source_url",
            "industry",
            "company",
            "relevance_score",
            "score_reasoning",
            "interest_categories",
            "category_reasoning",
        }
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")

        # Sort by relevance_score descending and prepare subset for agent
        df_sorted = df.sort("relevance_score", descending=True)

        # Create the subset view for the agent (index, title, relevance_score, interest_categories)
        df_subset = df_sorted.select(
            "index", "title", "relevance_score", "interest_categories"
        )

        # Format items for the prompt
        items_text = self._format_items_for_prompt(df_subset)

        # Calculate category distribution for context
        category_counts = self._get_category_distribution(df_sorted)

        prompt = f"""Select up to {MAX_TOTAL_ITEMS} items distributed across up to {MAX_CATEGORIES} categories for the newsletter.

Audience Profile:
{self.profile}

Selection Criteria:
1. Prefer items with higher relevance_score (4-5 = high priority, 3 = medium, 0-2 = low)
2. Balance category representation: pick top 3 most populated/relevant categories
3. Within each category, rank by relevance_score
4. Ensure diversity: avoid multiple items from same company unless exceptionally relevant
5. Each selected category must have at least 1 item

Available Items (sorted by relevance_score descending):
{items_text}

Category Distribution:
{category_counts}

Use the available tools to get full details on specific items when needed:
- get_item_details(index): Get complete details for a specific item
- get_items_by_category(category): Get all items in a specific category
- get_items_by_company(company): Check for company diversity
- get_high_score_items(min_score): Get all items with score >= min_score

Select the best items for the newsletter and return your selection as JSON.
The section_items should map category names to lists of objects with 'index', 'title', 'summary', and 'source_url' fields.
Use the full 'summary' text (not short_summary) from item details for the newsletter content."""

        deps = SectionCompilerDeps(df=df_sorted)

        _LOGGER.debug(f"Compiling sections from {len(df)} items")
        result = await self.agent.run(prompt, output_type=SectionSelection, deps=deps)
        _LOGGER.debug(f"{result=!r}")

        return result.output

    def _format_items_for_prompt(self, df: pl.DataFrame) -> str:
        """Format DataFrame subset as text for the prompt."""
        lines = []
        for row in df.iter_rows(named=True):
            categories = row["interest_categories"]
            if isinstance(categories, list):
                cat_str = ", ".join(categories)
            else:
                cat_str = str(categories)
            lines.append(
                f"[{row['index']}] (score={row['relevance_score']}) {row['title']}"
                f"\n    Categories: {cat_str}"
            )
        return "\n".join(lines)

    def _get_category_distribution(self, df: pl.DataFrame) -> str:
        """Get category distribution summary."""
        # Explode categories and count
        cat_counts = (
            df.select(pl.col("interest_categories").explode().alias("category"))
            .group_by("category")
            .len()
            .sort("len", descending=True)
        )
        lines = [
            f"- {row['category']}: {row['len']} items"
            for row in cat_counts.iter_rows(named=True)
        ]
        return "\n".join(lines)


# Tool functions for the agent


async def get_item_details(
    ctx: RunContext[SectionCompilerDeps], index: int
) -> ItemDetail:
    """Get complete details for a specific item by index.

    Args:
        index: The index of the item to retrieve.

    Returns:
        Full details of the item including summary, source_url, and all metadata.
    """
    df = ctx.deps.df
    row = df.filter(pl.col("index") == index)

    if row.is_empty():
        raise ValueError(f"No item found with index {index}")

    row_dict = row.to_dicts()[0]
    return ItemDetail(
        index=row_dict["index"],
        title=row_dict["title"],
        summary=row_dict["summary"],
        source_url=row_dict["source_url"],
        industry=row_dict["industry"],
        company=row_dict["company"],
        relevance_score=row_dict["relevance_score"],
        score_reasoning=row_dict["score_reasoning"],
        interest_categories=row_dict["interest_categories"],
        category_reasoning=row_dict["category_reasoning"],
    )


async def get_items_by_category(
    ctx: RunContext[SectionCompilerDeps], category: str
) -> list[dict[str, Any]]:
    """Get all items belonging to a specific category.

    Args:
        category: The category name to filter by.

    Returns:
        List of items with index, title, relevance_score, and company.
    """
    df = ctx.deps.df

    # Filter rows where the category is in interest_categories list
    filtered = df.filter(pl.col("interest_categories").list.contains(category)).select(
        "index", "title", "relevance_score", "company"
    )

    return filtered.to_dicts()


async def get_items_by_company(
    ctx: RunContext[SectionCompilerDeps], company: str
) -> list[dict[str, Any]]:
    """Get all items from a specific company.

    Args:
        company: The company name to filter by.

    Returns:
        List of items with index, title, and relevance_score.
    """
    df = ctx.deps.df
    filtered = df.filter(
        pl.col("company").str.to_lowercase().str.contains(company.lower())
    ).select("index", "title", "relevance_score", "interest_categories")

    return filtered.to_dicts()


async def get_high_score_items(
    ctx: RunContext[SectionCompilerDeps], min_score: int
) -> list[dict[str, Any]]:
    """Get all items with relevance_score >= min_score.

    Args:
        min_score: Minimum relevance score threshold (0-5).

    Returns:
        List of items with index, title, relevance_score, company, and categories.
    """
    df = ctx.deps.df
    filtered = df.filter(pl.col("relevance_score") >= min_score).select(
        "index", "title", "relevance_score", "company", "interest_categories"
    )

    return filtered.to_dicts()
