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
    full_summary: str = Field(
        ..., description="The full summary text for the newsletter item"
    )
    source_url: str = Field(..., description="The source URL for the item")


class LLMSectionSelection(BaseModel):
    """Lightweight output schema for LLM section selection (indices only)."""

    section_indices: dict[str, list[int]] = Field(
        ...,
        description="Maps category name to list of item indices from the DataFrame",
    )
    selection_reasoning: str = Field(
        ...,
        description="Brief explanation of selection choices",
    )

    @field_validator("section_indices")
    @classmethod
    def validate_section_indices(cls, v: dict[str, list[int]]) -> dict[str, list[int]]:
        """Validate section indices constraints."""
        if len(v) > MAX_CATEGORIES:
            raise ValueError(
                f"Maximum {MAX_CATEGORIES} categories allowed, got {len(v)}"
            )

        total_items = sum(len(indices) for indices in v.values())
        if total_items > MAX_TOTAL_ITEMS:
            raise ValueError(
                f"Maximum {MAX_TOTAL_ITEMS} total items allowed, got {total_items}"
            )

        # Ensure at least 1 item per category
        for category, indices in v.items():
            if len(indices) < 1:
                raise ValueError(f"Category '{category}' must have at least 1 item")

        # Ensure no index is repeated across all sections
        all_indices: list[int] = []
        for indices in v.values():
            all_indices.extend(indices)
        duplicates = {idx for idx in all_indices if all_indices.count(idx) > 1}
        if duplicates:
            raise ValueError(
                f"Duplicate indices found across sections: {sorted(duplicates)}"
            )

        return v


class SectionSelection(BaseModel):
    """Output schema for section compilation."""

    section_items: dict[str, list[SectionItem]] = Field(
        ...,
        description="Maps category name to list of section items with full_summary and source_url",
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
    full_summary: str
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

        model = config.COMPILER_MODEL
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

    def _build_cache_key(self, df: pl.DataFrame) -> tuple:
        """Build a hashable cache key from the DataFrame.

        Uses Polars hash_rows to create a unique key that captures the input state.
        """
        # Hash relevant columns for cache key
        df_for_key = df.sort("index").select("index", "title", "relevance_score")
        row_hashes = tuple(df_for_key.hash_rows().to_list())
        # Include profile hash to invalidate cache on profile changes
        return (row_hashes, self.profile[:300:3])

    @instrument()
    async def compile_sections(self, df: pl.DataFrame) -> SectionSelection:
        """Select final newsletter items from scored and categorized dataset.

        Args:
            df: DataFrame with columns: title, full_summary, source_url, industry,
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
            "full_summary",
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

        # Build cache key (used for both cache lookup and storage)
        cache_key = self._build_cache_key(df) if self.cache is not None else None

        # Check cache first if available
        if self.cache is not None and cache_key is not None:
            if await self.cache.contains(cache_key):
                _LOGGER.debug("cache hit for section compilation")
                return await self.cache.get_item(cache_key)
            _LOGGER.debug("cache miss for section compilation")

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
The section_indices should map category names to lists of item indices (integers).
You only need to return the indices - the full item details will be retrieved programmatically."""

        deps = SectionCompilerDeps(df=df_sorted)

        _LOGGER.debug(f"Compiling sections from {len(df)} items")
        result = await self.agent.run(
            prompt, output_type=LLMSectionSelection, deps=deps
        )
        _LOGGER.debug(f"{result=!r}")

        # Hydrate full SectionSelection from LLM's index selection
        section_selection = self._hydrate_section_selection(result.output, df_sorted)

        # Cache the result if caching is enabled
        if self.cache is not None and cache_key is not None:
            await self.cache.set_item(cache_key, section_selection)
            _LOGGER.debug("cached section compilation result")

        return section_selection

    def _hydrate_section_selection(
        self, llm_selection: LLMSectionSelection, df: pl.DataFrame
    ) -> SectionSelection:
        """Build full SectionSelection from LLM's index-only selection.

        Args:
            llm_selection: The LLM's output with category->indices mapping.
            df: DataFrame with full item details.

        Returns:
            Complete SectionSelection with hydrated item data.
        """
        # Build index lookup for efficient access
        index_to_row = {row["index"]: row for row in df.iter_rows(named=True)}

        section_items: dict[str, list[SectionItem]] = {}
        for category, indices in llm_selection.section_indices.items():
            items = []
            for idx in indices:
                if idx not in index_to_row:
                    _LOGGER.warning(f"Index {idx} not found in DataFrame, skipping")
                    continue
                row = index_to_row[idx]
                items.append(
                    SectionItem(
                        index=row["index"],
                        title=row["title"],
                        full_summary=row["full_summary"],
                        source_url=row["source_url"],
                    )
                )
            if items:  # Only add category if it has valid items
                section_items[category] = items

        return SectionSelection(
            section_items=section_items,
            selected_categories=list(section_items.keys()),
            selection_reasoning=llm_selection.selection_reasoning,
        )

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
        Full details of the item including full_summary, source_url, and all metadata.
    """
    df = ctx.deps.df
    row = df.filter(pl.col("index") == index)

    if row.is_empty():
        raise ValueError(f"No item found with index {index}")

    row_dict = row.to_dicts()[0]
    return ItemDetail(
        index=row_dict["index"],
        title=row_dict["title"],
        full_summary=row_dict["full_summary"],
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
