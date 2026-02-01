"""Category selection agent for news items."""

import asyncio
from logging import getLogger
from typing import Any, Literal, Self, Type

from logfire import instrument
from pydantic import BaseModel, Field, field_validator
from pydantic_ai import Agent

from newsletter.async_disk_cache import AsyncDiskCache
from newsletter.profile import load_audience_profile

_LOGGER = getLogger(__name__)

# Predefined categories for newsletter sections
ALLOWED_CATEGORIES = frozenset(
    {
        "AI Engineering",
        "LLMOps Tools",
        "Telecom Innovation",
        "Production ML",
        "Industry News",
        "Research Highlights",
        "RAG & Retrieval",
        "Prompt Engineering",
        "Model Fine-tuning",
        "AI Infrastructure",
    }
)

CategoryType = Literal[
    "AI Engineering",
    "LLMOps Tools",
    "Telecom Innovation",
    "Production ML",
    "Industry News",
    "Research Highlights",
    "RAG & Retrieval",
    "Prompt Engineering",
    "Model Fine-tuning",
    "AI Infrastructure",
]


class CategorySelection(BaseModel):
    """Output schema for category selection."""

    categories: list[str] = Field(
        ...,
        min_length=1,
        max_length=3,
        description="1-3 categories that best capture the audience interest angle",
    )
    reasoning: str = Field(
        ..., description="Brief explanation of why these categories were selected"
    )

    @field_validator("categories")
    @classmethod
    def validate_categories(cls, v: list[str]) -> list[str]:
        """Ensure all categories are from the allowed set."""
        invalid = set(v) - ALLOWED_CATEGORIES
        if invalid:
            raise ValueError(
                f"Invalid categories: {invalid}. Must be from: {sorted(ALLOWED_CATEGORIES)}"
            )
        return v


class CategoryAgent:
    """Async agent for assigning interest categories to news items.

    Attributes:
        agent: Underlying pydantic-ai Agent instance.
        profile: Textual audience profile for category selection.
        cache: Optional disk cache for faster development iteration.

    Recommended usage:
        import newsletter.config as config
        agent = await CategoryAgent.from_config(config)
        result = await agent.select_categories(
            title="...",
            short_summary="...",
            application_tags="...",
            tools_tags="...",
            techniques_tags="...",
        )
    """

    def __init__(self, agent: Agent, profile: str, cache: AsyncDiskCache | None = None):
        """Initialize the category selection agent."""
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
        _LOGGER.debug(f"initialised {self=!r}")

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
        """Create a CategoryAgent instance from configuration.

        Args:
            config: Configuration module with attributes
                CATEGORY_MODEL and
                AUDIENCE_PROFILE_PATH and
                CACHE_DIRECTORY.
        """
        agent = Agent(config.CATEGORY_MODEL, name=cls.__qualname__)
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
    async def select_categories(
        self,
        title: str,
        short_summary: str,
        application_tags: str,
        tools_tags: str,
        techniques_tags: str,
    ) -> CategorySelection:
        """Assign interest categories to a news item.

        Args:
            title: Headline of the news item.
            short_summary: Brief summary of the news item.
            application_tags: Comma-separated application tags from dataset.
            tools_tags: Comma-separated tools tags from dataset.
            techniques_tags: Comma-separated techniques tags from dataset.

        Returns:
            CategorySelection with 1-3 categories and reasoning.
        """
        if self.cache is not None:
            cache_key = (
                title,
                short_summary[:500:5],
                application_tags,
                tools_tags,
                techniques_tags,
                self.profile[:300:3],
            )
            if await self.cache.contains(cache_key):
                _LOGGER.debug(f"cache hit for {cache_key=!r}")
                return await self.cache.get_item(cache_key)
            _LOGGER.debug(f"cache miss for {cache_key=!r}")

        _LOGGER.debug(f"selecting categories for item with {title=!r}")

        # Format existing tags
        tags_section = ""
        if application_tags:
            tags_section += f"- Application Tags: {application_tags}\n"
        if tools_tags:
            tags_section += f"- Tools Tags: {tools_tags}\n"
        if techniques_tags:
            tags_section += f"- Techniques Tags: {techniques_tags}\n"

        prompt = f"""Select 1-3 interest categories for this news item based on the audience profile.

Audience Profile:
{self.profile}

News Item:
- Title: {title}
- Summary: {short_summary}
{tags_section}
Available Categories (select 1-3):
{chr(10).join(f"- {cat}" for cat in sorted(ALLOWED_CATEGORIES))}

Instructions:
1. Analyze the item content and how it relates to the audience's interests
2. Select categories that best capture WHY this item matters to the audience
3. Prefer specific categories over generic ones when applicable
4. Return exactly 1-3 categories from the allowed list above

Return your selection with brief reasoning as JSON.
Adhere to the schema : {CategorySelection.model_json_schema()}"""

        result = await self.agent.run(prompt, output_type=CategorySelection)
        _LOGGER.debug(f"{result=!r}")
        if self.cache is not None:
            await self.cache.set_item(cache_key, result.output)
            _LOGGER.debug(f"cached result for {cache_key=!r}")
        return result.output
