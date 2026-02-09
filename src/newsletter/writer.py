"""Newsletter writer agent for generating polished Markdown newsletters."""

import asyncio
from datetime import date, timedelta
from logging import getLogger
from pathlib import Path
from typing import Any, Self, Type

from logfire import instrument
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from newsletter.async_disk_cache import AsyncDiskCache
from newsletter.profile import load_audience_profile
from newsletter.section_compiler import SectionItem

_LOGGER = getLogger(__name__)


class NewsletterOutput(BaseModel):
    """Output schema for newsletter generation."""

    newsletter_markdown: str = Field(
        ..., description="The full Markdown content of the newsletter"
    )
    title: str = Field(..., description="The newsletter title for logging")


class NewsletterWriterAgent:
    """Async agent for generating polished Markdown newsletters.

    The agent takes categorized section items and produces a complete
    newsletter with introduction, categorized sections, and closing.

    Attributes:
        agent: Underlying pydantic-ai Agent instance.
        profile: Textual audience profile for tone and emphasis.
        cache: Optional disk cache for faster development iteration.

    Recommended usage:
        import newsletter.config as config
        agent = await NewsletterWriterAgent.from_config(config)
        result = await agent.write_newsletter(section_items)
    """

    def __init__(
        self,
        agent: Agent,
        profile: str,
        cutoff_days: int,
        cache: AsyncDiskCache | None = None,
    ):
        """Initialize the newsletter writer agent."""
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
        self.cutoff_days = cutoff_days
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
            f"cutoff_days={self.cutoff_days!r}, "
            f"cache={self.cache!r})"
        )

    @classmethod
    @instrument()
    async def from_config(cls: Type[Self], config: Any) -> Self:
        """Create a NewsletterWriterAgent instance from configuration.

        Args:
            config: Configuration module with attributes
                WRITER_MODEL and
                AUDIENCE_PROFILE_PATH and
                CACHE_DIRECTORY.
        """
        model = config.WRITER_MODEL

        agent = Agent(model, name=cls.__qualname__)

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

        return cls(agent, profile, config.CUTOFF_DAYS, cache)

    def _build_cache_key(self, section_items: dict[str, list[SectionItem]]) -> tuple:
        """Build a hashable cache key from section items."""
        # Create a tuple of (category, tuple of item indices) for caching
        items_key = tuple(
            (cat, tuple(item.index for item in items))
            for cat, items in sorted(section_items.items())
        )
        # Include profile hash to invalidate cache on profile changes
        return (items_key, self.profile[:300:3])

    def _format_section_items_for_prompt(
        self, section_items: dict[str, list[SectionItem]]
    ) -> str:
        """Format section items as text for the LLM prompt."""
        lines = []
        for category, items in section_items.items():
            lines.append(f"\n## {category}")
            for item in items:
                lines.append(f"\n### {item.title}")
                lines.append(f"Source: {item.source_url}")
                lines.append(f"Summary: {item.full_summary}")
        return "\n".join(lines)

    @instrument()
    async def write_newsletter(
        self,
        section_items: dict[str, list[SectionItem]],
    ) -> NewsletterOutput:
        """Generate a polished Markdown newsletter from section items.

        Args:
            section_items: Mapping of category names to lists of SectionItem objects.

        Returns:
            NewsletterOutput with full Markdown content and title.
        """
        if not section_items:
            raise ValueError("section_items cannot be empty")

        # Check cache first
        if self.cache is not None:
            cache_key = self._build_cache_key(section_items)
            if await self.cache.contains(cache_key):
                _LOGGER.debug("cache hit for newsletter generation")
                return await self.cache.get_item(cache_key)

        # Format items for prompt
        formatted_items = self._format_section_items_for_prompt(section_items)
        categories = list(section_items.keys())

        # Compute date range from cutoff_days
        end_date = date.today()
        start_date = end_date - timedelta(days=self.cutoff_days)
        date_info = f"\n\n## Coverage Period\nThis newsletter covers content from {start_date.isoformat()} to {end_date.isoformat()}.\n"

        prompt = f"""You are a professional newsletter writer. Generate a polished weekly newsletter in Markdown format.

## Audience Profile
{self.profile}{date_info}

## Available Section Items
{formatted_items}

## Instructions
Write a complete newsletter with the following structure:

1. **Title**: Create a catchy weekly newsletter header. Use a level-1 heading (# Title).

2. **Introduction**: Write a short engaging paragraph (2-3 sentences) teasing the key themes from this week's content.

3. **Categorized Sections**: Create one H2 section (## Category Name) for each of these categories: {", ".join(categories)}
   - Start each section with a brief intro sentence
   - Include a bulleted list of items
   - For each item, write a concise 1-2 sentence summary that distils the key insight
   - Include an inline Markdown link using the item's title and source URL: [Title](URL)

4. **Closing**: Write a short wrap-up (2-3 sentences) with a forward-looking note or call-to-action.

## Constraints
- Keep each section (intro + items) under ~150 words
- Maintain a professional but engaging tone matched to the audience profile
- Every item MUST include its link in the format [Title](source_url)
- Distill the full summaries down to concise, actionable insights

Generate the complete newsletter now:"""

        result = await self.agent.run(prompt, output_type=NewsletterOutput)
        output = result.output

        # Cache result
        if self.cache is not None:
            await self.cache.set_item(cache_key, output)

        _LOGGER.info(f"Generated newsletter: {output.title}")
        return output


async def save_newsletter(
    output: NewsletterOutput, output_path: str | Path | None = None
) -> Path:
    """Save newsletter Markdown to file.

    Args:
        output: The NewsletterOutput from the writer agent.
        output_path: Path to save the newsletter. Defaults to newsletter.md in project root.

    Returns:
        Path where the newsletter was saved.
    """
    import os

    if output_path is None:
        output_path = os.getenv("OUTPUT_PATH", "newsletter.md")

    path = Path(output_path)

    def _write_newsletter():
        # Ensure parent directories exist
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output.newsletter_markdown)

    # Write file in a thread to avoid blocking
    await asyncio.to_thread(_write_newsletter)

    _LOGGER.info(f"Saved newsletter to {path}")
    return path
