"""Newsletter writer agent for generating polished Markdown newsletters."""

import asyncio
import re
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

# Regex pattern to extract URLs from Markdown links: [text](url)
_MARKDOWN_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


class LinkValidationError(Exception):
    """Raised when generated newsletter contains invalid links."""

    def __init__(
        self,
        message: str,
        missing_links: set[str] | None = None,
        extra_links: set[str] | None = None,
    ):
        super().__init__(message)
        self.missing_links = missing_links or set()
        self.extra_links = extra_links or set()


def extract_urls_from_markdown(markdown: str) -> set[str]:
    """Extract all URLs from Markdown link syntax.

    Args:
        markdown: The Markdown text to extract URLs from.

    Returns:
        Set of unique URLs found in the markdown.
    """
    matches = _MARKDOWN_LINK_PATTERN.findall(markdown)
    return {url.strip() for _, url in matches}


def validate_newsletter_links(
    markdown: str, expected_urls: set[str]
) -> tuple[set[str], set[str]]:
    """Validate that newsletter markdown contains exactly the expected links.

    Args:
        markdown: The generated newsletter Markdown.
        expected_urls: Set of URLs that should appear in the newsletter.

    Returns:
        Tuple of (missing_urls, extra_urls). Both empty if valid.
    """
    found_urls = extract_urls_from_markdown(markdown)

    missing_urls = expected_urls - found_urls
    extra_urls = found_urls - expected_urls

    return missing_urls, extra_urls


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
        max_link_validation_retries: int,
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
        self.max_link_validation_retries = max_link_validation_retries
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
            f"max_link_validation_retries={self.max_link_validation_retries!r}, "
            f"cache={self.cache!r})"
        )

    @classmethod
    @instrument()
    async def from_config(cls: Type[Self], config: Any) -> Self:
        """Create a NewsletterWriterAgent instance from configuration.

        Args:
            config: Configuration module with attributes
                WRITER_MODEL,
                AUDIENCE_PROFILE_PATH,
                CACHE_DIRECTORY, and
                MAX_LINK_VALIDATION_RETRIES.
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

        return cls(
            agent,
            profile,
            config.CUTOFF_DAYS,
            config.MAX_LINK_VALIDATION_RETRIES,
            cache,
        )

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

    @staticmethod
    def _extract_expected_urls(section_items: dict[str, list[SectionItem]]) -> set[str]:
        """Extract all expected URLs from section items."""
        return {item.source_url for items in section_items.values() for item in items}

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

        Raises:
            ValueError: If section_items is empty.
            LinkValidationError: If link validation fails after max retries.
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

        # Extract expected URLs for validation
        expected_urls = self._extract_expected_urls(section_items)

        # Compute date range from cutoff_days
        end_date = date.today()
        start_date = end_date - timedelta(days=self.cutoff_days)
        date_info = f"\n\n## Coverage Period\nThis newsletter covers content from {start_date.isoformat()} to {end_date.isoformat()}.\n"

        base_prompt = f"""You are a professional newsletter writer. Generate a polished newsletter in Markdown format.

## Audience Profile
{self.profile}{date_info}

## Available Section Items
{formatted_items}

## Instructions
Write a complete newsletter with the following structure:

1. **Title**: Create a catchy newsletter header. Use a level-1 heading (# Title).

2. **Introduction**: Write a short engaging paragraph (2-3 sentences) teasing the key themes.

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

        last_error: LinkValidationError | None = None

        for attempt in range(self.max_link_validation_retries):
            if attempt == 0:
                prompt = base_prompt
            else:
                # Build retry prompt with specific feedback
                assert last_error is not None  # Set in previous iteration
                error_feedback = self._build_link_error_feedback(last_error)
                prompt = f"{base_prompt}\n\n## IMPORTANT - Previous Attempt Failed\n{error_feedback}"
                _LOGGER.warning(
                    f"Retrying newsletter generation (attempt {attempt + 1}/{self.max_link_validation_retries}) "
                    f"due to link validation failure"
                )

            result = await self.agent.run(prompt, output_type=NewsletterOutput)
            output = result.output

            # Validate links
            missing_urls, extra_urls = validate_newsletter_links(
                output.newsletter_markdown, expected_urls
            )

            if not missing_urls and not extra_urls:
                # Validation passed
                _LOGGER.debug("Link validation passed")
                break

            # Build error for retry or final exception
            error_parts = []
            if missing_urls:
                error_parts.append(f"Missing links: {missing_urls}")
            if extra_urls:
                error_parts.append(f"Unexpected/hallucinated links: {extra_urls}")

            last_error = LinkValidationError(
                f"Link validation failed: {'; '.join(error_parts)}",
                missing_links=missing_urls,
                extra_links=extra_urls,
            )
            _LOGGER.warning(str(last_error))
        else:
            # All retries exhausted
            raise last_error  # type: ignore[misc]

        # Cache result (only cache validated results)
        if self.cache is not None:
            await self.cache.set_item(cache_key, output)

        _LOGGER.info(f"Generated newsletter: {output.title}")
        return output

    def _build_link_error_feedback(self, error: LinkValidationError) -> str:
        """Build feedback message for retry prompt based on link validation error."""
        feedback_parts = []
        if error.missing_links:
            links_list = "\n".join(f"  - {url}" for url in error.missing_links)
            feedback_parts.append(
                f"The following source URLs were NOT included but MUST be:\n{links_list}"
            )
        if error.extra_links:
            links_list = "\n".join(f"  - {url}" for url in error.extra_links)
            feedback_parts.append(
                f"The following URLs were used but are NOT from the provided items (remove them):\n{links_list}"
            )
        return "\n\n".join(feedback_parts)


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
