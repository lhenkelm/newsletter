"""Relevance scoring agent for news items."""

import asyncio
from typing import Any, Self, Type

from logfire import instrument
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from logging import getLogger

from newsletter.async_disk_cache import AsyncDiskCache
from newsletter.profile import load_audience_profile

_LOGGER = getLogger(__name__)


class RelevanceScore(BaseModel):
    """Output schema for relevance scoring."""

    score: int = Field(
        ge=0,
        le=5,
        description="Relevance score from 0 (irrelevant) to 5 (high priority)",
    )
    reasoning: str = Field(description="Brief explanation of the score")


class ScoringAgent:
    """Async agent for scoring news items against audience profile.

    Attributes:
        agent: Underlying pydantic-ai Agent instance.
        profile: Textual audience profile for relevance scoring.
        cache: Optional disk cache for faster development iteration

    Recommended usage:

        import newsletter.config as config
        agent = await ScoringAgent.from_config(config)
        result = await agent.score_item(
            title="...",
            short_summary="...",
            industry="...",
            company="...",
        )
    """

    def __init__(self, agent: Agent, profile: str, cache: AsyncDiskCache | None = None):
        """Initialize the scoring agent."""
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
        """Create a ScoringAgent instance from configuration.

        Args:
            config: Configuration module with attributes
                SCORING_MODEL and
                AUDIENCE_PROFILE_PATH and
                CACHE_DIRECTORY.
        """
        agent = Agent(config.SCORING_MODEL, name=cls.__qualname__)
        async with asyncio.TaskGroup() as tg:
            profile_load_task = tg.create_task(
                load_audience_profile(config.AUDIENCE_PROFILE_PATH)
            )
            if config.CACHE_DIRECTORY:
                cache_init_task = tg.create_task(
                    AsyncDiskCache.from_cache_dir_path(
                        config.CACHE_DIRECTORY / cls.__qualname__
                    )
                )
        profile = await profile_load_task
        cache = None
        if config.CACHE_DIRECTORY:
            cache = await cache_init_task
        return cls(agent=agent, profile=profile, cache=cache)

    @instrument()
    async def score_item(
        self,
        title: str,
        short_summary: str,
        industry: str,
        company: str,
    ) -> RelevanceScore:
        """Score a single news item for relevance to the audience.

        Args:
            title: Headline of the news item.
            short_summary: Brief summary of the news item.
            industry: Industry associated with the item.
            company: Company associated with the item.

        Returns:
            RelevanceScore with integer score 0-5 and reasoning.
        """
        if self.cache is not None:
            cache_key = (
                title,
                short_summary[:500:5],
                industry,
                company,
                self.profile[:300:3],
            )
            if await self.cache.contains(cache_key):
                _LOGGER.debug(f"cache hit for {cache_key=!r}")
                return await self.cache.get_item(cache_key)
            _LOGGER.debug(f"cache miss for {cache_key=!r}")

        prompt = f"""Score this news item's relevance for the target audience.

Audience Profile:
{self.profile}

News Item:
- Title: {title}
- Summary: {short_summary}
- Industry: {industry}
- Company: {company}

Consider:
1. Industry fit with audience profile and AI engineering
2. Novelty and actionability
3. Direct impact potential for audience's AI/LLMOps work

Return a score from 0 (irrelevant) to 5 (high priority) with brief reasoning, in JSON format."""
        _LOGGER.debug(f"relevance-scoring item with {title=!r}")
        result = await self.agent.run(prompt, output_type=RelevanceScore)
        _LOGGER.debug(f"{result=!r}")
        if self.cache is not None:
            await self.cache.set_item(cache_key, result.output)
            _LOGGER.debug(f"cached result for {cache_key=!r}")
        return result.output
