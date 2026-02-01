"""Relevance scoring agent for news items."""

from pathlib import Path
from typing import Any, Self, Type

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from logging import getLogger

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

    Recommended usage:

        import newsletter.config as config
        agent = ScoringAgent.from_config(config)
        result = await agent.score_item(
            title="...",
            short_summary="...",
            industry="...",
            company="...",
        )
    """

    def __init__(self, agent: Agent, profile: str):
        """Initialize the scoring agent.

        Args:
            model: Model identifier for pydantic-ai Agent.
        """
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
        _LOGGER.debug(f"initialized {self!r}")

    @staticmethod
    def _load_audience_profile(
        profile_path: str | Path = "data/audience_profile.txt",
    ) -> str:
        """Load audience profile from file.

        Args:
            profile_path: Path to the audience profile text file.
        """
        _LOGGER.debug(f"loading profile from {profile_path=!r}")
        path = Path(profile_path)
        profile = path.read_text().strip()
        _LOGGER.debug(f"loaded {profile=!r}")
        return profile

    @classmethod
    def from_config(cls: Type[Self], config: Any) -> Self:
        """Create a ScoringAgent instance from configuration.

        Args:
            config: Configuration module with attributes
                SCORING_MODEL and
                AUDIENCE_PROFILE_PATH.
        """
        profile = cls._load_audience_profile(config.AUDIENCE_PROFILE_PATH)
        agent = Agent(config.SCORING_MODEL)
        return cls(agent=agent, profile=profile)

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
        return result.output
