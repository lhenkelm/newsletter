"""Relevance scoring agent for news items."""

from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_ai import Agent


class RelevanceScore(BaseModel):
    """Output schema for relevance scoring."""

    score: int = Field(
        ...,
        ge=0,
        le=5,
        description="Relevance score from 0 (irrelevant) to 5 (high priority)",
    )
    reasoning: str = Field(..., description="Brief explanation of the score")


class ScoringAgent:
    """Async agent for scoring news items against audience profile."""

    def __init__(self, model: str = "openai:gpt-4o-mini"):
        """Initialize the scoring agent.

        Args:
            model: Model identifier for pydantic-ai Agent.
        """
        self.agent = Agent(model)
        self.profile: str | None = None

    def load_audience_profile(
        self, profile_path: str | Path = "data/audience_profile.txt"
    ) -> None:
        """Load audience profile from file.

        Args:
            profile_path: Path to the audience profile text file.
        """
        path = Path(profile_path)
        self.profile = path.read_text().strip()

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

        Raises:
            ValueError: If audience profile has not been loaded.
        """
        if not self.profile:
            raise ValueError(
                "Audience profile not loaded. Call load_audience_profile() first."
            )

        prompt = f"""Score this news item's relevance for the target audience.

Audience Profile:
{self.profile}

News Item:
- Title: {title}
- Summary: {short_summary}
- Industry: {industry}
- Company: {company}

Consider:
1. Industry fit with telecom/network/AI engineering
2. Novelty and actionability
3. Direct impact potential for Odido's AI/LLMOps work

Return a score from 0 (irrelevant) to 5 (high priority) with brief reasoning."""

        result = await self.agent.run(prompt, output_type=RelevanceScore)
        return result.data
