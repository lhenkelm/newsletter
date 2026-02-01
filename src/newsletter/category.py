"""Category selection agent for news items."""

from logging import getLogger
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator
from pydantic_ai import Agent

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
    """Async agent for assigning interest categories to news items."""

    def __init__(self, model: str = "openai:gpt-4o-mini"):
        """Initialize the category selection agent.

        Args:
            model: Model identifier for pydantic-ai Agent.
        """
        self.agent = Agent(model)
        self.profile: str | None = None
        _LOGGER.debug(f"initialised {self=!r}")

    def load_audience_profile(
        self, profile_path: str | Path = "data/audience_profile.txt"
    ) -> None:
        """Load audience profile from file.

        Args:
            profile_path: Path to the audience profile text file.
        """
        _LOGGER.debug(f"loading profile from {profile_path=!r}")
        path = Path(profile_path)
        self.profile = path.read_text().strip()
        _LOGGER.debug(f"loaded {self.profile=!r}")

    async def select_categories(
        self,
        title: str,
        short_summary: str,
        application_tags: str | None = None,
        tools_tags: str | None = None,
        techniques_tags: str | None = None,
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

        Raises:
            ValueError: If audience profile has not been loaded.
        """
        if not self.profile:
            raise ValueError(
                "Audience profile not loaded. Call load_audience_profile() first."
            )

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
        return result.output
