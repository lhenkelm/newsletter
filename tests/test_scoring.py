"""Tests for the relevance scoring agent."""

import os
from unittest.mock import AsyncMock, patch

import pytest

from newsletter.scoring import RelevanceScore, ScoringAgent


@pytest.fixture
def mock_api_key(monkeypatch):
    """Set a fake API key for testing."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key-for-testing")


@pytest.fixture
def scoring_agent(mock_api_key):
    """Create a scoring agent instance."""
    agent = ScoringAgent()
    agent.load_audience_profile("data/audience_profile.txt")
    return agent


def test_scoring_agent_profile_loading():
    """Test that audience profile can be loaded."""
    agent = ScoringAgent.__new__(ScoringAgent)  # Create without calling __init__
    agent.agent = None  # Set dummy agent
    agent.profile = None

    agent.load_audience_profile("data/audience_profile.txt")

    assert agent.profile is not None
    assert len(agent.profile) > 0
    assert "Odido" in agent.profile or "AI" in agent.profile


@pytest.mark.asyncio
async def test_scoring_agent_without_profile(mock_api_key):
    """Test that the agent raises an error if profile is not loaded."""
    agent = ScoringAgent()

    with pytest.raises(ValueError, match="Audience profile not loaded"):
        await agent.score_item(
            title="Test Title",
            short_summary="Test summary",
            industry="Tech",
            company="Test Co",
        )


@pytest.mark.asyncio
async def test_scoring_agent_setup_and_run_mocked(scoring_agent):
    """Test that the scoring agent can be set up and run on a fixed example (mocked).

    This test uses a sample news item from the EDA notebook to verify:
    1. Agent can be instantiated
    2. Audience profile can be loaded
    3. Agent produces a valid score (0-5 integer)
    4. Agent provides reasoning

    This test mocks the actual API call to avoid requiring API keys in CI/CD.
    """
    # Fixed example based on the ZenML LLMOps dataset structure
    title = "Building Production-Ready RAG Systems with LangChain"
    short_summary = (
        "A comprehensive guide to implementing Retrieval-Augmented Generation "
        "systems using LangChain, covering prompt engineering, vector stores, "
        "and monitoring strategies for enterprise deployments."
    )
    industry = "Tech"
    company = "Various"

    # Mock the agent.run method to return a fake score
    mock_result = AsyncMock()
    mock_result.output = RelevanceScore(
        score=4,
        reasoning="Highly relevant for LLMOps: covers RAG implementation with practical tooling (LangChain) and monitoring strategies applicable to enterprise AI deployments.",
    )

    with patch.object(scoring_agent.agent, "run", return_value=mock_result):
        result = await scoring_agent.score_item(
            title=title,
            short_summary=short_summary,
            industry=industry,
            company=company,
        )

    # Verify the result structure
    assert isinstance(result, RelevanceScore)
    assert isinstance(result.score, int)
    assert 0 <= result.score <= 5
    assert isinstance(result.reasoning, str)
    assert len(result.reasoning) > 0


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="Requires OPENAI_API_KEY environment variable for live API calls",
)
@pytest.mark.asyncio
async def test_scoring_agent_live_api(scoring_agent):
    """Test the scoring agent with a live API call (requires API key).

    This test is skipped unless OPENAI_API_KEY is set.
    Run with: OPENAI_API_KEY=your-key pytest tests/test_scoring.py::test_scoring_agent_live_api -v
    """
    # Fixed example based on the ZenML LLMOps dataset structure
    title = "Building Production-Ready RAG Systems with LangChain"
    short_summary = (
        "A comprehensive guide to implementing Retrieval-Augmented Generation "
        "systems using LangChain, covering prompt engineering, vector stores, "
        "and monitoring strategies for enterprise deployments."
    )
    industry = "Tech"
    company = "Various"

    # Run the scoring agent with real API
    result = await scoring_agent.score_item(
        title=title,
        short_summary=short_summary,
        industry=industry,
        company=company,
    )

    # Verify the result structure
    assert isinstance(result, RelevanceScore)
    assert isinstance(result.score, int)
    assert 0 <= result.score <= 5
    assert isinstance(result.reasoning, str)
    assert len(result.reasoning) > 0
