"""Tests for the category selection agent."""

import os
from unittest.mock import AsyncMock, patch

import pytest

from newsletter.category import (
    ALLOWED_CATEGORIES,
    CategoryAgent,
    CategorySelection,
)


@pytest.fixture
def mock_api_key(monkeypatch):
    """Set a fake API key for testing."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key-for-testing")


@pytest.fixture
def category_agent(mock_api_key):
    """Create a category agent instance."""
    agent = CategoryAgent()
    agent.load_audience_profile("data/audience_profile.txt")
    return agent


class TestCategorySelection:
    """Tests for the CategorySelection model."""

    def test_valid_single_category(self):
        """Test that a single valid category is accepted."""
        selection = CategorySelection(
            categories=["AI Engineering"],
            reasoning="Relevant to AI engineering practices",
        )
        assert selection.categories == ["AI Engineering"]

    def test_valid_multiple_categories(self):
        """Test that multiple valid categories are accepted."""
        selection = CategorySelection(
            categories=["AI Engineering", "LLMOps Tools", "Production ML"],
            reasoning="Covers multiple relevant areas",
        )
        assert len(selection.categories) == 3

    def test_invalid_category_rejected(self):
        """Test that invalid categories raise validation error."""
        with pytest.raises(ValueError, match="Invalid categories"):
            CategorySelection(
                categories=["Invalid Category"],
                reasoning="This should fail",
            )

    def test_empty_categories_rejected(self):
        """Test that empty category list raises validation error."""
        with pytest.raises(ValueError):
            CategorySelection(
                categories=[],
                reasoning="This should fail",
            )

    def test_too_many_categories_rejected(self):
        """Test that more than 3 categories raises validation error."""
        with pytest.raises(ValueError):
            CategorySelection(
                categories=[
                    "AI Engineering",
                    "LLMOps Tools",
                    "Production ML",
                    "Industry News",
                ],
                reasoning="This should fail",
            )


class TestCategoryAgentProfileLoading:
    """Tests for profile loading functionality."""

    def test_profile_loading(self):
        """Test that audience profile can be loaded."""
        agent = CategoryAgent.__new__(CategoryAgent)
        agent.agent = None
        agent.profile = None

        agent.load_audience_profile("data/audience_profile.txt")

        assert agent.profile is not None
        assert len(agent.profile) > 0
        assert "Odido" in agent.profile or "AI" in agent.profile

    def test_profile_not_found_raises_error(self):
        """Test that missing profile file raises error."""
        agent = CategoryAgent.__new__(CategoryAgent)
        agent.agent = None
        agent.profile = None

        with pytest.raises(FileNotFoundError):
            agent.load_audience_profile("nonexistent/profile.txt")


class TestCategoryAgentSelection:
    """Tests for category selection functionality."""

    @pytest.mark.asyncio
    async def test_select_categories_without_profile(self, mock_api_key):
        """Test that the agent raises an error if profile is not loaded."""
        agent = CategoryAgent()

        with pytest.raises(ValueError, match="Audience profile not loaded"):
            await agent.select_categories(
                title="Test Title",
                short_summary="Test summary",
            )

    @pytest.mark.asyncio
    async def test_select_categories_mocked(self, category_agent):
        """Test category selection with mocked API response.

        Uses a sample news item to verify:
        1. Agent can process a news item
        2. Returns valid CategorySelection
        3. Categories are from allowed set
        """
        title = "Building Production-Ready RAG Systems with LangChain"
        short_summary = (
            "A comprehensive guide to implementing Retrieval-Augmented Generation "
            "systems using LangChain, covering prompt engineering, vector stores, "
            "and monitoring strategies for enterprise deployments."
        )
        application_tags = "chatbot, document_processing"
        tools_tags = "langchain, vector_store"
        techniques_tags = "rag, prompt_engineering"

        # Mock the agent.run method
        mock_result = AsyncMock()
        mock_result.output = CategorySelection(
            categories=["RAG & Retrieval", "LLMOps Tools"],
            reasoning="Covers RAG implementation with production tooling.",
        )

        with patch.object(category_agent.agent, "run", return_value=mock_result):
            result = await category_agent.select_categories(
                title=title,
                short_summary=short_summary,
                application_tags=application_tags,
                tools_tags=tools_tags,
                techniques_tags=techniques_tags,
            )

        assert isinstance(result, CategorySelection)
        assert len(result.categories) >= 1
        assert len(result.categories) <= 3
        assert all(cat in ALLOWED_CATEGORIES for cat in result.categories)
        assert len(result.reasoning) > 0

    @pytest.mark.asyncio
    async def test_select_categories_without_optional_tags(self, category_agent):
        """Test category selection works without optional tags."""
        title = "New Telecom AI Platform Announced"
        short_summary = (
            "Major telecom provider launches AI-powered network optimization."
        )

        mock_result = AsyncMock()
        mock_result.output = CategorySelection(
            categories=["Telecom Innovation", "AI Infrastructure"],
            reasoning="Directly relevant to telecom and AI infrastructure.",
        )

        with patch.object(category_agent.agent, "run", return_value=mock_result):
            result = await category_agent.select_categories(
                title=title,
                short_summary=short_summary,
            )

        assert isinstance(result, CategorySelection)
        assert all(cat in ALLOWED_CATEGORIES for cat in result.categories)


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="Requires OPENAI_API_KEY environment variable for live API calls",
)
class TestCategoryAgentLiveAPI:
    """Live API tests (skipped without API key)."""

    @pytest.mark.asyncio
    async def test_select_categories_live_api(self, category_agent):
        """Test category selection with live API call.

        Run with: OPENAI_API_KEY=your-key pytest tests/test_category.py::TestCategoryAgentLiveAPI -v
        """
        title = "Building Production-Ready RAG Systems with LangChain"
        short_summary = (
            "A comprehensive guide to implementing Retrieval-Augmented Generation "
            "systems using LangChain, covering prompt engineering, vector stores, "
            "and monitoring strategies for enterprise deployments."
        )
        application_tags = "chatbot, document_processing"
        tools_tags = "langchain, vector_store"
        techniques_tags = "rag, prompt_engineering"

        result = await category_agent.select_categories(
            title=title,
            short_summary=short_summary,
            application_tags=application_tags,
            tools_tags=tools_tags,
            techniques_tags=techniques_tags,
        )

        assert isinstance(result, CategorySelection)
        assert 1 <= len(result.categories) <= 3
        assert all(cat in ALLOWED_CATEGORIES for cat in result.categories)
        assert len(result.reasoning) > 0

    @pytest.mark.asyncio
    async def test_telecom_item_categorization(self, category_agent):
        """Test that telecom-related items get appropriate categories."""
        title = "5G Network Optimization Using Machine Learning"
        short_summary = (
            "Telecom operators are deploying ML models to optimize 5G network "
            "performance, reduce latency, and predict maintenance needs."
        )

        result = await category_agent.select_categories(
            title=title,
            short_summary=short_summary,
        )

        assert isinstance(result, CategorySelection)
        # Should likely include Telecom Innovation
        assert any(
            cat in ["Telecom Innovation", "Production ML", "AI Infrastructure"]
            for cat in result.categories
        )
