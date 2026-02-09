"""Tests for the newsletter writer agent."""

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from newsletter.section_compiler import SectionItem
from newsletter.writer import (
    NewsletterOutput,
    NewsletterWriterAgent,
    save_newsletter,
)


@pytest.fixture
def mock_api_key(monkeypatch):
    """Set a fake API key for testing."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key-for-testing")


@pytest.fixture
def sample_section_items():
    """Create sample section items for testing."""
    return {
        "AI Engineering": [
            SectionItem(
                index=0,
                title="Building RAG Systems with LangChain",
                full_summary="A comprehensive guide to implementing RAG systems using LangChain and vector databases for enterprise applications.",
                source_url="https://example.com/rag",
            ),
            SectionItem(
                index=1,
                title="OpenAI Releases GPT-5",
                full_summary="OpenAI announces GPT-5 with improved reasoning capabilities and reduced hallucination rates.",
                source_url="https://example.com/gpt5",
            ),
        ],
        "Telecom Innovation": [
            SectionItem(
                index=2,
                title="5G Network AI Optimization",
                full_summary="How AI is transforming 5G network management and optimization in telecom operations.",
                source_url="https://example.com/telecom",
            ),
        ],
    }


@pytest_asyncio.fixture
async def writer_agent(mock_api_key):
    """Create a newsletter writer agent instance."""

    class MockConfig:
        WRITER_MODEL = "openai:gpt-4o-mini"
        CATEGORY_MODEL = "openai:gpt-4o-mini"
        AUDIENCE_PROFILE_PATH = "./data/audience_profile.txt"
        CACHE_DIRECTORY = None
        CUTOFF_DAYS = 7

    agent = await NewsletterWriterAgent.from_config(MockConfig())
    return agent


class TestNewsletterWriterAgentInit:
    """Tests for NewsletterWriterAgent initialization."""

    def test_init_requires_agent_instance(self):
        """Test that init requires a proper Agent instance."""
        with pytest.raises(TypeError, match="expected 'agent' to be an instance"):
            NewsletterWriterAgent("not-an-agent", "profile", 7)

    def test_init_requires_string_profile(self, mock_api_key):
        """Test that init requires a string profile."""
        from pydantic_ai import Agent

        agent = Agent("openai:gpt-4o-mini")
        with pytest.raises(TypeError, match="expected 'profile' to be of type 'str'"):
            NewsletterWriterAgent(agent, 123, 7)

    def test_init_requires_non_empty_profile(self, mock_api_key):
        """Test that init requires a non-empty profile."""
        from pydantic_ai import Agent

        agent = Agent("openai:gpt-4o-mini")
        with pytest.raises(ValueError, match="Audience profile cannot be empty"):
            NewsletterWriterAgent(agent, "", 7)


class TestNewsletterWriterAgentFromConfig:
    """Tests for creating agent from config."""

    @pytest.mark.asyncio
    async def test_from_config_creates_agent(self, mock_api_key):
        """Test that from_config creates a valid agent."""

        class MockConfig:
            WRITER_MODEL = "openai:gpt-4o-mini"
            CATEGORY_MODEL = "openai:gpt-4o-mini"
            AUDIENCE_PROFILE_PATH = "./data/audience_profile.txt"
            CACHE_DIRECTORY = None
            CUTOFF_DAYS = 7

        agent = await NewsletterWriterAgent.from_config(MockConfig())
        assert agent.profile != ""
        assert agent.cache is None

    @pytest.mark.asyncio
    async def test_from_config_with_cache_directory(self, mock_api_key, tmp_path):
        """Test that from_config initializes cache when configured."""

        class MockConfig:
            WRITER_MODEL = "openai:gpt-4o-mini"
            CATEGORY_MODEL = "openai:gpt-4o-mini"
            AUDIENCE_PROFILE_PATH = "./data/audience_profile.txt"
            CACHE_DIRECTORY = tmp_path
            CUTOFF_DAYS = 7

        agent = await NewsletterWriterAgent.from_config(MockConfig())
        assert agent.cache is not None


class TestWriteNewsletter:
    """Tests for the write_newsletter method."""

    @pytest.mark.asyncio
    async def test_write_newsletter_rejects_empty_items(self, writer_agent):
        """Test that write_newsletter rejects empty section_items."""
        with pytest.raises(ValueError, match="section_items cannot be empty"):
            await writer_agent.write_newsletter({})

    @pytest.mark.asyncio
    async def test_write_newsletter_with_mocked_agent(
        self, writer_agent, sample_section_items
    ):
        """Test write_newsletter with mocked LLM response."""
        mock_result = MagicMock()
        mock_result.output = NewsletterOutput(
            newsletter_markdown="# Test Newsletter\n\nContent...",
            title="Test Newsletter",
        )

        with patch.object(
            writer_agent.agent, "run", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = mock_result
            result = await writer_agent.write_newsletter(sample_section_items)

            assert result.title == "Test Newsletter"


class TestSaveNewsletter:
    """Tests for the save_newsletter function."""

    @pytest.mark.asyncio
    async def test_save_newsletter_default_path(self, tmp_path, monkeypatch):
        """Test saving newsletter to default path."""
        monkeypatch.chdir(tmp_path)

        output = NewsletterOutput(
            newsletter_markdown="# Test Newsletter\n\nContent here...",
            title="Test Newsletter",
        )

        path = await save_newsletter(output)
        assert path == Path("newsletter.md")
        assert path.exists()
        assert path.read_text() == output.newsletter_markdown

    @pytest.mark.asyncio
    async def test_save_newsletter_custom_path(self, tmp_path):
        """Test saving newsletter to custom path."""
        custom_path = tmp_path / "custom" / "output.md"

        output = NewsletterOutput(
            newsletter_markdown="# Custom Newsletter\n\nContent...",
            title="Custom Newsletter",
        )

        path = await save_newsletter(output, custom_path)
        assert path == custom_path
        assert path.read_text() == output.newsletter_markdown


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY")
    or os.getenv("OPENAI_API_KEY", "").startswith("sk-test"),
    reason="Live API test requires real OPENAI_API_KEY",
)
class TestLiveAPI:
    """Live API tests - only run when real API key is available."""

    @pytest.mark.asyncio
    async def test_write_newsletter_live(self, sample_section_items):
        """Test write_newsletter with live OpenAI API."""

        class MockConfig:
            WRITER_MODEL = "openai:gpt-4o-mini"
            CATEGORY_MODEL = "openai:gpt-4o-mini"
            AUDIENCE_PROFILE_PATH = "./data/audience_profile.txt"
            CACHE_DIRECTORY = None
            CUTOFF_DAYS = 7

        agent = await NewsletterWriterAgent.from_config(MockConfig())
        result = await agent.write_newsletter(sample_section_items)

        assert len(result.newsletter_markdown) > 100
        assert result.title != ""
        # Check that markdown contains expected elements
        assert "#" in result.newsletter_markdown
        assert (
            "AI Engineering" in result.newsletter_markdown
            or "Telecom" in result.newsletter_markdown
        )
