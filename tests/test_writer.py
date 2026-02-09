"""Tests for the newsletter writer agent."""

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from newsletter.section_compiler import SectionItem
from newsletter.writer import (
    LinkValidationError,
    NewsletterOutput,
    NewsletterWriterAgent,
    extract_urls_from_markdown,
    save_newsletter,
    validate_newsletter_links,
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
        MAX_LINK_VALIDATION_RETRIES = 3

    agent = await NewsletterWriterAgent.from_config(MockConfig())
    return agent


class TestNewsletterWriterAgentInit:
    """Tests for NewsletterWriterAgent initialization."""

    def test_init_requires_agent_instance(self):
        """Test that init requires a proper Agent instance."""
        with pytest.raises(TypeError, match="expected 'agent' to be an instance"):
            NewsletterWriterAgent("not-an-agent", "profile", 7, 3)

    def test_init_requires_string_profile(self, mock_api_key):
        """Test that init requires a string profile."""
        from pydantic_ai import Agent

        agent = Agent("openai:gpt-4o-mini")
        with pytest.raises(TypeError, match="expected 'profile' to be of type 'str'"):
            NewsletterWriterAgent(agent, 123, 7, 3)

    def test_init_requires_non_empty_profile(self, mock_api_key):
        """Test that init requires a non-empty profile."""
        from pydantic_ai import Agent

        agent = Agent("openai:gpt-4o-mini")
        with pytest.raises(ValueError, match="Audience profile cannot be empty"):
            NewsletterWriterAgent(agent, "", 7, 3)


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
            MAX_LINK_VALIDATION_RETRIES = 3

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
            MAX_LINK_VALIDATION_RETRIES = 3

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
        # Markdown must include all expected links to pass validation
        valid_markdown = """# Test Newsletter

## AI Engineering
- [Building RAG Systems with LangChain](https://example.com/rag) - Great guide
- [OpenAI Releases GPT-5](https://example.com/gpt5) - Big announcement

## Telecom Innovation
- [5G Network AI Optimization](https://example.com/telecom) - Telecom AI
"""
        mock_result = MagicMock()
        mock_result.output = NewsletterOutput(
            newsletter_markdown=valid_markdown,
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
            MAX_LINK_VALIDATION_RETRIES = 3

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


class TestExtractUrlsFromMarkdown:
    """Tests for the extract_urls_from_markdown function."""

    def test_extract_single_url(self):
        """Test extracting a single URL from markdown."""
        markdown = "Check out [this link](https://example.com/page)."
        urls = extract_urls_from_markdown(markdown)
        assert urls == {"https://example.com/page"}

    def test_extract_multiple_urls(self):
        """Test extracting multiple URLs from markdown."""
        markdown = """
        Here are some links:
        - [Link 1](https://example.com/1)
        - [Link 2](https://example.com/2)
        - [Another link](https://other.org/page)
        """
        urls = extract_urls_from_markdown(markdown)
        assert urls == {
            "https://example.com/1",
            "https://example.com/2",
            "https://other.org/page",
        }

    def test_extract_no_urls(self):
        """Test with markdown containing no links."""
        markdown = "Just plain text without any links."
        urls = extract_urls_from_markdown(markdown)
        assert urls == set()

    def test_extract_duplicate_urls_returns_set(self):
        """Test that duplicate URLs are deduplicated."""
        markdown = "[Link 1](https://example.com) and [Link 2](https://example.com)"
        urls = extract_urls_from_markdown(markdown)
        assert urls == {"https://example.com"}

    def test_handles_urls_with_whitespace(self):
        """Test that URLs with surrounding whitespace are trimmed."""
        markdown = "[Link]( https://example.com/page )"
        urls = extract_urls_from_markdown(markdown)
        assert urls == {"https://example.com/page"}


class TestValidateNewsletterLinks:
    """Tests for the validate_newsletter_links function."""

    def test_valid_links_all_present(self):
        """Test when all expected links are present and no extras."""
        markdown = """
        # Newsletter
        - [Item 1](https://example.com/1)
        - [Item 2](https://example.com/2)
        """
        expected = {"https://example.com/1", "https://example.com/2"}

        missing, extra = validate_newsletter_links(markdown, expected)
        assert missing == set()
        assert extra == set()

    def test_missing_links_detected(self):
        """Test detection of missing links."""
        markdown = "[Only one](https://example.com/1)"
        expected = {"https://example.com/1", "https://example.com/2"}

        missing, extra = validate_newsletter_links(markdown, expected)
        assert missing == {"https://example.com/2"}
        assert extra == set()

    def test_extra_links_detected(self):
        """Test detection of hallucinated/extra links."""
        markdown = """
        - [Item 1](https://example.com/1)
        - [Hallucinated](https://fake.com/made-up)
        """
        expected = {"https://example.com/1"}

        missing, extra = validate_newsletter_links(markdown, expected)
        assert missing == set()
        assert extra == {"https://fake.com/made-up"}

    def test_both_missing_and_extra(self):
        """Test when both missing and extra links exist."""
        markdown = "[Wrong link](https://wrong.com)"
        expected = {"https://expected.com"}

        missing, extra = validate_newsletter_links(markdown, expected)
        assert missing == {"https://expected.com"}
        assert extra == {"https://wrong.com"}


class TestLinkValidationError:
    """Tests for the LinkValidationError exception."""

    def test_error_with_missing_links(self):
        """Test error construction with missing links."""
        error = LinkValidationError(
            "Validation failed",
            missing_links={"https://missing.com"},
        )
        assert error.missing_links == {"https://missing.com"}
        assert error.extra_links == set()

    def test_error_with_extra_links(self):
        """Test error construction with extra links."""
        error = LinkValidationError(
            "Validation failed",
            extra_links={"https://extra.com"},
        )
        assert error.missing_links == set()
        assert error.extra_links == {"https://extra.com"}

    def test_error_with_both(self):
        """Test error construction with both missing and extra links."""
        error = LinkValidationError(
            "Validation failed",
            missing_links={"https://missing.com"},
            extra_links={"https://extra.com"},
        )
        assert error.missing_links == {"https://missing.com"}
        assert error.extra_links == {"https://extra.com"}


class TestWriteNewsletterLinkValidation:
    """Tests for link validation in write_newsletter method."""

    @pytest.mark.asyncio
    async def test_valid_links_pass_validation(
        self, writer_agent, sample_section_items
    ):
        """Test that newsletter with valid links passes validation."""

        valid_markdown = """# Test Newsletter

## AI Engineering
- [Building RAG Systems](https://example.com/rag) - Great article
- [OpenAI Releases GPT-5](https://example.com/gpt5) - Big news

## Telecom Innovation
- [5G Network AI](https://example.com/telecom) - Innovation story
"""
        mock_result = MagicMock()
        mock_result.output = NewsletterOutput(
            newsletter_markdown=valid_markdown,
            title="Test Newsletter",
        )

        with patch.object(
            writer_agent.agent, "run", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = mock_result
            result = await writer_agent.write_newsletter(sample_section_items)

            assert result.title == "Test Newsletter"
            # Should only call once since validation passed
            assert mock_run.call_count == 1

    @pytest.mark.asyncio
    async def test_missing_link_triggers_retry(
        self, writer_agent, sample_section_items
    ):
        """Test that missing links trigger a retry."""
        # First response missing one link
        invalid_markdown = """# Newsletter
- [RAG](https://example.com/rag)
- [GPT-5](https://example.com/gpt5)
"""
        # Second response with all links
        valid_markdown = """# Newsletter
- [RAG](https://example.com/rag)
- [GPT-5](https://example.com/gpt5)
- [Telecom](https://example.com/telecom)
"""
        mock_result_invalid = MagicMock()
        mock_result_invalid.output = NewsletterOutput(
            newsletter_markdown=invalid_markdown,
            title="Test",
        )

        mock_result_valid = MagicMock()
        mock_result_valid.output = NewsletterOutput(
            newsletter_markdown=valid_markdown,
            title="Test",
        )

        with patch.object(
            writer_agent.agent, "run", new_callable=AsyncMock
        ) as mock_run:
            mock_run.side_effect = [mock_result_invalid, mock_result_valid]
            result = await writer_agent.write_newsletter(sample_section_items)

            assert mock_run.call_count == 2
            assert result.newsletter_markdown == valid_markdown

    @pytest.mark.asyncio
    async def test_max_retries_raises_error(self, writer_agent, sample_section_items):
        """Test that exceeding max retries raises LinkValidationError."""
        # Always return invalid markdown
        invalid_markdown = """# Newsletter
- [Only RAG](https://example.com/rag)
"""
        mock_result = MagicMock()
        mock_result.output = NewsletterOutput(
            newsletter_markdown=invalid_markdown,
            title="Test",
        )

        with patch.object(
            writer_agent.agent, "run", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = mock_result

            with pytest.raises(LinkValidationError) as exc_info:
                await writer_agent.write_newsletter(sample_section_items)

            # Should have tried max times (default is 3)
            assert mock_run.call_count == 3
            # Error should contain info about missing links
            assert len(exc_info.value.missing_links) > 0

    @pytest.mark.asyncio
    async def test_retry_prompt_includes_feedback(
        self, writer_agent, sample_section_items
    ):
        """Test that retry prompts include specific feedback about failures."""
        invalid_markdown = """# Newsletter
- [RAG](https://example.com/rag)
"""
        valid_markdown = """# Newsletter
- [RAG](https://example.com/rag)
- [GPT-5](https://example.com/gpt5)
- [Telecom](https://example.com/telecom)
"""
        mock_result_invalid = MagicMock()
        mock_result_invalid.output = NewsletterOutput(
            newsletter_markdown=invalid_markdown,
            title="Test",
        )
        mock_result_valid = MagicMock()
        mock_result_valid.output = NewsletterOutput(
            newsletter_markdown=valid_markdown,
            title="Test",
        )

        with patch.object(
            writer_agent.agent, "run", new_callable=AsyncMock
        ) as mock_run:
            mock_run.side_effect = [mock_result_invalid, mock_result_valid]
            await writer_agent.write_newsletter(sample_section_items)

            # Check second call includes error feedback in prompt
            second_call_args = mock_run.call_args_list[1]
            prompt = second_call_args[0][0]  # First positional arg is the prompt
            assert "Previous Attempt Failed" in prompt
            assert "MUST" in prompt


class TestExtractExpectedUrls:
    """Tests for the _extract_expected_urls static method."""

    def test_extracts_all_urls(self, sample_section_items):
        """Test that all URLs are extracted from section items."""
        urls = NewsletterWriterAgent._extract_expected_urls(sample_section_items)
        assert urls == {
            "https://example.com/rag",
            "https://example.com/gpt5",
            "https://example.com/telecom",
        }

    def test_empty_section_items(self):
        """Test with empty section items."""
        urls = NewsletterWriterAgent._extract_expected_urls({})
        assert urls == set()
