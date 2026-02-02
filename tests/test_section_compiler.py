"""Tests for the section compiler agent."""

import os
from unittest.mock import AsyncMock, patch

import polars as pl
import pytest
import pytest_asyncio

from newsletter.section_compiler import (
    MAX_CATEGORIES,
    MAX_TOTAL_ITEMS,
    SectionCompilerAgent,
    SectionItem,
    SectionSelection,
    get_item_details,
    get_items_by_category,
    get_items_by_company,
    get_high_score_items,
    SectionCompilerDeps,
    ItemDetail,
)


@pytest.fixture
def mock_api_key(monkeypatch):
    """Set a fake API key for testing."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key-for-testing")


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame with scored and categorized items."""
    return pl.DataFrame(
        {
            "index": [0, 1, 2, 3, 4],
            "title": [
                "Building RAG Systems with LangChain",
                "OpenAI Releases GPT-5",
                "Telecom AI: 5G Network Optimization",
                "Fine-tuning LLMs for Enterprise",
                "Prompt Engineering Best Practices",
            ],
            "summary": [
                "A comprehensive guide to implementing RAG systems...",
                "OpenAI announces GPT-5 with improved reasoning...",
                "How AI is transforming 5G network management...",
                "Enterprise strategies for LLM fine-tuning...",
                "Best practices for effective prompt engineering...",
            ],
            "source_url": [
                "https://example.com/rag",
                "https://example.com/gpt5",
                "https://example.com/telecom",
                "https://example.com/finetuning",
                "https://example.com/prompts",
            ],
            "industry": ["tech", "tech", "telecom", "enterprise", "tech"],
            "company": ["LangChain", "OpenAI", "Ericsson", "Microsoft", "Anthropic"],
            "relevance_score": [5, 4, 5, 3, 4],
            "score_reasoning": [
                "Highly relevant to RAG implementation",
                "Major AI release with industry impact",
                "Direct telecom relevance",
                "Enterprise LLM operations",
                "Practical prompt engineering guidance",
            ],
            "interest_categories": [
                ["RAG & Retrieval", "LLMOps Tools"],
                ["Industry News", "AI Engineering"],
                ["Telecom Innovation", "AI Infrastructure"],
                ["Model Fine-tuning", "Production ML"],
                ["Prompt Engineering", "AI Engineering"],
            ],
            "category_reasoning": [
                "RAG focus with tooling aspects",
                "Major industry news in AI",
                "Telecom-specific AI application",
                "Fine-tuning for production use",
                "Prompt engineering techniques",
            ],
        }
    )


@pytest_asyncio.fixture
async def section_compiler_agent(mock_api_key):
    """Create a section compiler agent instance."""

    class MockConfig:
        COMPILER_MODEL = "openai:gpt-4o-mini"
        CATEGORY_MODEL = "openai:gpt-4o-mini"
        AUDIENCE_PROFILE_PATH = "./data/audience_profile.txt"
        CACHE_DIRECTORY = None

    agent = await SectionCompilerAgent.from_config(MockConfig())
    return agent


class TestSectionSelection:
    """Tests for the SectionSelection model."""

    def test_valid_section_selection(self):
        """Test that a valid section selection is accepted."""
        selection = SectionSelection(
            section_items={
                "AI Engineering": [
                    SectionItem(
                        index=0,
                        title="Article 1 Title",
                        summary="Summary of article 1...",
                        source_url="https://example.com/1",
                    ),
                    SectionItem(
                        index=1,
                        title="Article 2 Title",
                        summary="Summary of article 2...",
                        source_url="https://example.com/2",
                    ),
                ],
                "RAG & Retrieval": [
                    SectionItem(
                        index=2,
                        title="RAG Article Title",
                        summary="Summary of RAG article...",
                        source_url="https://example.com/rag",
                    ),
                ],
            },
            selected_categories=["AI Engineering", "RAG & Retrieval"],
            selection_reasoning="Selected based on relevance and diversity.",
        )
        assert len(selection.selected_categories) == 2
        assert len(selection.section_items) == 2

    def test_max_categories_exceeded(self):
        """Test that more than MAX_CATEGORIES raises error."""
        with pytest.raises(ValueError, match=f"Maximum {MAX_CATEGORIES} categories"):
            SectionSelection(
                section_items={
                    "AI Engineering": [
                        SectionItem(
                            index=0,
                            title="Title 1",
                            summary="Summary...",
                            source_url="https://example.com/1",
                        )
                    ],
                    "RAG & Retrieval": [
                        SectionItem(
                            index=1,
                            title="Title 2",
                            summary="Summary...",
                            source_url="https://example.com/2",
                        )
                    ],
                    "LLMOps Tools": [
                        SectionItem(
                            index=2,
                            title="Title 3",
                            summary="Summary...",
                            source_url="https://example.com/3",
                        )
                    ],
                    "Industry News": [
                        SectionItem(
                            index=3,
                            title="Title 4",
                            summary="Summary...",
                            source_url="https://example.com/4",
                        )
                    ],
                },
                selected_categories=[
                    "AI Engineering",
                    "RAG & Retrieval",
                    "LLMOps Tools",
                ],
                selection_reasoning="Too many categories.",
            )

    def test_max_items_exceeded(self):
        """Test that more than MAX_TOTAL_ITEMS raises error."""
        items = [
            SectionItem(
                index=i,
                title=f"Title {i}",
                summary=f"Summary {i}...",
                source_url=f"https://example.com/{i}",
            )
            for i in range(12)
        ]
        with pytest.raises(ValueError, match=f"Maximum {MAX_TOTAL_ITEMS} total items"):
            SectionSelection(
                section_items={"AI Engineering": items},
                selected_categories=["AI Engineering"],
                selection_reasoning="Too many items.",
            )

    def test_empty_category_rejected(self):
        """Test that empty category lists are rejected."""
        with pytest.raises(ValueError, match="must have at least 1 item"):
            SectionSelection(
                section_items={
                    "AI Engineering": [],
                },
                selected_categories=["AI Engineering"],
                selection_reasoning="Empty category.",
            )


class TestToolFunctions:
    """Tests for the agent tool functions."""

    @pytest.mark.asyncio
    async def test_get_item_details(self, sample_dataframe):
        """Test get_item_details returns correct item."""

        class MockContext:
            deps = SectionCompilerDeps(df=sample_dataframe)

        result = await get_item_details(MockContext(), index=0)
        assert isinstance(result, ItemDetail)
        assert result.index == 0
        assert result.title == "Building RAG Systems with LangChain"
        assert result.relevance_score == 5

    @pytest.mark.asyncio
    async def test_get_item_details_not_found(self, sample_dataframe):
        """Test get_item_details raises error for missing index."""

        class MockContext:
            deps = SectionCompilerDeps(df=sample_dataframe)

        with pytest.raises(ValueError, match="No item found"):
            await get_item_details(MockContext(), index=999)

    @pytest.mark.asyncio
    async def test_get_items_by_category(self, sample_dataframe):
        """Test get_items_by_category filters correctly."""

        class MockContext:
            deps = SectionCompilerDeps(df=sample_dataframe)

        result = await get_items_by_category(MockContext(), category="AI Engineering")
        assert len(result) == 2  # GPT-5 and Prompt Engineering articles

    @pytest.mark.asyncio
    async def test_get_items_by_company(self, sample_dataframe):
        """Test get_items_by_company filters correctly."""

        class MockContext:
            deps = SectionCompilerDeps(df=sample_dataframe)

        result = await get_items_by_company(MockContext(), company="OpenAI")
        assert len(result) == 1
        assert result[0]["title"] == "OpenAI Releases GPT-5"

    @pytest.mark.asyncio
    async def test_get_high_score_items(self, sample_dataframe):
        """Test get_high_score_items filters correctly."""

        class MockContext:
            deps = SectionCompilerDeps(df=sample_dataframe)

        result = await get_high_score_items(MockContext(), min_score=5)
        assert len(result) == 2  # RAG and Telecom articles


class TestSectionCompilerAgent:
    """Tests for the section compiler agent."""

    @pytest.mark.asyncio
    async def test_compile_sections_mocked(
        self, section_compiler_agent, sample_dataframe
    ):
        """Test section compilation with mocked API response."""
        mock_result = AsyncMock()
        mock_result.output = SectionSelection(
            section_items={
                "RAG & Retrieval": [
                    SectionItem(
                        index=0,
                        title="Building RAG Systems with LangChain",
                        summary="A comprehensive guide to implementing RAG systems...",
                        source_url="https://example.com/rag",
                    ),
                ],
                "Telecom Innovation": [
                    SectionItem(
                        index=2,
                        title="Telecom AI: 5G Network Optimization",
                        summary="How AI is transforming 5G network management...",
                        source_url="https://example.com/telecom",
                    ),
                ],
                "AI Engineering": [
                    SectionItem(
                        index=1,
                        title="OpenAI Releases GPT-5",
                        summary="OpenAI announces GPT-5 with improved reasoning...",
                        source_url="https://example.com/gpt5",
                    ),
                ],
            },
            selected_categories=[
                "RAG & Retrieval",
                "Telecom Innovation",
                "AI Engineering",
            ],
            selection_reasoning="Selected highest scoring items with category diversity.",
        )

        with patch.object(
            section_compiler_agent.agent, "run", return_value=mock_result
        ):
            result = await section_compiler_agent.compile_sections(sample_dataframe)

        assert isinstance(result, SectionSelection)
        assert len(result.selected_categories) == 3
        assert "RAG & Retrieval" in result.selected_categories
        assert len(result.section_items) == 3

    @pytest.mark.asyncio
    async def test_compile_sections_missing_columns(self, section_compiler_agent):
        """Test that missing required columns raises ValueError."""
        incomplete_df = pl.DataFrame(
            {
                "index": [0],
                "title": ["Test"],
            }
        )

        with pytest.raises(ValueError, match="missing required columns"):
            await section_compiler_agent.compile_sections(incomplete_df)


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="Live API test requires OPENAI_API_KEY",
)
class TestSectionCompilerLive:
    """Live API tests for section compiler (skipped in CI)."""

    @pytest.mark.asyncio
    async def test_compile_sections_live(self, sample_dataframe):
        """Test actual API call for section compilation."""
        from newsletter import config

        agent = await SectionCompilerAgent.from_config(config)
        result = await agent.compile_sections(sample_dataframe)

        assert isinstance(result, SectionSelection)
        assert 1 <= len(result.selected_categories) <= MAX_CATEGORIES
        assert (
            sum(len(items) for items in result.section_items.values())
            <= MAX_TOTAL_ITEMS
        )
