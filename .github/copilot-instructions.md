# Copilot Instructions for Newsletter Project

## Project Overview
Automated AI newsletter generator that processes the ZenML LLMOps dataset from Hugging Face and generates weekly newsletters in Markdown format.

## Architecture

### Data Flow
1. **Ingest** → Hugging Face parquet dataset (`zenml/llmops-database`)
2. **Process** → Filter by date, categorize by tags, detect trends
3. **Generate** → LLM summarization (OpenAI via pydantic-ai) → Markdown newsletter output

### Project Structure
```
src/newsletter/       # Main package (installed via pyproject.toml)
main.py               # CLI entry point
notebooks/            # EDA and exploration
  eda.ipynb           # Jupyter notebook with Polars analysis
  manual_look.py      # Marimo notebook for interactive work
```

## Tech Stack & Conventions

### Package Management (CRITICAL)
- **Always use `uv`** - never use pip, poetry, or conda directly
- `uv add <package>` - install new dependencies
- `uv run <command>` - run any Python command or script
- `uv run python main.py` - run the CLI
- `uv run marimo edit notebooks/manual_look.py` - run Marimo
- Python 3.10+ required

### LLM Integration
- **Use pydantic-ai** with OpenAI backend
- Configuration via `.env` file (see `.env.example`)
- Example:
  ```python
  from pydantic_ai import Agent
  agent = Agent('openai:gpt-4o')
  ```

### Data Processing
- **Use Polars** (not pandas) for all data operations
- Load dataset directly from HuggingFace:
  ```python
  import polars as pl
  df = pl.read_parquet("hf://datasets/zenml/llmops-database/data/train-00000-of-00001.parquet")
  ```

### Dataset Schema
Key columns for newsletter generation:
- `created_at` - Timestamp for weekly filtering
- `title`, `summary` - Content for newsletter items
- `source_url` - Links to include
- `industry`, `company`, `year` - Categorical metadata
- `application_tags`, `tools_tags`, `techniques_tags`, `extra_tags` - Comma-separated, use `.str.split(",").explode().str.strip_chars()` to process

### Configuration
- Use `.env` for all config (API keys, settings)
- Provide `.env.example` with required variables

## Implementation Guidelines

### Newsletter Requirements
- Output format: Markdown
- Sections: Introduction, categorized items (Research Highlights, Industry News, Use Cases), closing
- Handle deduplication and fresh-item filtering

### Code Quality
- Modular code in `src/newsletter/`
- CLI interface in `main.py`
- Sample output in `newsletter.md`
- **Linting**: `uv run ruff check .` and `uv run ruff format .`
- **Type checking**: `uv run mypy src/`

### Automation (Bonus)
- GitHub Actions for weekly schedule
- Support personalization (technical vs non-technical audience)
