# Copilot Instructions for Newsletter Project

## Project Overview
Automated AI newsletter generator that processes the ZenML LLMOps dataset from Hugging Face and generates weekly newsletters in Markdown format.

## Architecture

### Data Flow
1. **Ingest** → Hugging Face parquet dataset (`zenml/llmops-database`)
2. **Process** → Filter by date, categorize by tags, detect trends
3. **Generate** → LLM summarization (pydantic-ai Agent) → Markdown newsletter output

### Project Structure
```
src/newsletter/       # Main package (installed via pyproject.toml)
main.py               # CLI entry point (module: `python -m newsletter.main`)
notebooks/            # EDA and exploration
  eda.ipynb           # Jupyter notebook with Polars analysis
  manual_look.py      # Marimo notebook for interactive work
```

## Tech Stack & Conventions

### Package Management (CRITICAL)
- **Use `uv` for local project tasks and dependency management** (preferred workflow for this repo).
  - `uv sync` to install project dependencies from the lockfile
  - `uv add <package>` to add new runtime dependencies
  - `uv run <command>` to run commands (tests, scripts, CLI)
- Recommended command to run the CLI: `uv run python -m newsletter.main` (this repo uses module-style entrypoints)
- Build backend: `hatchling` (see `pyproject.toml`).
- Python 3.12+ required

### LLM Integration
- **Use `pydantic-ai`** as the LLM integration layer. It accepts provider-prefixed model identifiers like `openai:gpt-4o-mini` or `ollama:<model-name>`.
- This repo ships defaults in `.env.example`:
  - `SCORING_MODEL=openai:gpt-4o-mini`
  - `CATEGORY_MODEL=openai:gpt-4o-mini`
- Ollama support: if you want to use local/open weights via Ollama, set `OLLAMA_BASE_URL` (and `OLLAMA_API_KEY` for cloud) in your `.env` and pass `ollama:<model>` to `Agent()`.
- Configuration via `.env` file (see `.env.example`).
- Example:
  ```python
  from pydantic_ai import Agent
  agent = Agent('openai:gpt-4o-mini')
  ```

### Caching and Development
- The scoring agent supports an optional disk cache (set `CACHE_DIRECTORY` in `.env`), implemented by `newsletter.async_disk_cache.AsyncDiskCache`. Use it to speed up iterative development and avoid repeat API calls.

### Data Processing
- **Use Polars** for all data operations (async scans and `collect_async` is used in this codebase).
- Example ingest uses `pl.scan_parquet(...)` combined with `collect_async()` to read parquet from Hugging Face.

### Dataset Schema
Key columns for newsletter generation:
- `created_at` - Timestamp for weekly filtering
- `title`, `short_summary`, `summary` - Content for newsletter items
- `source_url` - Links to include
- `industry`, `company`, `year` - Categorical metadata
- `application_tags`, `tools_tags`, `techniques_tags`, `extra_tags` - Comma-separated tags (use `.str.split(",").explode().str.strip_chars()` to process)

### Configuration
- Copy `.env.example` → `.env` and set secrets / overrides.
- Important env vars: `OPENAI_API_KEY`, `OLLAMA_BASE_URL` / `OLLAMA_API_KEY`, `SCORING_MODEL`, `CATEGORY_MODEL`, `CACHE_DIRECTORY`, `CUTOFF_DAYS`, `MIN_ITEMS`, `MAX_ITEMS`.

## Implementation Guidelines

### Newsletter Requirements
- Output format: Markdown
- Sections: Introduction, categorized items (Research Highlights, Industry News, Use Cases), closing
- Handle deduplication and fresh-item filtering

### Agents & Async Flow
- Agents (scoring, category, writer) are async and created via `Agent(model_spec)` from `pydantic-ai`.
- Scoring output uses a typed Pydantic model (`RelevanceScore`) with `score` (0-5) and `reasoning` string.
- Category selection and newsletter writing follow similar typed-output patterns.

### Testing Guidance
- Tests use `pytest` + `pytest-asyncio` for async fixtures and rely on mocking the agent's `run` method to avoid live API calls in CI.
- Live API tests are skipped unless `OPENAI_API_KEY` is set (see `tests/test_scoring.py`).

### Code Quality
- Linting: `uv run ruff check .` and `uv run ruff format .` (if `ruff` is available in environment).
- Type checking: `uv run mypy src/` (project config in `pyproject.toml`).
- Tests: `uv run -- pytest`.
