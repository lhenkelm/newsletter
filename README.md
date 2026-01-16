# Newsletter

Automated AI newsletter generator that processes the ZenML LLMOps dataset from Hugging Face.

## Installation

clone the repository and install the required packages. If you don't have `uv` installed, you can do so via [their instructions:](https://docs.astral.sh/uv/getting-started/installation/)

```shell
uv sync
```

## Configuration

Copy `.env.example` to `.env` and customize as needed:

Important: export any API keys (by default, the repo uses OpenAI, but you can change this in the configuration using env vars or a .env file) before running the pipeline.
```shell
cp .env.example .env
```

Available settings:
- `CUTOFF_DAYS` - Number of days to look back for recent items (default: 7)
- `MIN_ITEMS` - Minimum expected items, warns if fewer (default: 3)
- `MAX_ITEMS` - Maximum expected items, warns if more (default: 30)
- `SOURCE_URI` - Dataset URI (default: Hugging Face ZenML LLMOps dataset)

## Usage

Run the newsletter generator:

```shell
uv run python -m newsletter.main
```

## Testing

```shell
uv run -- pytest
```

