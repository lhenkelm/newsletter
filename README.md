# Newsletter

Automated AI newsletter generator that processes the ZenML LLMOps dataset from Hugging Face.

## Installation

clone the repository and install the required packages. If you don't have `uv` installed, you can do so via [their instructions:](https://docs.astral.sh/uv/getting-started/installation/)

```shell
uv sync
```

### Logfire Setup

This project uses Logfire for logging and observability. To set up Logfire with a free personal account, follow the instructions at [Logfire's Getting Started Guide](https://logfire.pydantic.dev/docs/#logfire).

## Configuration

Copy `.env.example` to `.env` and customize as needed:

Important: export any API keys (by default, the repo uses OpenAI, but you can change this in the configuration using env vars or a .env file) before running the pipeline.
```shell
cp .env.example .env
```
see `.env.example` for available configuration options.

## Usage

Run the newsletter generator:

```shell
uv run python -m newsletter.main
```

## Testing

```shell
uv run -- pytest
```

## Ideas for Improvement

 - Option to disable logfire/make it an optional feature and dependency
