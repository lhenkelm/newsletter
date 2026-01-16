# Relevance Scoring Agent
1. Goal: rank single ZenML news item for newsletter relevance.
2. Framework: async pydantic-ai Agent('openai:gpt-4o-mini') invoked inside pipeline.
3. Inputs: headline, industry, company, use_case short summary from Polars row.
4. Audience profile: load text from data/audience_profile.txt before scoring.
5. Prompt: compare item traits vs profile themes (industry fit, novelty, impact).
6. Output: integer score 0 (irrelevant) to 5 (high priority); reject other formats.
7. Schema: wrap response in BaseModel with conint(ge=0, le=5) for enforcement.
8. Async flow: await agent.score_item(row, profile) via asyncio.gather for batches.
9. Usage: persist scores as Polars column audience_score for filtering and logging.
