# Category Selection Agent
1. Goal: assign relevant interest categories to a news item based on headline and audience profile.
2. Framework: async pydantic-ai Agent('openai:gpt-4o-mini') invoked inside pipeline.
3. Inputs: headline (title), short_summary, existing tags (application_tags, tools_tags, techniques_tags) from Polars row.
4. Audience profile: load text from data/audience_profile.txt before categorization.
5. Prompt: analyze item content and profile themes to select 1-3 category labels that best capture audience interest angle.
6. Output: list of category strings from predefined set (e.g., "AI Engineering" , "LLMOps Tools", "Telecom Innovation", "Production ML", "Industry News").
7. Schema: wrap response in BaseModel with list[str] field validated against allowed categories.
8. Async flow: await agent.select_categories(row, profile) via asyncio.gather for batches.
9. Usage: persist categories as Polars column interest_categories (list type) for newsletter section grouping.
