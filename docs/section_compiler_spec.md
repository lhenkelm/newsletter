# Section Compiler Agent
1. Goal: select final newsletter items from scored and categorized dataset, choosing up to 10 items across up to 3 categories.
2. Framework: async pydantic-ai Agent('openai:gpt-4o-mini') invoked inside pipeline.
3. Inputs: augmented DataFrame with columns: title, summary, source_url, industry, company, relevance_score, score_reasoning, interest_categories, category_reasoning.
     - the agent recieves a subset of the dataframe to start with: all items, sorted by relevance_score descending,
        projected to only index, title, relevance_score, and interest_categories columns.
     - the agent also has access to the audience profile text in its instructions. 
     - The agent can refer to the full dataframe as needed for details on each item, using tools
     - The tools are implemented as functions that the agent can call that return details on items by index.
     - The tools follow a SELECT ... FROM df WHERE ... -style simplified SQL syntax.
4. Audience profile: load text from data/audience_profile.txt to inform selection priorities.
5. Selection criteria:
   - Prefer items with higher relevance_score (4-5 = high priority, 3 = medium, 0-2 = low).
   - Balance category representation: pick top 3 most populated/relevant categories.
   - Within each category, rank by relevance_score, then by recency (created_at).
   - Ensure diversity: avoid multiple items from same company unless exceptionally relevant.
6. Prompt: given candidate items with scores and categories, select up to 10 items distributed across up to 3 categories that maximize newsletter value for the audience.
7. Output: mapping of section names to list of item tuples, matching newsletter writer input format.
8. Schema: wrap response in BaseModel with fields:
   - section_items: dict[str, list[tuple[str, str]]] — maps category name to list of (long_summary, source_url) tuples.
   - selected_categories: list[str] (the 3 or fewer chosen sections).
   - selection_reasoning: str (brief explanation of choices).
9. Constraints: max 10 items total; max 3 categories; at least 1 item per selected category.
10. Output format example:
    ```python
    {
        "AI Engineering": [("Summary of first article...", "https://example.com/1"), ...],
        "Industry News": [("Summary of second article...", "https://example.com/2"), ...]
    }
    ```
11. Usage: output feeds directly into newsletter writer agent; section_items dict passed as-is to writer.
