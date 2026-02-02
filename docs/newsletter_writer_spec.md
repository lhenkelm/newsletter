# Newsletter Writer Agent
1. Goal: generate a polished Markdown newsletter from categorized items with summaries and links.
2. Framework: async pydantic-ai Agent('openai:gpt-4o-mini') invoked as final pipeline step.
3. Inputs: `SectionSelection.section_items` from the section compiler — a mapping of section categories to list of `SectionItem` objects.
   - `SectionItem` schema: `{ index: int, title: str, full_summary: str, source_url: str }`.
   - Example:
     ```python
     {
         "AI Engineering": [
             SectionItem(index=0, title="Building RAG Systems", full_summary="Full summary of article...", source_url="https://..."),
             ...
         ],
         "Industry News": [...]
     }
     ```
4. Audience profile: load text from data/audience_profile.txt to tailor tone and emphasis.
5. Newsletter structure (per assignment requirements):
   - **Title**: catchy weekly newsletter header with date range.
   - **Introduction**: short engaging paragraph (2-3 sentences) teasing key themes.
   - **Categorized sections**: one H2 section per category with:
     - Brief section intro (1 sentence).
     - Bulleted list of items, each with concise summary and inline Markdown link using item title.
   - **Closing section**: short wrap-up (2-3 sentences) with call-to-action or forward-looking note.
6. Prompt: given section→items mapping and audience profile, write a complete newsletter in Markdown with introduction, categorized sections containing summaries with links, and closing.
7. Output: raw Markdown string ready for file write.
8. Schema: wrap response in BaseModel with fields:
   - newsletter_markdown: str (the full Markdown content).
   - title: str (extracted newsletter title for logging).
9. Link formatting: each item summary must include inline link as `[item.title](item.source_url)`, using the title from the SectionItem.
10. Constraints:
    - Max ~150 words per section intro + items combined.
    - Keep summaries concise (1-2 sentences per item); distill `full_summary` down.
    - Maintain a consistent tone matched to the audience profile.
11. Async flow: `await writer_agent.write_newsletter(section_selection.section_items, profile)` as single call following pydantic-ai Agent patterns.
12. Usage: save returned newsletter_markdown to a configurable local path (default: `newsletter.md` in project root). Path can be set via `OUTPUT_PATH` environment variable.
