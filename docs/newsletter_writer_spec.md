# Newsletter Writer Agent
1. Goal: generate a polished Markdown newsletter from categorized items with summaries and links.
2. Framework: async pydantic-ai Agent('openai:gpt-4o-mini') invoked as final pipeline step.
3. Inputs: mapping of section categories to list of item tuples, each containing (long_summary: str, source_url: str).
   - Example: `{"AI Engineering": [("Summary of article...", "https://..."), ...], "Industry News": [...]}`.
4. Audience profile: load text from data/audience_profile.txt to tailor tone and emphasis.
5. Newsletter structure (per assignment requirements):
   - **Title**: catchy weekly newsletter header with date range.
   - **Introduction**: short engaging paragraph (2-3 sentences) teasing key themes.
   - **Categorized sections**: one H2 section per category with:
     - Brief section intro (1 sentence).
     - Bulleted list of items, each with concise summary and inline Markdown link.
   - **Closing section**: short wrap-up (2-3 sentences) with call-to-action or forward-looking note.
6. Prompt: given section→items mapping and audience profile, write a complete newsletter in Markdown with introduction, categorized sections containing summaries with links, and closing.
7. Output: raw Markdown string ready for file write.
8. Schema: wrap response in BaseModel with fields:
   - newsletter_markdown: str (the full Markdown content).
   - title: str (extracted newsletter title for logging).
9. Link formatting: each item summary must include inline link as `[Title or phrase](source_url)`.
10. Constraints:
    - Max ~150 words per section intro + items combined.
    - Keep summaries concise (1-2 sentences per item).
    - Maintain a consistent tone matched to the audience profile.
11. Async flow: await agent.write_newsletter(section_items, profile) as single call.
12. Usage: save returned newsletter_markdown to `newsletter.md` in project root.
