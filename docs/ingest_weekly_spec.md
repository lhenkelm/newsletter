# MVP Spec: weekly ingest
- Function: `load_recent_items(cutoff_days: int = 7, min_items: int = 3, max_items: int = 30) -> pl.DataFrame`.
- Source: `pl.scan_parquet("hf://datasets/zenml/llmops-database/data/train-00000-of-00001.parquet")`.
- Filter: keep rows where `created_at >= (now_utc - cutoff_days)`; parse/convert `created_at` to datetime if needed.
- Warning: if row count after filter is `< min_items` or `> max_items`, emit a warning with the count.
- place it in src/newsletter/ingest.py. 