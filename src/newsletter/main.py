"""Main entry point for the newsletter generator."""

from newsletter.config import IngestConfig
from newsletter.ingest import load_recent_items


def main() -> None:
    """Main function to run the newsletter generator."""
    config = IngestConfig.from_env()
    df = load_recent_items(
        cutoff_days=config.cutoff_days,
        min_items=config.min_items,
        max_items=config.max_items,
        source_uri=config.source_uri,
    )
    print(f"Loaded {len(df)} recent items")


if __name__ == "__main__":
    main()
