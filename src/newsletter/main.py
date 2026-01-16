"""Main entry point for the newsletter generator."""

from newsletter import config
from newsletter.ingest import load_recent_items


def main() -> None:
    """Main function to run the newsletter generator."""
    df = load_recent_items(
        cutoff_days=config.CUTOFF_DAYS,
        min_items=config.MIN_ITEMS,
        max_items=config.MAX_ITEMS,
        source_uri=config.SOURCE_URI,
    )
    print(f"Loaded {len(df)} recent items")


if __name__ == "__main__":
    main()
