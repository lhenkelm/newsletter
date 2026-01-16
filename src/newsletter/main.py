"""Main entry point for the newsletter generator."""

from newsletter.ingest import load_recent_items


def main() -> None:
    """Main function to run the newsletter generator."""
    df = load_recent_items()
    print(f"Loaded {len(df)} recent items")


if __name__ == "__main__":
    main()
