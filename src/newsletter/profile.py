"""
Audience profiles are used for personalization and to better fit a given use-case

They are stored on disk and loaded at pipeline initialisation.
This module contains re-used code for handling audience profiles.
"""

import asyncio
from logging import getLogger
from pathlib import Path

from async_lru import alru_cache

_LOGGER = getLogger(__name__)


@alru_cache
async def load_audience_profile(
    profile_path: str | Path = "data/audience_profile.txt",
) -> str:
    """Load audience profile from file.

    File contents are cached per-event-loop, since multiple agent instances may need to load them.

    Args:
        profile_path: Path to the audience profile text file.

    Returns:
        Audience profile description (text string).
    """
    _LOGGER.debug(f"loading profile from {profile_path=!r}")
    path = Path(profile_path)
    profile = await asyncio.to_thread(lambda: path.read_text())
    profile = profile.strip()
    _LOGGER.debug(f"loaded {profile=!r}")
    return profile
