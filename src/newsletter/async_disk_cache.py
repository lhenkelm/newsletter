"""
Thin thread-based async wrapper around disk cache for storing LLM responses.
"""

from pathlib import Path
from typing import Any, Callable, Hashable, MutableMapping, Self, Type

from diskcache import Cache

import asyncio

from logfire import instrument


class AsyncDiskCache[K: Hashable, V]:
    """
    Asynchronous disk cache wrapper around a synchronous disk cache.

    This class provides async methods to get and set items in a disk cache,
    using asyncio locks to ensure safe concurrent access.

    There are no "public" fields; use the class methods and async methods.

    Usage:
        cache = await AsyncDiskCache.from_cache_dir_path(Path("/path/to/cache"))
        await cache.set_item("key", "value")
        await cache.contains("key")  # True
        await cache.get_item("miss")  # False
        value = await cache.get_item("key")
    """

    # sync access to _LOCK_BY_CANONICAL_PATH, globally
    # this is ok because only on creation
    _NEW_LOCK_LOCK = asyncio.Lock()
    # sync access to the caches on disk, per path
    _LOCKS_BY_CANONICAL_PATH: MutableMapping[Path, asyncio.Lock] = {}

    def __init__(self, cache_impl: Cache, lock=asyncio.Lock):
        """
        Initialize the async disk cache.

        Note: use `from_cache_dir_path` to create an instance.

        Args:
            cache_impl: Synchronous disk cache implementation.
            lock: Asyncio lock to use for synchronizing access. Must be held
                when accessing the cache_impl. Should have been registered in
                AsyncDiskCache._LOCKS_BY_CANONICAL_PATH.
        """
        self._cache_impl = cache_impl
        self._lock = lock

    @classmethod
    @instrument()
    async def from_cache_dir_path(cls: Type[Self], cache_dir_path: Path) -> Self:
        """
        Create an AsyncDiskCache instance from a cache directory path.

        This function is not thread-safe. It should only be called from a single
        thread at a time for a given cache_dir_path.

        Args:
            cache_dir_path: Path to the cache directory.

        Returns:
            An AsyncDiskCache instance.
        """
        cache_dir_path = await asyncio.to_thread(lambda: cache_dir_path.resolve())
        async with cls._NEW_LOCK_LOCK:
            if cache_dir_path in cls._LOCKS_BY_CANONICAL_PATH:
                lock = cls._LOCKS_BY_CANONICAL_PATH[cache_dir_path]
            else:
                lock = asyncio.Lock()
                cls._LOCKS_BY_CANONICAL_PATH[cache_dir_path] = lock
        async with lock:
            cache = await asyncio.to_thread(lambda: Cache(cache_dir_path))
        return cls(cache, lock)

    async def _locked_to_thread[R](self, func: Callable[[], R]) -> R:
        """
        Execute a function in a thread while holding the internal lock.

        Used to ensure thread-safe access to the cache implementation.
        """
        async with self._lock:
            return await asyncio.to_thread(func)

    @instrument()
    async def contains(self, key: K) -> bool:
        """
        Check if a key is in the cache.

        Equal to `key in cache`, but async.

        Args:
            key: Key to check.

        Returns:
            True if the key is in the cache, False otherwise.
        """
        return await self._locked_to_thread(lambda: key in self._cache_impl)

    @instrument()
    async def get_item(self, key: K) -> V:
        """
        Get an item from the cache.

        Note: this is equivalent to `cache[key]`, but async.
        Unlike `Mapping.get`, this will raise KeyError
        if the key is not found.

        Args:
            key: Key to retrieve.

        Returns:
            The value associated with the key.
        """
        return await self._locked_to_thread(lambda: self._cache_impl[key])

    @instrument()
    async def set_item(self, key: K, value: V) -> None:
        """
        Set an item in the cache.

        Note: this is equivalent to `cache[key] = value`, but async.

        Args:
            key: Key to set.
            value: Value to associate with the key.
        """
        await self._locked_to_thread(lambda: _set_item(self._cache_impl, key, value))


def _set_item(mapping: MutableMapping, key: Hashable, value: Any) -> None:
    """
    Set a key-value pair in a mutable mapping.

    Just a helper function to be called in a thread.
    """
    mapping[key] = value
