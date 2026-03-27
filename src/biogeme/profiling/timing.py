from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Generic, TypeVar

import jax

T = TypeVar("T")


def block_until_ready(value: Any) -> None:
    """Recursively block on JAX results so timings reflect actual execution."""
    if value is None:
        return
    if isinstance(value, tuple | list):
        for item in value:
            block_until_ready(item)
        return
    if isinstance(value, dict):
        for item in value.values():
            block_until_ready(item)
        return
    try:
        jax.block_until_ready(value)
    except (TypeError, AttributeError):
        # Non-JAX values do not need synchronization.
        return


def timed_call(
    function: Callable[..., T], *args: Any, **kwargs: Any
) -> tuple[T, float]:
    """Execute a callable, block until ready, and return result and elapsed time."""
    start = perf_counter()
    result = function(*args, **kwargs)
    block_until_ready(result)
    elapsed = perf_counter() - start
    return result, elapsed


@dataclass
class TimedBlock(AbstractContextManager["TimedBlock"], Generic[T]):
    """Simple context manager for wall-clock timing."""

    label: str | None = None
    elapsed: float = 0.0
    start_time: float = 0.0

    def __enter__(self) -> "TimedBlock":
        self.start_time = perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.elapsed = perf_counter() - self.start_time
        return None
