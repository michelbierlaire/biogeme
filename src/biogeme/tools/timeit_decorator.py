import functools
import inspect
import logging
import time
from typing import Any, Callable, TypeVar, cast

from biogeme.tools import format_elapsed_time

F = TypeVar("F", bound=Callable[..., Any])


def timeit_old(
    logger: logging.Logger | None = None,
    level: int = logging.INFO,
    label: str | None = None,
    threshold_ms: float = 10.0,  # <── new parameter (default: 10 ms)
) -> Callable[[F], F]:
    """
    Decorator factory that measures wall-clock time of a function and reports it
    only if the elapsed time exceeds a threshold.

    Args:
        logger: Logger to use (default: None -> print).
        level: Logging level if logger is provided.
        label: Optional label to override function.__qualname__ in the message.
        threshold_ms: Minimum duration (in milliseconds) to report (default: 10.0).

    Usage:
        @timeit()                         # prints if >10 ms
        @timeit(threshold_ms=1.0)         # prints if >1 ms
        @timeit(logging.getLogger())      # logs instead of printing
        @timeit(label="my_step")          # custom label
    """

    def _decorator(func: F) -> F:

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                name = label or func.__qualname__
                start = time.perf_counter()
                try:
                    return await func(*args, **kwargs)
                finally:
                    elapsed = (time.perf_counter() - start) * 1000.0  # ms
                    if elapsed >= threshold_ms:
                        msg = f"{name} finished in {format_elapsed_time(elapsed)}"
                        if logger:
                            logger.log(level, msg)
                        else:
                            print(msg)

            return cast(F, wrapper)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            name = label or func.__qualname__
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = (time.perf_counter() - start) * 1000.0  # ms
                if elapsed >= threshold_ms:
                    msg = f"{name} finished in {format_elapsed_time(elapsed)}"
                    if logger:
                        logger.log(level, msg)
                    else:
                        print(msg)

        return cast(F, wrapper)

    return _decorator
