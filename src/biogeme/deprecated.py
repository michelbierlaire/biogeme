import warnings
import functools
from typing import Callable, Any


def deprecated(
    new_func: Callable[..., Any]
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(old_func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(old_func)
        def wrapper(*args, **kwargs) -> Any:
            msg = f"{old_func.__name__} is deprecated; use {new_func.__name__} instead."
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            return new_func(*args, **kwargs)  # Directly call the new function

        return wrapper

    return decorator
