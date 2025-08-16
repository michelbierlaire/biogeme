"""Decorators for deprecated functions and parameters

Michel Bierlaire
Mon Jun 17 20:50:19 2024
"""

import functools
import inspect
import logging
import os
from typing import Any, Callable

from biogeme.exceptions import BiogemeError

logger = logging.getLogger(__name__)

RAISE_EXCEPTION = False

if os.environ.get('TOX_ENV_NAME') is not None and RAISE_EXCEPTION:
    raise Exception(
        "Exception raised during testing with tox. Remove the exception raised by the deprecated functions"
    )


def deprecated(
    new_func: Callable[..., Any],
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator for deprecated functions. It calls the new version of the function

    :param new_func:
    :return:
    """

    def decorator(old_func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(old_func)
        def wrapper(*args, **kwargs) -> Any:
            msg = f"{old_func.__name__} is deprecated; use {new_func.__name__} instead."
            if RAISE_EXCEPTION:
                raise BiogemeError('Deprecated')
            logger.warning(msg)
            return new_func(*args, **kwargs)  # Directly call the new function

        wrapper.__deprecated__ = True
        wrapper.__newname__ = new_func.__name__
        return wrapper

    return decorator


def deprecated_parameters(obsolete_params: dict[str, str | None]):

    def decorator(func):
        func_signature = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            processed_kwargs = {}
            for name, value in list(kwargs.items()):
                if name in obsolete_params:
                    new_name = obsolete_params[name]
                    if new_name:
                        msg = f"Parameter '{name}' is deprecated; use '{new_name}={value}' instead."
                        logger.warning(msg)
                        processed_kwargs[new_name] = value
                    else:
                        msg = (
                            f"Parameter '{name}' is deprecated and is ignored. "
                            f"It will be removed in a future version.",
                        )
                        logger.warning(msg)
                else:
                    processed_kwargs[name] = value

            return func(*args, **processed_kwargs)

        return wrapper

    return decorator


def issue_deprecation_warning(text: str) -> None:
    logger.warning(text)
