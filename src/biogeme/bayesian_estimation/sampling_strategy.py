"""
Defines the strategy for sampling the MCMC based on hardware configuration and user's preferences.

Michel Bierlaire
Mon Oct 27 2025, 17:01:13

"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


SAMPLER_STRATEGIES_DESCRIPTION = {
    "numpyro-parallel": "one chain per device",
    "numpyro-vectorized": "all chains on one device",
    "pymc": "default PyMC sampler on CPU",
}


def describe_strategies() -> str:
    return ', '.join(
        [f"'{name}' ({desc})" for name, desc in SAMPLER_STRATEGIES_DESCRIPTION.items()]
    )


@dataclass(frozen=True)
class SamplingConfig:
    backend: str
    chain_method: str | None
    cores: int | None
    target_accept: float
    init: str | None
    max_treedepth: int | None
    nuts_kwargs: dict[str, Any] | None


# ------------------------- Hardware helpers (single responsibility) -------------------------


def _jax_available() -> bool:
    """
    Return whether JAX/NumPyro sampling is importable.

    :returns: ``True`` if both JAX and NumPyro sampling can be imported, ``False`` otherwise.
    :rtype: bool
    """
    try:
        import jax  # noqa: F401
        from pymc.sampling.jax import sample_numpyro_nuts  # noqa: F401
        import numpyro  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        return False
    return True


def _jax_devices_summary() -> tuple[int, list[str]]:
    """
    Get a summary of available JAX devices.

    :returns: A pair ``(n_devices, platforms)`` where ``n_devices`` is the number of devices detected and ``platforms`` is the list of platform names.
    :rtype: tuple[int, list[str]]
    """
    try:
        import jax
    except (ImportError, ModuleNotFoundError):
        return 0, []
    devices = jax.devices()
    n_dev = len(devices)
    platforms = sorted({getattr(d, "platform", "unknown") for d in devices})
    return n_dev, platforms


def _cpu_core_count() -> int:
    """
    Best-effort CPU core count.

    :returns: The detected CPU core count, falling back to ``1`` if it cannot be determined.
    :rtype: int
    """
    try:
        import os

        return os.cpu_count() or 1
    except ImportError:
        return 1


def _select_chain_method(n_dev: int) -> str:
    """
    Select a NumPyro chain method given the number of devices.

    :param n_dev: Number of available JAX devices.
    :type n_dev: int
    :returns: ``'parallel'`` if more than one device is available, otherwise ``'vectorized'``.
    :rtype: str
    """
    return "parallel" if n_dev > 1 else "vectorized"


# ------------------------- Public factory -------------------------


def make_sampling_config(
    strategy: str,
    target_accept: float,
) -> SamplingConfig:
    """
    Create a :class:`SamplingConfig` from a short strategy string.

    :param strategy: One of ``'automatic'``, ``'numpyro-parallel'``, ``'numpyro-vectorized'``, or ``'pymc'``.
    :type strategy: str
    :param target_accept: Target acceptance rate.
    :type target_accept: float
    :returns: A ready-to-use configuration object.
    :rtype: SamplingConfig
    :raises ValueError: If ``strategy`` is not one of the allowed values.
    """
    key = (strategy or "").strip().lower()

    if key == "automatic":
        # Prefer JAX/NumPyro when available. If multiple devices → parallel chains; else vectorized.
        if _jax_available():
            n_dev, platforms = _jax_devices_summary()
            method = _select_chain_method(n_dev if n_dev else 1)
            logger.info(
                "Auto sampling: JAX available (devices=%s, platforms=%s) → numpyro/%s",
                n_dev,
                ",".join(platforms) if platforms else "unknown",
                method,
            )
            return SamplingConfig(
                backend="numpyro",
                chain_method=method,
                cores=None,
                target_accept=target_accept,
                init=None,
                max_treedepth=None,
                nuts_kwargs=None,
            )
        # Fallback to PyMC with a sensible cores default
        cores_count = _cpu_core_count()
        logger.info("Auto sampling: JAX not available → PyMC (cores=%s)", cores_count)
        return SamplingConfig(
            backend="pymc",
            chain_method=None,
            cores=cores_count,
            target_accept=target_accept,
            init=None,
            max_treedepth=None,
            nuts_kwargs=None,
        )

    if key == "numpyro-parallel":
        if not _jax_available():
            logger.warning(
                "Requested numpyro-parallel but JAX/NumPyro not available; falling back to PyMC."
            )
            return SamplingConfig(
                backend="pymc",
                chain_method=None,
                cores=_cpu_core_count(),
                target_accept=target_accept,
                init=None,
                max_treedepth=None,
                nuts_kwargs=None,
            )
        return SamplingConfig(
            backend="numpyro",
            chain_method="parallel",
            cores=None,
            target_accept=target_accept,
            init=None,
            max_treedepth=None,
            nuts_kwargs=None,
        )

    if key == "numpyro-vectorized":
        if not _jax_available():
            logger.warning(
                "Requested numpyro-vectorized but JAX/NumPyro not available; falling back to PyMC."
            )
            return SamplingConfig(
                backend="pymc",
                chain_method=None,
                cores=_cpu_core_count(),
                target_accept=target_accept,
                init=None,
                max_treedepth=None,
                nuts_kwargs=None,
            )
        return SamplingConfig(
            backend="numpyro",
            chain_method="vectorized",
            cores=None,
            target_accept=target_accept,
            init=None,
            max_treedepth=None,
            nuts_kwargs=None,
        )

    if key == "pymc":
        return SamplingConfig(
            backend="pymc",
            chain_method=None,
            cores=_cpu_core_count(),
            target_accept=target_accept,
            init=None,
            max_treedepth=None,
            nuts_kwargs=None,
        )

    raise ValueError(
        f"Unknown sampling strategy: {strategy!r}. Allowed values are: 'automatic', 'numpyro-parallel', 'numpyro-vectorized', 'pymc'."
    )
