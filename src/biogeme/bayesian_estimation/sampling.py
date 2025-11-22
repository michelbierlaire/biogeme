"""
Sampling using MCMC with simplified configuration.

Michel Bierlaire
Mon Oct 27 2025, 17:04:31
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .sampling_strategy import SamplingConfig

logger = logging.getLogger(__name__)


def run_sampling(
    *,
    model: Any,
    draws: int,
    tune: int,
    chains: int,
    config: SamplingConfig,
    starting_values: dict[str, float] | None = None,
):
    """Execute sampling according to a simplified configuration. Returns (idata, used_numpyro: bool)."""
    if config.backend == "numpyro":
        # Import lazily so environments without JAX/NumPyro can still import this module
        from pymc.sampling.jax import sample_numpyro_nuts

        kwargs = {
            "model": model,
            "draws": draws,
            "tune": tune,
            "chains": chains,
            "target_accept": config.target_accept,
        }
        if config.chain_method is not None:
            kwargs["chain_method"] = config.chain_method
        if config.nuts_kwargs is not None:
            kwargs["nuts_kwargs"] = config.nuts_kwargs

        # If starting values are provided, jitter them per chain and pass as initvals
        if starting_values:

            def jitter(
                values: dict[str, float], scale: float = 0.01
            ) -> dict[str, float]:
                return {k: v + scale * np.random.randn() for k, v in values.items()}

            initvals = [jitter(starting_values) for _ in range(chains)]
            kwargs["initvals"] = initvals

        idata = sample_numpyro_nuts(**kwargs)
        return idata, True

    # PyMC multiprocessing path
    import pymc as pm

    kwargs = {
        "model": model,
        "draws": draws,
        "tune": tune,
        "chains": chains,
        "cores": config.cores if config.cores is not None else 1,
        "target_accept": config.target_accept,
    }
    if config.init is not None:
        kwargs["init"] = config.init
    if config.max_treedepth is not None:
        kwargs["max_treedepth"] = config.max_treedepth

    # If starting values are provided, jitter them per chain and pass as initvals
    if starting_values:

        def jitter(values: dict[str, float], scale: float = 0.01) -> dict[str, float]:
            return {k: v + scale * np.random.randn() for k, v in values.items()}

        initvals = [jitter(starting_values) for _ in range(chains)]
        kwargs["initvals"] = initvals

    idata = pm.sample(**kwargs)
    return idata, False
