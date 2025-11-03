"""
Sampling using MCMC.

Michel Bierlaire
Mon Oct 27 2025, 17:04:31
"""

from __future__ import annotations

import logging
from typing import Any

from .sampling_strategy import SamplerPlan, SamplerPlanner

logger = logging.getLogger(__name__)


def choose_plan(
    *,
    chains: int,
    planner: SamplerPlanner | None,
    plan: SamplerPlan | str | dict[str, Any] | None,
) -> SamplerPlan:
    """Resolve the effective sampling plan from user input or auto planner."""
    if plan is None:
        if planner is None:
            raise ValueError("planner must be provided when plan=None (auto mode).")
        eff = planner.plan(chains=int(chains))
    else:
        eff = SamplerPlan.from_user_plan(plan)

        # Minimal enrichment for PyMC cores (common expectation: cores=chains)
        if eff.backend == "pymc" and eff.cores is None:
            eff = SamplerPlan(
                "pymc", None, int(chains), note=eff.note or "user-defined cores=chains"
            )

    logger.info(
        "Sampler plan resolved: backend=%s, chain_method=%s, cores=%s. %s",
        eff.backend,
        eff.chain_method,
        eff.cores,
        eff.note,
    )
    return eff


def run_sampling(
    *,
    model: Any,
    draws: int,
    tune: int,
    chains: int,
    target_accept: float,
    planner: SamplerPlanner | None = None,
    plan: SamplerPlan | str | dict[str, Any] | None = None,
):
    """Execute sampling according to a plan. Returns (idata, used_numpyro: bool).

    Parameters
    ----------
    model : Any
        A PyMC model instance (or context-managed model).
    draws : int
        Number of posterior draws.
    tune : int
        Number of warmup/adaptation steps.
    chains : int
        Number of MCMC chains.
    target_accept : float
        NUTS target accept probability.
    planner : SamplerPlanner | None
        Required when `plan is None` (auto mode). Ignored if `plan` is provided.
    plan : SamplerPlan | str | dict[str, Any] | None
        Optional user plan. If None, uses `planner.plan(chains)`.

    Returns
    -------
    (idata, used_numpyro: bool)
    """
    eff_plan = choose_plan(chains=chains, planner=planner, plan=plan)

    if eff_plan.backend == "numpyro":
        # Import lazily so environments without JAX/NumPyro can still import this module
        from pymc.sampling.jax import sample_numpyro_nuts

        idata = sample_numpyro_nuts(
            model=model,
            draws=draws,
            tune=tune,
            chains=chains,
            chain_method=eff_plan.chain_method,  # 'vectorized' or 'parallel'
            target_accept=target_accept,
        )
        return idata, True

    # PyMC multiprocessing path
    import pymc as pm

    idata = pm.sample(
        model=model,
        draws=draws,
        tune=tune,
        chains=chains,
        cores=eff_plan.cores if eff_plan.cores is not None else 1,
        target_accept=target_accept,
    )
    return idata, False
