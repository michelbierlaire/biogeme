from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import TypeAlias

import pymc as pm
from pymc.distributions import Distribution

PyMcDistributionFactory: TypeAlias = Callable[[str], Distribution]

pymc_distributions: dict[str, PyMcDistributionFactory] = {
    "Cauchy": pm.Cauchy,  # Heavy-tailed distribution defined by location and scale
    "ChiSquared": pm.ChiSquared,  # Distribution of a sum of squared standard normal variables
    "Exponential": pm.Exponential,  # Memoryless distribution for positive values, rate parameter
    "Flat": pm.Flat,  # Improper uniform prior over all real numbers (no bounds)
    "Gumbel": pm.Gumbel,  # Distribution of the maximum of samples from exponential families
    "HalfCauchy": pm.HalfCauchy,  # Positive-only Cauchy, often used as prior for scale parameters
    "HalfFlat": pm.HalfFlat,  # Improper uniform prior on positive reals
    "HalfNormal": pm.HalfNormal,  # Positive part of a normal distribution
    "Logistic": pm.Logistic,  # S-shaped distribution, similar to normal but with heavier tails
    "LogNormal": pm.LogNormal,  # Distribution of a variable whose log is normal
    "Normal": pm.Normal,  # Gaussian distribution defined by mean and standard deviation
    "TruncatedNormal": pm.TruncatedNormal,  # Normal distribution limited to a given interval
    "Uniform": pm.Uniform,  # Proper uniform distribution between specified bounds
    "UniformSym": partial(pm.Uniform, lower=-1, upper=1),
    "Weibull": pm.Weibull,  # Flexible distribution for modeling lifetimes or failure times
}


def get_distribution(
    name: str, the_dict: dict[str, PyMcDistributionFactory]
) -> PyMcDistributionFactory:
    """Return a PyMC continuous distribution factory by name, ignoring case.

    The returned callable can be used like any PyMC distribution constructor,
    for example::

        dist = get_distribution("Normal")
        rv = dist("beta_time", mu=0.0, sigma=1.0)

    """
    keymap: dict[str, PyMcDistributionFactory] = {
        k.lower(): v for k, v in the_dict.items()
    }
    the_distribution = keymap.get(name.lower())
    if the_distribution is None:
        error_msg = (
            f"{name} is not a valid distribution. Available distributions are "
            f"{get_list_of_available_distributions()}"
        )
        raise ValueError(error_msg)

    return the_distribution


def get_list_of_available_distributions() -> list[str]:
    return list(pymc_distributions.keys())
