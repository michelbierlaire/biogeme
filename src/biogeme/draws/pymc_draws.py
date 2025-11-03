from functools import partial

import pymc as pm

pymc_distributions = {
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


def get_distribution(name, default=None):
    """Return a PyMC continuous distribution class by name, ignoring case."""
    keymap = {k.lower(): v for k, v in pymc_distributions.items()}
    the_distribution = keymap.get(name.lower(), default)
    return partial(the_distribution)


def get_list_of_available_distributions():
    return list(pymc_distributions.keys())
